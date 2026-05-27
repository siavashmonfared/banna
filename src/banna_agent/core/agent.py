"""The transition-function driver.

`run_policy(state, policy, llm, tools, log)` runs the inner loop until:
  - the policy proposes FINAL_ANSWER, or
  - the budget trips (wall / steps / tokens / cost), or
  - the driver hits a non-recoverable error.

This is the level-1 loop described in the project plan. All policies
(ReAct, verifier_retry, best_first) share it. Level-2 search procedures
(best-first over plans, MCTS) *wrap* this function — they don't rewrite it.
"""
from __future__ import annotations

import time as _time
from typing import Any

from ..llm.base import LLMClient
from ..tools.base import ToolRegistry, invoke_tool
from .budget import BudgetTracker
from .events import EventKind, EventLog, emit
from .state import AgentState
from .types import (
    Action,
    ActionKind,
    BudgetReason,
    Observation,
)
from .user_io import UserIO


# Tools whose risk is high enough to require a per-call confirmation
# when a UserIO is supplied. Pass 1 of permissions only gates
# `run_shell` — everything else still runs through unchanged. Pass 2
# extends this to `run_python` / `write_file` / `read_url` once the
# UX shape has settled.
_GATED_TOOLS: dict[str, str] = {
    "run_shell": "exec",
}

# Max backoff-retries for a *retryable* provider error (rate limit /
# transient 5xx) on a single propose() call. Waits are 2,4,8,16s (cap
# 30) and are charged off the task wall budget (see the propose loop).
_MAX_LLM_RETRIES: int = 4


def _call_signature(tool_name: str, args: dict[str, Any]) -> str:
    """Stable hash for an (allow_always)-able call signature."""
    import hashlib
    import json
    try:
        canon = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        canon = repr(sorted(args.items()) if isinstance(args, dict) else args)
    return f"{tool_name}:{hashlib.sha256(canon.encode()).hexdigest()[:16]}"


# Tools that must ALWAYS execute, even on an identical repeat call:
# side-effecting (run_shell / run_python / memory / plan), terminal
# (final_answer / ask_user), or stateful — browser_* depend on the
# current page, so the same args at different times mean different things.
# Every other tool is a pure read of a stable resource within one task
# run (search / read_url / read_file / pdf_* / xlsx_* / calculator), so a
# byte-identical repeat returns the same thing. We skip those repeats and
# nudge the model instead of re-paying the latency — this is the anti-loop
# guard for the L3 search-loopers (24 searches across 2 unique queries).
_NO_DEDUP_TOOLS: frozenset[str] = frozenset({
    "run_shell", "run_python", "final_answer", "ask_user", "plan", "memory",
    "browser_open", "browser_view", "browser_find", "browser_next",
    "browser_click", "browser_back",
})


def _dup_lookup(state: AgentState, name: str, args: dict[str, Any]) -> str | None:
    """Return a one-line preview of a prior identical call, or None.

    None when the tool isn't dedup-eligible, the call is novel, or no
    cache exists yet — i.e. "go ahead and execute".
    """
    if name in _NO_DEDUP_TOOLS:
        return None
    cache: dict[str, str] | None = getattr(state, "_call_cache", None)
    if not cache:
        return None
    return cache.get(_call_signature(name, args))


def _dup_record(state: AgentState, name: str, args: dict[str, Any],
                preview: str) -> None:
    """Remember an executed call so a later identical one is skipped."""
    if name in _NO_DEDUP_TOOLS:
        return
    cache: dict[str, str] | None = getattr(state, "_call_cache", None)
    if cache is None:
        cache = {}
        state._call_cache = cache  # type: ignore[attr-defined]
    cache.setdefault(_call_signature(name, args), preview or "(no preview)")


def _duplicate_observation(name: str, preview: str) -> "Observation":
    """The synthetic, model-visible result for a skipped duplicate call."""
    note = (
        f"DUPLICATE CALL SKIPPED. You already called `{name}` with these "
        f"exact arguments earlier in this task, so it was not run again. "
        f"The previous result was: {preview or '(no preview)'}. "
        f"Do not repeat it — use that result, call `{name}` with different "
        f"arguments, or commit your answer with final_answer."
    )
    return Observation(
        ok=True,
        data={"duplicate_skipped": True, "note": note,
              "previous_result": preview},
        wall_s=0.0,
    )


def _brief_tool_result(result: Any, ok: bool) -> str:
    """One-line preview of a tool result for live display.

    Picks a useful snippet without the noise of a full dict dump:
    n hits / n results / first text / stdout. Returns "" when nothing
    useful is extractable so the display can fall back to "(ok)".
    """
    if not ok:
        return ""
    if isinstance(result, dict):
        # If the inner tool flagged its own failure (e.g. run_python with
        # a non-zero return code), surface its error summary first so the
        # live display doesn't show "ok" for a script that actually
        # crashed.
        if result.get("ok") is False and result.get("error"):
            return f"error: {str(result['error'])[:120]}"
        for key, fmt in (
            ("hits", lambda v: f"{len(v)} hits" if isinstance(v, list) else ""),
            ("results", lambda v: f"{len(v)} results" if isinstance(v, list) else ""),
            ("value", lambda v: f"value: {str(v)[:80]}"),
            ("summary", lambda v: f"'{str(v)[:80]}'" if v else ""),
            ("text", lambda v: f"'{str(v)[:80]}'" if v else ""),
            ("stdout", lambda v: f"stdout: '{str(v)[:80]}'" if v else ""),
            ("answer", lambda v: f"answer: {str(v)[:80]}"),
        ):
            if key in result and result[key]:
                preview = fmt(result[key])
                if preview:
                    return preview
    if isinstance(result, str) and result.strip():
        return f"'{result[:80]}'"
    return ""


def _execute(
    state: AgentState,
    action: Action,
    tools: ToolRegistry,
    log: EventLog | None,
    user_io: UserIO | None = None,
    budget: "BudgetTracker | None" = None,
) -> Observation:
    """Run one Action; return the Observation.

    For THINK actions the observation is trivial. For TOOL_CALL the
    handler is dispatched through `tools`; exceptions are caught and
    converted to an error observation (the agent loop never dies).
    For FINAL_ANSWER the observation mirrors the answer and marks
    readiness to terminate.

    For ASK_USER, the loop blocks on `user_io.ask(...)` when a UserIO
    is provided; in batch mode (user_io=None) the action degrades to
    a synthetic THINK so the trace is comparable to a non-interactive
    run.
    """
    # Lift any LLM token usage the policy attached to action.meta onto the
    # Observation. Policies record `tokens_in/out` from their LLM call's
    # Usage object; without this lift, the budget tracker sees 0 every
    # tick because Observations are otherwise built fresh in this function.
    _t_in = int(action.meta.get("tokens_in") or 0)
    _t_out = int(action.meta.get("tokens_out") or 0)

    if action.kind == ActionKind.THINK:
        return Observation(ok=True, text=action.text, wall_s=0.0,
                           tokens_in=_t_in, tokens_out=_t_out)

    if action.kind == ActionKind.FINAL_ANSWER:
        return Observation(ok=True, text=action.answer, wall_s=0.0,
                           tokens_in=_t_in, tokens_out=_t_out)

    if action.kind == ActionKind.ASK_USER:
        return _execute_ask_user(state, action, log, user_io,
                                 t_in=_t_in, t_out=_t_out, budget=budget)

    if action.kind == ActionKind.TOOL_CALL:
        name = action.tool_name or ""
        # Permission gate (Pass 1): only `run_shell` is gated, and
        # only when a UserIO is attached. Batch / GAIA runs (no
        # UserIO) bypass entirely so existing eval traces don't shift.
        if user_io is not None and name in _GATED_TOOLS:
            decision_obs = _gate_tool_call(
                state, name, action.tool_args, user_io, log,
                step_idx=len(state.trace.steps), budget=budget,
            )
            if decision_obs is not None:
                # Denied; synthetic error Observation returned. Don't
                # invoke the tool.
                return decision_obs
        tool = tools.get(name)
        step_idx = len(state.trace.steps)
        emit(
            log,
            run_id=state.trace.run_id,
            step=step_idx,
            kind=EventKind.TOOL_CALL,
            tool_name=name,
            arguments=action.tool_args,
        )
        if tool is None:
            available = ", ".join(sorted(tools.names())) or "(none)"
            err = (
                f"unknown tool: {name!r}. "
                f"This tool is not registered. Available tools: {available}. "
                f"Pick one of the available tools, or call `final_answer` "
                f"if you have enough information to answer."
            )
            obs = Observation(ok=False, error=err, data={"error": err})
            emit(log, run_id=state.trace.run_id, step=step_idx,
                 kind=EventKind.TOOL_RESULT, ok=False, error=obs.error,
                 preview="")
            return obs
        # Anti-loop: skip a byte-identical repeat of a dedup-eligible
        # tool call and hand the model back its prior result instead.
        dup = _dup_lookup(state, name, action.tool_args)
        if dup is not None:
            emit(log, run_id=state.trace.run_id, step=step_idx,
                 kind=EventKind.TOOL_RESULT, ok=True,
                 preview=f"duplicate-skipped ({dup})", duplicate_skipped=True)
            return _duplicate_observation(name, dup)
        # Capture evidence count BEFORE the tool runs so the
        # auto-registration delta can be reported live.
        state._ev_before_tool = len(state.evidence)  # type: ignore[attr-defined]
        inv = invoke_tool(tool, action.tool_args)
        _dup_record(state, name, action.tool_args,
                    _brief_tool_result(inv.result, inv.ok))
        emit(
            log,
            run_id=state.trace.run_id,
            step=step_idx,
            kind=EventKind.TOOL_RESULT,
            ok=inv.ok,
            wall_s=inv.wall_s,
            error=inv.error,
            preview=_brief_tool_result(inv.result, inv.ok),
            evidence_before=getattr(state, "_ev_before_tool", len(state.evidence)),
        )
        obs = Observation(
            ok=inv.ok,
            data=inv.result if isinstance(inv.result, dict) else {"result": inv.result},
            error=inv.error,
            wall_s=inv.wall_s,
            tokens_in=_t_in,
            tokens_out=_t_out,
        )
        # If the tool returned grounded info, auto-register evidence and
        # inject the assigned evidence_id back into the result dict so
        # the model can cite it on its next `final_answer` call (Phase 7
        # citation grounding). Without the round-trip, the model has no
        # way to refer to a specific piece of evidence it just saw.
        if inv.ok and isinstance(inv.result, dict):
            hits = inv.result.get("hits")
            if isinstance(hits, list):
                for h in hits:
                    if isinstance(h, dict) and h.get("url"):
                        ev = state.add_evidence(
                            source=h["url"],
                            content=h.get("snippet") or h.get("title") or "",
                            origin_step=step_idx,
                            meta={"tool": name, "title": h.get("title", "")},
                        )
                        h["evidence_id"] = ev.evidence_id
            elif inv.result.get("url"):
                ev = state.add_evidence(
                    source=inv.result["url"],
                    content=(inv.result.get("title") or "") + "\n" + (inv.result.get("text") or "")[:1000],
                    origin_step=step_idx,
                    meta={"tool": name},
                )
                inv.result["evidence_id"] = ev.evidence_id
        return obs

    if action.kind == ActionKind.TOOL_BATCH:
        return _execute_batch(state, action, tools, log, user_io, budget=budget)

    raise ValueError(f"unknown ActionKind: {action.kind}")


def _execute_ask_user(
    state: AgentState,
    action: Action,
    log: EventLog | None,
    user_io: UserIO | None,
    *,
    t_in: int,
    t_out: int,
    budget: "BudgetTracker | None" = None,
) -> Observation:
    """Block on a user response, or degrade to a synthetic THINK in batch mode.

    The question lives in ``action.text``. The user's reply becomes the
    Observation's ``text`` and is appended to ``state.metadata['user_replies']``
    so the policy can fold it into the next prompt as a user-role turn.
    """
    question = (action.text or "").strip()
    step_idx = len(state.trace.steps)
    if user_io is None:
        # Batch mode: degrade to a marker THINK so the trace is
        # comparable to a non-interactive run. The model sees the
        # marker on its next tick and should commit a best-guess
        # answer (the system prompt for react+ instructs this).
        marker = "(no user available — proceeding with best guess)"
        emit(log, run_id=state.trace.run_id, step=step_idx,
             kind=EventKind.ASK_USER, question=question, answered=False,
             batch_mode=True)
        return Observation(ok=True, text=marker, wall_s=0.0,
                           tokens_in=t_in, tokens_out=t_out)

    emit(log, run_id=state.trace.run_id, step=step_idx,
         kind=EventKind.ASK_USER, question=question, answered=False,
         batch_mode=False)
    # Human think-time must not count against the agent's wall budget,
    # which bounds *agent* compute. Pause the timer across the blocking
    # ask() and resume once we have a reply.
    if budget is not None:
        budget.pause()
    try:
        reply = user_io.ask(question)
    except (EOFError, KeyboardInterrupt):
        reply = ""
    finally:
        if budget is not None:
            budget.resume()
    reply = (reply or "").strip()
    # Stash for the policy to fold into the prompt.
    state.metadata.setdefault("user_replies", []).append({
        "step": step_idx,
        "question": question,
        "reply": reply,
    })
    emit(log, run_id=state.trace.run_id, step=step_idx,
         kind=EventKind.ASK_USER, question=question, answered=True,
         reply_chars=len(reply))
    return Observation(
        ok=bool(reply),
        text=reply or "(no reply)",
        data={"question": question, "reply": reply},
        wall_s=0.0,
        tokens_in=t_in,
        tokens_out=t_out,
    )


def _gate_tool_call(
    state: AgentState,
    tool_name: str,
    args: dict[str, Any],
    user_io: UserIO,
    log: EventLog | None,
    *,
    step_idx: int,
    budget: "BudgetTracker | None" = None,
) -> Observation | None:
    """Ask the user whether this call is allowed. Returns:

    * None — call is approved, caller should proceed with the tool invocation.
    * Observation(ok=False) — call was denied; caller skips invocation
      and surfaces the synthetic error to the policy.

    "allow_always" memoizes the call signature in
    ``state.metadata['allowed_call_sigs']`` so the user isn't pestered
    with the same prompt repeatedly in one session.
    """
    sig = _call_signature(tool_name, args or {})
    allowed = state.metadata.setdefault("allowed_call_sigs", set())
    if sig in allowed:
        emit(log, run_id=state.trace.run_id, step=step_idx,
             kind=EventKind.PERMISSION, tool_name=tool_name,
             decision="allow_always_remembered", sig=sig)
        return None
    risk = _GATED_TOOLS.get(tool_name, "exec")
    # Don't bill the user's decision time against the agent wall budget.
    if budget is not None:
        budget.pause()
    try:
        decision = user_io.confirm(tool_name=tool_name, args=args or {}, risk=risk)
    except (EOFError, KeyboardInterrupt):
        decision = "deny"
    finally:
        if budget is not None:
            budget.resume()
    emit(log, run_id=state.trace.run_id, step=step_idx,
         kind=EventKind.PERMISSION, tool_name=tool_name,
         decision=decision, sig=sig)
    if decision == "allow_always":
        allowed.add(sig)
        return None
    if decision == "allow_once":
        return None
    # Deny: synthesize an error Observation the policy will see as a
    # tool failure. The model can then pick another tool, ask the user
    # to relax the restriction, or commit a "can't do it" final answer.
    err = (
        f"user denied tool call: {tool_name}. "
        f"You do not have permission to run this tool with these args. "
        f"Pick a different tool, ask the user with `ask_user` whether to "
        f"proceed differently, or commit a final answer explaining why "
        f"you can't complete the request."
    )
    return Observation(ok=False, error=err, data={"error": err})


def _auto_register_evidence(
    state: AgentState, tool_name: str, result: Any, step_idx: int,
) -> None:
    """Mirror of the single-tool evidence auto-registration, factored out
    so the batch dispatcher can call it per sub-result.
    """
    if not isinstance(result, dict):
        return
    hits = result.get("hits")
    if isinstance(hits, list):
        for h in hits:
            if isinstance(h, dict) and h.get("url"):
                ev = state.add_evidence(
                    source=h["url"],
                    content=h.get("snippet") or h.get("title") or "",
                    origin_step=step_idx,
                    meta={"tool": tool_name, "title": h.get("title", "")},
                )
                h["evidence_id"] = ev.evidence_id
        return
    if result.get("url"):
        ev = state.add_evidence(
            source=result["url"],
            content=(result.get("title") or "") + "\n" + (result.get("text") or "")[:1000],
            origin_step=step_idx,
            meta={"tool": tool_name},
        )
        result["evidence_id"] = ev.evidence_id


def _execute_batch(
    state: AgentState,
    action: Action,
    tools: ToolRegistry,
    log: EventLog | None,
    user_io: UserIO | None = None,
    budget: "BudgetTracker | None" = None,
) -> Observation:
    """Dispatch a TOOL_BATCH action: run all sub-calls in parallel.

    Sub-calls live in ``action.meta["batch_calls"]`` as a list of
    ``{"name": str, "args": dict}``. We use a small ThreadPoolExecutor —
    each tool call is usually a network round-trip, so threads are the
    right concurrency primitive (the GIL is released around I/O).

    Returns one combined Observation:
      - ``ok`` = all sub-calls succeeded
      - ``data["batch"]`` = list of {name, ok, result, error, wall_s}
      - ``wall_s`` = max of sub-call wall times (wall-clock, not summed)

    Each sub-call still emits its own TOOL_CALL + TOOL_RESULT event so
    existing log consumers / display code keep working unchanged. The
    new ``TOOL_BATCH`` event brackets the group.
    """
    from concurrent.futures import ThreadPoolExecutor

    raw_calls = (action.meta or {}).get("batch_calls") or []
    calls: list[dict[str, Any]] = []
    for c in raw_calls:
        if isinstance(c, dict) and c.get("name"):
            calls.append({"name": str(c["name"]), "args": dict(c.get("args") or {})})
    if not calls:
        return Observation(ok=True, wall_s=0.0)

    step_idx = len(state.trace.steps)
    names = [c["name"] for c in calls]
    emit(
        log,
        run_id=state.trace.run_id,
        step=step_idx,
        kind=EventKind.TOOL_BATCH,
        tool_names=names,
        n=len(calls),
    )

    # Pre-flight permission gate for gated tools in the batch. We
    # *serialize* the prompts (one at a time) — issuing modal prompts
    # inside the parallel pool would interleave terminal output. After
    # all decisions are collected, the pool runs the allowed sub-calls
    # in parallel; denied sub-calls get a synthetic error result.
    denied: dict[int, str] = {}
    if user_io is not None:
        for i, call in enumerate(calls):
            if call["name"] in _GATED_TOOLS:
                gate = _gate_tool_call(
                    state, call["name"], call["args"], user_io, log,
                    step_idx=step_idx, budget=budget,
                )
                if gate is not None:
                    denied[i] = gate.error or "user denied tool call"

    def _run_one(call: dict[str, Any]) -> dict[str, Any]:
        name = call["name"]
        args = call["args"]
        tool = tools.get(name)
        if tool is None:
            available = ", ".join(sorted(tools.names())) or "(none)"
            err = (
                f"unknown tool: {name!r}. "
                f"This tool is not registered. Available tools: {available}."
            )
            return {"name": name, "ok": False, "result": None,
                    "error": err, "wall_s": 0.0, "args": args}
        # Anti-loop: same skip-and-nudge as the single-call path, applied
        # per sub-call so a batch of repeated searches doesn't re-execute.
        dup = _dup_lookup(state, name, args)
        if dup is not None:
            return {"name": name, "ok": True,
                    "result": {"duplicate_skipped": True,
                               "note": _duplicate_observation(name, dup).data["note"],
                               "previous_result": dup},
                    "error": None, "wall_s": 0.0, "args": args}
        inv = invoke_tool(tool, args)
        _dup_record(state, name, args, _brief_tool_result(inv.result, inv.ok))
        return {"name": name, "ok": inv.ok, "result": inv.result,
                "error": inv.error, "wall_s": inv.wall_s, "args": args}

    # Cap parallelism. 5 is enough for the GAIA-shaped workloads we see
    # (1–3 read_url/search in a batch); raising it further risks rate
    # limits on the provider side.
    runnable_indices = [i for i in range(len(calls)) if i not in denied]
    runnable_calls = [calls[i] for i in runnable_indices]
    if runnable_calls:
        max_workers = max(1, min(len(runnable_calls), 5))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            run_results = list(pool.map(_run_one, runnable_calls))
    else:
        run_results = []
    # Stitch results back in original order, splicing synthetic
    # denied-error results into the slots whose calls were denied.
    results: list[dict[str, Any]] = []
    rr = iter(run_results)
    for i, call in enumerate(calls):
        if i in denied:
            results.append({
                "name": call["name"], "ok": False, "result": None,
                "error": denied[i], "wall_s": 0.0, "args": call["args"],
            })
        else:
            results.append(next(rr))

    # Re-emit per-call TOOL_CALL/TOOL_RESULT events in submission order so
    # log consumers see a deterministic sequence.
    all_ok = True
    max_wall = 0.0
    for r in results:
        emit(
            log,
            run_id=state.trace.run_id,
            step=step_idx,
            kind=EventKind.TOOL_CALL,
            tool_name=r["name"],
            arguments=r["args"],
            in_batch=True,
        )
        emit(
            log,
            run_id=state.trace.run_id,
            step=step_idx,
            kind=EventKind.TOOL_RESULT,
            ok=r["ok"],
            wall_s=r["wall_s"],
            error=r["error"],
            preview=_brief_tool_result(r["result"], r["ok"]),
            in_batch=True,
        )
        if r["ok"]:
            _auto_register_evidence(state, r["name"], r["result"], step_idx)
        else:
            all_ok = False
        if r["wall_s"] and r["wall_s"] > max_wall:
            max_wall = r["wall_s"]

    # Slim payload: drop full sub-results from `data` to keep the
    # Observation small. Each sub-call's tool output is already on the
    # tool result events; what the policy needs back is the summary list.
    summary = [
        {"name": r["name"], "ok": r["ok"], "error": r["error"],
         "wall_s": r["wall_s"], "result": r["result"]}
        for r in results
    ]
    _t_in = int((action.meta or {}).get("tokens_in") or 0)
    _t_out = int((action.meta or {}).get("tokens_out") or 0)
    return Observation(
        ok=all_ok,
        text=None,
        data={"batch": summary, "n": len(summary)},
        error=None if all_ok else "; ".join(
            f"{r['name']}: {r['error']}" for r in results if not r["ok"]
        ),
        wall_s=max_wall,
        tokens_in=_t_in,
        tokens_out=_t_out,
    )


def run_policy(
    state: AgentState,
    policy: Any,
    *,
    llm: LLMClient,
    tools: ToolRegistry,
    log: EventLog | None = None,
    compactor: Any = None,
    user_io: UserIO | None = None,
) -> AgentState:
    """Run the inner loop until done or budget-capped. Returns the state.

    `policy` must have `propose(state, *, llm, tools) -> Action`.
    `compactor`, if supplied, is a `memory.compactor.TraceCompactor`-shaped
    object: `should_compact(state) -> bool` and `compact(state) -> dict`.
    Called at each tick before the policy proposes.

    `user_io`, if supplied, enables human-in-the-loop behavior:
    `ASK_USER` actions block on `user_io.ask(...)`, and gated tool
    calls (currently `run_shell`) prompt via `user_io.confirm(...)`.
    When `None` (the default), the loop is in batch mode — `ASK_USER`
    degrades to a marker THINK and gated tools auto-allow. This keeps
    GAIA / CI runs producing identical traces to the old behavior.
    """
    tracker = BudgetTracker(state.budget)
    tracker.start()
    emit(
        log,
        run_id=state.trace.run_id,
        step=-1,
        kind=EventKind.RUN_START,
        question=state.question,
        policy=getattr(policy, "name", type(policy).__name__),
    )

    while True:
        reason = tracker.check()
        if reason != BudgetReason.OK:
            emit(log, run_id=state.trace.run_id, step=len(state.trace.steps),
                 kind=EventKind.BUDGET, reason=reason.value)
            # Best-effort synthesis when we hit the budget without ever
            # committing. Avoids 17% of GAIA tasks ending with
            # `pred_answer=null`. Synthesis is opt-in per policy: the
            # default `Policy` Protocol doesn't require it; if the
            # method is absent or returns None we simply terminate
            # without an answer.
            if not state.is_done:
                synth = getattr(policy, "synthesize_on_exhaustion", None)
                if callable(synth):
                    try:
                        synth_action = synth(state, llm=llm, tools=tools)
                    except Exception as exc:
                        emit(log, run_id=state.trace.run_id,
                             step=len(state.trace.steps),
                             kind=EventKind.ERROR,
                             error=f"{type(exc).__name__}: {exc}",
                             where="policy.synthesize_on_exhaustion")
                        synth_action = None
                    if synth_action is not None:
                        # Don't bill against `steps_used` — budget already
                        # exhausted. Token/cost accounting still applies.
                        synth_meta = dict(synth_action.meta or {})
                        synth_meta.setdefault("repair", True)
                        synth_meta.setdefault("synthesis_on_exhaustion", True)
                        synth_action.meta = synth_meta
                        obs = _execute(state, synth_action, tools, log, user_io,
                                       budget=tracker)
                        state.append_step(synth_action, obs)
                        emit(log, run_id=state.trace.run_id,
                             step=len(state.trace.steps) - 1,
                             kind=EventKind.PROPOSE,
                             kind_of_action=synth_action.kind.value,
                             tool_name=None,
                             has_answer=True,
                             action_text=(synth_action.answer or "")[:240],
                             is_error=False)
            break

        if compactor is not None and compactor.should_compact(state):
            info = compactor.compact(state)
            emit(log, run_id=state.trace.run_id, step=len(state.trace.steps),
                 kind=EventKind.COMPACT, **info)

        # Propose, retrying transient provider errors (rate limits, 5xx)
        # with exponential backoff. The backoff sleep is wrapped in
        # tracker.pause()/resume() so it does NOT count against the task's
        # wall budget — a low-tier (rate-limited) account shouldn't fail
        # tasks it would otherwise solve just because it had to wait. A
        # non-retryable error (bad key, model can't tool-call) or an
        # exhausted retry budget terminates the run as before.
        from ..llm.base import ProviderError
        action = None
        retry_n = 0
        while True:
            try:
                action = policy.propose(state, llm=llm, tools=tools)
                break
            except ProviderError as exc:
                if not exc.retryable or retry_n >= _MAX_LLM_RETRIES:
                    emit(log, run_id=state.trace.run_id,
                         step=len(state.trace.steps),
                         kind=EventKind.ERROR,
                         error=f"{type(exc).__name__}: {exc}",
                         where="policy.propose",
                         provider_error=True, retryable=exc.retryable)
                    break
                retry_n += 1
                backoff_s = min(2.0 ** retry_n, 30.0)  # 2,4,8,16 → cap 30
                emit(log, run_id=state.trace.run_id,
                     step=len(state.trace.steps),
                     kind=EventKind.ERROR, error=f"{type(exc).__name__}: {exc}",
                     where="policy.propose.retry",
                     provider_error=True, retryable=True,
                     attempt=retry_n, backoff_s=backoff_s)
                tracker.pause()
                try:
                    _time.sleep(backoff_s)
                finally:
                    tracker.resume()
                continue
            except Exception as exc:
                emit(log, run_id=state.trace.run_id,
                     step=len(state.trace.steps),
                     kind=EventKind.ERROR, error=f"{type(exc).__name__}: {exc}",
                     where="policy.propose",
                     provider_error=False, retryable=None)
                break
        if action is None:
            break

        # Truncate action text/answer so the event log stays small but
        # still readable for THINK / FINAL_ANSWER and useful for debugging
        # error-THINKs that policies emit when an LLM call fails.
        _txt = action.text or action.answer or ""
        if len(_txt) > 240:
            _txt = _txt[:237] + "…"
        emit(log, run_id=state.trace.run_id, step=len(state.trace.steps),
             kind=EventKind.PROPOSE, kind_of_action=action.kind.value,
             tool_name=action.tool_name, has_answer=action.answer is not None,
             action_text=_txt, is_error=bool(action.meta.get("error")))

        # Submit the model's literal answer string. We do not canonicalize
        # or otherwise rewrite the answer — canonicalization at this point
        # was destroying correct answers (e.g. "INT. THE CASTLE - DAY 1."
        # → "int. castle day 1" when gold was "THE CASTLE"). The GAIA
        # scorer applies its own normalization for the comparison; that
        # is appropriate and stays untouched.

        obs = _execute(state, action, tools, log, user_io, budget=tracker)
        step = state.append_step(action, obs)

        # Estimate USD cost for this tick's LLM usage. Provider/model are
        # set by the policy on action.meta when it lifted the LLM reply
        # (react.py / planner_react.py / verifier_retry.py). When the
        # action came from a non-LLM path (a synthetic THINK, a tool
        # result projection), provider may be missing — in that case we
        # treat the cost as 0 rather than guessing.
        tick_cost = 0.0
        meta = action.meta or {}
        provider = str(meta.get("provider") or "")
        model = str(meta.get("model") or "")
        if provider and model and (obs.tokens_in or obs.tokens_out):
            from ..llm.pricing import estimate_cost
            tick_cost, _known = estimate_cost(
                provider, model, obs.tokens_in, obs.tokens_out,
            )

        tracker.tick(
            step=False,
            tokens_in=obs.tokens_in,
            tokens_out=obs.tokens_out,
            cost_usd=tick_cost,
        )

        emit(log, run_id=state.trace.run_id, step=step.idx,
             kind=EventKind.OBSERVATION,
             ok=obs.ok, wall_s=obs.wall_s,
             tokens_in=obs.tokens_in, tokens_out=obs.tokens_out,
             # Running totals so the live display can show "step N/M ·
             # 25s/60s · 4.2k tok" without re-walking the trace.
             evidence_count=len(state.evidence),
             cumulative_tokens_in=state.budget.tokens_in,
             cumulative_tokens_out=state.budget.tokens_out,
             cumulative_wall_s=state.budget.elapsed_wall_s,
             max_steps=state.budget.max_steps,
             max_wall_s=state.budget.max_wall_s)

        if action.kind == ActionKind.FINAL_ANSWER:
            break

    emit(log, run_id=state.trace.run_id, step=len(state.trace.steps),
         kind=EventKind.RUN_END,
         is_done=state.is_done,
         final_answer=state.trace.final_answer,
         steps_used=state.budget.steps_used,
         budget_reason=tracker.check().value)
    return state
