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
) -> Observation:
    """Run one Action; return the Observation.

    For THINK actions the observation is trivial. For TOOL_CALL the
    handler is dispatched through `tools`; exceptions are caught and
    converted to an error observation (the agent loop never dies).
    For FINAL_ANSWER the observation mirrors the answer and marks
    readiness to terminate.
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

    if action.kind == ActionKind.TOOL_CALL:
        name = action.tool_name or ""
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
        # Capture evidence count BEFORE the tool runs so the
        # auto-registration delta can be reported live.
        state._ev_before_tool = len(state.evidence)  # type: ignore[attr-defined]
        inv = invoke_tool(tool, action.tool_args)
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

    raise ValueError(f"unknown ActionKind: {action.kind}")


def run_policy(
    state: AgentState,
    policy: Any,
    *,
    llm: LLMClient,
    tools: ToolRegistry,
    log: EventLog | None = None,
    compactor: Any = None,
) -> AgentState:
    """Run the inner loop until done or budget-capped. Returns the state.

    `policy` must have `propose(state, *, llm, tools) -> Action`.
    `compactor`, if supplied, is a `memory.compactor.TraceCompactor`-shaped
    object: `should_compact(state) -> bool` and `compact(state) -> dict`.
    Called at each tick before the policy proposes.
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
                        obs = _execute(state, synth_action, tools, log)
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

        try:
            action = policy.propose(state, llm=llm, tools=tools)
        except Exception as exc:
            emit(log, run_id=state.trace.run_id, step=len(state.trace.steps),
                 kind=EventKind.ERROR, error=f"{type(exc).__name__}: {exc}",
                 where="policy.propose")
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

        obs = _execute(state, action, tools, log)
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
