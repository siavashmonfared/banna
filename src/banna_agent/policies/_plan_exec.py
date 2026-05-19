"""Shared plan executor + scorer.

BFS, DFS, and best-first all operate on multiple candidate Plans. They
differ in which plan they execute next; the *execution* of a single
plan step is identical.

The executor runs a *single* plan step through one LLM call + optional
tool dispatch (via the driver's `_execute` path). It does NOT run the
full ReAct loop per step — that would blow budgets and lose the point
of search. One LLM call per step, one tool call if requested, one
observation.

Scoring: a Plan's partial score is a blend of:
  - LLM-rated step quality (1-10) when the planner returned scores
  - evidence count added so far
  - verifier signals on intermediate answers (when verifier is wired)
  - inverse of tokens used (cost penalty)

The actual verifier hook is optional in week 2 (day 8-10). Today the
scorer is heuristic + LLM-optional.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from ..core.agent import _execute
from ..core.events import EventLog
from ..core.state import AgentState
from ..core.types import Action, ActionKind, Observation
from ..llm.base import ContentBlock, LLMClient, Message
from ..tools.base import ToolRegistry
from ._planning import Plan


# ---------------------------------------------------------------------------
# Step-level ReAct-shaped call
# ---------------------------------------------------------------------------


STEP_EXECUTOR_SYSTEM = (
    "You are a focused worker. You will receive the overall question, the "
    "full plan with progress markers, and a single CURRENT SUBQUESTION. "
    "Resolve only that subquestion.\n\n"
    "You may make several tool calls in sequence — for example: `search` "
    "to find a relevant URL, then `read_url` to fetch its content, then "
    "give a short text answer with the actual value you extracted.\n\n"
    "When you have an answer, return TEXT (no more tool calls). The text "
    "must contain the actual fact / value / number — DO NOT list URL "
    "titles or echo search snippets. If you cannot resolve the "
    "subquestion from the tools, say so plainly in one sentence and "
    "stop.\n\n"
    "Do not attempt the whole main question — only this subquestion. "
    "Keep your final text answer terse."
)


@dataclass
class StepResult:
    """Outcome of executing one plan step."""

    step_idx: int
    subquestion: str
    tool_name: str | None
    tool_args: dict[str, Any]
    ok: bool
    answer: str
    observation_data: dict[str, Any]
    tokens_in: int
    tokens_out: int
    wall_s: float
    error: str | None = None


def _format_plan_context(plan: Plan, cursor: int, main_question: str) -> str:
    lines = [
        f"Main question: {main_question}",
        "",
        "Plan:",
    ]
    for i, s in enumerate(plan.steps):
        mark = ">" if i == cursor else ("✓" if i < cursor else "-")
        lines.append(f"  {mark} {i+1}. {s}")
    lines.extend([
        "",
        f"Current subquestion ({cursor+1}/{len(plan.steps)}): {plan.steps[cursor]}",
    ])
    if cursor > 0 and plan.step_results:
        lines.extend(["", "Previously resolved:"])
        for i in range(cursor):
            r = plan.step_results[i] if i < len(plan.step_results) else {}
            res = r.get("resolution") or r.get("answer") or "(pending)"
            lines.append(f"  {i+1}. {plan.steps[i]} → {res}")
    return "\n".join(lines)


def execute_plan_step(
    plan: Plan,
    cursor: int,
    *,
    main_question: str,
    llm: LLMClient,
    tools: ToolRegistry,
    state: AgentState,
    log: EventLog | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_inner_steps: int = 3,
) -> StepResult:
    """Mini-ReAct loop scoped to one plan step.

    Each inner iteration is one LLM call. The model either:
      - **emits a tool call** → we dispatch it, append the step, and
        loop with the tool's result added to a *self-contained* inner
        message history (so the next iteration can read what the tool
        returned without polluting the global trace projection);
      - **emits text** → we commit that as the subquestion's answer
        and return.

    Capped at `max_inner_steps` to bound cost. If the cap is hit
    without a clean text answer, the last tool's summary is used as
    the answer so the plan still has *something* to score against.

    Why multi-call: many subquestions need search → read_url → extract
    to produce a value. Limiting to a single LLM call meant the model
    saw search hit titles but couldn't fetch the actual content; the
    plan's "answer" became a list of URL titles, which the synthesizer
    then had nothing to compute from.

    Tokens accumulate across all inner iterations and are stamped onto
    every action.meta so the budget tracker sees real costs.
    """
    base_prompt = _format_plan_context(plan, cursor, main_question)
    inner_history: list[Message] = [Message(
        role="user",
        content=[ContentBlock(kind="text", text=base_prompt)],
    )]

    accum_tokens_in = 0
    accum_tokens_out = 0
    last_tool_name: str | None = None
    last_tool_args: dict[str, Any] = {}
    last_obs_data: dict[str, Any] = {}
    last_obs_ok = True  # only flips false if a tool actually fails
    last_tool_summary = ""
    last_wall_s = 0.0
    any_tool_attempted = False

    for inner in range(max_inner_steps):
        kwargs: dict[str, Any] = {
            "messages": list(inner_history),
            "tools": tools.to_tool_specs(),
            "system": STEP_EXECUTOR_SYSTEM,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model:
            kwargs["model"] = model

        try:
            reply = llm.chat(**kwargs)
        except Exception as exc:
            return StepResult(
                step_idx=cursor, subquestion=plan.steps[cursor],
                tool_name=last_tool_name, tool_args=last_tool_args,
                ok=False, answer="",
                observation_data=last_obs_data,
                tokens_in=accum_tokens_in,
                tokens_out=accum_tokens_out,
                wall_s=last_wall_s,
                error=f"{type(exc).__name__}: {exc}",
            )

        accum_tokens_in += reply.usage.tokens_in
        accum_tokens_out += reply.usage.tokens_out

        if reply.has_tool_calls:
            call = reply.tool_calls[0]
            action = Action(
                kind=ActionKind.TOOL_CALL,
                tool_name=call.name,
                tool_args=dict(call.arguments),
                meta={
                    "plan_step": cursor,
                    "tokens_in": reply.usage.tokens_in,
                    "tokens_out": reply.usage.tokens_out,
                    "inner_step": inner,
                },
            )
            obs = _execute(state, action, tools, log)
            state.append_step(action, obs)
            any_tool_attempted = True

            last_tool_name = call.name
            last_tool_args = dict(call.arguments)
            last_obs_ok = obs.ok
            last_obs_data = obs.data if obs.ok else {"error": obs.error}
            last_tool_summary = _summarize_tool_result(obs.data)
            last_wall_s = obs.wall_s

            # Build the next iteration's prompt by appending the
            # assistant tool_use turn + the user tool_result turn.
            # This keeps the inner loop self-contained — no read of
            # state.trace, no risk of trace pollution from prior
            # plan steps.
            inner_history.append(Message(
                role="assistant",
                content=[ContentBlock(
                    kind="tool_use",
                    id=f"inner_{inner}",
                    name=call.name,
                    arguments=dict(call.arguments),
                )],
            ))
            inner_history.append(Message(
                role="user",
                content=[ContentBlock(
                    kind="tool_result",
                    id=f"inner_{inner}",
                    name=call.name,
                    result=last_obs_data,
                    is_error=not obs.ok,
                )],
            ))
            continue  # next inner iteration

        # Text response — that's the subquestion's answer.
        text = reply.text.strip()
        if text:
            action = Action(
                kind=ActionKind.THINK,
                text=f"[plan_step {cursor+1}] {text[:400]}",
                meta={
                    "plan_step": cursor,
                    "tokens_in": accum_tokens_in,
                    "tokens_out": accum_tokens_out,
                    "inner_step": inner,
                },
            )
            obs = Observation(
                ok=True, text=text,
                tokens_in=accum_tokens_in, tokens_out=accum_tokens_out,
            )
            state.append_step(action, obs)
            return StepResult(
                step_idx=cursor, subquestion=plan.steps[cursor],
                tool_name=last_tool_name, tool_args=last_tool_args,
                ok=True, answer=text,
                observation_data=last_obs_data,
                tokens_in=accum_tokens_in, tokens_out=accum_tokens_out,
                wall_s=last_wall_s,
            )
        # Empty text + no tool — nothing useful from this iteration; bail.
        break

    # Cap reached or empty reply — fall back to the last tool's summary
    # so the plan has *something* to score against. Record a synthetic
    # THINK so the trace reflects what happened.
    answer = last_tool_summary or "(no answer)"
    action = Action(
        kind=ActionKind.THINK,
        text=f"[plan_step {cursor+1}] {answer[:400]}",
        meta={
            "plan_step": cursor,
            "tokens_in": accum_tokens_in,
            "tokens_out": accum_tokens_out,
            "max_inner_reached": True,
        },
    )
    # `ok` reflects whether the last tool *actually* succeeded — not just
    # whether we have a non-empty summary. An unknown-tool error
    # produces a truthy summary string ({"error": "unknown ..."}) but
    # ok must be False.
    step_ok = any_tool_attempted and last_obs_ok and bool(last_tool_summary)
    obs = Observation(
        ok=step_ok, text=answer,
        tokens_in=accum_tokens_in, tokens_out=accum_tokens_out,
    )
    state.append_step(action, obs)
    return StepResult(
        step_idx=cursor, subquestion=plan.steps[cursor],
        tool_name=last_tool_name, tool_args=last_tool_args,
        ok=step_ok, answer=answer,
        observation_data=last_obs_data,
        tokens_in=accum_tokens_in, tokens_out=accum_tokens_out,
        wall_s=last_wall_s,
    )


def _summarize_tool_result(data: dict[str, Any]) -> str:
    """Pick a human-readable string from an arbitrary tool result dict.

    Prefers common fields ('value', 'summary', 'text', 'answer',
    'stdout') and falls back to a compact summary of search hits or a
    truncated stringified dict.

    The empty-string check is deliberate: many tool outputs include
    a `summary` key whose value is `""` (search returns empty summary
    when the cascade lands on YaCy/DuckDuckGo, which don't emit one).
    Treating `""` as "the answer" produces empty plan-step resolutions
    and breaks BFS/DFS plan scoring.
    """
    if not isinstance(data, dict):
        return str(data)[:400]
    for key in ("value", "summary", "text", "answer", "stdout"):
        v = data.get(key)
        if v:  # skip None and empty strings
            return str(v)[:400]
    # Search-tool shape: surface a compact list of hits so plan scoring
    # has *something* to score against.
    hits = data.get("hits")
    if isinstance(hits, list) and hits:
        bits: list[str] = []
        for h in hits[:5]:
            if not isinstance(h, dict):
                continue
            title = (h.get("title") or "").strip()
            url = (h.get("url") or "").strip()
            snippet = (h.get("snippet") or "").strip()
            if title and snippet:
                bits.append(f"{title} — {snippet[:120]}")
            elif title:
                bits.append(title)
            elif url:
                bits.append(url)
        if bits:
            return " | ".join(bits)[:600]
    return str(data)[:400]


# ---------------------------------------------------------------------------
# Final-answer synthesizer
# ---------------------------------------------------------------------------


def stash_pending_tokens(state: AgentState, tokens_in: int, tokens_out: int) -> None:
    """Accumulate tokens from out-of-band LLM calls into state.metadata.

    Plan-based policies make LLM calls that don't go through an Action
    (e.g. `propose_candidate_plans` and `propose_plan`). Their token
    cost is real but the budget tracker only sees Observations. Stash
    the tokens here, then drain them onto the next returned action's
    meta via `drain_pending_tokens`.
    """
    if not (tokens_in or tokens_out):
        return
    state.metadata["_pending_tokens_in"] = (
        int(state.metadata.get("_pending_tokens_in", 0)) + int(tokens_in)
    )
    state.metadata["_pending_tokens_out"] = (
        int(state.metadata.get("_pending_tokens_out", 0)) + int(tokens_out)
    )


def drain_pending_tokens(state: AgentState, action_meta: dict[str, Any]) -> None:
    """Move pending tokens from state.metadata onto an action's meta.

    Mutates `action_meta` in place. Called by plan-based policies just
    before returning an Action so the driver's `_execute` lift credits
    the tokens to `state.budget`.
    """
    pending_in = int(state.metadata.pop("_pending_tokens_in", 0))
    pending_out = int(state.metadata.pop("_pending_tokens_out", 0))
    if not (pending_in or pending_out):
        return
    action_meta["tokens_in"] = int(action_meta.get("tokens_in") or 0) + pending_in
    action_meta["tokens_out"] = int(action_meta.get("tokens_out") or 0) + pending_out


SYNTHESIZER_SYSTEM = (
    "You are a careful synthesizer. You will receive a main question, the "
    "subquestions a previous worker resolved, their resolutions, and a "
    "list of source-grounded evidence (URL — snippet pairs).\n\n"
    "Produce ONE concise final answer to the main question. Rules:\n"
    "  • DO NOT echo URL titles or copy evidence URLs into your answer.\n"
    "  • DO NOT list the search results — synthesize them into a single "
    "answer to the main question.\n"
    "  • When the question asks for a number: give the number with "
    "units. If you have intermediate values (e.g. a population and an "
    "area), COMPUTE the requested ratio yourself and report the "
    "computed value.\n"
    "  • When the question asks for a name, year, or list: give just "
    "that, terse, no preamble.\n"
    "  • If the gathered evidence does not contain the specific value "
    "the question asks for, say so plainly in ONE sentence (e.g. "
    "\"The gathered evidence lists Iceland-related sources but does "
    "not contain the population number\") and stop. Do not propose "
    "new tool calls."
)


def synthesize_final_answer(
    main_question: str,
    plan: Plan,
    state: AgentState,
    *,
    llm: LLMClient,
    model: str | None = None,
    max_tokens: int = 400,
    max_evidence: int = 8,
) -> tuple[str, int, int]:
    """Run one LLM call to produce a final answer from the plan's
    resolutions plus the auto-collected evidence on `state`.

    Returns ``(answer_text, tokens_in, tokens_out)``.

    The previous behavior — returning whatever string the *last* tool
    call produced — broke whenever the last step happened to be a
    side-effecting tool (memory.write, run_python with no return,
    etc.). The synthesizer reads the *full* plan trajectory and the
    auto-registered evidence, so the answer reflects all the work
    rather than only the trailing step.
    """
    lines: list[str] = [f"Main question: {main_question}", ""]
    if state.evidence:
        lines.append("Evidence (URL — snippet):")
        for ev in state.evidence[-max_evidence:]:
            src = (ev.source or "")[:80]
            content = (ev.content or "").strip().replace("\n", " ")[:200]
            lines.append(f"  - [{src}] {content}")
        lines.append("")
    lines.append("Subquestions resolved:")
    for i, step in enumerate(plan.steps):
        r = plan.step_results[i] if i < len(plan.step_results) else {}
        ans = (r.get("resolution") or r.get("answer") or "(unresolved)")
        lines.append(f"  {i+1}. {step}")
        lines.append(f"      → {str(ans)[:280]}")
    lines.extend([
        "",
        "Now produce the final answer to the main question. Be terse.",
    ])
    prompt = "\n".join(lines)

    kwargs: dict[str, Any] = {
        "messages": [Message(
            role="user",
            content=[ContentBlock(kind="text", text=prompt)],
        )],
        "system": SYNTHESIZER_SYSTEM,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if model:
        kwargs["model"] = model

    try:
        reply = llm.chat(**kwargs)
    except Exception as exc:
        return (
            f"(synthesizer failed: {type(exc).__name__}: {exc})",
            0, 0,
        )
    text = reply.text.strip() or "(synthesizer returned empty)"
    return text, reply.usage.tokens_in, reply.usage.tokens_out


# ---------------------------------------------------------------------------
# Plan scorer
# ---------------------------------------------------------------------------


@dataclass
class PlanScore:
    total: float
    coverage: float       # fraction of plan steps that resolved ok
    evidence_count: int   # new evidence items added
    cost_tokens: int
    penalty: float = 0.0


def score_plan(
    plan: Plan,
    *,
    evidence_before: int,
    evidence_after: int,
    verifier: Callable[[str], bool] | None = None,
) -> PlanScore:
    """Compute a partial score for a plan's execution so far.

    score = coverage - cost_penalty  (+verifier_bonus if applicable)

    Coverage rewards plans that successfully resolve their steps; cost
    penalizes token usage. Verifier (when supplied) boosts plans whose
    final answer passes a grounded check.
    """
    resolved = sum(1 for r in plan.step_results if r.get("ok"))
    coverage = resolved / max(1, len(plan.steps))
    tokens = sum(int(r.get("tokens_in", 0) + r.get("tokens_out", 0))
                 for r in plan.step_results)
    cost_penalty = math.log1p(tokens) / 20.0  # diminishing
    bonus = 0.0
    if verifier is not None and plan.final_answer:
        try:
            if verifier(plan.final_answer):
                bonus = 0.25
        except Exception:
            bonus = 0.0
    total = coverage + bonus - cost_penalty
    return PlanScore(
        total=total,
        coverage=coverage,
        evidence_count=max(0, evidence_after - evidence_before),
        cost_tokens=tokens,
        penalty=cost_penalty,
    )
