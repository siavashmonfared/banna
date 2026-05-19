"""Planner-ReAct — two-stage policy.

Stage 1: LLM writes a plan (ordered list of subquestions) once, at the
start of the run. Stored in `state.metadata["plan"]`.

Stage 2: On each subsequent tick, the inner ReAct decision rule applies
— but with the *current subquestion* prepended to the context so the
model stays focused. When the model emits a final answer, we advance
to the next subquestion; when all subquestions are exhausted, we emit
the final answer of the last step as the task's final answer.

Why this matters: the `plan` tool we have is great for the model to
externalize state *during* reasoning, but it's optional and the model
has to remember to use it. Planner-ReAct forces decomposition up front,
which dramatically helps Level 2/3 multi-hop tasks that ReAct tends to
solve in a single undifferentiated burst.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.state import AgentState
from ..core.types import Action, ActionKind
from ..llm.base import ContentBlock, LLMClient, Message, ToolSpec
from ..tools.base import ToolRegistry
from ._plan_exec import drain_pending_tokens, stash_pending_tokens
from ._planning import Plan, propose_plan
from .react import DEFAULT_SYSTEM_PROMPT, ReActPolicy


@dataclass
class PlannerReActPolicy:
    """Plan-once, execute-step-by-step.

    The planner runs once (lazily, on the first `propose` call) and
    caches its plan in `state.metadata["_planner_react_plan"]`. A
    ReActPolicy instance handles per-subquestion execution. When the
    inner ReAct produces a `FINAL_ANSWER`, we intercept it:
      - if this is the last subquestion, pass the answer through as
        the task's final answer;
      - otherwise, rewrite the action as a THINK that records the
        subquestion's resolution and advance the cursor. The next tick
        will run ReAct on the next subquestion.
    """

    name: str = "planner_react"
    planner_system: str | None = None                  # override DEFAULT_PLANNER_SYSTEM
    executor_system: str = DEFAULT_SYSTEM_PROMPT
    model: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    # --- lazy planning helpers -------------------------------------------

    def _ensure_plan(self, state: AgentState, llm: LLMClient) -> Plan:
        """Return the cached plan, creating it on first call."""
        cached = state.metadata.get("_planner_react_plan")
        if isinstance(cached, Plan):
            return cached
        plan = propose_plan(
            llm,
            state.question,
            system=self.planner_system or _DEFAULT_PLANNER_WITH_FORMAT,
            model=self.model,
        )
        if not plan.steps:
            # Fallback: treat the whole question as a single step.
            plan = Plan(steps=[state.question], meta={"fallback": True})
        # Credit the planner LLM call's tokens via the pending-token
        # mechanism. Without this they're lost.
        stash_pending_tokens(
            state,
            int(plan.meta.get("tokens_in") or 0),
            int(plan.meta.get("tokens_out") or 0),
        )
        state.metadata["_planner_react_plan"] = plan
        state.metadata["_planner_react_cursor"] = 0
        return plan

    def _current_subquestion(self, state: AgentState, plan: Plan) -> tuple[int, str]:
        cursor: int = state.metadata.get("_planner_react_cursor", 0)
        if cursor >= len(plan.steps):
            # Shouldn't happen in healthy operation, but be safe.
            cursor = len(plan.steps) - 1
        return cursor, plan.steps[cursor]

    def _advance(self, state: AgentState, plan: Plan, resolution: str) -> bool:
        """Record a sub-answer and advance the cursor. Returns True if
        the plan is now exhausted (caller should emit FINAL_ANSWER)."""
        cursor: int = state.metadata.get("_planner_react_cursor", 0)
        plan.with_step_result(cursor, {"subquestion": plan.steps[cursor],
                                       "resolution": resolution})
        cursor += 1
        state.metadata["_planner_react_cursor"] = cursor
        return cursor >= len(plan.steps)

    # --- Policy.propose --------------------------------------------------

    def propose(
        self,
        state: AgentState,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
    ) -> Action:
        action = self._propose(state, llm=llm, tools=tools)
        drain_pending_tokens(state, action.meta)
        return action

    def _propose(
        self,
        state: AgentState,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
    ) -> Action:
        plan = self._ensure_plan(state, llm)
        if not plan.steps:
            return Action(
                kind=ActionKind.FINAL_ANSWER,
                answer="(planner failed)",
                meta={"policy": self.name, "error": "empty plan"},
            )

        cursor, current = self._current_subquestion(state, plan)

        # Inner ReAct call, scoped to the current subquestion.
        react = ReActPolicy(
            system_prompt=self.executor_system,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            extra=self.extra,
        )
        # Wrap state.question to include plan context + current subquestion.
        inner_state = _clone_for_subquestion(state, plan, cursor, current)
        action = react.propose(inner_state, llm=llm, tools=tools)

        if action.kind == ActionKind.FINAL_ANSWER:
            resolution = action.answer or action.text or ""
            plan_done = self._advance(state, plan, resolution)
            if plan_done:
                # Pass through as the task's final answer.
                return Action(
                    kind=ActionKind.FINAL_ANSWER,
                    answer=resolution,
                    text=resolution,
                    meta={
                        **action.meta,
                        "policy": self.name,
                        "plan_step": cursor,
                        "plan_len": len(plan.steps),
                    },
                )
            # More subquestions to go. Convert to a THINK so the driver
            # records progress without terminating.
            return Action(
                kind=ActionKind.THINK,
                text=f"[subq {cursor+1}/{len(plan.steps)}] {current} → {resolution}",
                meta={
                    "policy": self.name,
                    "plan_step": cursor,
                    "plan_len": len(plan.steps),
                    "resolved_subquestion": True,
                },
            )

        # Tool call or THINK — pass through, adorning with plan context.
        action.meta = {
            **action.meta,
            "policy": self.name,
            "plan_step": cursor,
            "plan_len": len(plan.steps),
        }
        return action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DEFAULT_PLANNER_WITH_FORMAT = (
    "You are a research planner. Decompose the question into 2-6 concrete "
    "subquestions. Each subquestion should be answerable with a single tool "
    "call (search, read_url, calculator, read_file, run_python) plus one "
    "reasoning step. The last subquestion produces the final answer.\n\n"
    'Return ONLY JSON: {"plan": ["subq 1", "subq 2", "..."]}'
)


def _clone_for_subquestion(
    state: AgentState,
    plan: Plan,
    cursor: int,
    current: str,
) -> AgentState:
    """Return a shallow clone of `state` scoped to the current subquestion.

    Two important details, both critical for plan execution to actually
    progress instead of looping:

    1. **Resolutions in the preamble.** Each previously-resolved plan
       step appears as ``✓ N. <subq>  →  <resolution>`` so subquestion N
       can see what 1..N-1 actually answered. Without this, "compute
       density" can't see the population/area numbers it needs.

    2. **Per-subquestion trace filter.** ReAct's history projection
       (`react.ReActPolicy._history`) iterates `state.trace.steps` and
       turns them into LLM message turns. If we hand it the full trace
       — including subquestion 1's search call + result — the LLM sees
       its own prior tool call right before the new prompt and tends
       to mimic it (qwen3-coder is especially prone to this), repeating
       the same search instead of moving on. Filtering the wrapped
       state's trace to *only* this subquestion's steps cuts that
       failure mode at the source.

    Steps are tagged via `Action.meta["plan_step"]`, set by
    `PlannerReActPolicy.propose` when it returns each tick's action.
    """
    original_question = state.question

    # --- preamble: plan + prior resolutions inline -----------------------
    preamble_lines = [
        f"Main question: {original_question}",
        "",
        "Plan:",
    ]
    for i, step in enumerate(plan.steps):
        mark = ">" if i == cursor else ("✓" if i < cursor else "-")
        line = f"  {mark} {i+1}. {step}"
        if i < cursor and i < len(plan.step_results):
            res = (plan.step_results[i] or {}).get("resolution") or "(no resolution)"
            line += f"  →  {str(res)[:240]}"
        preamble_lines.append(line)
    preamble_lines.extend([
        "",
        f"Current subquestion ({cursor+1}/{len(plan.steps)}): {current}",
        "",
        "Answer ONLY the current subquestion. If this is the final subquestion, "
        "produce the final answer to the main question; otherwise produce a "
        "short intermediate result that later steps can build on.",
    ])
    composite = "\n".join(preamble_lines)

    # --- trace filter: only this subquestion's steps ----------------------
    filtered_steps = [
        s for s in state.trace.steps
        if s.action.meta.get("plan_step", cursor) == cursor
    ]

    class _FakeTrace:
        """Just enough Trace shape for ReActPolicy._history."""
        pass

    ft = _FakeTrace()
    ft.steps = filtered_steps
    ft.run_id = state.trace.run_id
    ft.question = state.trace.question
    ft.final_answer = state.trace.final_answer
    ft.started_at = state.trace.started_at

    # --- wrapper that looks like AgentState but with the scoped trace ----
    class _Wrapper:
        pass

    w = _Wrapper()
    w.question = composite
    w.trace = ft
    w.evidence = state.evidence
    w.claims = state.claims
    w.budget = state.budget
    w.metadata = state.metadata
    w.state_id = state.state_id
    w.parent_state_id = state.parent_state_id
    w.is_done = state.is_done
    w.last_step = state.last_step
    w.append_step = state.append_step
    w.add_evidence = state.add_evidence
    w.add_claim = state.add_claim
    w.evidence_for = state.evidence_for
    return w  # type: ignore[return-value]
