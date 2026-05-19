"""Best-first search over plans.

Priority queue semantics:
  1. Propose N candidate plans.
  2. Execute one step of the currently best-scored plan. Advance its
     cursor by one.
  3. Rescore all plans. The plan with the highest (coverage - cost +
     verifier_bonus) score is selected next tick.
  4. When any plan finishes all its steps with a passing answer, return it.

Key differences from BFS:
  - BFS front-loads expansion (all first steps) then descends greedily.
  - Best-first interleaves: after each step, the active plan can change
    based on the score. This allows early commitment when one plan is
    clearly winning, but recovers when a plan stalls.

This is the workshop-paper baseline every other policy is measured
against. Implementation is intentionally close to textbook best-first:
priority queue, one pop-push per tick.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..core.state import AgentState
from ..core.types import Action, ActionKind
from ..llm.base import LLMClient
from ..tools.base import ToolRegistry
from ._plan_exec import (
    drain_pending_tokens,
    execute_plan_step,
    score_plan,
    stash_pending_tokens,
    synthesize_final_answer,
)
from ._planning import Plan, propose_candidate_plans


@dataclass
class BestFirstOverPlansPolicy:
    """Best-first search via a rescored priority list.

    `verifier` is an optional callable `(answer: str) -> bool` that
    awards a score bonus when a plan's current answer passes a grounded
    check. Wired in week 2 when verifiers land.
    """

    name: str = "best_first_over_plans"
    n_candidates: int = 3
    model: str | None = None
    max_tokens: int = 1024
    temperature_branches: float = 0.7
    temperature_exec: float = 0.0
    verifier: Callable[[str], bool] | None = None
    # Max steps *per plan* before we consider it stalled (prevents one
    # plan monopolizing the budget).
    max_steps_per_plan: int = 8

    def _ensure_setup(self, state: AgentState, llm: LLMClient) -> list[Plan]:
        if state.metadata.get("_bf_plans") is None:
            plans = propose_candidate_plans(
                llm, state.question,
                n_candidates=self.n_candidates,
                model=self.model,
                temperature=self.temperature_branches,
            ) or []
            t_in = sum(int(p.meta.get("tokens_in") or 0) for p in plans)
            t_out = sum(int(p.meta.get("tokens_out") or 0) for p in plans)
            stash_pending_tokens(state, t_in, t_out)
            state.metadata["_bf_plans"] = plans
            state.metadata["_bf_cursors"] = [0] * len(plans)
            state.metadata["_bf_evidence_baselines"] = [0] * len(plans)
        return state.metadata["_bf_plans"]

    def _select_next(self, state: AgentState) -> int | None:
        plans: list[Plan] = state.metadata["_bf_plans"]
        cursors: list[int] = state.metadata["_bf_cursors"]
        baselines: list[int] = state.metadata["_bf_evidence_baselines"]
        best_idx = None
        best_score = -float("inf")
        for i, plan in enumerate(plans):
            if cursors[i] >= len(plan.steps) or cursors[i] >= self.max_steps_per_plan:
                continue
            # Score the plan as-is (partial execution).
            sc = score_plan(
                plan,
                evidence_before=baselines[i],
                evidence_after=len(state.evidence),
                verifier=self.verifier,
            )
            if sc.total > best_score:
                best_score = sc.total
                best_idx = i
        return best_idx

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
        plans = self._ensure_setup(state, llm)
        if not plans:
            return Action(
                kind=ActionKind.FINAL_ANSWER,
                answer="(best-first: no candidate plans)",
                meta={"policy": self.name, "error": "empty_candidates"},
            )
        cursors: list[int] = state.metadata["_bf_cursors"]

        # Check for a finished plan; synthesize its final answer.
        for i, plan in enumerate(plans):
            if cursors[i] >= len(plan.steps) and plan.step_results:
                if not plan.final_answer:
                    text, t_in, t_out = synthesize_final_answer(
                        state.question, plan, state,
                        llm=llm, model=self.model,
                    )
                    plan.final_answer = text
                    return Action(
                        kind=ActionKind.FINAL_ANSWER,
                        answer=text,
                        meta={"policy": self.name, "branch": i,
                              "plan_steps": len(plan.steps),
                              "tokens_in": t_in, "tokens_out": t_out},
                    )

        # Pick the current best plan.
        active_idx = self._select_next(state)
        if active_idx is None:
            # Every plan exhausted; return best-effort answer.
            final = _pick_best_answer(plans)
            return Action(
                kind=ActionKind.FINAL_ANSWER,
                answer=final or "(best-first: all branches stalled)",
                meta={"policy": self.name, "phase": "exhausted"},
            )

        plan = plans[active_idx]
        cursor = cursors[active_idx]
        result = execute_plan_step(
            plan, cursor=cursor,
            main_question=state.question,
            llm=llm, tools=tools, state=state,
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature_exec,
        )
        plan.with_step_result(cursor, _result_to_dict(result))
        cursors[active_idx] = cursor + 1

        return Action(
            kind=ActionKind.THINK,
            text=f"[best-first b{active_idx} s{cursor+1}/{len(plan.steps)}] "
                 f"{result.answer[:100]}",
            meta={"policy": self.name, "branch": active_idx,
                  "plan_step": cursor, "ok": result.ok},
        )


def _pick_best_answer(plans: list[Plan]) -> str:
    for p in plans:
        if p.step_results:
            ans = p.step_results[-1].get("answer")
            if ans:
                return ans
    return ""


def _result_to_dict(r) -> dict[str, Any]:
    return {
        "subquestion": r.subquestion,
        "ok": r.ok,
        "tool_name": r.tool_name,
        "answer": r.answer,
        "tokens_in": r.tokens_in,
        "tokens_out": r.tokens_out,
        "error": r.error,
        "resolution": r.answer,
    }
