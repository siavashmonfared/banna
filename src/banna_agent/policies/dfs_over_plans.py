"""DFS over plans — depth-first search with backtracking.

Semantics:
  1. Propose N candidate plans up front, ordered as a stack.
  2. Pop the top plan; execute it end-to-end (step by step until
     either its last step produces an answer, or a step fails hard).
  3. If the plan's final answer passes a cheap quality filter — return
     it. Otherwise backtrack: pop the next plan from the stack and
     restart execution.
  4. If all plans exhaust without success, return the best candidate's
     last answer or an error marker.

Why DFS: good when each plan has a low probability of succeeding but
high cost of switching. Minimizes exploration; maximizes commitment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.state import AgentState
from ..core.types import Action, ActionKind
from ..llm.base import LLMClient
from ..tools.base import ToolRegistry
from ._plan_exec import (
    drain_pending_tokens,
    execute_plan_step,
    stash_pending_tokens,
    synthesize_final_answer,
)
from ._planning import Plan, propose_candidate_plans


@dataclass
class DFSOverPlansPolicy:
    """Depth-first search over candidate plans, with backtracking."""

    name: str = "dfs_over_plans"
    n_candidates: int = 3
    model: str | None = None
    max_tokens: int = 1024
    temperature_branches: float = 0.7
    temperature_exec: float = 0.0

    def _ensure_setup(
        self, state: AgentState, llm: LLMClient,
    ) -> tuple[list[Plan], int, int]:
        if state.metadata.get("_dfs_plans") is None:
            candidates = propose_candidate_plans(
                llm, state.question,
                n_candidates=self.n_candidates,
                model=self.model,
                temperature=self.temperature_branches,
            ) or []
            t_in = sum(int(p.meta.get("tokens_in") or 0) for p in candidates)
            t_out = sum(int(p.meta.get("tokens_out") or 0) for p in candidates)
            stash_pending_tokens(state, t_in, t_out)
            state.metadata["_dfs_plans"] = candidates
            state.metadata["_dfs_active_idx"] = 0
            state.metadata["_dfs_cursor"] = 0
        return (
            state.metadata["_dfs_plans"],
            state.metadata.get("_dfs_active_idx", 0),
            state.metadata.get("_dfs_cursor", 0),
        )

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
        candidates, active_idx, cursor = self._ensure_setup(state, llm)
        if not candidates:
            return Action(
                kind=ActionKind.FINAL_ANSWER,
                answer="(dfs: no candidate plans)",
                meta={"policy": self.name, "error": "empty_candidates"},
            )

        # All plans exhausted?
        if active_idx >= len(candidates):
            best = _pick_best(candidates)
            if best is not None and best.final_answer:
                return Action(
                    kind=ActionKind.FINAL_ANSWER,
                    answer=best.final_answer,
                    meta={"policy": self.name, "phase": "exhausted",
                          "branch": "best"},
                )
            return Action(
                kind=ActionKind.FINAL_ANSWER,
                answer="(dfs: all branches failed)",
                meta={"policy": self.name, "phase": "exhausted"},
            )

        plan = candidates[active_idx]

        # Finished executing this plan's steps?
        if cursor >= len(plan.steps):
            text, t_in, t_out = synthesize_final_answer(
                state.question, plan, state,
                llm=llm, model=self.model,
            )
            if _answer_looks_acceptable(text):
                plan.final_answer = text
                return Action(
                    kind=ActionKind.FINAL_ANSWER,
                    answer=text,
                    meta={"policy": self.name, "phase": "accept",
                          "branch": active_idx,
                          "tokens_in": t_in, "tokens_out": t_out},
                )
            # Backtrack: try the next plan.
            state.metadata["_dfs_active_idx"] = active_idx + 1
            state.metadata["_dfs_cursor"] = 0
            return Action(
                kind=ActionKind.THINK,
                text=f"[dfs backtrack] branch {active_idx} answer "
                     f"{text!r} didn't pass filter; trying next",
                meta={"policy": self.name, "phase": "backtrack",
                      "branch": active_idx,
                      "tokens_in": t_in, "tokens_out": t_out},
            )

        # Execute next step of current plan.
        result = execute_plan_step(
            plan, cursor=cursor,
            main_question=state.question,
            llm=llm, tools=tools, state=state,
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature_exec,
        )
        plan.with_step_result(cursor, _result_to_dict(result))
        state.metadata["_dfs_cursor"] = cursor + 1
        return Action(
            kind=ActionKind.THINK,
            text=f"[dfs b{active_idx} s{cursor+1}/{len(plan.steps)}] "
                 f"{result.answer[:100]}",
            meta={"policy": self.name, "phase": "descend",
                  "branch": active_idx, "plan_step": cursor,
                  "ok": result.ok},
        )


def _answer_looks_acceptable(ans: str) -> bool:
    """Minimal quality filter — rejects empty strings and obvious failures.

    A verifier-based filter is a week-2 upgrade. Today: non-empty and not
    an error marker."""
    if not ans or not ans.strip():
        return False
    lowered = ans.lower().strip()
    bad_markers = ("i don't know", "cannot determine", "unknown",
                   "i am unable", "error", "(none)")
    return not any(lowered.startswith(b) for b in bad_markers)


def _pick_best(candidates: list[Plan]) -> Plan | None:
    """Return the plan whose last step has a non-empty answer, if any."""
    for p in candidates:
        if p.step_results and p.step_results[-1].get("answer"):
            return p
    return candidates[0] if candidates else None


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
