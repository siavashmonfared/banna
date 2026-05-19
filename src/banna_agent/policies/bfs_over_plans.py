"""BFS over plans — breadth-first search.

Semantics:
  1. Propose N candidate plans up front.
  2. Execute each plan's *first* step (one tool call each). This is the
     "breadth" expansion.
  3. Score the plans by coverage + evidence + cost.
  4. Continue executing the *best-scored* plan through its remaining
     steps. The other plans' first-step work is already in the trace
     (evidence is merged), so the winner benefits from broad exploration.
  5. The last step's answer becomes the final answer.

Why BFS: good for questions where the first move decides a lot and you
want to cheaply compare alternatives before committing. Pays a fixed
cost (N first-steps) for diversification.

Unlike true textbook BFS over a graph, this variant expands one layer
then greedily descends, because deeper BFS is expensive for agents.
We call it BFS because the *shape* of the search front is breadth-first
(all siblings before any descendant).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.events import EventLog
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
class BFSOverPlansPolicy:
    """Breadth-first search over candidate plans."""

    name: str = "bfs_over_plans"
    n_candidates: int = 3
    model: str | None = None
    max_tokens: int = 1024
    temperature_branches: float = 0.7  # higher for diversity
    temperature_exec: float = 0.0
    # Internal state: cached list of candidate plans, the best one picked
    # after first-step exploration, and the cursor into that plan.

    def _ensure_setup(
        self, state: AgentState, llm: LLMClient, tools: ToolRegistry,
    ) -> tuple[list[Plan], Plan | None, int]:
        cached = state.metadata.get("_bfs_plans")
        if cached is None:
            candidates = propose_candidate_plans(
                llm, state.question,
                n_candidates=self.n_candidates,
                model=self.model,
                temperature=self.temperature_branches,
            )
            if not candidates:
                candidates = []
            # Credit the planner LLM call's tokens via the pending-token
            # mechanism so they show up in state.budget. Tokens live on
            # each plan.meta from _planning.propose_candidate_plans.
            t_in = sum(int(p.meta.get("tokens_in") or 0) for p in candidates)
            t_out = sum(int(p.meta.get("tokens_out") or 0) for p in candidates)
            stash_pending_tokens(state, t_in, t_out)
            state.metadata["_bfs_plans"] = candidates
            state.metadata["_bfs_phase"] = "expand"
            state.metadata["_bfs_cursor"] = 0
            state.metadata["_bfs_winner_idx"] = None
        candidates = state.metadata["_bfs_plans"]
        winner_idx = state.metadata.get("_bfs_winner_idx")
        winner = candidates[winner_idx] if winner_idx is not None else None
        cursor = state.metadata.get("_bfs_cursor", 0)
        return candidates, winner, cursor

    def propose(
        self,
        state: AgentState,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
    ) -> Action:
        action = self._propose(state, llm=llm, tools=tools)
        # Drain pending tokens (planner / synthesizer call costs) onto
        # whichever Action we're about to return. The driver's _execute
        # then lifts them onto the Observation and into the budget.
        drain_pending_tokens(state, action.meta)
        return action

    def _propose(
        self,
        state: AgentState,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
    ) -> Action:
        candidates, winner, cursor = self._ensure_setup(state, llm, tools)
        if not candidates:
            return Action(
                kind=ActionKind.FINAL_ANSWER,
                answer="(bfs: no candidate plans)",
                meta={"policy": self.name, "error": "empty_candidates"},
            )

        phase = state.metadata.get("_bfs_phase", "expand")

        # ---- Phase 1: expand — execute first step of each candidate ---
        if phase == "expand":
            idx = state.metadata.get("_bfs_expand_idx", 0)
            if idx < len(candidates):
                ev_before = len(state.evidence)
                plan = candidates[idx]
                result = execute_plan_step(
                    plan, cursor=0,
                    main_question=state.question,
                    llm=llm, tools=tools, state=state,
                    model=self.model, max_tokens=self.max_tokens,
                    temperature=self.temperature_exec,
                )
                plan.with_step_result(0, _result_to_dict(result))
                plan.meta["evidence_added"] = len(state.evidence) - ev_before
                state.metadata["_bfs_expand_idx"] = idx + 1
                return Action(
                    kind=ActionKind.THINK,
                    text=f"[bfs expand {idx+1}/{len(candidates)}] "
                         f"{plan.steps[0][:80]} → {result.answer[:80]}",
                    meta={"policy": self.name, "phase": "expand",
                          "branch": idx,
                          "ok": result.ok},
                )
            # All candidates expanded — pick the winner and descend.
            scored = [
                (i, p, score_plan(
                    p, evidence_before=0,
                    evidence_after=p.meta.get("evidence_added", 0),
                ))
                for i, p in enumerate(candidates)
            ]
            scored.sort(key=lambda x: x[2].total, reverse=True)
            winner_idx = scored[0][0]
            state.metadata["_bfs_winner_idx"] = winner_idx
            state.metadata["_bfs_phase"] = "descend"
            state.metadata["_bfs_cursor"] = 1
            winner = candidates[winner_idx]
            return Action(
                kind=ActionKind.THINK,
                text=(f"[bfs descend] chose branch {winner_idx} "
                      f"({len(winner.steps)} steps, score="
                      f"{scored[0][2].total:.3f})"),
                meta={"policy": self.name, "phase": "descend",
                      "winner_idx": winner_idx,
                      "winner_score": scored[0][2].total},
            )

        # ---- Phase 2: descend — execute remaining steps of winner -----
        assert winner is not None
        if cursor >= len(winner.steps):
            return self._finalize(winner, state, llm)

        result = execute_plan_step(
            winner, cursor=cursor,
            main_question=state.question,
            llm=llm, tools=tools, state=state,
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature_exec,
        )
        winner.with_step_result(cursor, _result_to_dict(result))
        state.metadata["_bfs_cursor"] = cursor + 1

        # If this was the last step, finalize via synthesizer on this tick.
        if cursor + 1 >= len(winner.steps):
            return self._finalize(winner, state, llm)
        return Action(
            kind=ActionKind.THINK,
            text=f"[bfs step {cursor+1}/{len(winner.steps)}] {result.answer[:120]}",
            meta={"policy": self.name, "phase": "descend",
                  "plan_step": cursor, "ok": result.ok},
        )

    def _finalize(self, winner: Plan, state: AgentState, llm: LLMClient) -> Action:
        """Run the synthesizer to produce the user-facing final answer.

        Replaces the old ``result.answer`` shortcut, which exposed
        whichever string the *last* tool call happened to return — a
        memory.write receipt, a Python sandbox stdout, etc. The
        synthesizer reads the full plan trajectory + auto-collected
        evidence and produces an answer to the *main* question.
        """
        text, t_in, t_out = synthesize_final_answer(
            state.question, winner, state,
            llm=llm, model=self.model,
        )
        winner.final_answer = text
        return Action(
            kind=ActionKind.FINAL_ANSWER,
            answer=text,
            meta={
                "policy": self.name, "phase": "done",
                "plan_steps": len(winner.steps),
                "tokens_in": t_in, "tokens_out": t_out,
            },
        )


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
