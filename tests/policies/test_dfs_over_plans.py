"""Tests for DFSOverPlansPolicy."""
from __future__ import annotations

import json
from dataclasses import dataclass, field


from banna_agent.core.agent import run_policy
from banna_agent.core.state import AgentState
from banna_agent.core.types import Budget
from banna_agent.llm.base import ContentBlock, LLMReply, Usage
from banna_agent.policies.dfs_over_plans import DFSOverPlansPolicy
from banna_agent.tools.base import ToolRegistry


@dataclass
class _ScriptedLLM:
    replies: list[LLMReply] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    provider: str = "s"

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0) if self.replies else LLMReply(
            provider="s", model="m", content=[], stop_reason="end_turn"
        )


def _text(t: str) -> LLMReply:
    return LLMReply(
        provider="s", model="m",
        content=[ContentBlock(kind="text", text=t)],
        stop_reason="end_turn",
        usage=Usage(tokens_in=5, tokens_out=2),
    )


def _plans_reply(plans: list[list[str]]) -> LLMReply:
    return _text(json.dumps({"plans": plans}))


# ---------------------------------------------------------------------------
# Happy path: first plan succeeds, no backtrack
# ---------------------------------------------------------------------------


def test_dfs_first_plan_succeeds() -> None:
    llm = _ScriptedLLM([
        _plans_reply([["a1", "a2"], ["b1"]]),
        _text("resolution a1"),
        _text("final answer from a2"),
        _text("the synthesized answer"),  # synthesizer for plan a
    ])
    state = AgentState(question="Q", budget=Budget(max_steps=10, max_wall_s=5.0))
    state = run_policy(state, DFSOverPlansPolicy(n_candidates=2),
                       llm=llm, tools=ToolRegistry())
    assert state.is_done
    # Final answer is the synthesizer's output, not the last step's result.
    # Post-Phase-2: literal model answer submitted (no canonicalize).
    assert state.trace.final_answer == "the synthesized answer"


# ---------------------------------------------------------------------------
# Backtrack path: first plan's answer fails filter, second plan succeeds
# ---------------------------------------------------------------------------


def test_dfs_backtracks_on_unacceptable_answer() -> None:
    llm = _ScriptedLLM([
        _plans_reply([["a1"], ["b1"]]),
        _text("i don't know"),       # branch 0 step 0
        _text("i don't know either"),  # synthesizer for branch 0 → rejected
        _text("correct answer"),      # branch 1 step 0
        _text("the correct synthesized answer"),  # synthesizer for branch 1
    ])
    state = AgentState(question="Q", budget=Budget(max_steps=10, max_wall_s=5.0))
    state = run_policy(state, DFSOverPlansPolicy(n_candidates=2),
                       llm=llm, tools=ToolRegistry())
    assert state.is_done
    assert state.trace.final_answer == "the correct synthesized answer"


def test_dfs_exhausts_all_branches() -> None:
    llm = _ScriptedLLM([
        _plans_reply([["a"], ["b"]]),
        _text("unknown"),
        _text("cannot determine"),
    ])
    state = AgentState(question="Q", budget=Budget(max_steps=10, max_wall_s=5.0))
    state = run_policy(state, DFSOverPlansPolicy(n_candidates=2),
                       llm=llm, tools=ToolRegistry())
    assert state.is_done
    # Best-effort: either returns one of the rejected answers or the "all failed" marker.
    assert state.trace.final_answer is not None


def test_dfs_handles_empty_candidates() -> None:
    llm = _ScriptedLLM([_text("garbage")])
    state = AgentState(question="Q", budget=Budget(max_steps=5, max_wall_s=5.0))
    state = run_policy(state, DFSOverPlansPolicy(n_candidates=2),
                       llm=llm, tools=ToolRegistry())
    assert state.is_done
    assert "no candidate plans" in (state.trace.final_answer or "")


def test_dfs_plan_cached() -> None:
    llm = _ScriptedLLM([
        _plans_reply([["a"]]),
        _text("answer"),
    ])
    state = AgentState(question="Q", budget=Budget(max_steps=10, max_wall_s=5.0))
    state = run_policy(state, DFSOverPlansPolicy(n_candidates=1),
                       llm=llm, tools=ToolRegistry())
    planner_calls = sum(
        1 for c in llm.calls
        if "research planner" in c.get("system", "").lower()
    )
    assert planner_calls == 1
