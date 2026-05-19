"""Tests for BFSOverPlansPolicy."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from banna_agent.core.agent import run_policy
from banna_agent.core.state import AgentState
from banna_agent.core.types import Budget
from banna_agent.llm.base import ContentBlock, LLMReply, Usage
from banna_agent.policies.bfs_over_plans import BFSOverPlansPolicy
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


def _text(t: str, tok_out=5) -> LLMReply:
    return LLMReply(
        provider="s", model="m",
        content=[ContentBlock(kind="text", text=t)],
        stop_reason="end_turn",
        usage=Usage(tokens_in=10, tokens_out=tok_out),
    )


def _plans_reply(plans: list[list[str]]) -> LLMReply:
    return _text(json.dumps({"plans": plans}))


# ---------------------------------------------------------------------------
# End-to-end: 2 candidates, expand, descend, final answer
# ---------------------------------------------------------------------------


def test_bfs_expands_candidates_then_descends_best() -> None:
    llm = _ScriptedLLM([
        _plans_reply([["a1", "a2"], ["b1", "b2"]]),   # propose plans
        _text("a1 result (short)", tok_out=2),         # expand branch 0 step 0
        _text("b1 result (long high quality)", tok_out=2),  # expand branch 1 step 0
        # descend phase kicks in — winner is picked in a THINK tick
        _text("winning step 2 result"),                # winner step 1
        _text("the synthesized final answer"),         # synthesizer call
    ])
    state = AgentState(question="Q", budget=Budget(max_steps=10, max_wall_s=5.0))
    state = run_policy(state, BFSOverPlansPolicy(n_candidates=2),
                       llm=llm, tools=ToolRegistry())
    assert state.is_done
    # Final answer comes from the synthesizer call, not from the last step's
    # raw tool result. (Pre-fix the answer was result.answer; that exposed
    # memory.write receipts and other side-effecting tool returns to users.)
    # Post-Phase-2: literal model answer submitted (no canonicalize).
    assert state.trace.final_answer == "the synthesized final answer"
    # Metadata reflects BFS progress.
    assert state.metadata["_bfs_phase"] == "descend"
    assert state.metadata["_bfs_winner_idx"] in (0, 1)


def test_bfs_handles_empty_candidates() -> None:
    llm = _ScriptedLLM([_text("not valid json")])
    state = AgentState(question="Q", budget=Budget(max_steps=5, max_wall_s=5.0))
    state = run_policy(state, BFSOverPlansPolicy(n_candidates=2),
                       llm=llm, tools=ToolRegistry())
    assert state.is_done
    assert "no candidate plans" in (state.trace.final_answer or "")


def test_bfs_plans_cached_across_ticks() -> None:
    llm = _ScriptedLLM([
        _plans_reply([["a1"], ["b1"]]),
        _text("a answer"),
        _text("b answer"),
        # descend phase
    ])
    state = AgentState(question="Q", budget=Budget(max_steps=10, max_wall_s=5.0))
    state = run_policy(state, BFSOverPlansPolicy(n_candidates=2),
                       llm=llm, tools=ToolRegistry())
    # Planner call made once; step executor called twice (one per branch's first step).
    system_prompts = [c.get("system", "") for c in llm.calls]
    planner_count = sum(1 for s in system_prompts if "research planner" in s.lower())
    assert planner_count == 1


def test_bfs_single_step_plans_finalize_after_expand() -> None:
    # Both plans have 1 step each. After expansion, descend-phase should
    # immediately finalize via the synthesizer.
    llm = _ScriptedLLM([
        _plans_reply([["step one"], ["step alt"]]),
        _text("answer one"),
        _text("answer alt"),
        _text("synthesized answer from step results"),  # synthesizer
    ])
    state = AgentState(question="Q", budget=Budget(max_steps=10, max_wall_s=5.0))
    state = run_policy(state, BFSOverPlansPolicy(n_candidates=2),
                       llm=llm, tools=ToolRegistry())
    assert state.is_done
    assert state.trace.final_answer == "synthesized answer from step results"
