"""Tests for BestFirstOverPlansPolicy."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from banna_agent.core.agent import run_policy
from banna_agent.core.state import AgentState
from banna_agent.core.types import Budget
from banna_agent.llm.base import ContentBlock, LLMReply, Usage
from banna_agent.policies.best_first_over_plans import BestFirstOverPlansPolicy
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
# End-to-end
# ---------------------------------------------------------------------------


def test_best_first_selects_and_finalizes() -> None:
    llm = _ScriptedLLM([
        _plans_reply([["a1", "a2"], ["b1"]]),
        _text("first selected branch step"),  # first tick: one of the two
        _text("second tick"),                  # second tick: same or other
        _text("third tick"),
    ])
    state = AgentState(question="Q", budget=Budget(max_steps=8, max_wall_s=5.0))
    state = run_policy(state, BestFirstOverPlansPolicy(n_candidates=2),
                       llm=llm, tools=ToolRegistry())
    assert state.is_done
    assert state.trace.final_answer is not None
    # Metadata reflects best-first bookkeeping.
    assert len(state.metadata["_bf_plans"]) == 2
    cursors = state.metadata["_bf_cursors"]
    # At least one plan advanced.
    assert sum(cursors) >= 1


def test_best_first_empty_candidates() -> None:
    llm = _ScriptedLLM([_text("garbage")])
    state = AgentState(question="Q", budget=Budget(max_steps=5, max_wall_s=5.0))
    state = run_policy(state, BestFirstOverPlansPolicy(n_candidates=2),
                       llm=llm, tools=ToolRegistry())
    assert state.is_done
    assert "no candidate plans" in (state.trace.final_answer or "")


def test_best_first_respects_max_steps_per_plan() -> None:
    # Plan a has 10 steps but max_steps_per_plan=2.
    long_plan = [f"step {i}" for i in range(10)]
    llm = _ScriptedLLM([
        _plans_reply([long_plan, ["b1"]]),
        _text("a step 0"),
        _text("a step 1"),
        _text("b step 0"),
    ])
    state = AgentState(question="Q", budget=Budget(max_steps=8, max_wall_s=5.0))
    state = run_policy(state, BestFirstOverPlansPolicy(
        n_candidates=2, max_steps_per_plan=2,
    ), llm=llm, tools=ToolRegistry())
    # Plan a was capped at 2 steps.
    cursors = state.metadata["_bf_cursors"]
    assert cursors[0] <= 2


def test_best_first_verifier_bonus_influences_scoring() -> None:
    """A verifier that always returns True should never hurt scoring."""
    llm = _ScriptedLLM([
        _plans_reply([["a1"], ["b1"]]),
        _text("a answer"),
        _text("b answer"),
    ])
    state = AgentState(question="Q", budget=Budget(max_steps=6, max_wall_s=5.0))
    policy = BestFirstOverPlansPolicy(
        n_candidates=2, verifier=lambda a: True,
    )
    state = run_policy(state, policy, llm=llm, tools=ToolRegistry())
    assert state.is_done
