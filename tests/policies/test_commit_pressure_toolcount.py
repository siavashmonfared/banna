"""Commit-pressure nudge keys on tool-call count, not just steps.

Regression for the L3 search-loopers: a reply that emits several parallel
tool calls is one TOOL_BATCH *step*, so a model can fire 2-3x as many
tool calls as steps and slip under a step-only threshold. The nudge must
count the underlying calls so batching can't hide a loop.
"""
from __future__ import annotations

from banna_agent.core.state import AgentState
from banna_agent.core.types import Action, ActionKind, Budget, Observation
from banna_agent.policies.react import ReActPolicy


def _state_with_batches(n_batches: int, calls_per_batch: int,
                        max_steps: int = 10) -> AgentState:
    state = AgentState(question="?", budget=Budget(max_steps=max_steps))
    for _ in range(n_batches):
        action = Action(
            kind=ActionKind.TOOL_BATCH,
            meta={"batch_calls": [{"name": "search", "args": {"q": str(i)}}
                                  for i in range(calls_per_batch)]},
        )
        state.append_step(action, Observation(ok=True, data={"n": calls_per_batch}))
    state.budget.steps_used = n_batches
    return state


def test_nudge_fires_on_batched_tool_volume() -> None:
    # 4 batch-steps x 2 calls = 8 tool calls > 0.6 * 10, but only 4 steps
    # (4/10 = 0.4, under the step threshold). Counting calls must trip it.
    policy = ReActPolicy(commit_pressure_threshold=0.6)
    state = _state_with_batches(n_batches=4, calls_per_batch=2, max_steps=10)
    assert state.budget.steps_used == 4  # would NOT trip a step-only check
    msg = policy._step_pressure_message(state)
    assert msg is not None, "8 tool calls should trip the commit nudge"
    assert "commit" in msg.content[0].text.lower()


def test_nudge_silent_when_well_under_budget() -> None:
    policy = ReActPolicy(commit_pressure_threshold=0.6)
    state = _state_with_batches(n_batches=1, calls_per_batch=2, max_steps=10)
    assert policy._step_pressure_message(state) is None
