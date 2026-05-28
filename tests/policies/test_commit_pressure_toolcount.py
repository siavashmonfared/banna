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


def test_duplicate_skipped_calls_do_not_count() -> None:
    # 8 single tool calls but all duplicate-skipped (a loop the dedup guard
    # already neutralized) must NOT trip commit-pressure — otherwise the
    # nudge fires on a loop and can cut the model off as it pivots.
    policy = ReActPolicy(commit_pressure_threshold=0.6)
    state = AgentState(question="?", budget=Budget(max_steps=10))
    for _ in range(8):
        action = Action(kind=ActionKind.TOOL_CALL, tool_name="search",
                        tool_args={"q": "same"})
        state.append_step(action, Observation(ok=True, data={"duplicate_skipped": True}))
    state.budget.steps_used = 8
    # steps_used is 8 but they're all skips → productive count 0, and the
    # nudge keys on max(steps, productive). steps_used dominates here, so it
    # still fires on raw steps — the point is the *productive* tally is 0.
    # Use a fresh trace where steps_used is low to isolate the call tally:
    state2 = AgentState(question="?", budget=Budget(max_steps=10))
    for _ in range(8):
        action = Action(
            kind=ActionKind.TOOL_BATCH,
            meta={"batch_calls": [{"name": "search", "args": {"q": "same"}}]},
        )
        state2.append_step(action, Observation(
            ok=True, data={"batch": [{"name": "search",
                                      "result": {"duplicate_skipped": True}}]}))
    state2.budget.steps_used = 2  # only the proposes that weren't dups
    assert policy._step_pressure_message(state2) is None, (
        "duplicate-skipped batch calls must not inflate the commit-pressure tally"
    )


def test_productive_batch_calls_still_count() -> None:
    # Same shape but the sub-calls are real (not skipped) → they count.
    policy = ReActPolicy(commit_pressure_threshold=0.6)
    state = AgentState(question="?", budget=Budget(max_steps=10))
    for _ in range(4):
        action = Action(
            kind=ActionKind.TOOL_BATCH,
            meta={"batch_calls": [{"name": "search", "args": {"q": str(i)}}
                                  for i in range(2)]},
        )
        state.append_step(action, Observation(
            ok=True, data={"batch": [{"name": "search", "result": {"hits": []}},
                                     {"name": "search", "result": {"hits": []}}]}))
    state.budget.steps_used = 4
    assert policy._step_pressure_message(state) is not None  # 8 productive calls
