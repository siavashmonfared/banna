"""Unit tests for core.types and core.state.

These tests exercise the *substrate contract* — the behaviors every policy and
verifier silently depends on. If any of these ever break, something much
bigger is wrong.
"""
from __future__ import annotations

import pytest

from banna_agent.core.state import AgentState
from banna_agent.core.types import (
    Action,
    ActionKind,
    Budget,
    BudgetReason,
    Observation,
)


# ---------------------------------------------------------------------------
# AgentState construction
# ---------------------------------------------------------------------------


def test_agentstate_defaults_are_empty() -> None:
    s = AgentState(question="What is 2+2?")
    assert s.question == "What is 2+2?"
    assert s.trace.question == "What is 2+2?"
    assert s.trace.steps == []
    assert s.evidence == []
    assert s.claims == []
    assert not s.is_done
    assert s.last_step is None


def test_trace_question_synced_from_state() -> None:
    s = AgentState(question="Q?")
    assert s.trace.question == "Q?"


def test_state_ids_are_unique() -> None:
    a = AgentState(question="Q")
    b = AgentState(question="Q")
    assert a.state_id != b.state_id
    assert a.trace.run_id != b.trace.run_id


# ---------------------------------------------------------------------------
# append_step — budget accounting + trace ordering
# ---------------------------------------------------------------------------


def test_append_step_increments_indices() -> None:
    s = AgentState(question="Q")
    for i in range(3):
        s.append_step(
            Action(kind=ActionKind.THINK, text=f"thought {i}"),
            Observation(ok=True, text=f"thought {i}"),
        )
    assert [step.idx for step in s.trace.steps] == [0, 1, 2]


def test_append_step_updates_step_counter() -> None:
    """append_step now owns step counting only; tokens and wall are
    accrued by the driver's BudgetTracker.tick (single source of truth
    fixes the prior double-count: tokens were added here AND in tick)."""
    s = AgentState(question="Q", budget=Budget(max_steps=10, max_wall_s=100.0))
    s.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="search", tool_args={"q": "x"}),
        Observation(ok=True, data={"hits": []}, wall_s=1.5, tokens_in=20, tokens_out=5),
    )
    assert s.budget.steps_used == 1
    # tokens are not accrued by append_step — only by tracker.tick().
    assert s.budget.tokens_in == 0
    assert s.budget.tokens_out == 0


def test_final_answer_marks_state_done() -> None:
    s = AgentState(question="Q")
    assert not s.is_done
    s.append_step(
        Action(kind=ActionKind.FINAL_ANSWER, answer="42"),
        Observation(ok=True, text="42"),
    )
    assert s.is_done
    assert s.trace.final_answer == "42"


def test_last_step_returns_latest() -> None:
    s = AgentState(question="Q")
    s.append_step(
        Action(kind=ActionKind.THINK, text="first"),
        Observation(ok=True, text="first"),
    )
    s.append_step(
        Action(kind=ActionKind.THINK, text="second"),
        Observation(ok=True, text="second"),
    )
    assert s.last_step is not None
    assert s.last_step.action.text == "second"


# ---------------------------------------------------------------------------
# Evidence + Claim linking
# ---------------------------------------------------------------------------


def test_add_evidence_and_claim_linkage() -> None:
    s = AgentState(question="Q")
    ev = s.add_evidence(source="http://example.com", content="Netflix ARPU was $11.64")
    cl = s.add_claim(text="Netflix 2023 ARPU = $11.64", supports=[ev.evidence_id])
    assert s.evidence_for(cl) == [ev]


def test_claim_without_supports_has_no_evidence() -> None:
    s = AgentState(question="Q")
    s.add_evidence(source="x", content="y")
    cl = s.add_claim(text="unsupported")
    assert s.evidence_for(cl) == []


# ---------------------------------------------------------------------------
# Budget — the check() state machine
# ---------------------------------------------------------------------------


def test_budget_ok_by_default() -> None:
    b = Budget()
    assert b.check() == BudgetReason.OK


def test_budget_trips_on_steps() -> None:
    b = Budget(max_steps=2)
    b.steps_used = 2
    assert b.check() == BudgetReason.STEPS


def test_budget_trips_on_wall() -> None:
    b = Budget(max_wall_s=1.0)
    b.elapsed_wall_s = 1.5
    assert b.check() == BudgetReason.WALL


def test_budget_trips_on_tokens_when_capped() -> None:
    b = Budget(max_tokens_total=100)
    b.tokens_in = 60
    b.tokens_out = 50
    assert b.check() == BudgetReason.TOKENS


def test_budget_trips_on_cost_when_capped() -> None:
    b = Budget(max_cost_usd=0.10)
    b.cost_usd = 0.15
    assert b.check() == BudgetReason.COST


def test_uncapped_token_and_cost_are_telemetry_only() -> None:
    b = Budget()  # no max_tokens_total, no max_cost_usd
    b.tokens_in = 10**9
    b.cost_usd = 10**6
    assert b.check() == BudgetReason.OK


def test_budget_remaining_helpers() -> None:
    b = Budget(max_steps=5, max_wall_s=10.0)
    b.steps_used = 2
    b.elapsed_wall_s = 3.0
    assert b.remaining_steps() == 3
    assert b.remaining_wall_s() == pytest.approx(7.0)


def test_budget_remaining_clamped_to_zero() -> None:
    b = Budget(max_steps=3, max_wall_s=2.0)
    b.steps_used = 10
    b.elapsed_wall_s = 10.0
    assert b.remaining_steps() == 0
    assert b.remaining_wall_s() == 0.0


# ---------------------------------------------------------------------------
# Repair-step accounting (C1)
# ---------------------------------------------------------------------------


def test_repair_step_does_not_tick_main_budget() -> None:
    """A THINK tagged `meta['repair']=True` doesn't consume steps_used."""
    s = AgentState(question="Q")
    for _ in range(3):
        s.append_step(
            Action(
                kind=ActionKind.THINK,
                text="[empty_reply] foo",
                meta={"repair": True, "empty_reply": True},
            ),
            Observation(ok=True, text="x"),
        )
    assert s.budget.steps_used == 0
    assert s.budget.repair_steps_used == 3


def test_productive_step_resets_repair_streak() -> None:
    s = AgentState(question="Q")
    s.append_step(
        Action(kind=ActionKind.THINK, text="[empty_reply]", meta={"repair": True}),
        Observation(ok=True, text=""),
    )
    assert s.budget.repair_steps_used == 1
    s.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="search", tool_args={"q": "x"}),
        Observation(ok=True),
    )
    assert s.budget.repair_steps_used == 0
    assert s.budget.steps_used == 1


def test_normal_think_still_consumes_main_budget() -> None:
    s = AgentState(question="Q")
    s.append_step(
        Action(kind=ActionKind.THINK, text="ordinary reasoning"),
        Observation(ok=True),
    )
    assert s.budget.steps_used == 1
    assert s.budget.repair_steps_used == 0
