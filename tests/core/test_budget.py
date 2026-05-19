"""Unit tests for BudgetTracker."""
from __future__ import annotations

import time

from banna_agent.core.budget import BudgetTracker
from banna_agent.core.types import Budget, BudgetReason


def test_tracker_starts_and_ticks() -> None:
    b = Budget(max_steps=5, max_wall_s=10.0)
    t = BudgetTracker(b)
    t.start()
    t.tick(tokens_in=10, tokens_out=5)
    assert b.steps_used == 1
    assert b.tokens_in == 10
    assert b.tokens_out == 5
    assert b.elapsed_wall_s >= 0.0


def test_tracker_auto_starts_on_first_tick() -> None:
    b = Budget(max_steps=5, max_wall_s=10.0)
    t = BudgetTracker(b)
    t.tick()
    assert t.started


def test_tracker_trips_steps_budget() -> None:
    b = Budget(max_steps=2, max_wall_s=100.0)
    t = BudgetTracker(b)
    t.start()
    t.tick(); t.tick()
    assert t.check() == BudgetReason.STEPS


def test_tracker_wall_clock_is_updated_on_check() -> None:
    b = Budget(max_steps=100, max_wall_s=0.05)
    t = BudgetTracker(b)
    t.start()
    time.sleep(0.08)
    assert t.check() == BudgetReason.WALL


def test_tracker_step_flag_toggles_step_counter() -> None:
    b = Budget(max_steps=10, max_wall_s=100.0)
    t = BudgetTracker(b)
    t.start()
    t.tick(step=False, tokens_in=5)
    assert b.steps_used == 0
    assert b.tokens_in == 5


def test_tracker_cost_accumulates() -> None:
    b = Budget(max_steps=10, max_wall_s=100.0, max_cost_usd=0.01)
    t = BudgetTracker(b)
    t.start()
    t.tick(cost_usd=0.006)
    assert t.check() == BudgetReason.OK
    t.tick(cost_usd=0.006)
    assert t.check() == BudgetReason.COST


def test_max_repair_steps_trips_separate_axis() -> None:
    """Repair steps have their own ceiling; tripping it surfaces as
    BudgetReason.REPAIR_STEPS, not STEPS."""
    b = Budget(max_steps=100, max_wall_s=100.0, max_repair_steps=3)
    b.repair_steps_used = 3
    t = BudgetTracker(b)
    t.start()
    assert t.check() == BudgetReason.REPAIR_STEPS
