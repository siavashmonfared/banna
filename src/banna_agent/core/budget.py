"""Budget enforcement helpers.

The Budget dataclass already lives in `types.py`. This module adds the
monotonic-timer lifecycle the driver uses:

    tracker = BudgetTracker(budget)
    tracker.start()
    ...
    tracker.tick(tokens_in=..., tokens_out=..., cost_usd=...)
    reason = tracker.check()  # OK if may continue

The tracker is deliberately outside the dataclass so that `AgentState`
can still be dataclass-copied cheaply when we introduce branching.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .types import Budget, BudgetReason


@dataclass
class BudgetTracker:
    """Wraps a Budget with a live monotonic timer.

    Not a subclass — pure composition so `Budget` stays a plain data
    carrier.
    """

    budget: Budget
    _t0: float | None = field(default=None, init=False, repr=False)
    _paused_at: float | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        self._t0 = time.monotonic()
        self.budget.started_mono = self._t0

    @property
    def started(self) -> bool:
        return self._t0 is not None

    def pause(self) -> None:
        """Stop the wall clock while blocked on a human.

        The wall budget bounds *agent* compute, not how long a person
        takes to answer an `ask_user` prompt or a permission box. Time
        spent paused is excluded from `elapsed_wall_s`. Idempotent.
        """
        if self._t0 is not None and self._paused_at is None:
            self._paused_at = time.monotonic()

    def resume(self) -> None:
        """Resume after a `pause()`, shifting the start forward so the
        paused interval never counts. No-op if not paused.
        """
        if self._t0 is not None and self._paused_at is not None:
            self._t0 += time.monotonic() - self._paused_at
            self.budget.started_mono = self._t0
        self._paused_at = None

    def tick(
        self,
        *,
        step: bool = True,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Advance the tracker. Called by the driver after each step."""
        if self._t0 is None:
            self.start()
        self.budget.elapsed_wall_s = time.monotonic() - self._t0  # type: ignore[arg-type]
        if step:
            self.budget.steps_used += 1
        self.budget.tokens_in += tokens_in
        self.budget.tokens_out += tokens_out
        self.budget.cost_usd += cost_usd

    def check(self) -> BudgetReason:
        if self._t0 is not None:
            self.budget.elapsed_wall_s = time.monotonic() - self._t0
        return self.budget.check()
