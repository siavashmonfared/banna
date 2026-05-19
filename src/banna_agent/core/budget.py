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

    def start(self) -> None:
        self._t0 = time.monotonic()
        self.budget.started_mono = self._t0

    @property
    def started(self) -> bool:
        return self._t0 is not None

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
