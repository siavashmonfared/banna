"""AgentState — the typed object every Policy transforms.

This is the load-bearing abstraction of the project. A Policy is a function
`(state) -> Action`. The driver runs `observe(state, action) -> Observation`
and appends `Step(action, observation)` to the trace.

Options reserved for future use:
- week 2, `with_step(step) -> AgentState` (immutable copy-on-write) is added
  when best-first branching needs cheap forks. Today the methods mutate.
- `parent_state_id` / `branch_id` fields are placeholders for MCTS later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import (
    Action,
    Budget,
    Claim,
    Evidence,
    Observation,
    Step,
    Trace,
    _short_id,
)


@dataclass
class AgentState:
    """Everything a policy needs to decide what to do next.

    Fields:
      question        — the task prompt (immutable over the run).
      trace           — ordered Steps; `trace.final_answer` set when done.
      evidence        — corpus of source-grounded facts the agent has collected.
      claims          — propositions asserted toward the final answer.
      budget          — enforced resource ceilings.
      metadata        — free-form per-run data (task id, provider, policy, etc).
    """

    question: str
    trace: Trace = field(default_factory=lambda: Trace(question=""))
    evidence: list[Evidence] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Reserved for week-2 branching; today every state has no parent.
    state_id: str = field(default_factory=lambda: _short_id("st"))
    parent_state_id: str | None = None

    def __post_init__(self) -> None:
        # Keep trace.question in sync with state.question.
        if not self.trace.question:
            self.trace.question = self.question

    # --- mutation helpers --------------------------------------------------

    def append_step(self, action: Action, observation: Observation) -> Step:
        """Append (action, observation) as the next Step.

        Step counting lives here; *token* accounting is owned by the
        driver's BudgetTracker.tick() so we don't double-count the same
        observation. (Pre-fix bug: tokens were added in both places, so
        every "tok/task" number in reports was 2× the true LLM usage.)
        Wall-time is overwritten by tracker.check() each loop, so the
        running sum here is benign but vestigial.

        Repair steps — empty_reply / commit_required / verifier_retry
        THINKs marked with `meta["repair"]=True` — are tracked on a
        separate axis (`budget.repair_steps_used`) and do NOT consume
        `budget.steps_used`. Any productive step resets the repair
        counter so a single hiccup mid-trace doesn't poison the rest.
        """
        step = Step(
            idx=len(self.trace.steps),
            action=action,
            observation=observation,
        )
        self.trace.steps.append(step)
        is_repair = bool((action.meta or {}).get("repair"))
        if is_repair:
            self.budget.repair_steps_used += 1
        else:
            self.budget.steps_used += 1
            # A productive step clears any prior repair streak.
            self.budget.repair_steps_used = 0
        if action.answer is not None:
            self.trace.final_answer = action.answer
        return step

    def add_evidence(
        self,
        *,
        source: str,
        content: str,
        origin_step: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Evidence:
        ev = Evidence(
            source=source,
            content=content,
            origin_step=origin_step,
            meta=meta or {},
        )
        self.evidence.append(ev)
        return ev

    def add_claim(
        self,
        *,
        text: str,
        supports: list[str] | None = None,
        origin_step: int | None = None,
    ) -> Claim:
        cl = Claim(
            text=text,
            supports=supports or [],
            origin_step=origin_step,
        )
        self.claims.append(cl)
        return cl

    # --- convenience queries ----------------------------------------------

    @property
    def last_step(self) -> Step | None:
        return self.trace.steps[-1] if self.trace.steps else None

    @property
    def is_done(self) -> bool:
        return self.trace.final_answer is not None

    def evidence_for(self, claim: Claim) -> list[Evidence]:
        supported = set(claim.supports)
        return [ev for ev in self.evidence if ev.evidence_id in supported]
