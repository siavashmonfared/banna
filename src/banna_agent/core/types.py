"""Core typed substrate.

Every policy (ReAct, verifier_retry, best_first, eventually MCTS) operates on
`AgentState` via this vocabulary. The rule is: the substrate is the thing that
persists; the text representation is one *projection* of it, not the source
of truth.

Design choice — option (b): dataclasses are **mutable** for now. Policies call
`state.append_step(...)` / `state.add_evidence(...)` directly. When best-first
branching forces cheap copies in week 2, we'll refactor to frozen dataclasses
with `state.with_step(...)` returning a new AgentState. That refactor is the
point — it's where you learn why immutability matters for search.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _now_iso() -> str:
    """UTC timestamp, ISO 8601, second resolution. Used for replayable logs."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _short_id(prefix: str) -> str:
    """Short unique id like 'step_a1b2c3' — enough to correlate log lines."""
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Actions & observations — the smallest unit of work
# ---------------------------------------------------------------------------


class ActionKind(str, Enum):
    """What a policy can propose the agent do next."""

    THINK = "think"            # internal reasoning, no side effect
    TOOL_CALL = "tool_call"    # invoke a tool with arguments
    TOOL_BATCH = "tool_batch"  # invoke ≥2 independent tools concurrently
    FINAL_ANSWER = "final_answer"  # terminate with an answer


@dataclass
class Action:
    """An action proposed by a policy.

    Exactly one of `text` / (`tool_name`,`tool_args`) / `answer` is populated,
    matched to `kind`. We keep this as a single type (rather than a sum type)
    because the whole point of this project is that ReAct, verifier_retry,
    and best_first all produce *the same Action shape* — only their *policy*
    for choosing differs.
    """

    kind: ActionKind
    text: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None
    # Free-form metadata — e.g. best_first plan id, model name, branch score.
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """The result of executing an Action.

    For THINK actions, `text` is the same as Action.text and `ok=True`.
    For TOOL_CALL, `data` is the tool's dict return or error payload.
    For FINAL_ANSWER, `text` echoes the answer and `ok=True` — the driver
    stops at this point.
    """

    ok: bool
    text: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # Wall time the action took, in seconds. Useful for cost-normalized eval.
    wall_s: float = 0.0
    # Token telemetry when the action was an LLM call; 0 when it was a tool.
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class Step:
    """One tick of the transition function: (action, observation)."""

    idx: int
    action: Action
    observation: Observation
    step_id: str = field(default_factory=lambda: _short_id("step"))
    ts: str = field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# Evidence & claims — the substrate the verifier will read in week 2
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """A piece of external information the agent is relying on.

    Created automatically when a tool call returns a source-grounded result
    (search hit, URL fetch, file read). Policies can also append Evidence
    by hand when they paraphrase a document.
    """

    evidence_id: str = field(default_factory=lambda: _short_id("ev"))
    source: str = ""          # URL, file path, tool name, or free text
    content: str = ""         # extracted text / summary
    origin_step: int | None = None  # which Step produced this
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Claim:
    """A proposition the agent asserts toward the final answer.

    Verifiers (week 2) read `claims` and attach ClaimCheck results. Every
    claim should ideally cite ≥1 Evidence. Unsupported claims are the main
    GAIA failure mode the citation verifier catches.
    """

    claim_id: str = field(default_factory=lambda: _short_id("cl"))
    text: str = ""
    supports: list[str] = field(default_factory=list)  # Evidence ids
    origin_step: int | None = None
    # Populated by verifiers — not the policy. Kept on the claim itself so a
    # branch of search can drop/keep a claim independently of its siblings.
    verdicts: dict[str, str] = field(default_factory=dict)  # verifier_name -> "ok"|"fail"|"warn"


# ---------------------------------------------------------------------------
# Trace — the replayable transcript of the whole run
# ---------------------------------------------------------------------------


@dataclass
class Trace:
    """Ordered list of Steps plus a top-level question.

    A Trace is *projectable* to a prompt (for the next LLM call) and
    *serializable* to JSONL (for the event log). Both projections are
    defined elsewhere — this type is the in-memory source of truth.
    """

    question: str
    steps: list[Step] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: _short_id("run"))
    started_at: str = field(default_factory=_now_iso)
    # Final answer once FINAL_ANSWER is produced, else None.
    final_answer: str | None = None


# ---------------------------------------------------------------------------
# Budget — enforced by the driver each tick
# ---------------------------------------------------------------------------


class BudgetReason(str, Enum):
    """Why the driver stopped, if it did."""

    OK = "ok"
    WALL = "budget_wall"
    STEPS = "budget_steps"
    TOKENS = "budget_tokens"
    COST = "budget_cost"
    REPAIR_STEPS = "budget_repair_steps"


@dataclass
class Budget:
    """Soft budget with enforcement at tick boundaries.

    `max_wall_s` and `max_steps` are enforced. `max_tokens_total` and
    `max_cost_usd` are telemetry by default but can be made hard limits
    by the driver.

    Mirrors the shape of Banna's `agents_lib/protocol.py::Budget` so
    the AgentRunner adapter is a trivial copy.
    """

    max_wall_s: float = 60.0
    max_steps: int = 8
    max_tokens_total: int | None = None
    max_cost_usd: float | None = None
    # Cap on consecutive repair steps (empty_reply / commit_required /
    # verifier_retry). These are exempt from `max_steps` because they
    # don't represent productive model work, but a runaway repair loop
    # still needs its own ceiling so we don't burn forever.
    max_repair_steps: int = 6

    # Live counters — incremented by the driver as steps execute.
    started_mono: float | None = None
    elapsed_wall_s: float = 0.0
    steps_used: int = 0
    # Repair steps consumed since the last non-repair step. Reset to 0
    # on any productive step (TOOL_CALL / non-repair THINK / FINAL_ANSWER).
    repair_steps_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.steps_used)

    def remaining_wall_s(self) -> float:
        return max(0.0, self.max_wall_s - self.elapsed_wall_s)

    def check(self) -> BudgetReason:
        """Return OK if we may keep going, else the reason to stop."""
        if self.elapsed_wall_s >= self.max_wall_s:
            return BudgetReason.WALL
        if self.steps_used >= self.max_steps:
            return BudgetReason.STEPS
        if self.repair_steps_used >= self.max_repair_steps:
            return BudgetReason.REPAIR_STEPS
        if (
            self.max_tokens_total is not None
            and (self.tokens_in + self.tokens_out) >= self.max_tokens_total
        ):
            return BudgetReason.TOKENS
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            return BudgetReason.COST
        return BudgetReason.OK
