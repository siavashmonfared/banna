"""Best-of-N policy — run an inner policy N times and pick a winner.

The N runs are independent: each starts from a fresh copy of the state
(same question, same per-run budget), and the inner policy's own
verifier_retry loop runs to completion inside each. After N candidates
land, a selector picks one. Two selectors are supported:

  - ``majority_vote`` — canonicalize each candidate (lowercase + strip
    surrounding punctuation) and pick the mode. Ties broken by lowest
    token usage so cheaper trajectories win ties. Free.

  - ``llm_judge`` — one LLM call rating the candidates with the
    question + answer + a brief reasoning summary visible. Cost is one
    extra short completion. Returns the best index.

Why this lives as a Policy (instead of a separate driver function):
the existing CLI / experiment runner is keyed on `--policy`, so making
this a Policy means no infrastructure change — pick `best_of_n` and
everything else (budget tracking, event logging, cost tally, on_run_end
hooks) keeps working. The first `propose()` call performs all N inner
runs and emits a single FINAL_ANSWER; subsequent calls (if any) replay
the cached winner.

Per-candidate telemetry is recorded into
``state.metadata["best_of_n"]["candidates"]`` so the failure forensics
can inspect why a particular trajectory was discarded.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.state import AgentState
from ..core.types import Action, ActionKind, Budget
from ..llm.base import ContentBlock, LLMClient, Message
from ..tools.base import ToolRegistry
from .react import ReActPolicy
from .verifier_retry import VerifierRetryPolicy


def _canonical_answer(s: str) -> str:
    """Cheap normalization for vote bucketing — lowercase, strip outer
    whitespace and surrounding punctuation. Conservatively *narrow*: we
    don't drop articles or strip units, because that would merge
    "the castle" with "castle" (often wrong intent collapse) and could
    silently flip a vote. The scorer's own normalization runs later on
    the winning string; this is only for grouping equivalents."""
    if not s:
        return ""
    out = s.strip().lower()
    # Trim trailing punctuation that the model often adds without meaning.
    out = re.sub(r"[\s\.,!?;:\"'`]+$", "", out)
    out = re.sub(r"^[\s\.,!?;:\"'`]+", "", out)
    return out


@dataclass
class _Candidate:
    idx: int
    answer: str
    tokens_in: int
    tokens_out: int
    n_steps: int
    finished_reason: str  # "final" | "budget_steps" | "budget_wall" | ...


def _select_majority(cands: list[_Candidate]) -> int:
    """Vote on canonical answers. Ties: lowest token usage wins.

    Empty/whitespace canonical strings get zero weight — an empty
    answer from a budget-exhausted run shouldn't outvote two real
    answers that happened to disagree.
    """
    buckets: dict[str, list[_Candidate]] = {}
    for c in cands:
        key = _canonical_answer(c.answer)
        if not key:
            continue
        buckets.setdefault(key, []).append(c)
    if not buckets:
        return 0  # nothing usable; first candidate as fallback
    # Sort buckets by (count desc, min(tokens_in + tokens_out) asc).
    ranked = sorted(
        buckets.items(),
        key=lambda kv: (
            -len(kv[1]),
            min(c.tokens_in + c.tokens_out for c in kv[1]),
        ),
    )
    winning_bucket = ranked[0][1]
    # Within the winning bucket, prefer the cheapest trajectory.
    chosen = min(winning_bucket, key=lambda c: c.tokens_in + c.tokens_out)
    return chosen.idx


def _select_llm_judge(
    cands: list[_Candidate],
    *,
    question: str,
    llm: LLMClient,
    model: str | None,
) -> int:
    """One short LLM call ranking the candidates.

    Empty answers are pre-filtered (the judge would otherwise feel
    obligated to rank them). On parse failure, falls back to majority.
    """
    keep = [c for c in cands if _canonical_answer(c.answer)]
    if not keep:
        return _select_majority(cands)
    if len(keep) == 1:
        return keep[0].idx

    listing = "\n".join(
        f"  ({i + 1}) {c.answer!r}" for i, c in enumerate(keep)
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Candidate answers from {len(keep)} independent attempts:\n"
        f"{listing}\n\n"
        f"Reply with ONLY the index number (1, 2, ...) of the candidate "
        f"that best answers the question. No explanation."
    )
    try:
        reply = llm.chat(
            messages=[
                Message(role="user", content=[ContentBlock(kind="text", text=prompt)])
            ],
            model=model,
            max_tokens=8,
            temperature=0.0,
        )
        m = re.search(r"\b([0-9]+)\b", reply.text or "")
        if m:
            pick = int(m.group(1)) - 1
            if 0 <= pick < len(keep):
                return keep[pick].idx
    except Exception:
        pass
    return _select_majority(cands)


@dataclass
class BestOfNPolicy:
    """Run ``inner`` N times on fresh state copies; pick a winner.

    Fields:
      n            - number of independent trajectories (default 3)
      inner        - the policy run on each (default verifier_retry+react)
      selector     - "majority_vote" | "llm_judge"
      judge_model  - model passed to the judge call (None → llm default)
    """

    name: str = "best_of_n"
    n: int = 3
    selector: str = "majority_vote"
    inner: Any = field(default_factory=lambda: VerifierRetryPolicy(
        inner=ReActPolicy(),
    ))
    judge_model: str | None = None

    # ------------------------------------------------------------------
    def propose(
        self,
        state: AgentState,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
    ) -> Action:
        bn = state.metadata.setdefault("best_of_n", {})

        # Idempotent: if we've already chosen a winner on a prior tick,
        # return the same FINAL_ANSWER. (The driver only calls us until
        # FINAL_ANSWER, so this is a defensive guard.)
        if "winner_answer" in bn:
            return Action(
                kind=ActionKind.FINAL_ANSWER,
                answer=bn["winner_answer"],
                meta={
                    "policy": self.name,
                    "selector": self.selector,
                    "n_candidates": self.n,
                    "cached": True,
                },
            )

        # Run N independent inner trajectories on fresh state copies.
        # Each gets its own budget (copy-by-value of the outer one) so
        # the inner driver's BudgetTracker can enforce per-run caps
        # independently. We deliberately copy.copy (shallow) the budget
        # dataclass — it only holds primitives.
        from ..core.agent import run_policy  # local import: avoid cycle
        candidates: list[_Candidate] = []
        for i in range(self.n):
            sub_state = AgentState(
                question=state.question,
                budget=copy.copy(state.budget),
            )
            run_policy(sub_state, self.inner, llm=llm, tools=tools)
            answer = sub_state.trace.final_answer or ""
            reason = "final" if sub_state.is_done else (
                sub_state.budget.check().value if sub_state.budget else "unknown"
            )
            candidates.append(_Candidate(
                idx=i,
                answer=answer,
                tokens_in=sub_state.budget.tokens_in,
                tokens_out=sub_state.budget.tokens_out,
                n_steps=len(sub_state.trace.steps),
                finished_reason=str(reason),
            ))

        # Selector.
        if self.selector == "llm_judge":
            winner_idx = _select_llm_judge(
                candidates,
                question=state.question,
                llm=llm,
                model=self.judge_model,
            )
        else:
            winner_idx = _select_majority(candidates)

        winner = candidates[winner_idx]
        bn["winner_answer"] = winner.answer
        bn["winner_idx"] = winner_idx
        bn["selector"] = self.selector
        bn["candidates"] = [
            {
                "idx": c.idx,
                "answer": c.answer,
                "tokens_in": c.tokens_in,
                "tokens_out": c.tokens_out,
                "n_steps": c.n_steps,
                "finished_reason": c.finished_reason,
            }
            for c in candidates
        ]

        # Aggregate token usage onto the action.meta so the outer driver's
        # BudgetTracker / cost estimator picks up the full multi-run cost.
        # Provider/model lift makes the cost lookup work in core/agent.py.
        return Action(
            kind=ActionKind.FINAL_ANSWER,
            answer=winner.answer,
            meta={
                "policy": self.name,
                "selector": self.selector,
                "n_candidates": self.n,
                "winner_idx": winner_idx,
                "tokens_in": sum(c.tokens_in for c in candidates),
                "tokens_out": sum(c.tokens_out for c in candidates),
                "provider": getattr(llm, "provider", ""),
                "model": getattr(llm, "model", "") or "",
            },
        )

    # ------------------------------------------------------------------
    def on_run_end(self, state: AgentState, **kwargs: Any) -> Any:
        """Forward to the inner policy if it implements one."""
        hook = getattr(self.inner, "on_run_end", None)
        if hook is None:
            return None
        return hook(state, **kwargs)
