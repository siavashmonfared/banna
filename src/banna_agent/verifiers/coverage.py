"""CoverageVerifier — every Claim should be backed by ≥1 Evidence.

GAIA's most common silent failure is *unsupported claims*: the model
produces a statement that happens to be true (or plausibly true) but
never grounds it in a tool-fetched source. Coverage checks the
substrate-level structural property: `Claim.supports` is non-empty.

Verdicts:
    ok   → claim has at least one supporting Evidence id
    warn → claim is computational/derived (arithmetic-style) and
           expecting an external citation would be wrong
    fail → claim looks factual (proper nouns, definite assertions)
           and has no supports
    skip → no claims in state — there's nothing to check

Heuristic for "computational" claims (warn instead of fail):
  - text contains an arithmetic expression and an equality
  - text is short and numeric
That mirrors what `ArithmeticVerifier` would treat as in-scope.

We also flag broken citations (a `supports` id that doesn't exist in
state.evidence) — that's structurally a fail regardless of content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.state import AgentState
from .base import ClaimCheck


_FACTUAL_CUE = re.compile(
    r"\b(?:was|is|are|were|the|on|in|at|by|of)\b.*"
    r"\b[A-Z][a-zA-Z]+\b",  # at least one capitalized word — proper-noun-ish
)
_ARITH_LIKE = re.compile(r"[\d+\-*/^()]+\s*(?:=|equals)\s*[-+]?\d")


def _looks_computational(text: str) -> bool:
    if not text:
        return False
    if _ARITH_LIKE.search(text):
        return True
    # Pure-numeric short claim ("the answer is 42").
    if len(text) < 60 and re.search(r"\b\d+(?:\.\d+)?\b", text):
        # No proper noun → probably computational.
        if not re.search(r"\b[A-Z][a-z]{2,}\b", text):
            return True
    return False


@dataclass
class CoverageVerifier:
    """Structural check: claims should cite evidence."""

    name: str = "coverage"

    def check(
        self,
        state: AgentState,
        proposed_answer: str | None = None,
    ) -> list[ClaimCheck]:
        out: list[ClaimCheck] = []
        ev_ids = {ev.evidence_id for ev in state.evidence}

        if not state.claims:
            return out  # nothing to grade

        for cl in state.claims:
            # Broken-citation case (structurally a fail, regardless).
            broken = [s for s in cl.supports if s not in ev_ids]
            if broken:
                out.append(ClaimCheck(
                    claim_id=cl.claim_id,
                    verifier_name=self.name,
                    verdict="fail",
                    detail=f"broken citation(s): {', '.join(broken[:3])}",
                    meta={
                        "broken": broken,
                        "nudge": (
                            f"You cited evidence IDs {broken[:3]} that "
                            f"don't exist in this run. Copy evidence_ids "
                            f"from your prior tool calls' JSON responses, "
                            f"or omit evidence_ids if you have no real "
                            f"citation."
                        ),
                    },
                ))
                continue

            if cl.supports:
                out.append(ClaimCheck(
                    claim_id=cl.claim_id,
                    verifier_name=self.name,
                    verdict="ok",
                    detail=f"supported by {len(cl.supports)} evidence",
                ))
                continue

            # No supports — is the claim computational or factual?
            if _looks_computational(cl.text):
                out.append(ClaimCheck(
                    claim_id=cl.claim_id,
                    verifier_name=self.name,
                    verdict="warn",
                    detail="claim is computational; arithmetic verifier should grade it",
                ))
            else:
                out.append(ClaimCheck(
                    claim_id=cl.claim_id,
                    verifier_name=self.name,
                    verdict="fail",
                    detail="factual claim has no supporting evidence",
                    meta={
                        "nudge": (
                            f"Your claim ({cl.text[:80]!r}) is factual but "
                            f"has no supporting evidence. Before committing, "
                            f"run `search` or `read_url` to locate a source, "
                            f"then pass that evidence_id in `final_answer`."
                        ),
                    },
                ))
        return out
