"""Verifier protocol — the project's grounded-check substrate.

The README calls this "the research signal of the project": rather than
trust an LLM-judge, every verifier is a small, deterministic procedure
that can independently check one aspect of an agent's output.

Contract:
  * `Verifier.check(state, proposed_answer=None) -> list[ClaimCheck]`
  * Each ClaimCheck names the verifier, the claim it relates to, the
    verdict (`ok` / `fail` / `warn` / `skip`), a short detail string,
    and an optional confidence score.
  * Verifiers may also examine `state.trace` for evidence of work done
    (e.g. arithmetic looks at THINK steps with expressions).

Verdicts:
  * `ok`     — verifier ran and the claim holds.
  * `fail`   — verifier ran and the claim *does not* hold (actionable).
  * `warn`   — soft signal; the claim is suspicious but not refuted.
  * `skip`   — verifier wasn't applicable (e.g. arithmetic on a
               string-valued claim).

Side effects:
  * `apply_to_state(state, checks)` writes the verdicts back onto the
    Claim objects via `claim.verdicts[verifier_name] = verdict`. This is
    what `verifier_retry` and segment-promotion logic read.

Most verifiers are pure functions of state, so they're cheap to run
after every step (or at FINAL_ANSWER time). The two exceptions are
citation (which can fetch URLs) and any future LLM-judge fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

from ..core.state import AgentState


Verdict = Literal["ok", "fail", "warn", "skip"]


@dataclass
class ClaimCheck:
    """One verifier's verdict on one claim (or on the proposed answer).

    `claim_id` is the Claim.claim_id when the check is about a claim;
    when the check is about the FINAL_ANSWER itself (e.g. the format
    verifier looking at the proposed answer string), use the sentinel
    `claim_id="__answer__"`.
    """

    claim_id: str
    verifier_name: str
    verdict: Verdict
    detail: str = ""
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "verifier_name": self.verifier_name,
            "verdict": self.verdict,
            "detail": self.detail,
            "score": self.score,
            "meta": dict(self.meta),
        }


# Sentinel used when a check applies to the FINAL_ANSWER, not to a Claim.
ANSWER_CLAIM_ID = "__answer__"


@runtime_checkable
class Verifier(Protocol):
    """Stateless or near-stateless grounded checker."""

    name: str

    def check(
        self,
        state: AgentState,
        proposed_answer: str | None = None,
    ) -> list[ClaimCheck]: ...


def apply_to_state(state: AgentState, checks: Sequence[ClaimCheck]) -> None:
    """Write verdicts onto the matching Claim objects in state.

    Checks targeting the proposed answer (claim_id == ANSWER_CLAIM_ID)
    are not written to any Claim — the policy reads them directly off
    the returned list.
    """
    by_id = {cl.claim_id: cl for cl in state.claims}
    for c in checks:
        if c.claim_id == ANSWER_CLAIM_ID:
            continue
        cl = by_id.get(c.claim_id)
        if cl is None:
            continue
        cl.verdicts[c.verifier_name] = c.verdict


def run_all(
    verifiers: Sequence[Verifier],
    state: AgentState,
    *,
    proposed_answer: str | None = None,
    apply: bool = True,
) -> list[ClaimCheck]:
    """Run every verifier and aggregate their checks.

    Convenience for callers (verifier_retry, segment promotion). When
    `apply=True`, verdicts are also written back onto Claim objects.
    """
    out: list[ClaimCheck] = []
    for v in verifiers:
        try:
            out.extend(v.check(state, proposed_answer=proposed_answer))
        except Exception as exc:
            out.append(ClaimCheck(
                claim_id=ANSWER_CLAIM_ID,
                verifier_name=getattr(v, "name", type(v).__name__),
                verdict="warn",
                detail=f"verifier crashed: {type(exc).__name__}: {exc}",
            ))
    if apply:
        apply_to_state(state, out)
    return out


def summarize(checks: Sequence[ClaimCheck]) -> dict[str, int]:
    """Count verdicts by kind. Useful for logging / debug summaries."""
    out = {"ok": 0, "fail": 0, "warn": 0, "skip": 0}
    for c in checks:
        out[c.verdict] = out.get(c.verdict, 0) + 1
    return out


def has_failures(checks: Sequence[ClaimCheck]) -> bool:
    return any(c.verdict == "fail" for c in checks)


def default_verifiers(
    *,
    command_verifier: "Verifier | None" = None,
) -> list[Verifier]:
    """Build the v1 default verifier set.

    Imported lazily to avoid pulling all four implementations whenever a
    caller only wants the protocol types.

    `command_verifier`, when provided, is appended — typically a
    `CommandVerifier` built via `verifiers.default_command_verifier()`.
    Off by default so QA-style runs (GAIA) don't shell out to pytest.
    """
    from .arithmetic import ArithmeticVerifier
    from .citation import CitationVerifier
    from .coverage import CoverageVerifier
    from .format import FormatVerifier

    out: list[Verifier] = [
        ArithmeticVerifier(),
        CitationVerifier(),
        FormatVerifier(),
        CoverageVerifier(),
    ]
    if command_verifier is not None:
        out.append(command_verifier)
    return out
