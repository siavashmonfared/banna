"""Tests for verifiers.base — protocol + run_all + apply_to_state."""
from __future__ import annotations

from dataclasses import dataclass


from banna_agent.core.state import AgentState
from banna_agent.core.types import Claim
from banna_agent.verifiers.base import (
    ANSWER_CLAIM_ID,
    ClaimCheck,
    Verifier,
    apply_to_state,
    default_verifiers,
    has_failures,
    run_all,
    summarize,
)


@dataclass
class _FixedVerifier:
    name: str = "fixed"
    verdicts: list[str] = None  # type: ignore[assignment]

    def check(self, state, proposed_answer=None):
        out = []
        for cl, v in zip(state.claims, self.verdicts or []):
            out.append(ClaimCheck(claim_id=cl.claim_id, verifier_name=self.name,
                                  verdict=v, detail=""))
        return out


def test_default_verifiers_satisfy_protocol() -> None:
    for v in default_verifiers():
        assert isinstance(v, Verifier)


def test_run_all_aggregates_and_applies() -> None:
    s = AgentState(question="?")
    s.claims.append(Claim(text="a"))
    s.claims.append(Claim(text="b"))
    v = _FixedVerifier(verdicts=["ok", "fail"])
    checks = run_all([v], s)
    assert len(checks) == 2
    # Verdicts written back onto Claims.
    assert s.claims[0].verdicts["fixed"] == "ok"
    assert s.claims[1].verdicts["fixed"] == "fail"


def test_run_all_apply_false_does_not_mutate() -> None:
    s = AgentState(question="?")
    s.claims.append(Claim(text="a"))
    v = _FixedVerifier(verdicts=["ok"])
    run_all([v], s, apply=False)
    assert "fixed" not in s.claims[0].verdicts


def test_summarize_counts() -> None:
    checks = [
        ClaimCheck(claim_id="a", verifier_name="x", verdict="ok"),
        ClaimCheck(claim_id="a", verifier_name="y", verdict="fail"),
        ClaimCheck(claim_id="a", verifier_name="z", verdict="warn"),
        ClaimCheck(claim_id="a", verifier_name="w", verdict="fail"),
    ]
    s = summarize(checks)
    assert s == {"ok": 1, "fail": 2, "warn": 1, "skip": 0}


def test_has_failures() -> None:
    assert not has_failures([
        ClaimCheck(claim_id="a", verifier_name="x", verdict="ok"),
        ClaimCheck(claim_id="b", verifier_name="x", verdict="warn"),
    ])
    assert has_failures([
        ClaimCheck(claim_id="a", verifier_name="x", verdict="ok"),
        ClaimCheck(claim_id="b", verifier_name="x", verdict="fail"),
    ])


def test_crashing_verifier_recovered_as_warn() -> None:
    class _Boom:
        name = "boom"
        def check(self, state, proposed_answer=None):
            raise RuntimeError("oops")
    s = AgentState(question="?")
    checks = run_all([_Boom()], s)
    assert checks[0].verdict == "warn"
    assert "oops" in checks[0].detail


def test_apply_skips_answer_sentinel() -> None:
    s = AgentState(question="?")
    s.claims.append(Claim(text="a"))
    apply_to_state(s, [
        ClaimCheck(claim_id=ANSWER_CLAIM_ID, verifier_name="x", verdict="fail"),
    ])
    # No claim was touched.
    assert s.claims[0].verdicts == {}


