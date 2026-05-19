"""CoverageVerifier tests."""
from __future__ import annotations

import pytest

from banna_agent.core.state import AgentState
from banna_agent.core.types import Claim
from banna_agent.verifiers.coverage import CoverageVerifier


def test_no_claims_is_no_checks() -> None:
    s = AgentState(question="?")
    checks = CoverageVerifier().check(s)
    assert checks == []


def test_supported_claim_is_ok() -> None:
    s = AgentState(question="?")
    ev = s.add_evidence(source="x", content="...")
    s.claims.append(Claim(text="claim", supports=[ev.evidence_id]))
    checks = CoverageVerifier().check(s)
    assert checks[0].verdict == "ok"


def test_factual_unsupported_claim_is_fail() -> None:
    s = AgentState(question="?")
    s.claims.append(Claim(text="The capital of Iceland is Reykjavik."))
    checks = CoverageVerifier().check(s)
    assert checks[0].verdict == "fail"


def test_computational_unsupported_claim_is_warn() -> None:
    s = AgentState(question="?")
    s.claims.append(Claim(text="47 * 83 = 3901"))
    checks = CoverageVerifier().check(s)
    assert checks[0].verdict == "warn"


def test_broken_citation_is_fail() -> None:
    s = AgentState(question="?")
    s.claims.append(Claim(text="The answer", supports=["ev_doesnotexist"]))
    checks = CoverageVerifier().check(s)
    assert checks[0].verdict == "fail"
    assert "broken" in checks[0].detail
