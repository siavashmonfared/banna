"""CitationVerifier tests."""
from __future__ import annotations

import pytest

from banna_agent.core.state import AgentState
from banna_agent.core.types import Claim, Evidence
from banna_agent.verifiers.citation import CitationVerifier


def _state_with(claim_text: str, supports_content: list[str]) -> tuple[AgentState, list[Evidence]]:
    s = AgentState(question="?")
    evs = []
    for c in supports_content:
        ev = s.add_evidence(source="http://example.com/x", content=c)
        evs.append(ev)
    s.claims.append(Claim(text=claim_text, supports=[ev.evidence_id for ev in evs]))
    return s, evs


def test_high_overlap_is_ok() -> None:
    s, _ = _state_with(
        "Iceland has a small population density.",
        ["The population density of Iceland is among the lowest in Europe."],
    )
    checks = CitationVerifier(min_overlap=0.05).check(s)
    assert len(checks) == 1
    assert checks[0].verdict == "ok"


def test_no_overlap_is_fail() -> None:
    s, _ = _state_with(
        "The cube root of 1331 is 11.",
        ["This evidence is about something completely unrelated like apples and oranges."],
    )
    checks = CitationVerifier().check(s)
    assert checks[0].verdict == "fail"


def test_unsupported_claim_is_skipped() -> None:
    """A claim with no supports is the coverage verifier's job."""
    s = AgentState(question="?")
    s.claims.append(Claim(text="some claim", supports=[]))
    checks = CitationVerifier().check(s)
    assert checks[0].verdict == "skip"


def test_broken_citation_is_fail() -> None:
    s = AgentState(question="?")
    s.claims.append(Claim(text="claim", supports=["ev_doesnotexist"]))
    checks = CitationVerifier().check(s)
    assert checks[0].verdict == "fail"
    assert "broken" in checks[0].detail


def test_empty_evidence_content_is_warn() -> None:
    s = AgentState(question="?")
    ev = s.add_evidence(source="http://example.com/x", content="")
    s.claims.append(Claim(text="meaningful claim text here", supports=[ev.evidence_id]))
    checks = CitationVerifier().check(s)
    assert checks[0].verdict == "warn"
