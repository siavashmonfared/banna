"""Phase 5 gate: real / fabricated / unreachable citation verdicts.

The gate (per the plan): a real citation gets ``ok``, a fabricated one
gets ``fail`` with ``meta.error_class == 'evidence_not_found'``, and an
unreachable URL gets ``warn`` with ``meta.error_class == 'tool_error'``.

We prime an on-disk HTTP cache with one canned page, then install a
replay-mode cache as the singleton so the verifier's default fetcher
gets deterministic bytes. The "unreachable" case uses a URL we never
recorded — replay mode then raises HttpCacheMiss, which the verifier
catches and tags as tool_error.

Plus: a non-URL evidence with empty content, and a numeric-claim
support check (claim cites a year that's only in *one* of two pieces
of evidence — still ok).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from banna_agent.core.state import AgentState
from banna_agent.core.types import Claim
from banna_agent.tools import _http_cache as hc
from banna_agent.verifiers.citation import (
    CitationVerifier,
    default_citation_verifier,
)


PAGE_HTML = (
    b"<html><body>"
    b"<p>Topic Z was first observed in 1923 by researchers in Oslo. "
    b"Its formal classification followed three years later, in 1926.</p>"
    b"</body></html>"
)


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    monkeypatch.delenv("BANNA_HTTP_CACHE", raising=False)
    hc.set_cache(None)
    yield
    hc.set_cache(None)


def _prime(tmp_path: Path) -> None:
    """Record the canned page through a fake live request."""

    def fake_live(method, url, params, data, json_body, headers, timeout):
        return hc.CachedResponse(
            status_code=200, url=url,
            headers={"Content-Type": "text/html"},
            content=PAGE_HTML, from_cache=False,
        )

    rec = hc.HttpCache(root=tmp_path, mode="record", _live_request=fake_live)
    rec.fetch("GET", "https://example.test/article",
              headers={"User-Agent": "banna_agent/0.1", "Accept": "*/*"})


# ---------------------------------------------------------------------------
# Gate: real citation ⇒ ok.
# ---------------------------------------------------------------------------


def test_real_citation_with_supporting_text_is_ok(tmp_path: Path) -> None:
    _prime(tmp_path)
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))

    state = AgentState(question="?")
    ev = state.add_evidence(source="https://example.test/article", content="")
    state.claims.append(Claim(
        text="Topic Z was first observed in 1923.",
        supports=[ev.evidence_id],
    ))
    checks = default_citation_verifier().check(state)
    assert len(checks) == 1
    assert checks[0].verdict == "ok"
    assert checks[0].score > 0
    assert "1923" in (checks[0].meta.get("claim_numbers") or [])


# ---------------------------------------------------------------------------
# Gate: fabricated citation ⇒ fail with evidence_not_found.
# ---------------------------------------------------------------------------


def test_fabricated_year_against_real_page_fails(tmp_path: Path) -> None:
    _prime(tmp_path)
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))

    state = AgentState(question="?")
    ev = state.add_evidence(source="https://example.test/article", content="")
    # Page mentions 1923 and 1926; the claim cites 1947 which isn't there.
    state.claims.append(Claim(
        text="Topic Z was first observed in 1947.",
        supports=[ev.evidence_id],
    ))
    checks = default_citation_verifier().check(state)
    assert checks[0].verdict == "fail"
    assert checks[0].meta.get("error_class") == "evidence_not_found"
    assert "1947" in checks[0].meta.get("missing_numbers", [])


def test_topical_jaccard_alone_isnt_enough_when_a_number_is_off(tmp_path: Path) -> None:
    _prime(tmp_path)
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))

    state = AgentState(question="?")
    ev = state.add_evidence(source="https://example.test/article", content="")
    # Strong topical overlap (topic z, observed, researchers, oslo) but
    # the claim's number doesn't appear → still a fail.
    state.claims.append(Claim(
        text="Topic Z was observed by Oslo researchers in 1965.",
        supports=[ev.evidence_id],
    ))
    checks = default_citation_verifier().check(state)
    assert checks[0].verdict == "fail"
    assert checks[0].meta.get("error_class") == "evidence_not_found"


# ---------------------------------------------------------------------------
# Gate: unreachable URL ⇒ warn with tool_error.
# ---------------------------------------------------------------------------


def test_unreachable_url_yields_warn_tool_error(tmp_path: Path) -> None:
    _prime(tmp_path)
    # Replay mode → unknown URL raises HttpCacheMiss → fetcher raises →
    # verifier tags as tool_error.
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))

    state = AgentState(question="?")
    ev = state.add_evidence(source="https://example.test/missing", content="")
    state.claims.append(Claim(
        text="Topic Z was first observed in 1923.",
        supports=[ev.evidence_id],
    ))
    checks = default_citation_verifier().check(state)
    assert checks[0].verdict == "warn"
    assert checks[0].meta.get("error_class") == "tool_error"
    assert "errors" in checks[0].meta
    assert any("HttpCacheMiss" in e or "missing" in e for e in checks[0].meta["errors"])


# ---------------------------------------------------------------------------
# Mixed evidence — one good URL covers the number, ok overall.
# ---------------------------------------------------------------------------


def test_number_supported_by_one_of_several_evidences(tmp_path: Path) -> None:
    _prime(tmp_path)
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))

    state = AgentState(question="?")
    # Evidence A: empty + URL → refetched, contains "1923".
    ev_a = state.add_evidence(source="https://example.test/article", content="")
    # Evidence B: inline text covering the second number "1926".
    ev_b = state.add_evidence(
        source="local://notes",
        content="The official classification of Topic Z took place in 1926.",
    )
    state.claims.append(Claim(
        text="Topic Z was first observed in 1923 and formally classified in 1926.",
        supports=[ev_a.evidence_id, ev_b.evidence_id],
    ))
    checks = default_citation_verifier(min_overlap=0.05).check(state)
    assert checks[0].verdict == "ok"


# ---------------------------------------------------------------------------
# Offline-safe default: with use_http_cache=False, empty URL evidence
# stays as a warn (v1 behavior).
# ---------------------------------------------------------------------------


def test_offline_safe_default_does_not_refetch(tmp_path: Path) -> None:
    # No cache primed; with use_http_cache=False the verifier MUST NOT
    # call the network, just warn that it can't grade.
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))
    state = AgentState(question="?")
    ev = state.add_evidence(source="https://example.test/article", content="")
    state.claims.append(Claim(
        text="Topic Z was first observed in 1923.",
        supports=[ev.evidence_id],
    ))
    checks = default_citation_verifier(use_http_cache=False).check(state)
    assert checks[0].verdict == "warn"
    assert checks[0].meta.get("error_class") == "evidence_not_found"


# ---------------------------------------------------------------------------
# require_number_match=False reverts to v1 (overlap-only).
# ---------------------------------------------------------------------------


def test_require_number_match_false_relaxes_to_v1(tmp_path: Path) -> None:
    _prime(tmp_path)
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))

    state = AgentState(question="?")
    ev = state.add_evidence(source="https://example.test/article", content="")
    # Same wrong-year case as before, but numeric check is off → topical
    # Jaccard alone wins (the claim shares enough tokens with the page
    # text to clear min_overlap=0.05).
    state.claims.append(Claim(
        text="Topic Z was observed by Oslo researchers in 1965.",
        supports=[ev.evidence_id],
    ))
    cv = CitationVerifier(
        min_overlap=0.05,
        require_number_match=False,
        fetch_url=None,
    )
    # Need the cache fetcher for refetch; rebuild with it.
    from banna_agent.verifiers.citation import _default_cache_fetcher
    cv.fetch_url = _default_cache_fetcher
    checks = cv.check(state)
    assert checks[0].verdict == "ok"
