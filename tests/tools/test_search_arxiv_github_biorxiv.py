"""Tests for the arxiv / github / biorxiv backends (Phase 6.6)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from banna_agent.tools import _http_cache as hc
from banna_agent.tools.search.backends.arxiv import ArxivBackend
from banna_agent.tools.search.backends.biorxiv import BiorxivBackend
from banna_agent.tools.search.backends.github import GitHubBackend


@dataclass
class _Resp:
    status_code: int
    headers: dict
    content: bytes
    url: str = ""

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    monkeypatch.delenv("BANNA_HTTP_CACHE", raising=False)
    hc.set_cache(None)
    yield
    hc.set_cache(None)


def _install_request(monkeypatch, body: bytes, *, status: int = 200, content_type: str = "application/json"):
    captured: dict[str, Any] = {}

    def fake_request(method, url, **kw):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = dict(kw.get("headers") or {})
        return _Resp(status_code=status, headers={"Content-Type": content_type},
                    content=body, url=url)

    import requests
    monkeypatch.setattr(requests, "request", fake_request)
    return captured


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------


ARXIV_ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2310.12345v1</id>
    <title>A Study of Topic X</title>
    <summary>We investigate Topic X across three regimes.</summary>
    <published>2023-10-19T17:00:00Z</published>
    <author><name>Alice Researcher</name></author>
    <author><name>Bob Coauthor</name></author>
    <category term="cs.LG" />
    <category term="stat.ML" />
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2309.99999v2</id>
    <title>Topic Y Revisited</title>
    <summary>A follow-up to prior Topic Y results.</summary>
    <published>2023-09-30T10:00:00Z</published>
    <author><name>Carol Solo</name></author>
    <category term="physics.ao-ph" />
  </entry>
</feed>
"""


def test_arxiv_parses_atom_entries(monkeypatch) -> None:
    cap = _install_request(monkeypatch, ARXIV_ATOM, content_type="application/atom+xml")
    out = ArxivBackend().search("topic x", n=5)
    assert len(out) == 2
    r0 = out[0]
    assert r0.url == "http://arxiv.org/abs/2310.12345v1"
    assert r0.title == "A Study of Topic X"
    assert "Topic X" in r0.snippet
    assert r0.meta["authors"] == ["Alice Researcher", "Bob Coauthor"]
    assert "cs.LG" in r0.meta["categories"]
    assert r0.published_at == "2023-10-19T17:00:00Z"
    # URL must include `all:` prefix for an unprefixed query.
    assert "all%3Atopic" in cap["url"]


def test_arxiv_respects_field_prefix(monkeypatch) -> None:
    cap = _install_request(monkeypatch, ARXIV_ATOM, content_type="application/atom+xml")
    ArxivBackend().search("ti:transformer au:vaswani", n=2)
    # Caller-supplied prefix must NOT be wrapped in another `all:`.
    assert "all%3Ati" not in cap["url"]
    assert "ti%3Atransformer" in cap["url"]


def test_arxiv_since_filter_appends_submitted_date(monkeypatch) -> None:
    cap = _install_request(monkeypatch, b"<feed/>", content_type="application/atom+xml")
    ArxivBackend().search("topic z", since="week", n=1)
    # The `submittedDate:[YYYYMMDD0000 TO *]` clause must be in the URL.
    assert "submittedDate" in cap["url"]


def test_arxiv_empty_query_returns_empty(monkeypatch) -> None:
    _install_request(monkeypatch, b"<feed/>")
    out = ArxivBackend().search("", n=5)
    assert out == []


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------


GH_REPOS = json.dumps({
    "items": [
        {
            "html_url": "https://github.com/foo/bar",
            "full_name": "foo/bar",
            "description": "A small lib for X.",
            "stargazers_count": 4200,
            "forks_count": 117,
            "open_issues_count": 9,
            "language": "Python",
            "default_branch": "main",
            "created_at": "2021-04-01T12:00:00Z",
            "score": 0.93,
        },
        {
            "html_url": "https://github.com/baz/qux",
            "full_name": "baz/qux",
            "description": "Tool for Y.",
            "stargazers_count": 12,
            "language": "Rust",
            "score": 0.42,
        },
    ]
}).encode()


def test_github_repository_search(monkeypatch) -> None:
    cap = _install_request(monkeypatch, GH_REPOS)
    out = GitHubBackend().search("topic x language:python stars:>100", n=10)
    assert len(out) == 2
    r0 = out[0]
    assert r0.url == "https://github.com/foo/bar"
    assert r0.title == "foo/bar"
    assert r0.snippet == "A small lib for X."
    assert r0.meta["stars"] == 4200
    assert r0.meta["language"] == "Python"
    assert r0.meta["kind"] == "repository"
    # Repositories endpoint by default.
    assert "/search/repositories" in cap["url"]


def test_github_uses_token_when_present(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "secret-xyz")
    cap = _install_request(monkeypatch, GH_REPOS)
    GitHubBackend().search("anything", n=1)
    assert cap["headers"].get("Authorization") == "Bearer secret-xyz"


def test_github_mode_overload_switches_endpoint(monkeypatch) -> None:
    cap = _install_request(monkeypatch, json.dumps({"items": []}).encode())
    GitHubBackend().search("ratelimit handling", since="mode:issues", n=3)
    assert "/search/issues" in cap["url"]


def test_github_issues_result_shape(monkeypatch) -> None:
    body = json.dumps({"items": [{
        "html_url": "https://github.com/foo/bar/issues/42",
        "title": "Crash on startup",
        "body": "Reproducible by …",
        "number": 42,
        "state": "open",
        "created_at": "2024-01-01T00:00:00Z",
        "score": 1.0,
    }]}).encode()
    _install_request(monkeypatch, body)
    out = GitHubBackend().search("crash", since="mode:issues", n=5)
    assert len(out) == 1
    assert out[0].meta["kind"] == "issue"
    assert out[0].meta["number"] == 42
    assert out[0].meta["state"] == "open"


# ---------------------------------------------------------------------------
# bioRxiv (Europe PMC)
# ---------------------------------------------------------------------------


PMC_BODY = json.dumps({
    "resultList": {
        "result": [
            {
                "title": "Some Preprint Title.",
                "doi": "10.1101/2023.01.01.000001",
                "authorString": "Smith J, Lee K, Garcia M",
                "abstractText": "We report a finding about X in mouse cortex.",
                "firstPublicationDate": "2023-02-15",
                "citedByCount": "7",
                "source": "PPR",
                "fullTextUrlList": {"fullTextUrl": [
                    {"documentStyle": "html", "url": "https://www.biorxiv.org/content/10.1101/abc"},
                ]},
            },
            {
                "title": "Another Preprint",
                "doi": "10.1101/2023.02.02.000002",
                "authorString": "Doe A",
                "abstractText": "Methods and results follow.",
                "firstPublicationDate": "2023-03-01",
                # No fullTextUrlList → falls back to DOI URL.
            },
        ],
    }
}).encode()


def test_biorxiv_parses_europepmc_results(monkeypatch) -> None:
    cap = _install_request(monkeypatch, PMC_BODY)
    out = BiorxivBackend().search("topic z mouse cortex", n=10)
    assert len(out) == 2
    r0 = out[0]
    assert r0.title == "Some Preprint Title"  # trailing dot stripped
    assert r0.meta["doi"] == "10.1101/2023.01.01.000001"
    assert "Smith J" in r0.meta["authors"]
    assert r0.meta["citation_count"] == 7
    assert r0.url.startswith("https://www.biorxiv.org/")
    # Second result falls back to DOI URL.
    assert out[1].url == "https://doi.org/10.1101/2023.02.02.000002"
    # Query qualifier present.
    assert "SRC%3APPR" in cap["url"]


def test_biorxiv_since_filter(monkeypatch) -> None:
    cap = _install_request(monkeypatch, b'{"resultList":{"result":[]}}')
    BiorxivBackend().search("topic y", since="month")
    assert "FIRST_PDATE" in cap["url"]


# ---------------------------------------------------------------------------
# Registry plumbing.
# ---------------------------------------------------------------------------


def test_registry_resolves_new_backend_names() -> None:
    from banna_agent.tools.search.tool import _BACKEND_FACTORIES
    for name in ("arxiv", "biorxiv", "medrxiv", "github"):
        assert name in _BACKEND_FACTORIES, f"{name!r} not registered"
