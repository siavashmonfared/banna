"""Tests for the search backend cascade + per-backend parsers."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from banna_agent.tools.search import (
    DEFAULT_AUTO_CASCADE,
    CascadeBackend,
    DuckDuckGoBackend,
    GoogleGroundedBackend,
    SearchResult,
    TavilyBackend,
    WebSearchBackend,
    YaCyBackend,
    make_search_tool,
    search,
)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_each_backend_satisfies_protocol() -> None:
    for B in (YaCyBackend, DuckDuckGoBackend, GoogleGroundedBackend, TavilyBackend):
        assert isinstance(B(), WebSearchBackend)


def test_default_cascade_includes_yacy_first() -> None:
    assert DEFAULT_AUTO_CASCADE[0] == "yacy"
    assert "duckduckgo" in DEFAULT_AUTO_CASCADE
    assert "google" in DEFAULT_AUTO_CASCADE


# ---------------------------------------------------------------------------
# Cascade behavior
# ---------------------------------------------------------------------------


@dataclass
class _FakeBackend:
    name: str
    source_tier: str = "web"
    hits: list = None  # type: ignore[assignment]
    raise_exc: Exception | None = None

    def search(self, query, *, n=10, since=None):
        if self.raise_exc is not None:
            raise self.raise_exc
        return list(self.hits or [])


def _r(url, source="x") -> SearchResult:
    return SearchResult(url=url, title=url, source=source)


def test_cascade_dedupes_by_url_first_wins() -> None:
    a = _FakeBackend(name="a", hits=[_r("http://x", "a"), _r("http://y", "a")])
    b = _FakeBackend(name="b", hits=[_r("http://y", "b"), _r("http://z", "b")])
    cb = CascadeBackend([a, b], per_backend_min=10)
    out = cb.search("q", n=10)
    assert [r.url for r in out] == ["http://x", "http://y", "http://z"]
    assert [r.source for r in out] == ["a", "a", "b"]


def test_cascade_falls_through_on_exception() -> None:
    a = _FakeBackend(name="a", raise_exc=RuntimeError("yacy down"))
    b = _FakeBackend(name="b", hits=[_r("http://z", "b")])
    cb = CascadeBackend([a, b])
    out = cb.search("q", n=5)
    assert [r.url for r in out] == ["http://z"]
    # Error captured for diagnostic.
    assert cb.last_errors == [("a", "RuntimeError: yacy down")]


def test_cascade_stops_when_n_reached() -> None:
    a = _FakeBackend(name="a", hits=[_r(f"http://x{i}", "a") for i in range(5)])
    b_called = []
    class _B:
        name = "b"
        source_tier = "web"
        def search(self, query, *, n=10, since=None):
            b_called.append(True)
            return [_r("http://b1", "b")]
    cb = CascadeBackend([a, _B()], per_backend_min=3)
    out = cb.search("q", n=3)
    assert len(out) == 3
    assert b_called == []  # b never invoked


def test_cascade_continues_when_below_per_backend_min() -> None:
    a = _FakeBackend(name="a", hits=[_r("http://x", "a")])
    b = _FakeBackend(name="b", hits=[_r("http://y", "b")])
    cb = CascadeBackend([a, b], per_backend_min=2)
    out = cb.search("q", n=10)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Top-level search() dispatch
# ---------------------------------------------------------------------------


def test_search_unknown_backend_raises() -> None:
    with pytest.raises(ValueError):
        search("q", backend="not_a_real_backend")


def test_search_returns_normalized_payload(monkeypatch) -> None:
    # The yacy/ddg branches in `_make_backend` construct fresh instances.
    # Patch the factory dict via a key that doesn't have special-case
    # construction (`tavily` works) so our fake gets used.
    class _FakeBE:
        name = "tavily"
        source_tier = "paid"
        def search(self, query, *, n=10, since=None):
            return [SearchResult(
                url="http://test/foo", title="foo", snippet="bar",
                source="tavily", source_tier="paid",
            )]

    from banna_agent.tools.search import tool as tool_mod
    monkeypatch.setitem(tool_mod._BACKEND_FACTORIES, "tavily", _FakeBE)
    out = search("q", backend="tavily")
    assert out["query"] == "q"
    assert out["backend"] == "tavily"
    assert out["n_results"] == 1
    assert out["hits"][0]["url"] == "http://test/foo"
    assert out["hits"][0]["source"] == "tavily"
    assert "fetched_at" in out["hits"][0]


def test_search_failure_propagates_in_single_backend_mode(monkeypatch) -> None:
    """When a single backend is selected and it raises, search() propagates.

    Errors are only captured (and surfaced in the `errors` list) when
    multiple backends are running through `CascadeBackend`.
    """
    class _Boom:
        name = "tavily"
        source_tier = "paid"
        def search(self, query, *, n=10, since=None):
            raise RuntimeError("nope")

    from banna_agent.tools.search import tool as tool_mod
    monkeypatch.setitem(tool_mod._BACKEND_FACTORIES, "tavily", _Boom)
    with pytest.raises(RuntimeError, match="nope"):
        search("q", backend="tavily")


def test_cascade_failure_captured_in_errors(monkeypatch) -> None:
    """In cascade mode, a backend's exception is captured and the run continues."""
    class _Boom:
        name = "tavily"
        source_tier = "paid"
        def search(self, query, *, n=10, since=None):
            raise RuntimeError("nope")

    class _Ok:
        name = "duckduckgo"
        source_tier = "web"
        def search(self, query, *, n=10, since=None):
            return [SearchResult(url="http://ok/", source="duckduckgo")]

    from banna_agent.tools.search import tool as tool_mod
    monkeypatch.setitem(tool_mod._BACKEND_FACTORIES, "tavily", _Boom)
    monkeypatch.setitem(tool_mod._BACKEND_FACTORIES, "duckduckgo", _Ok)
    out = search("q", backend="tavily,duckduckgo")
    # The cascade still produced a hit from the second backend.
    assert out["n_results"] == 1
    assert out["hits"][0]["url"] == "http://ok/"
    # The first backend's error is captured.
    assert out["errors"] and "RuntimeError" in out["errors"][0]["error"]


# ---------------------------------------------------------------------------
# DuckDuckGo HTML parsing
# ---------------------------------------------------------------------------


def test_yacy_relevance_filter_drops_off_topic_hits(monkeypatch) -> None:
    """YaCy returning hits whose title/snippet don't cover the query are dropped."""
    from banna_agent.tools.search.backends import yacy as yacy_mod

    payload = {
        "channels": [{
            "items": [
                # Off-topic: snippet has 'Iceland' (in a country list) but no
                # 'population'. Should be dropped under title-or-all-snippet.
                {"link": "https://aqr.com/contact",
                 "title": "Contact AQR",
                 "description": "Guyana Haiti Honduras Iceland India Indonesia"},
                # On-topic via title.
                {"link": "https://en.wikipedia.org/wiki/Demographics_of_Iceland",
                 "title": "Demographics of Iceland - Wikipedia",
                 "description": "Iceland is a Nordic island country..."},
                # On-topic via full-snippet coverage (title doesn't mention it).
                {"link": "https://example.com/europe",
                 "title": "Europe demographics overview",
                 "description": "The population of Iceland is around 372,000."},
                # Off-topic: neither title nor snippet covers both tokens.
                {"link": "https://example.com/random",
                 "title": "Stack Overflow Q&A",
                 "description": "How do I print a list?"},
            ]
        }]
    }

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return payload

    import requests as real_requests
    monkeypatch.setattr(real_requests, "get", lambda *a, **kw: _Resp())

    out = yacy_mod.YaCyBackend().search("Iceland population", n=10)
    urls = [r.url for r in out]
    assert "https://en.wikipedia.org/wiki/Demographics_of_Iceland" in urls
    assert "https://example.com/europe" in urls
    assert "https://aqr.com/contact" not in urls
    assert "https://example.com/random" not in urls


def test_yacy_relevance_filter_disabled(monkeypatch) -> None:
    """min_query_overlap=0 disables the filter — every YaCy hit passes."""
    from banna_agent.tools.search.backends import yacy as yacy_mod

    payload = {"channels": [{"items": [
        {"link": "https://aqr.com/contact",
         "title": "Contact AQR",
         "description": "no overlap at all"},
    ]}]}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return payload

    import requests as real_requests
    monkeypatch.setattr(real_requests, "get", lambda *a, **kw: _Resp())

    out = yacy_mod.YaCyBackend(min_query_overlap=0).search("Iceland population", n=10)
    assert len(out) == 1


def test_ddg_unwrap_redirect_preserves_target() -> None:
    from banna_agent.tools.search.backends.duckduckgo import _unwrap_redirect
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ffoo&rut=abc"
    assert _unwrap_redirect(wrapped) == "https://example.com/foo"


def test_ddg_unwrap_redirect_passes_through_direct_url() -> None:
    from banna_agent.tools.search.backends.duckduckgo import _unwrap_redirect
    assert _unwrap_redirect("https://example.com/foo") == "https://example.com/foo"


def test_ddg_parser_on_minimal_html(monkeypatch) -> None:
    # Don't actually hit DDG; mock requests.request (the HTTP cache shim
    # delegates here when mode is `off`).
    from banna_agent.tools.search.backends import duckduckgo as ddg_mod
    from banna_agent.tools import _http_cache as _hc
    monkeypatch.delenv("BANNA_HTTP_CACHE", raising=False)
    _hc.set_cache(None)

    body = """
        <html><body>
          <div class="result">
            <a class="result__a" href="https://example.com/a">First</a>
            <span class="result__snippet">a snippet of text</span>
          </div>
          <div class="result">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fb">Second</a>
            <div class="result__body">another snippet</div>
          </div>
        </body></html>
        """

    class _Resp:
        status_code = 200
        text = body
        content = body.encode("utf-8")
        url = "https://html.duckduckgo.com/html/"
        headers: dict = {}
        def raise_for_status(self): pass

    import requests as real_requests
    monkeypatch.setattr(real_requests, "request", lambda *a, **kw: _Resp())
    out = ddg_mod.DuckDuckGoBackend().search("hello", n=10)
    assert len(out) == 2
    assert out[0].url == "https://example.com/a"
    assert out[0].title == "First"
    assert "snippet" in out[0].snippet
    # Redirect-wrapped URL got unwrapped.
    assert out[1].url == "https://example.com/b"


# ---------------------------------------------------------------------------
# make_search_tool factory
# ---------------------------------------------------------------------------


def test_make_search_tool_default_auto() -> None:
    tool = make_search_tool()
    assert tool.name == "search"
    # Schema includes the backend selector.
    assert "backend" in tool.input_schema["properties"]


def test_make_search_tool_handler_passes_query(monkeypatch) -> None:
    """The factory's handler should call search() with the query and backend."""
    captured = {}
    from banna_agent.tools.search import tool as tool_mod

    def fake_search(query, **kwargs):
        captured["query"] = query
        captured["backend"] = kwargs.get("backend")
        return {"query": query, "backend": captured["backend"], "n_results": 0,
                "hits": [], "errors": []}

    monkeypatch.setattr(tool_mod, "search", fake_search)
    tool = make_search_tool(backend="duckduckgo")
    tool.handler({"query": "hello"})
    assert captured["query"] == "hello"
    assert captured["backend"] == "duckduckgo"
