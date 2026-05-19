"""Unit tests for the URL reader tool.

Network calls are mocked via monkeypatching `requests.request` (the
URL reader now routes through `tools._http_cache.cached_request`, which
delegates to `requests.request` when the cache mode is off).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from banna_agent.tools import _http_cache as _hc
from banna_agent.tools.url_reader import make_url_reader_tool, read_url


@dataclass
class _FakeResponse:
    status_code: int
    headers: dict
    text: str

    @property
    def content(self) -> bytes:
        return self.text.encode("utf-8")

    @property
    def url(self) -> str:
        return "https://example.com/"


@pytest.fixture(autouse=True)
def _reset_cache_singleton(monkeypatch):
    # Ensure cache mode is `off` for every test, regardless of env.
    monkeypatch.delenv("BANNA_HTTP_CACHE", raising=False)
    _hc.set_cache(None)
    yield
    _hc.set_cache(None)


def _install_fake_get(monkeypatch, resp: _FakeResponse) -> dict:
    captured: dict = {}

    def fake_request(method, url, params=None, data=None, json=None,
                     headers=None, timeout=None, allow_redirects=True):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        captured["allow_redirects"] = allow_redirects
        return resp

    import requests
    monkeypatch.setattr(requests, "request", fake_request)
    return captured


def test_read_url_html_extracts_text_and_title(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <html>
      <head><title>Hello Title</title></head>
      <body>
        <script>var x = 1;</script>
        <nav>skip me</nav>
        <p>Hello world</p>
        <p>second paragraph</p>
      </body>
    </html>
    """
    _install_fake_get(
        monkeypatch,
        _FakeResponse(status_code=200, headers={"Content-Type": "text/html; charset=utf-8"}, text=html),
    )
    out = read_url("https://example.com/page")
    assert out["status"] == 200
    assert out["content_type"] == "text/html"
    assert out["title"] == "Hello Title"
    assert "Hello world" in out["text"]
    assert "second paragraph" in out["text"]
    assert "var x = 1" not in out["text"]  # scripts stripped
    assert "skip me" not in out["text"]    # nav stripped
    assert out["truncated"] is False


def test_read_url_text_content_returned_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_get(
        monkeypatch,
        _FakeResponse(status_code=200, headers={"Content-Type": "text/plain"}, text="plain body"),
    )
    out = read_url("https://example.com/t.txt")
    assert out["text"] == "plain body"
    assert out["title"] == ""


def test_read_url_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    big = "x" * 10_000
    _install_fake_get(
        monkeypatch,
        _FakeResponse(status_code=200, headers={"Content-Type": "text/plain"}, text=big),
    )
    out = read_url("https://example.com", max_chars=100)
    assert len(out["text"]) == 100
    assert out["truncated"] is True


def test_handler_rejects_non_http_urls() -> None:
    tool = make_url_reader_tool()
    with pytest.raises(ValueError):
        tool.handler({"url": "file:///etc/passwd"})
    with pytest.raises(ValueError):
        tool.handler({"url": "not a url"})


def test_handler_sends_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_get(
        monkeypatch,
        _FakeResponse(status_code=200, headers={"Content-Type": "text/plain"}, text="ok"),
    )
    tool = make_url_reader_tool()
    tool.handler({"url": "https://example.com"})
    assert "banna_agent" in captured["headers"]["User-Agent"]


def test_url_reader_tool_capabilities() -> None:
    tool = make_url_reader_tool()
    assert tool.capabilities == frozenset({"network", "read"})
    assert tool.name == "read_url"
