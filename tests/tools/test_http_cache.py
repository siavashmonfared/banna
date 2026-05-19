"""HttpCache gate tests for Phase 1.

The gate (per the plan): record then replay returns identical bytes
without any live network call. We exercise the cache directly with an
injected `_live_request` so we don't need real `requests` plumbing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from banna_agent.tools import _http_cache as hc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_live(body: bytes = b"hello", status: int = 200, headers: dict | None = None):
    calls = {"n": 0}

    def live(method, url, params, data, json_body, headers_, timeout):
        calls["n"] += 1
        return hc.CachedResponse(
            status_code=status,
            url=url,
            headers=headers or {"Content-Type": "text/plain"},
            content=body,
            from_cache=False,
        )

    return live, calls


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    monkeypatch.delenv("BANNA_HTTP_CACHE", raising=False)
    hc.set_cache(None)
    yield
    hc.set_cache(None)


# ---------------------------------------------------------------------------
# Gate 1: record → replay returns identical bytes, zero live calls on replay.
# ---------------------------------------------------------------------------


def test_record_then_replay(tmp_path: Path) -> None:
    live, calls = _make_live(body=b"ground truth bytes")
    rec_cache = hc.HttpCache(root=tmp_path, mode="record", _live_request=live)
    r1 = rec_cache.fetch("GET", "https://example.com/x", params={"q": "a"})
    assert r1.from_cache is False
    assert r1.content == b"ground truth bytes"
    assert calls["n"] == 1

    # Replay against the same on-disk store; no live calls allowed.
    def boom(*a, **kw):
        raise AssertionError("replay must not hit live")

    rep_cache = hc.HttpCache(root=tmp_path, mode="replay", _live_request=boom)
    r2 = rep_cache.fetch("GET", "https://example.com/x", params={"q": "a"})
    assert r2.from_cache is True
    assert r2.content == b"ground truth bytes"
    assert rep_cache.stats() == {"hits": 1, "misses": 0,
                                 "bytes_read": len(b"ground truth bytes"),
                                 "bytes_written": 0}


def test_replay_miss_raises(tmp_path: Path) -> None:
    cache = hc.HttpCache(root=tmp_path, mode="replay")
    with pytest.raises(hc.HttpCacheMiss):
        cache.fetch("GET", "https://example.com/nowhere")


def test_replay_or_record_falls_through(tmp_path: Path) -> None:
    live, calls = _make_live(body=b"new")
    cache = hc.HttpCache(root=tmp_path, mode="replay_or_record", _live_request=live)
    r1 = cache.fetch("GET", "https://example.com/y")
    assert r1.from_cache is False
    assert calls["n"] == 1
    # Second call: same key now on disk.
    r2 = cache.fetch("GET", "https://example.com/y")
    assert r2.from_cache is True
    assert r2.content == b"new"
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Cache key sensitivity.
# ---------------------------------------------------------------------------


def test_key_depends_on_url_method_params_body(tmp_path: Path) -> None:
    live, calls = _make_live(body=b"x")
    cache = hc.HttpCache(root=tmp_path, mode="record", _live_request=live)
    cache.fetch("GET", "https://e.com/a")
    cache.fetch("GET", "https://e.com/a", params={"q": "1"})  # different
    cache.fetch("POST", "https://e.com/a")                      # different method
    cache.fetch("GET", "https://e.com/b")                       # different url
    cache.fetch("GET", "https://e.com/a", data={"k": "v"})      # different body
    assert calls["n"] == 5


def test_key_ignores_header_jitter(tmp_path: Path) -> None:
    live, calls = _make_live(body=b"x")
    cache = hc.HttpCache(root=tmp_path, mode="replay_or_record", _live_request=live)
    cache.fetch("GET", "https://e.com/a", headers={"User-Agent": "A"})
    cache.fetch("GET", "https://e.com/a", headers={"User-Agent": "B"})
    # Same key — only one live call.
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Env-driven singleton.
# ---------------------------------------------------------------------------


def test_get_cache_off_by_default() -> None:
    c = hc.get_cache()
    assert c.mode == "off"


def test_get_cache_parses_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BANNA_HTTP_CACHE", f"replay_or_record:{tmp_path}")
    hc.set_cache(None)
    c = hc.get_cache()
    assert c.mode == "replay_or_record"
    assert c.root == tmp_path.resolve()


def test_cached_request_off_mode_delegates(monkeypatch) -> None:
    # Mode `off` should call requests.request once and return a CachedResponse.
    captured = {}

    class _Resp:
        status_code = 200
        url = "https://e.com/"
        headers: dict = {}
        content = b"ok"

    def fake_request(method, url, **kw):
        captured["method"] = method
        captured["url"] = url
        return _Resp()

    import requests
    monkeypatch.setattr(requests, "request", fake_request)
    r = hc.cached_request("GET", "https://e.com/")
    assert isinstance(r, hc.CachedResponse)
    assert r.content == b"ok"
    assert captured == {"method": "GET", "url": "https://e.com/"}
