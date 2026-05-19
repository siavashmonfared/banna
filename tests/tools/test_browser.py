"""Browser-tool gate tests for Phase 2.

The Phase 2 gate: a scripted multi-hop fixture (page A → linked page B
→ extract a fact) passes end-to-end with the HTTP cache in replay mode.
We prime the cache by recording two synthetic pages through an injected
fake `_live_request`, then construct a replay-only cache, install it
as the singleton, and drive the browser through `open → click → find`.

The gate path is the load-bearing test. Unit tests cover view paging,
back, find/next, click-by-text, the fetch budget, and the PDF hint.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from banna_agent.tools import _http_cache as hc
from banna_agent.tools.browser import (
    BrowserSession,
    make_browser_tools,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


PAGE_A_HTML = b"""<!doctype html>
<html><head><title>Page A - Index</title></head>
<body>
  <header><nav>skip me</nav></header>
  <main>
    <article>
      <h1>Index of Topics</h1>
      <p>Welcome to the index. Pick a topic below.</p>
      <ul>
        <li><a href="/wiki/topic_x">Topic X</a></li>
        <li><a href="https://example.test/wiki/topic_y">Topic Y</a></li>
      </ul>
    </article>
  </main>
  <footer>copyright nope</footer>
</body></html>
"""

PAGE_B_HTML = b"""<!doctype html>
<html><head><title>Topic X</title></head>
<body>
  <main>
    <h1>Topic X</h1>
    <p>Topic X was founded in <b>1847</b> by a group of researchers.</p>
    <p>It is best known for the Topic X principle.</p>
  </main>
</body></html>
"""


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    monkeypatch.delenv("BANNA_HTTP_CACHE", raising=False)
    hc.set_cache(None)
    yield
    hc.set_cache(None)


def _prime_cache(tmp_path: Path) -> None:
    """Record two canned pages through a fake live request, then leave
    the disk store ready for a replay-mode cache to consume."""
    canned = {
        "https://example.test/wiki/index": (PAGE_A_HTML, "text/html"),
        "https://example.test/wiki/topic_x": (PAGE_B_HTML, "text/html"),
        "https://example.test/wiki/topic_y": (b"<html><body>unused</body></html>", "text/html"),
    }

    def fake_live(method, url, params, data, json_body, headers, timeout):
        body, ctype = canned[url]
        return hc.CachedResponse(
            status_code=200, url=url, headers={"Content-Type": ctype},
            content=body, from_cache=False,
        )

    rec = hc.HttpCache(root=tmp_path, mode="record", _live_request=fake_live)
    for url in canned:
        rec.fetch("GET", url, headers={"User-Agent": "test", "Accept": "*/*"})


# ---------------------------------------------------------------------------
# Gate: multi-hop in replay mode.
# ---------------------------------------------------------------------------


def test_multi_hop_open_click_find_in_replay_mode(tmp_path: Path) -> None:
    _prime_cache(tmp_path)

    def boom(*a, **kw):
        raise AssertionError("replay must not call live HTTP")

    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay", _live_request=boom))

    tools = make_browser_tools()
    tools_by_name = {t.name: t for t in tools}

    # 1. Open index page.
    r1 = tools_by_name["browser_open"].handler({"url": "https://example.test/wiki/index"})
    assert r1["ok"] is True
    assert r1["title"] == "Page A - Index"
    assert "skip me" not in r1["text_chunk"]   # nav stripped
    assert "copyright nope" not in r1["text_chunk"]  # footer stripped
    # Links must be indexed and absolute.
    hrefs = [l["href"] for l in r1["links"]]
    assert "https://example.test/wiki/topic_x" in hrefs

    # 2. Click "Topic X" by text.
    r2 = tools_by_name["browser_click"].handler({"text": "Topic X"})
    assert r2["ok"] is True
    assert r2["title"] == "Topic X"
    assert r2["current_url"] == "https://example.test/wiki/topic_x"

    # 3. Find the year and read its snippet.
    r3 = tools_by_name["browser_find"].handler({"query": "1847"})
    assert r3["ok"] is True
    assert r3["match_count"] == 1
    assert "1847" in r3["snippet"]
    assert "founded in" in r3["snippet"].lower()

    # 4. Back out to page A, confirm state.
    r4 = tools_by_name["browser_back"].handler({})
    assert r4["ok"] is True
    assert r4["title"] == "Page A - Index"


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_view_pagination(tmp_path: Path) -> None:
    _prime_cache(tmp_path)
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))

    sess = BrowserSession(chunk_chars=20)
    r = sess.open("https://example.test/wiki/topic_x")
    assert r["ok"] is True
    assert r["has_more"] is True
    assert len(r["text_chunk"]) == 20
    r2 = sess.view(start=r["view_end"])
    assert r2["view_start"] == 20
    assert r2["text_chunk"]  # non-empty


def test_find_next_cycles(tmp_path: Path) -> None:
    _prime_cache(tmp_path)
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))
    sess = BrowserSession()
    sess.open("https://example.test/wiki/topic_x")
    r1 = sess.find("Topic X")
    n = r1["match_count"]
    assert n >= 2
    r2 = sess.next_match()
    assert r2["match_index"] == 1
    # Cycle through all matches; ends back at 0.
    for _ in range(n - 1):
        r2 = sess.next_match()
    assert r2["match_index"] == 0


def test_back_with_empty_history_is_error(tmp_path: Path) -> None:
    _prime_cache(tmp_path)
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))
    sess = BrowserSession()
    sess.open("https://example.test/wiki/index")
    r = sess.back()
    assert r["ok"] is False
    assert "history is empty" in r["error"]


def test_click_by_nth(tmp_path: Path) -> None:
    _prime_cache(tmp_path)
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))
    sess = BrowserSession()
    sess.open("https://example.test/wiki/index")
    r = sess.click(nth=0)
    assert r["ok"] is True
    assert r["current_url"].endswith("/topic_x")


def test_click_with_no_match_is_error(tmp_path: Path) -> None:
    _prime_cache(tmp_path)
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))
    sess = BrowserSession()
    sess.open("https://example.test/wiki/index")
    r = sess.click(text="not present anywhere")
    assert r["ok"] is False
    assert "matched" in r["error"]


def test_fetch_budget_enforced(tmp_path: Path) -> None:
    _prime_cache(tmp_path)
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))
    sess = BrowserSession(fetch_budget=1)
    sess.open("https://example.test/wiki/index")
    # Second open should refuse before fetching.
    r = sess.open("https://example.test/wiki/topic_x")
    assert r["ok"] is False
    assert "budget" in r["error"]


def test_pdf_url_returns_hint_not_garbage(tmp_path: Path) -> None:
    def fake_live(method, url, params, data, json_body, headers, timeout):
        return hc.CachedResponse(
            status_code=200, url=url,
            headers={"Content-Type": "application/pdf"},
            content=b"%PDF-1.4 ...binary...", from_cache=False,
        )

    rec = hc.HttpCache(root=tmp_path, mode="record", _live_request=fake_live)
    rec.fetch("GET", "https://example.test/paper.pdf",
              headers={"User-Agent": "test", "Accept": "*/*"})
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))

    sess = BrowserSession()
    r = sess.open("https://example.test/paper.pdf")
    assert r["ok"] is True
    assert r["content_type"].endswith("pdf")
    assert "PDF detected" in r["text_chunk"]


def test_make_browser_tools_returns_six_tools_sharing_session(tmp_path: Path) -> None:
    _prime_cache(tmp_path)
    hc.set_cache(hc.HttpCache(root=tmp_path, mode="replay"))
    tools = make_browser_tools()
    names = sorted(t.name for t in tools)
    assert names == sorted([
        "browser_open", "browser_view", "browser_find",
        "browser_next", "browser_click", "browser_back",
    ])
    # The shared session means opening via tool A is visible to tool B.
    by_name = {t.name: t for t in tools}
    by_name["browser_open"].handler({"url": "https://example.test/wiki/index"})
    r = by_name["browser_find"].handler({"query": "Welcome"})
    assert r["ok"] is True
    assert r["match_count"] >= 1
