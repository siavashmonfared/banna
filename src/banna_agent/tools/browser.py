"""Stateful browser tool.

GAIA L2/L3 routinely require opening a page, scanning for a fact, then
following a link. `tools.url_reader.read_url` is one-shot and stateless;
it can't do step-2 cheaply because every call refetches and the LLM
has to copy-paste URLs around. The browser keeps a per-session

  * history stack          (back())
  * current page state     (URL, title, extracted text, indexed links)
  * find-cursor            (last query, match offsets, next-match pos)
  * fetch counter / budget (hard cap per session, prevents runaway)

…and exposes a small set of actions the LLM can compose.

All HTTP goes through `tools._http_cache.cached_request`, so a
record-then-replay benchmark run is deterministic and zero-cost on
replay.

PDFs are detected here but not yet read — Phase 3 wires `pdfplumber`.
For now, opening a PDF URL returns a hint so the LLM doesn't get back
mojibake.

The session is shared across the six tool actions returned by
`make_browser_tools()`; each call to that factory creates a fresh
session, so one GAIA task gets one browser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

if TYPE_CHECKING:
    from .base import JsonTool


_USER_AGENT = "banna_agent/0.1 (+https://github.com/)"
_DEFAULT_TIMEOUT_S = 20.0
_DEFAULT_FETCH_BUDGET = 25
_DEFAULT_CHUNK_CHARS = 1500
_MAX_LINKS_PER_PAGE = 200
_MAX_TEXT_CHARS = 200_000  # hard cap on stored page text


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Link:
    nth: int
    text: str
    href: str


@dataclass
class PageState:
    url: str
    title: str
    text: str
    links: list[Link]
    content_type: str
    # Find-cursor state.
    last_query: str = ""
    match_positions: list[int] = field(default_factory=list)
    match_cursor: int = 0

    @property
    def total_chars(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class BrowserSession:
    fetch_budget: int = _DEFAULT_FETCH_BUDGET
    fetch_count: int = 0
    timeout_s: float = _DEFAULT_TIMEOUT_S
    chunk_chars: int = _DEFAULT_CHUNK_CHARS
    history: list[PageState] = field(default_factory=list)
    current: PageState | None = None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def open(self, url: str, *, start: int = 0, n: int | None = None) -> dict[str, Any]:
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return _err("'url' must be an http(s) string")
        if self.fetch_count >= self.fetch_budget:
            return _err(f"fetch budget exhausted ({self.fetch_budget})")

        page, status, err = self._fetch(url)
        if err:
            return _err(err, status=status)
        # Push the previous page onto history before swapping.
        if self.current is not None:
            self.history.append(self.current)
        self.current = page
        return self._render_view(start=start, n=n)

    def view(self, start: int = 0, n: int | None = None) -> dict[str, Any]:
        if self.current is None:
            return _err("no current page; call browser_open first")
        return self._render_view(start=start, n=n)

    def find(self, query: str) -> dict[str, Any]:
        if self.current is None:
            return _err("no current page; call browser_open first")
        if not isinstance(query, str) or not query.strip():
            return _err("'query' must be a non-empty string")
        text = self.current.text
        q = query.lower()
        # Case-insensitive substring scan over the extracted text.
        positions: list[int] = []
        i = 0
        lower = text.lower()
        while True:
            j = lower.find(q, i)
            if j < 0:
                break
            positions.append(j)
            i = j + max(1, len(q))
        self.current.last_query = query
        self.current.match_positions = positions
        self.current.match_cursor = 0
        if not positions:
            return {
                "ok": True, "match_count": 0, "query": query,
                "current_url": self.current.url, "snippet": "",
            }
        return self._snippet_at(positions[0], query)

    def next_match(self) -> dict[str, Any]:
        if self.current is None:
            return _err("no current page; call browser_open first")
        if not self.current.match_positions:
            return _err("no active find cursor; call browser_find first")
        self.current.match_cursor = (self.current.match_cursor + 1) % len(self.current.match_positions)
        pos = self.current.match_positions[self.current.match_cursor]
        return self._snippet_at(pos, self.current.last_query)

    def click(self, *, nth: int | None = None, text: str | None = None) -> dict[str, Any]:
        if self.current is None:
            return _err("no current page; call browser_open first")
        target: Link | None = None
        if nth is not None:
            for link in self.current.links:
                if link.nth == int(nth):
                    target = link
                    break
            if target is None:
                return _err(f"no link with nth={nth}; visible links: 0..{len(self.current.links) - 1}")
        elif text:
            needle = text.lower()
            for link in self.current.links:
                if needle in link.text.lower():
                    target = link
                    break
            if target is None:
                return _err(f"no link text matched {text!r}")
        else:
            return _err("click requires either 'nth' or 'text'")
        return self.open(target.href)

    def back(self) -> dict[str, Any]:
        if not self.history:
            return _err("history is empty; nowhere to go back to")
        prev = self.history.pop()
        self.current = prev
        return self._render_view(start=0, n=None)

    def state(self) -> dict[str, Any]:
        """Inspection helper for the LLM (and tests)."""
        return {
            "current_url": self.current.url if self.current else "",
            "title": self.current.title if self.current else "",
            "history_depth": len(self.history),
            "fetch_count": self.fetch_count,
            "fetch_budget": self.fetch_budget,
            "total_chars": self.current.total_chars if self.current else 0,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch(self, url: str) -> tuple[PageState | None, int, str | None]:
        """Live (or cached) fetch + parse. Returns (page, status, err)."""
        from ._http_cache import cached_request  # local to keep import light

        self.fetch_count += 1
        try:
            resp = cached_request(
                "GET", url,
                headers={"User-Agent": _USER_AGENT, "Accept": "*/*"},
                timeout=self.timeout_s,
            )
        except Exception as exc:
            return None, 0, f"fetch failed: {type(exc).__name__}: {exc}"

        status = resp.status_code
        ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        final_url = resp.url or url

        if "pdf" in ctype or url.lower().endswith(".pdf"):
            page = PageState(
                url=final_url,
                title=_basename(final_url),
                text=(
                    f"[PDF detected at {final_url}; content-type={ctype or 'unknown'}. "
                    "The browser does not yet extract PDFs — use `read_file` on a local "
                    "path, or wait for the Phase 3 PDF reader.]"
                ),
                links=[],
                content_type=ctype or "application/pdf",
            )
            return page, status, None

        body = resp.text or ""
        if "html" not in ctype and not body.lstrip().startswith("<"):
            # Treat as plain text: no link extraction.
            page = PageState(
                url=final_url, title="", text=body[:_MAX_TEXT_CHARS],
                links=[], content_type=ctype or "text/plain",
            )
            return page, status, None

        title, text, links = _parse_html(body, base_url=final_url)
        page = PageState(
            url=final_url,
            title=title,
            text=text[:_MAX_TEXT_CHARS],
            links=links[:_MAX_LINKS_PER_PAGE],
            content_type=ctype or "text/html",
        )
        return page, status, None

    def _render_view(self, *, start: int, n: int | None) -> dict[str, Any]:
        assert self.current is not None
        page = self.current
        chunk_n = self.chunk_chars if n is None else max(1, int(n))
        start = max(0, int(start))
        end = min(page.total_chars, start + chunk_n)
        return {
            "ok": True,
            "current_url": page.url,
            "title": page.title,
            "content_type": page.content_type,
            "total_chars": page.total_chars,
            "view_start": start,
            "view_end": end,
            "has_more": end < page.total_chars,
            "text_chunk": page.text[start:end],
            "links": [
                {"nth": l.nth, "text": _short(l.text, 80), "href": l.href}
                for l in page.links
            ],
            "fetch_count": self.fetch_count,
        }

    def _snippet_at(self, pos: int, query: str, *, pad: int = 240) -> dict[str, Any]:
        assert self.current is not None
        text = self.current.text
        lo = max(0, pos - pad)
        hi = min(len(text), pos + len(query) + pad)
        return {
            "ok": True,
            "current_url": self.current.url,
            "query": query,
            "match_count": len(self.current.match_positions),
            "match_index": self.current.match_cursor,
            "match_pos": pos,
            "snippet": text[lo:hi],
        }


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------


def _parse_html(html: str, *, base_url: str) -> tuple[str, str, list[Link]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "", _strip_tags_naive(html), []

    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Strip noisy chrome before any extraction.
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    # Prefer the main content container if the page provides one.
    main = soup.find("main") or soup.find("article") or soup.body or soup

    # Link extraction: dedupe by href, keep stable order.
    links: list[Link] = []
    seen: set[str] = set()
    for a in main.find_all("a", href=True) if hasattr(main, "find_all") else []:
        href = a.get("href", "").strip()
        if not href or href.startswith(("javascript:", "mailto:", "#")):
            continue
        absolute = urljoin(base_url, href)
        if absolute in seen:
            continue
        seen.add(absolute)
        text = a.get_text(" ", strip=True) or absolute
        links.append(Link(nth=len(links), text=text, href=absolute))

    text_out = main.get_text(separator="\n", strip=True)
    # Squeeze obvious whitespace runs without destroying paragraph breaks.
    text_out = re.sub(r"\n{3,}", "\n\n", text_out)
    return title, text_out, links


def _strip_tags_naive(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html)


# ---------------------------------------------------------------------------
# JsonTool factories
# ---------------------------------------------------------------------------


def make_browser_tools(
    *,
    fetch_budget: int = _DEFAULT_FETCH_BUDGET,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    chunk_chars: int = _DEFAULT_CHUNK_CHARS,
) -> tuple["JsonTool", ...]:
    """Build a fresh BrowserSession and expose its actions as JsonTools.

    Returns a tuple of six tools, all sharing the one session. The
    caller registers them in a ToolRegistry; do not mix tools across
    sessions.
    """
    from .base import JsonTool

    session = BrowserSession(
        fetch_budget=fetch_budget,
        timeout_s=timeout_s,
        chunk_chars=chunk_chars,
    )

    open_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http(s) URL to open."},
            "start": {"type": "integer", "default": 0, "description": "Char offset for the initial view."},
            "n": {"type": "integer", "description": "Chars to include in the initial view (default ~chunk size)."},
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    view_schema = {
        "type": "object",
        "properties": {
            "start": {"type": "integer", "default": 0},
            "n": {"type": "integer"},
        },
        "additionalProperties": False,
    }
    find_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    click_schema = {
        "type": "object",
        "properties": {
            "nth": {"type": "integer", "description": "Link index from the most recent view."},
            "text": {"type": "string", "description": "Substring match on visible link text."},
        },
        "additionalProperties": False,
    }
    empty_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    tools = (
        JsonTool(
            name="browser_open",
            description=(
                "Open a URL in the stateful browser. Returns the page title, content type, "
                "a chunk of extracted main-text, and an indexed list of links. "
                "Subsequent actions (browser_view, browser_find, browser_click, browser_back) "
                "operate on this page until another open/click is issued."
            ),
            input_schema=open_schema,
            handler=lambda a: session.open(a["url"], start=int(a.get("start", 0)), n=a.get("n")),
            capabilities=frozenset({"network", "read"}),
        ),
        JsonTool(
            name="browser_view",
            description=(
                "Read another chunk of the current page. Pages are presented in slices of "
                "~chunk_chars; pass `start` to advance. The response's `has_more` field "
                "tells you whether more text exists past the returned slice."
            ),
            input_schema=view_schema,
            handler=lambda a: session.view(start=int(a.get("start", 0)), n=a.get("n")),
            capabilities=frozenset({"read"}),
        ),
        JsonTool(
            name="browser_find",
            description=(
                "Case-insensitive substring search on the current page's extracted text. "
                "Returns match count + a snippet around the first hit; advance with browser_next."
            ),
            input_schema=find_schema,
            handler=lambda a: session.find(a["query"]),
            capabilities=frozenset({"read"}),
        ),
        JsonTool(
            name="browser_next",
            description="Advance the find cursor to the next match on the current page.",
            input_schema=empty_schema,
            handler=lambda a: session.next_match(),
            capabilities=frozenset({"read"}),
        ),
        JsonTool(
            name="browser_click",
            description=(
                "Follow one of the links from the current page. Specify the link by `nth` "
                "(integer index from the latest view's `links`) or by `text` (substring match). "
                "Pushes the current page onto the history stack so browser_back can return."
            ),
            input_schema=click_schema,
            handler=lambda a: session.click(nth=a.get("nth"), text=a.get("text")),
            capabilities=frozenset({"network", "read"}),
        ),
        JsonTool(
            name="browser_back",
            description="Return to the previous page on the history stack.",
            input_schema=empty_schema,
            handler=lambda a: session.back(),
            capabilities=frozenset({"read"}),
        ),
    )
    return tools


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _err(msg: str, *, status: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "error": msg}
    if status:
        out["status"] = status
    return out


def _short(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _basename(url: str) -> str:
    path = urlparse(url).path
    return path.rsplit("/", 1)[-1] or url
