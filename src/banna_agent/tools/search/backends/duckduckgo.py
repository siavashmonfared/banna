"""DuckDuckGo backend — HTML scrape, no API key.

DDG offers two surfaces:
  1. The instant-answer JSON at api.duckduckgo.com — covers a small
     subset of queries (Wikipedia infoboxes, calculator, etc.) and is
     useless for general web search.
  2. The HTML results page at html.duckduckgo.com — what we use here.

We POST to the HTML endpoint, parse the result list with BeautifulSoup
(already a dep via the url_reader tool), and map each `.result` block
to a SearchResult. The DOM is unstable in principle but has been
reasonably consistent for years; we keep selectors flexible.

No auth. No rate-limit headers exposed; ~30 requests/minute is the
informally-tolerated ceiling. We don't enforce that here — callers
should keep search frequency reasonable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from ..base import SearchResult


_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0 Safari/537.36"
)


def _unwrap_redirect(href: str) -> str:
    """DDG sometimes wraps result links in `/l/?uddg=<encoded>`. Decode it."""
    if not href:
        return href
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path in ("/l/", "/l"):
        qs = parse_qs(parsed.query)
        target = qs.get("uddg", [""])[0]
        if target:
            return unquote(target)
    return href


@dataclass
class DuckDuckGoBackend:
    """Scrape DDG HTML results (no API key, no SDK dep)."""

    name: str = "duckduckgo"
    source_tier: str = "web"
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "DUCKDUCKGO_BASE_URL", "https://html.duckduckgo.com/html/"
        )
    )
    user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "DUCKDUCKGO_USER_AGENT", _DEFAULT_USER_AGENT
        )
    )
    timeout_s: float = 8.0

    def search(
        self,
        query: str,
        *,
        n: int = 10,
        since: str | None = None,
    ) -> list[SearchResult]:
        from ..._http_cache import cached_request  # noqa: WPS433
        from bs4 import BeautifulSoup

        # `df` is DDG's date filter; mirrors common 'since' conventions.
        df_map = {"day": "d", "week": "w", "month": "m", "year": "y"}
        df = df_map.get((since or "").lower(), "")

        data = {"q": query}
        if df:
            data["df"] = df

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = cached_request(
            "POST", self.base_url, data=data, headers=headers, timeout=self.timeout_s,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        results: list[SearchResult] = []
        # Each result lives in a div.result (sometimes also div.web-result).
        for block in soup.select("div.result, div.web-result"):
            # Title + URL
            a = block.select_one("a.result__a, a.result__url")
            if a is None:
                continue
            href = _unwrap_redirect((a.get("href") or "").strip())
            if not href or href.startswith("javascript:"):
                continue
            title = a.get_text(" ", strip=True)
            # Snippet
            snip_node = block.select_one(
                ".result__snippet, .result__snippet a, .result__body"
            )
            snippet = snip_node.get_text(" ", strip=True) if snip_node else ""
            results.append(SearchResult(
                url=href,
                title=title,
                snippet=snippet[:400],
                source=self.name,
                source_tier=self.source_tier,
            ))
            if len(results) >= n:
                break

        return results
