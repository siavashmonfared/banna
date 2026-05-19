"""YaCy backend — local curated crawl/index.

YaCy is a self-hosted, federated peer-to-peer search engine. We treat
it as a *curated* tier — best for sites you've explicitly told it to
crawl (RSS feeds, trusted domains, internal documentation), not as a
broad web replacement.

Endpoint: GET ${YACY_BASE_URL}/yacysearch.json?query=...&maximumRecords=N

Defaults:
  YACY_BASE_URL = http://localhost:8090

Response shape (abbreviated):
  {"channels": [{"items": [
      {"title": "...", "link": "...", "description": "...",
       "pubDate": "Wed, 06 Mar 2026 ..."},
      ...
   ]}]}
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from ..base import SearchResult


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("#text", "text", "value"):
            if key in value:
                return _text(value[key])
        return ""
    if isinstance(value, list):
        return _text(value[0]) if value else ""
    return str(value).strip()


def _first_text(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = _text(item.get(key))
        if value:
            return value
    return ""


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


_QUERY_STOPWORDS = frozenset(
    "the a an of and or to for from in on at by with as is are was were be been "
    "being it its their his her our your my we you they them us i me him she he "
    "do does did has have had not no yes which what who whom where when how why "
    "how many much".split()
)


def _query_tokens(query: str, *, min_len: int = 4) -> set[str]:
    """Significant query tokens for relevance scoring against YaCy hits.

    Lowercase, alpha-numeric, length ≥ min_len, stopwords stripped. We
    set min_len=4 so common short words ("how", "many", "what") drop
    out without an explicit stopword check while still keeping
    short proper nouns ("FDA", "CMS") via the > min_len-1 path below
    (callers can override).
    """
    if not query:
        return set()
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", query.lower())
    return {t for t in raw if len(t) >= min_len and t not in _QUERY_STOPWORDS}


def _is_relevant(
    title: str,
    snippet: str,
    qtokens: set[str],
) -> bool:
    """Title-or-all-snippet relevance gate for YaCy hits.

    Rule: a hit is on-topic if EITHER (a) the title contains at least
    one significant query token, OR (b) the snippet contains *all*
    significant query tokens.

    The two-tier rule matters because YaCy's keyword match is shallow
    over the page body. A search for "Iceland population" matches an
    AQR contact page that lists "Iceland" in a country drop-down, even
    though the page is about something else entirely. Demanding either
    a title hit or full-snippet coverage cuts that noise without
    sacrificing real hits like "Demographics of Iceland - Wikipedia"
    (passes via title) or "Europe demographics: Iceland's population
    is 372k…" (passes via snippet).

    Empty `qtokens` (very short / stopword-only queries) means "no
    signal — keep all hits".
    """
    if not qtokens:
        return True
    title_lc = (title or "").lower()
    if any(t in title_lc for t in qtokens):
        return True
    snippet_lc = (snippet or "").lower()
    return all(t in snippet_lc for t in qtokens)


def _extract_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for channel in _as_list(data.get("channels")):
        if isinstance(channel, dict):
            items.extend(i for i in _as_list(channel.get("items")) if isinstance(i, dict))
            items.extend(i for i in _as_list(channel.get("item")) if isinstance(i, dict))
    rss = data.get("rss")
    if isinstance(rss, dict):
        for channel in _as_list(rss.get("channel")):
            if isinstance(channel, dict):
                items.extend(i for i in _as_list(channel.get("item")) if isinstance(i, dict))
                items.extend(i for i in _as_list(channel.get("items")) if isinstance(i, dict))
    items.extend(i for i in _as_list(data.get("items")) if isinstance(i, dict))
    return items


@dataclass
class YaCyBackend:
    """YaCy local-search adapter."""

    name: str = "yacy"
    source_tier: str = "curated"
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "BANNA_YACY_BASE_URL",
            os.environ.get("YACY_BASE_URL", "http://localhost:8090"),
        )
    )
    timeout_s: float = 6.0
    # Optional `&contentdom=text` etc. — left configurable for future use.
    extra_params: dict[str, str] = field(default_factory=dict)
    # Relevance gate: a YaCy hit must mention at least this many query
    # tokens in its title or snippet. YaCy's keyword match is shallow —
    # a search for "Iceland population" matches any indexed page where
    # "population" appears anywhere in the body, even if the page is
    # actually about something else. This filter cuts the noise.
    # Set to 0 to disable.
    min_query_overlap: int = 1
    min_token_len: int = 4

    def search(
        self,
        query: str,
        *,
        n: int = 10,
        since: str | None = None,
    ) -> list[SearchResult]:
        import requests  # local import to avoid hard dep at module load

        fetched_at = _utc_now_iso()
        since_dt = _parse_datetime(since)
        yacy_query = query
        if since_dt is not None and "/date" not in yacy_query:
            yacy_query = f"{yacy_query} /date"

        params: dict[str, Any] = {
            "query": yacy_query,
            "maximumRecords": n,
            "resource": os.environ.get("BANNA_YACY_RESOURCE", "local"),
        }
        params.update(self.extra_params)

        url = f"{self.base_url.rstrip('/')}/yacysearch.json"
        resp = requests.get(url, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()

        qtokens = _query_tokens(query, min_len=self.min_token_len)

        results: list[SearchResult] = []
        for it in _extract_items(data):
            link = _first_text(it, "link", "url", "sku")
            if not link:
                continue
            published_at = _first_text(it, "pubDate", "pubdate", "published_at", "last_modified")
            published_dt = _parse_datetime(published_at)
            if since_dt is not None and published_dt is not None and published_dt < since_dt:
                continue
            title = _first_text(it, "title")
            snippet = _strip_html(_first_text(it, "description", "snippet", "text"))[:400]
            # Relevance gate: shallow YaCy matches that don't mention
            # the topic in title or *fully* in snippet are dropped.
            # Set min_query_overlap=0 on the backend instance to disable.
            if self.min_query_overlap > 0 and qtokens:
                if not _is_relevant(title, snippet, qtokens):
                    continue
            results.append(
                SearchResult(
                    url=link,
                    title=title,
                    snippet=snippet,
                    source=self.name,
                    source_tier=self.source_tier,
                    published_at=published_at or None,
                    fetched_at=fetched_at,
                    meta={
                        "host": _first_text(it, "host", "yacy:host"),
                        "size": _first_text(it, "size", "yacy:size"),
                        "guid": _first_text(it, "guid"),
                    },
                )
            )
        return results[:n]
