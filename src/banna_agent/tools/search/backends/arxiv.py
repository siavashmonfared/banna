"""arXiv backend — abstract / metadata search via the public API.

Endpoint: ``http://export.arxiv.org/api/query`` (Atom XML, no API key).
We use the standard ``search_query=`` parameter, which honors the
field-prefix syntax (``ti:``, ``au:``, ``abs:``, ``cat:``, ``all:``).
Bare queries get an implicit ``all:`` prefix.

`since` is mapped to arxiv's `submittedDate:[YYYYMMDDhhmm TO *]`
filter — useful when the agent wants recent preprints.

All HTTP goes through ``tools._http_cache.cached_request`` so the
record/replay machinery applies.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from urllib.parse import quote

from ..base import SearchResult


_FEED_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def _since_to_clause(since: str | None) -> str:
    """Convert 'day' | 'week' | 'month' | 'year' (or YYYYMMDD) to arxiv filter."""
    if not since:
        return ""
    days = {"day": 1, "week": 7, "month": 31, "year": 365}.get(since.lower())
    if days is not None:
        start = dt.date.today() - dt.timedelta(days=days)
        return f" AND submittedDate:[{start.strftime('%Y%m%d')}0000 TO *]"
    # Caller may pass a raw YYYYMMDD string.
    if re.fullmatch(r"\d{8}", since):
        return f" AND submittedDate:[{since}0000 TO *]"
    return ""


@dataclass
class ArxivBackend:
    """Abstract / metadata search across arxiv.org."""

    name: str = "arxiv"
    source_tier: str = "official"
    timeout_s: float = 15.0

    def search(
        self,
        query: str,
        *,
        n: int = 10,
        since: str | None = None,
    ) -> list[SearchResult]:
        from ..._http_cache import cached_request

        q = query.strip()
        if not q:
            return []
        # If the caller didn't use field prefixes, default to all:
        if not re.search(r"\b(ti|au|abs|cat|all):", q):
            q = f"all:{q}"
        q = q + _since_to_clause(since)

        url = (
            "http://export.arxiv.org/api/query"
            f"?search_query={quote(q)}"
            f"&start=0&max_results={int(n)}"
            "&sortBy=relevance&sortOrder=descending"
        )
        resp = cached_request(
            "GET", url,
            headers={"User-Agent": "banna_agent/0.1", "Accept": "application/atom+xml"},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()

        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            return []

        results: list[SearchResult] = []
        for entry in root.findall(f"{_FEED_NS}entry"):
            url_node = entry.find(f"{_FEED_NS}id")
            title_node = entry.find(f"{_FEED_NS}title")
            summary_node = entry.find(f"{_FEED_NS}summary")
            published_node = entry.find(f"{_FEED_NS}published")
            href = (url_node.text or "").strip() if url_node is not None else ""
            if not href:
                continue
            title = " ".join((title_node.text or "").split()) if title_node is not None else ""
            summary = " ".join((summary_node.text or "").split()) if summary_node is not None else ""
            authors = _extract_authors(entry)
            categories = _extract_categories(entry)
            published = (published_node.text or "").strip() if published_node is not None else None
            results.append(SearchResult(
                url=href,
                title=title,
                snippet=summary[:400],
                source=self.name,
                source_tier=self.source_tier,
                published_at=published,
                meta={"authors": authors, "categories": categories},
            ))
            if len(results) >= n:
                break
        return results


def _extract_authors(entry) -> list[str]:
    out: list[str] = []
    for a in entry.findall(f"{_FEED_NS}author"):
        name = a.find(f"{_FEED_NS}name")
        if name is not None and (name.text or "").strip():
            out.append(name.text.strip())
    return out


def _extract_categories(entry) -> list[str]:
    out: list[str] = []
    for c in entry.findall(f"{_FEED_NS}category"):
        term = c.get("term")
        if term:
            out.append(term)
    return out
