"""Tavily backend — paid commercial web search API.

Kept around as a paid-tier option in the cascade. Not in the default
auto cascade because of cost; users add it explicitly via the
`backend=` parameter or env override.

Endpoint: POST https://api.tavily.com/search
Auth: TAVILY_API_KEY
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..base import SearchResult


@dataclass
class TavilyBackend:
    """Tavily search-as-a-service."""

    name: str = "tavily"
    source_tier: str = "paid"
    timeout_s: float = 20.0

    def search(
        self,
        query: str,
        *,
        n: int = 10,
        since: str | None = None,
    ) -> list[SearchResult]:
        import requests

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set for tavily backend")

        body: dict[str, Any] = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": n,
            "include_answer": True,
        }
        if since:
            # Tavily uses `time_range` for recency: "day"|"week"|"month"|"year"
            body["time_range"] = since

        resp = requests.post(
            "https://api.tavily.com/search", json=body, timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for r in data.get("results") or []:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            results.append(SearchResult(
                url=url,
                title=(r.get("title") or "").strip(),
                snippet=(r.get("content") or "").strip()[:400],
                source=self.name,
                source_tier=self.source_tier,
                published_at=r.get("published_date"),
                score=float(r.get("score") or 0.0),
                meta={"answer": data.get("answer") or ""},
            ))
        return results[:n]
