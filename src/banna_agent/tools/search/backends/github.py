"""GitHub backend — repo / code / issue search via the REST API.

Endpoint: ``https://api.github.com/search/repositories`` (default),
plus optional ``code`` and ``issues`` modes selected via the ``since``
parameter's first segment if it starts with ``mode:`` (overloaded
because the SearchBackend protocol doesn't expose mode otherwise).

Auth: optional ``GITHUB_TOKEN`` env var bumps the rate limit from 10
to 30 requests/minute. Unauth still works; just don't run a thousand
queries in a benchmark cell.

The GitHub `q=` syntax supports qualifiers (``language:python``,
``stars:>500``, ``user:foo``) — pass them inline in the query.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote

from ..base import SearchResult


_ENDPOINTS = {
    "repositories": "https://api.github.com/search/repositories",
    "code": "https://api.github.com/search/code",
    "issues": "https://api.github.com/search/issues",
}


def _parse_mode(since: str | None) -> tuple[str, str | None]:
    """Returns (mode, residual_since_or_none).

    Overload convention: if `since` is ``mode:code`` or ``mode:issues``,
    that selects the endpoint; otherwise it's passed through unused
    (GitHub search doesn't expose a recency filter in the API path).
    """
    if since and since.startswith("mode:"):
        mode = since.split(":", 1)[1].strip().lower()
        if mode not in _ENDPOINTS:
            return "repositories", None
        return mode, None
    return "repositories", since


@dataclass
class GitHubBackend:
    """Search across GitHub repos / code / issues."""

    name: str = "github"
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

        q = (query or "").strip()
        if not q:
            return []
        mode, _ = _parse_mode(since)
        url = f"{_ENDPOINTS[mode]}?q={quote(q)}&per_page={int(n)}"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "banna_agent/0.1",
        }
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        resp = cached_request("GET", url, headers=headers, timeout=self.timeout_s)
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            return []
        items = data.get("items") or []
        results: list[SearchResult] = []
        for it in items[:n]:
            r = _to_result(it, mode, self.name, self.source_tier)
            if r is not None:
                results.append(r)
        return results


def _to_result(it: dict, mode: str, source: str, source_tier: str) -> SearchResult | None:
    if mode == "repositories":
        url = (it.get("html_url") or "").strip()
        if not url:
            return None
        return SearchResult(
            url=url,
            title=(it.get("full_name") or it.get("name") or "").strip(),
            snippet=(it.get("description") or "").strip()[:400],
            source=source,
            source_tier=source_tier,
            published_at=it.get("created_at"),
            score=float(it.get("score") or 0.0),
            meta={
                "stars": int(it.get("stargazers_count") or 0),
                "language": it.get("language") or "",
                "forks": int(it.get("forks_count") or 0),
                "open_issues": int(it.get("open_issues_count") or 0),
                "default_branch": it.get("default_branch") or "",
                "kind": "repository",
            },
        )
    if mode == "code":
        url = (it.get("html_url") or "").strip()
        repo = (it.get("repository") or {}).get("full_name") or ""
        if not url:
            return None
        return SearchResult(
            url=url,
            title=f"{repo}: {it.get('path') or it.get('name') or ''}",
            snippet=(it.get("text_matches") or [{}])[0].get("fragment", "")[:400] if it.get("text_matches") else "",
            source=source,
            source_tier=source_tier,
            score=float(it.get("score") or 0.0),
            meta={"repo": repo, "path": it.get("path") or "", "kind": "code"},
        )
    if mode == "issues":
        url = (it.get("html_url") or "").strip()
        if not url:
            return None
        return SearchResult(
            url=url,
            title=(it.get("title") or "").strip(),
            snippet=(it.get("body") or "").strip()[:400],
            source=source,
            source_tier=source_tier,
            published_at=it.get("created_at"),
            score=float(it.get("score") or 0.0),
            meta={
                "number": int(it.get("number") or 0),
                "state": it.get("state") or "",
                "kind": "issue",
                "is_pr": "pull_request" in it,
            },
        )
    return None
