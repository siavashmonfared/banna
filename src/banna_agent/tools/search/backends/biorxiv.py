"""bioRxiv backend — preprint search via Europe PMC.

bioRxiv itself doesn't expose a free-text search API (the
``api.biorxiv.org`` endpoints are DOI-lookup only). Europe PMC
indexes bioRxiv and medRxiv preprints and offers a clean JSON search
at:

    https://www.ebi.ac.uk/europepmc/webservices/rest/search

We restrict to preprints with the ``SRC:PPR`` qualifier. Each result
is mapped to a SearchResult with DOI, abstract, authors, journal
(biorxiv / medrxiv) in meta.

No API key required. All HTTP goes through
``tools._http_cache.cached_request``.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from urllib.parse import quote

from ..base import SearchResult


_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _since_to_clause(since: str | None) -> str:
    if not since:
        return ""
    days = {"day": 1, "week": 7, "month": 31, "year": 365}.get(since.lower())
    if days is not None:
        start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        return f" AND FIRST_PDATE:[{start} TO *]"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", since):
        return f" AND FIRST_PDATE:[{since} TO *]"
    return ""


@dataclass
class BiorxivBackend:
    """Preprint search restricted to bioRxiv / medRxiv via Europe PMC."""

    name: str = "biorxiv"
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
        # SRC:PPR limits to preprints (bioRxiv, medRxiv, ChemRxiv, etc).
        # Add the preprint-server filter to keep it tighter for our use.
        full_q = f"({q}) AND SRC:PPR{_since_to_clause(since)}"
        url = (
            f"{_ENDPOINT}?query={quote(full_q)}"
            f"&format=json&pageSize={int(n)}&resultType=core"
        )
        resp = cached_request(
            "GET", url,
            headers={"User-Agent": "banna_agent/0.1", "Accept": "application/json"},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            return []
        result_list = (data.get("resultList") or {}).get("result") or []
        out: list[SearchResult] = []
        for r in result_list[:n]:
            doi = (r.get("doi") or "").strip()
            url_link = ""
            for u in (r.get("fullTextUrlList") or {}).get("fullTextUrl") or []:
                if u.get("documentStyle") in ("html", "doi") and u.get("url"):
                    url_link = u["url"]; break
            if not url_link and doi:
                url_link = f"https://doi.org/{doi}"
            if not url_link:
                continue
            authors = [a.strip() for a in (r.get("authorString") or "").split(",") if a.strip()]
            out.append(SearchResult(
                url=url_link,
                title=(r.get("title") or "").strip().rstrip("."),
                snippet=(r.get("abstractText") or "").strip()[:400],
                source=self.name,
                source_tier=self.source_tier,
                published_at=r.get("firstPublicationDate") or r.get("pubYear"),
                meta={
                    "doi": doi,
                    "authors": authors,
                    "preprint_server": (r.get("bookOrReportDetails") or {}).get("publisher")
                                       or r.get("source")
                                       or "",
                    "citation_count": int(r.get("citedByCount") or 0),
                },
            ))
        return out
