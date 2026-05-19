"""Google-grounded search via Gemini's `google_search` tool.

This is *not* the Programmable Search Engine REST API. It's the
grounding feature exposed through Gemini's chat endpoint: the model
runs a Google search internally and returns `groundingMetadata` with
URLs, titles, and the snippets the model actually used.

That's the right surface for our use because the response carries
provider-attributed citations — exactly what the citation verifier
reads downstream. The downside is that you pay for one Gemini-Flash
call per search, and quotas apply.

Endpoint: POST generativelanguage.googleapis.com/v1beta/models/
          gemini-2.5-flash:generateContent  (with tools=[{google_search:{}}])

Auth: GOOGLE_API_KEY (also accepted: GOOGLE_SEARCH_API_KEY).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from ..base import SearchResult


@dataclass
class GoogleGroundedBackend:
    """Gemini-grounded search. Falls back to raise() if no key is set."""

    name: str = "google"
    source_tier: str = "grounded"
    model: str = "gemini-2.5-flash"
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com",
        )
    )
    timeout_s: float = 20.0
    max_output_tokens: int = 1024

    def search(
        self,
        query: str,
        *,
        n: int = 10,
        since: str | None = None,
    ) -> list[SearchResult]:
        import requests

        api_key = (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_SEARCH_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set for google-grounded search backend"
            )

        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        prompt = f"Web search: {query}\n\nReturn a concise summary and cite sources."
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        resp = requests.post(
            url, params={"key": api_key}, json=body, timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        summary = ""
        cands = data.get("candidates") or []
        if cands:
            parts = cands[0].get("content", {}).get("parts", [])
            summary = "".join(p.get("text", "") for p in parts)

        for cand in data.get("candidates", []):
            gm = cand.get("groundingMetadata") or {}
            chunks = gm.get("groundingChunks") or []
            for i, chunk in enumerate(chunks):
                web = chunk.get("web") or {}
                hit_url = web.get("uri")
                if not hit_url:
                    continue
                results.append(SearchResult(
                    url=hit_url,
                    title=(web.get("title") or "").strip(),
                    snippet="",
                    source=self.name,
                    source_tier=self.source_tier,
                    meta={"chunk_index": i, "summary": summary},
                ))
            # Attach the model-paraphrased snippets to the corresponding chunks.
            supports = gm.get("groundingSupports") or []
            for s in supports:
                seg = s.get("segment", {}).get("text", "") or ""
                idxs = s.get("groundingChunkIndices") or []
                for i in idxs:
                    if 0 <= i < len(results) and not results[i].snippet:
                        results[i].snippet = seg[:400]
        return results[:n]
