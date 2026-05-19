"""Embedding providers.

Three backends, all behind `EmbeddingProvider`:

  HashEmbedder   - zero-dep, deterministic, 256-d character 5-gram hashing.
                   Semantically weak but stable and hermetic for tests.
                   This is the default.

  OpenAIEmbedder - text-embedding-3-small (1536-d) via `openai` SDK.
  GeminiEmbedder - text-embedding-004 via Generative Language API.

Swap transparently — every store takes an optional `embedder` at
construction. If embedder is None, stores fall back to substring/token
scoring (commit A behavior).
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """One method. Returns `len(texts)` vectors, each of fixed dimension."""

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# HashEmbedder — deterministic, dep-free, default
# ---------------------------------------------------------------------------


def _ngrams(s: str, n: int = 5) -> list[str]:
    s = s.lower()
    if len(s) < n:
        return [s]
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def _hash_bucket(s: str, dim: int) -> int:
    h = hashlib.sha1(s.encode("utf-8")).digest()
    # Use 8 bytes → 64-bit int → mod dim.
    return int.from_bytes(h[:8], "big") % dim


@dataclass
class HashEmbedder:
    """Character-5-gram hashing into a fixed-dim float vector.

    Each 5-gram increments one bucket; the vector is then L2-normalized.
    Deterministic, fast, zero-dep. Good enough to beat substring-only
    ranking on short/fuzzy queries and perfect for tests.
    """

    dim: int = 256
    n: int = 5

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for gram in _ngrams(t, self.n):
                vec[_hash_bucket(gram, self.dim)] += 1.0
            out.append(_l2_normalize(vec))
        return out


def _l2_normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return v
    return [x / norm for x in v]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine on already-normalized vectors is just a dot product.
    We don't assume normalization — do the full computation so callers
    can pass raw provider embeddings."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------


@dataclass
class OpenAIEmbedder:
    """OpenAI text-embedding-3-small (1536-d).

    Requires OPENAI_API_KEY unless `sdk` is injected (tests use a fake).
    """

    model: str = "text-embedding-3-small"
    api_key: str | None = None
    sdk: Any = None
    dim: int = 1536

    def _client(self) -> Any:
        if self.sdk is not None:
            return self.sdk
        import openai
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set for embeddings.")
        self.sdk = openai.OpenAI(api_key=key)
        return self.sdk

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._client()
        resp = client.embeddings.create(model=self.model, input=list(texts))
        # SDK returns objects with `.data[i].embedding`; tests pass dicts.
        data = getattr(resp, "data", None) or resp.get("data", [])
        out: list[list[float]] = []
        for item in data:
            emb = getattr(item, "embedding", None) or item.get("embedding")
            out.append(list(emb or []))
        return out


# ---------------------------------------------------------------------------
# GeminiEmbedder
# ---------------------------------------------------------------------------


@dataclass
class GeminiEmbedder:
    """Gemini text-embedding-004 via Generative Language API.

    HTTP-only (no google-generativeai dep). `http_post` can be injected
    for tests to avoid network.
    """

    model: str = "text-embedding-004"
    api_key: str | None = None
    base_url: str = "https://generativelanguage.googleapis.com"
    http_post: Any = None
    dim: int = 768

    def _post(self):
        if self.http_post is not None:
            return self.http_post
        import requests
        return requests.post

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        key = self.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_SEARCH_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY not set for embeddings.")
        url = f"{self.base_url}/v1beta/models/{self.model}:batchEmbedContents"
        body = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                }
                for t in texts
            ]
        }
        post = self._post()
        resp = post(url, params={"key": key}, json=body, timeout=60.0)
        if hasattr(resp, "raise_for_status"):
            resp.raise_for_status()
        data = resp.json() if hasattr(resp, "json") else resp
        return [
            list((e.get("values") or []))
            for e in (data.get("embeddings") or [])
        ]
