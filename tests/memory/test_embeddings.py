"""Unit tests for embedding providers."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from banna_agent.memory.embeddings import (
    GeminiEmbedder,
    HashEmbedder,
    OpenAIEmbedder,
    cosine_similarity,
)


# ---------------------------------------------------------------------------
# HashEmbedder
# ---------------------------------------------------------------------------


def test_hash_embedder_produces_fixed_dim() -> None:
    e = HashEmbedder(dim=64)
    vecs = e.embed(["hello world", "foo bar"])
    assert len(vecs) == 2
    assert all(len(v) == 64 for v in vecs)


def test_hash_embedder_is_deterministic() -> None:
    e = HashEmbedder(dim=128)
    a1 = e.embed(["consistency"])[0]
    a2 = e.embed(["consistency"])[0]
    assert a1 == a2


def test_hash_embedder_vectors_are_l2_normalized() -> None:
    e = HashEmbedder(dim=128)
    v = e.embed(["hello world"])[0]
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-9 or norm == 0.0


def test_hash_embedder_similar_strings_correlate() -> None:
    e = HashEmbedder(dim=256)
    v1, v2, v3 = e.embed(["netflix arpu", "netflix arpu 2023", "bananas"])
    assert cosine_similarity(v1, v2) > cosine_similarity(v1, v3)


def test_hash_embedder_short_text() -> None:
    e = HashEmbedder(dim=64, n=5)
    v = e.embed(["hi"])[0]  # shorter than n
    assert len(v) == 64


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_identity() -> None:
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], []) == 0.0
    assert cosine_similarity([], [1.0, 0.0]) == 0.0


def test_cosine_length_mismatch_returns_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# OpenAIEmbedder with fake SDK
# ---------------------------------------------------------------------------


@dataclass
class _FakeOpenAIEmbResponse:
    data: list = field(default_factory=list)


@dataclass
class _FakeOpenAIEmb:
    def __init__(self, vecs: list[list[float]]) -> None:
        self.vecs = vecs
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return {
            "data": [{"embedding": v} for v in self.vecs],
        }


@dataclass
class _FakeOpenAISDK:
    embeddings: _FakeOpenAIEmb


def test_openai_embedder_calls_sdk() -> None:
    sdk = _FakeOpenAISDK(embeddings=_FakeOpenAIEmb([[0.1, 0.2], [0.3, 0.4]]))
    e = OpenAIEmbedder(sdk=sdk)
    vecs = e.embed(["a", "b"])
    assert vecs == [[0.1, 0.2], [0.3, 0.4]]
    assert sdk.embeddings.last_kwargs["input"] == ["a", "b"]
    assert sdk.embeddings.last_kwargs["model"] == "text-embedding-3-small"


def test_openai_embedder_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    e = OpenAIEmbedder()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        e.embed(["x"])


# ---------------------------------------------------------------------------
# GeminiEmbedder with fake HTTP
# ---------------------------------------------------------------------------


@dataclass
class _FakeResp:
    status_code: int
    _json: dict

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._json


def _make_post(payload: dict, captured: dict):
    def fake_post(url, params=None, json=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["body"] = json
        return _FakeResp(200, payload)
    return fake_post


def test_gemini_embedder_parses_batch_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    captured: dict = {}
    payload = {
        "embeddings": [
            {"values": [0.1, 0.2, 0.3]},
            {"values": [0.4, 0.5, 0.6]},
        ]
    }
    e = GeminiEmbedder(http_post=_make_post(payload, captured))
    vecs = e.embed(["a", "b"])
    assert vecs == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    body = captured["body"]
    assert len(body["requests"]) == 2
    assert body["requests"][0]["content"]["parts"][0]["text"] == "a"


def test_gemini_embedder_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_SEARCH_API_KEY", raising=False)
    e = GeminiEmbedder()
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        e.embed(["x"])
