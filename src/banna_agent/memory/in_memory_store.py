"""In-memory dict-backed store.

Use for tests and smoke runs. No persistence across processes.

Scoring:
- If an `embedder` is set and the entry has an embedding, cosine is used.
- Otherwise, substring + token-overlap scoring is the fallback.
- The two are composable: entries without embeddings still participate
  in the search via substring scoring.
"""
from __future__ import annotations

from typing import Any

from .base import (
    Memory,
    MemoryEntry,
    MemoryKind,
    MemoryQuery,
    entry_matches_filters,
    substring_score,
)
from .embeddings import EmbeddingProvider, cosine_similarity


class InMemoryStore:
    """A dict-backed, process-local Memory implementation."""

    def __init__(self, embedder: EmbeddingProvider | None = None) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self.embedder = embedder

    # --- Memory protocol --------------------------------------------------

    def write(self, entry: MemoryEntry) -> str:
        if self.embedder is not None and entry.embedding is None:
            entry.embedding = self.embedder.embed([entry.content])[0]
        self._entries[entry.id] = entry
        return entry.id

    def read_by_id(self, entry_id: str) -> MemoryEntry | None:
        return self._entries.get(entry_id)

    def search(self, q: MemoryQuery) -> list[tuple[MemoryEntry, float]]:
        query_vec: list[float] | None = None
        if self.embedder is not None:
            query_vec = self.embedder.embed([q.query])[0]
        scored: list[tuple[MemoryEntry, float]] = []
        for entry in self._entries.values():
            if not entry_matches_filters(entry, q):
                continue
            score = _score_entry(entry, q.query, query_vec)
            if score <= 0.0:
                continue
            scored.append((entry, score))
        scored.sort(key=lambda x: (-x[1], -_ts_rank(x[0].created_at)))
        return scored[: q.k]

    def delete(self, entry_id: str) -> bool:
        return self._entries.pop(entry_id, None) is not None

    def all(self, *, kind: MemoryKind | None = None) -> list[MemoryEntry]:
        entries = list(self._entries.values())
        if kind is not None:
            entries = [e for e in entries if e.kind == kind]
        entries.sort(key=lambda e: e.created_at)
        return entries

    def clear(self) -> None:
        self._entries.clear()

    # --- convenience -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)


def _ts_rank(ts: str) -> float:
    """Stable float rank from an ISO timestamp; fine for ordering."""
    try:
        return float("".join(c for c in ts if c.isdigit()) or "0")
    except ValueError:
        return 0.0


def _score_entry(
    entry: MemoryEntry,
    query: str,
    query_vec: list[float] | None,
) -> float:
    """Combine cosine (if available) and substring scoring.

    When both signals are present, blend them 70/30 in favor of cosine —
    cosine is the semantically richer signal but substring catches exact
    lexical matches (dates, IDs, proper nouns) that embedding can blur.
    """
    sub = substring_score(query, entry.content)
    if query_vec is None or entry.embedding is None:
        return sub
    cos = cosine_similarity(query_vec, entry.embedding)
    cos = max(cos, 0.0)  # cosine can be negative with sparse vectors
    return 0.7 * cos + 0.3 * sub


# Runtime conformance: silence type checkers (and assert at import time).
_check: Memory = InMemoryStore()
