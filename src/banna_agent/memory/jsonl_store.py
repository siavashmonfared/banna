"""JSONL-backed persistent store.

Append-only log file (one `MemoryEntry` per line) + in-memory cache.
Reads go through the cache; writes update both. `delete` appends a
tombstone record so we preserve append-only semantics; `clear` truncates
the file.

This is the default store for cross-task learning in experiments.
Supports up to ~10K entries comfortably; past that, use Chroma.
"""
from __future__ import annotations

import json
from pathlib import Path
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


_TOMBSTONE_KIND = "__tombstone__"


class JSONLStore:
    """Persistent Memory backed by a JSONL file."""

    def __init__(self, path: str | Path, embedder: EmbeddingProvider | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self._cache: dict[str, MemoryEntry] = {}
        self.embedder = embedder
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._load()

    # --- persistence -----------------------------------------------------

    def _load(self) -> None:
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("__tombstone__"):
                self._cache.pop(rec.get("id", ""), None)
                continue
            entry = MemoryEntry.from_dict(rec)
            self._cache[entry.id] = entry

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # --- Memory protocol -------------------------------------------------

    def write(self, entry: MemoryEntry) -> str:
        if self.embedder is not None and entry.embedding is None:
            entry.embedding = self.embedder.embed([entry.content])[0]
        self._cache[entry.id] = entry
        self._append(entry.to_dict())
        return entry.id

    def read_by_id(self, entry_id: str) -> MemoryEntry | None:
        return self._cache.get(entry_id)

    def search(self, q: MemoryQuery) -> list[tuple[MemoryEntry, float]]:
        query_vec: list[float] | None = None
        if self.embedder is not None:
            query_vec = self.embedder.embed([q.query])[0]
        scored: list[tuple[MemoryEntry, float]] = []
        for entry in self._cache.values():
            if not entry_matches_filters(entry, q):
                continue
            sub = substring_score(q.query, entry.content)
            if query_vec is not None and entry.embedding is not None:
                cos = max(cosine_similarity(query_vec, entry.embedding), 0.0)
                score = 0.7 * cos + 0.3 * sub
            else:
                score = sub
            if score <= 0.0:
                continue
            scored.append((entry, score))
        scored.sort(key=lambda x: (-x[1], x[0].created_at))
        return scored[: q.k]

    def delete(self, entry_id: str) -> bool:
        if entry_id not in self._cache:
            return False
        del self._cache[entry_id]
        self._append({"__tombstone__": True, "id": entry_id})
        return True

    def all(self, *, kind: MemoryKind | None = None) -> list[MemoryEntry]:
        entries = list(self._cache.values())
        if kind is not None:
            entries = [e for e in entries if e.kind == kind]
        entries.sort(key=lambda e: e.created_at)
        return entries

    def clear(self) -> None:
        self._cache.clear()
        self.path.write_text("")

    # --- convenience -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._cache)


_check: Memory = JSONLStore.__new__(JSONLStore)  # type: ignore[arg-type]
