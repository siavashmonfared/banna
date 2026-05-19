"""Memory protocol and entry types.

`Memory` is a Protocol — every store (InMemory, JSONL, Chroma) satisfies it.
`MemoryEntry` is the canonical shape persisted and retrieved. `MemoryQuery`
is the search parameter bag.

Design notes:
- `content` is the human-readable payload. Searchable.
- `kind` partitions memory semantically: facts (external truths the agent
  has verified), lessons (agent-authored reflections), skills (callable
  code artifacts), examples ((question,answer) pairs), scratch (ephemeral).
- `source_task_id` and `created_at` give provenance for the citation
  verifier in week 2; `verified_by` records which verifiers have signed
  off on this entry (empty = unverified).
- `embedding` is optional. `None` means "this store doesn't use vectors"
  (InMemory/JSONL-without-embedder); a list means cosine search is
  available. Commit B wires this in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

from ..core.types import _now_iso


MemoryKind = Literal["fact", "lesson", "skill", "example", "scratch"]


def _short_id() -> str:
    return f"mem_{uuid4().hex[:8]}"


@dataclass
class MemoryEntry:
    """One persistent item of knowledge or skill.

    `content` is authoritative; `metadata` is free-form per-kind payload
    (e.g. for skills: `{"name", "signature", "code", "examples"}`).
    """

    content: str
    kind: MemoryKind = "fact"
    id: str = field(default_factory=_short_id)
    source_task_id: str | None = None
    created_at: str = field(default_factory=_now_iso)
    confidence: float = 1.0
    verified_by: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind,
            "source_task_id": self.source_task_id,
            "created_at": self.created_at,
            "confidence": self.confidence,
            "verified_by": list(self.verified_by),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "embedding": list(self.embedding) if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=d.get("id", _short_id()),
            content=d.get("content", ""),
            kind=d.get("kind", "fact"),
            source_task_id=d.get("source_task_id"),
            created_at=d.get("created_at", _now_iso()),
            confidence=float(d.get("confidence", 1.0)),
            verified_by=list(d.get("verified_by") or []),
            tags=list(d.get("tags") or []),
            metadata=dict(d.get("metadata") or {}),
            embedding=list(d["embedding"]) if d.get("embedding") else None,
        )


@dataclass
class MemoryQuery:
    """Search parameters for a Memory.search() call.

    - `query` is a free-text query. Stores may use substring, token
      overlap, or cosine-over-embedding; all return (entry, score) tuples
      with scores higher-is-better.
    - `kind_filter` restricts to one kind. None = all kinds.
    - `tags_filter` restricts to entries whose `tags` is a superset.
    - `min_confidence` drops low-confidence entries.
    """

    query: str
    k: int = 5
    kind_filter: MemoryKind | None = None
    tags_filter: list[str] = field(default_factory=list)
    min_confidence: float = 0.0


@runtime_checkable
class Memory(Protocol):
    """The shape every store satisfies."""

    def write(self, entry: MemoryEntry) -> str: ...
    def read_by_id(self, entry_id: str) -> MemoryEntry | None: ...
    def search(self, q: MemoryQuery) -> list[tuple[MemoryEntry, float]]: ...
    def delete(self, entry_id: str) -> bool: ...
    def all(self, *, kind: MemoryKind | None = None) -> list[MemoryEntry]: ...
    def clear(self) -> None: ...


# ---------------------------------------------------------------------------
# Utilities shared by stores
# ---------------------------------------------------------------------------


def substring_score(query: str, content: str) -> float:
    """Lightweight scoring: 1.0 for exact match, else fraction of query
    tokens present in content. Case-insensitive."""
    q = query.strip().lower()
    c = content.strip().lower()
    if not q:
        return 0.0
    if q == c:
        return 1.0
    if q in c:
        return 0.8
    q_tokens = {tok for tok in q.split() if tok}
    if not q_tokens:
        return 0.0
    c_tokens = set(c.split())
    overlap = len(q_tokens & c_tokens)
    return overlap / max(1, len(q_tokens))


def entry_matches_filters(entry: MemoryEntry, q: MemoryQuery) -> bool:
    """Apply kind / tags / min_confidence filters (not the query itself)."""
    if q.kind_filter is not None and entry.kind != q.kind_filter:
        return False
    if q.min_confidence > 0.0 and entry.confidence < q.min_confidence:
        return False
    if q.tags_filter:
        required = set(q.tags_filter)
        if not required.issubset(set(entry.tags)):
            return False
    return True
