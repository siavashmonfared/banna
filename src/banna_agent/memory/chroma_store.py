"""Optional Chroma-backed store (`chromadb.PersistentClient`).

Only useful at scale (≥10K entries). Lazy-imports `chromadb` so the
rest of the codebase doesn't pay the dep cost. Tests are
skipped if the package isn't installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    Memory,
    MemoryEntry,
    MemoryKind,
    MemoryQuery,
    entry_matches_filters,
)
from .embeddings import EmbeddingProvider


class ChromaStore:
    """Chroma-backed persistent Memory. Requires `chromadb` installed."""

    def __init__(
        self,
        path: str | Path,
        *,
        collection: str = "banna_memory",
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "`chromadb` not installed. `pip install chromadb` to use ChromaStore."
            ) from exc

        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedder

    # --- Memory protocol -------------------------------------------------

    def write(self, entry: MemoryEntry) -> str:
        emb = entry.embedding
        if emb is None and self.embedder is not None:
            emb = self.embedder.embed([entry.content])[0]
            entry.embedding = emb
        md = self._metadata(entry)
        if emb is not None:
            self._collection.upsert(
                ids=[entry.id],
                documents=[entry.content],
                embeddings=[list(emb)],
                metadatas=[md],
            )
        else:
            self._collection.upsert(
                ids=[entry.id], documents=[entry.content], metadatas=[md]
            )
        return entry.id

    def read_by_id(self, entry_id: str) -> MemoryEntry | None:
        res = self._collection.get(ids=[entry_id], include=["documents", "metadatas", "embeddings"])
        if not res.get("ids") or not res["ids"][0]:
            return None
        return self._to_entry(
            entry_id,
            res["documents"][0],
            (res.get("metadatas") or [{}])[0],
            (res.get("embeddings") or [None])[0],
        )

    def search(self, q: MemoryQuery) -> list[tuple[MemoryEntry, float]]:
        # Build a Chroma 'where' filter for kind / min_confidence.
        where: dict[str, Any] = {}
        if q.kind_filter is not None:
            where["kind"] = q.kind_filter
        query_embedding = None
        if self.embedder is not None:
            query_embedding = self.embedder.embed([q.query])[0]
        res = self._collection.query(
            query_texts=[q.query] if query_embedding is None else None,
            query_embeddings=[list(query_embedding)] if query_embedding is not None else None,
            n_results=max(q.k, 1),
            where=where or None,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        embs = (res.get("embeddings") or [[None] * len(ids)])[0]

        out: list[tuple[MemoryEntry, float]] = []
        for i, eid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            entry = self._to_entry(eid, docs[i], meta, embs[i] if i < len(embs) else None)
            if not entry_matches_filters(entry, q):
                continue
            # cosine distance -> similarity
            dist = float(dists[i]) if i < len(dists) else 1.0
            sim = max(0.0, 1.0 - dist)
            out.append((entry, sim))
        return out[: q.k]

    def delete(self, entry_id: str) -> bool:
        existing = self._collection.get(ids=[entry_id])
        if not existing.get("ids") or not existing["ids"]:
            return False
        self._collection.delete(ids=[entry_id])
        return True

    def all(self, *, kind: MemoryKind | None = None) -> list[MemoryEntry]:
        where = {"kind": kind} if kind is not None else None
        res = self._collection.get(
            where=where, include=["documents", "metadatas", "embeddings"]
        )
        out: list[MemoryEntry] = []
        for i, eid in enumerate(res.get("ids", [])):
            meta = (res.get("metadatas") or [{}])[i]
            doc = (res.get("documents") or [""])[i]
            emb = (res.get("embeddings") or [None])[i] if res.get("embeddings") else None
            out.append(self._to_entry(eid, doc, meta, emb))
        out.sort(key=lambda e: e.created_at)
        return out

    def clear(self) -> None:
        ids = self._collection.get().get("ids", [])
        if ids:
            self._collection.delete(ids=ids)

    # --- helpers --------------------------------------------------------

    @staticmethod
    def _metadata(entry: MemoryEntry) -> dict[str, Any]:
        # Chroma metadata values must be scalar; serialize lists as CSV.
        return {
            "kind": entry.kind,
            "source_task_id": entry.source_task_id or "",
            "created_at": entry.created_at,
            "confidence": float(entry.confidence),
            "verified_by": ",".join(entry.verified_by),
            "tags": ",".join(entry.tags),
        }

    @staticmethod
    def _to_entry(
        eid: str, document: str, meta: dict[str, Any], embedding: Any
    ) -> MemoryEntry:
        def _split(s: Any) -> list[str]:
            if not s:
                return []
            return [p for p in str(s).split(",") if p]

        return MemoryEntry(
            id=eid,
            content=document or "",
            kind=meta.get("kind", "fact"),
            source_task_id=(meta.get("source_task_id") or None) or None,
            created_at=meta.get("created_at", ""),
            confidence=float(meta.get("confidence", 1.0)),
            verified_by=_split(meta.get("verified_by")),
            tags=_split(meta.get("tags")),
            embedding=list(embedding) if embedding is not None else None,
        )
