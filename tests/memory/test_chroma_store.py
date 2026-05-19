"""Optional Chroma-backed store tests. Skipped if chromadb not installed."""
from __future__ import annotations

from pathlib import Path

import pytest

chromadb = pytest.importorskip("chromadb")

from banna_agent.memory.base import MemoryEntry, MemoryQuery  # noqa: E402
from banna_agent.memory.chroma_store import ChromaStore  # noqa: E402
from banna_agent.memory.embeddings import HashEmbedder  # noqa: E402


def test_chroma_write_read_roundtrip(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "cdb", embedder=HashEmbedder(dim=64))
    eid = store.write(MemoryEntry(content="Netflix ARPU is $11.64", kind="fact"))
    got = store.read_by_id(eid)
    assert got is not None
    assert got.content.startswith("Netflix")


def test_chroma_search_returns_ranked(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "cdb2", embedder=HashEmbedder(dim=128))
    store.write(MemoryEntry(content="Netflix ARPU 2023"))
    store.write(MemoryEntry(content="Apple TV plans"))
    store.write(MemoryEntry(content="Amazon Prime shipping rates"))
    hits = store.search(MemoryQuery(query="netflix arpu", k=3))
    assert hits
    assert "netflix" in hits[0][0].content.lower()


def test_chroma_delete(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "cdb3", embedder=HashEmbedder(dim=64))
    eid = store.write(MemoryEntry(content="gone"))
    assert store.delete(eid) is True
    assert store.read_by_id(eid) is None


def test_chroma_kind_filter(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "cdb4", embedder=HashEmbedder(dim=64))
    store.write(MemoryEntry(content="x", kind="skill"))
    store.write(MemoryEntry(content="x", kind="lesson"))
    skills = store.all(kind="skill")
    assert len(skills) == 1
    assert skills[0].kind == "skill"


def test_chroma_clear(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "cdb5", embedder=HashEmbedder(dim=64))
    store.write(MemoryEntry(content="a"))
    store.write(MemoryEntry(content="b"))
    store.clear()
    assert store.all() == []


def test_chroma_protocol_conformance(tmp_path: Path) -> None:
    from banna_agent.memory.base import Memory
    store = ChromaStore(tmp_path / "cdb6", embedder=HashEmbedder(dim=64))
    assert isinstance(store, Memory)
