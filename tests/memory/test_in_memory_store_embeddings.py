"""InMemoryStore with an embedder: cosine ranking becomes the primary signal."""
from __future__ import annotations

from banna_agent.memory.base import MemoryEntry, MemoryQuery
from banna_agent.memory.embeddings import HashEmbedder
from banna_agent.memory.in_memory_store import InMemoryStore


def test_write_auto_embeds_entry() -> None:
    store = InMemoryStore(embedder=HashEmbedder(dim=64))
    eid = store.write(MemoryEntry(content="Netflix ARPU"))
    got = store.read_by_id(eid)
    assert got is not None
    assert got.embedding is not None
    assert len(got.embedding) == 64


def test_search_uses_cosine_when_embedder_present() -> None:
    """With HashEmbedder: the semantically-closer entry should outrank an
    entry that only shares one token."""
    store = InMemoryStore(embedder=HashEmbedder(dim=256))
    store.write(MemoryEntry(content="Netflix average revenue per user was 11.64"))
    store.write(MemoryEntry(content="Apple TV average revenue per user"))
    store.write(MemoryEntry(content="Bananas are yellow"))
    hits = store.search(MemoryQuery(query="netflix arpu 2023", k=3))
    assert hits
    assert "Netflix" in hits[0][0].content


def test_search_works_with_entries_missing_embeddings() -> None:
    """Entries added before an embedder was attached still participate via
    substring scoring."""
    store = InMemoryStore()  # no embedder
    store.write(MemoryEntry(content="Netflix ARPU 2023"))
    # now attach an embedder post-hoc; existing entry has no embedding
    store.embedder = HashEmbedder(dim=64)
    hits = store.search(MemoryQuery(query="netflix"))
    assert len(hits) == 1


def test_substring_still_contributes_signal() -> None:
    """Even with cosine, exact substring presence should still score > 0."""
    store = InMemoryStore(embedder=HashEmbedder(dim=64))
    store.write(MemoryEntry(content="lorem ipsum"))
    hits = store.search(MemoryQuery(query="lorem"))
    assert hits
    assert hits[0][1] > 0
