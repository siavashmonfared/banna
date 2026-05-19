"""Unit tests for MemoryEntry / MemoryQuery / filter helpers."""
from __future__ import annotations

from banna_agent.memory.base import (
    MemoryEntry,
    MemoryQuery,
    entry_matches_filters,
    substring_score,
)


def test_entry_defaults() -> None:
    e = MemoryEntry(content="hello")
    assert e.kind == "fact"
    assert e.confidence == 1.0
    assert e.verified_by == []
    assert e.tags == []
    assert e.metadata == {}
    assert e.embedding is None
    assert e.id.startswith("mem_")
    assert e.created_at  # ISO string


def test_entry_roundtrip() -> None:
    e = MemoryEntry(
        content="Netflix ARPU was $11.64",
        kind="fact",
        tags=["netflix", "arpu"],
        confidence=0.9,
        verified_by=["arithmetic"],
        metadata={"year": 2023},
    )
    d = e.to_dict()
    e2 = MemoryEntry.from_dict(d)
    assert e2.id == e.id
    assert e2.content == e.content
    assert e2.tags == e.tags
    assert e2.verified_by == e.verified_by
    assert e2.confidence == 0.9
    assert e2.metadata == {"year": 2023}


def test_query_defaults() -> None:
    q = MemoryQuery(query="x")
    assert q.k == 5
    assert q.kind_filter is None
    assert q.tags_filter == []
    assert q.min_confidence == 0.0


def test_substring_score_exact_match() -> None:
    assert substring_score("hello world", "hello world") == 1.0


def test_substring_score_contained() -> None:
    assert 0.7 < substring_score("netflix arpu", "netflix arpu is $11.64") <= 0.8


def test_substring_score_partial_tokens() -> None:
    s = substring_score("netflix arpu number", "netflix revenue numbers")
    assert 0 < s < 1


def test_substring_score_empty() -> None:
    assert substring_score("", "anything") == 0.0


def test_entry_matches_kind_filter() -> None:
    e = MemoryEntry(content="x", kind="lesson")
    assert entry_matches_filters(e, MemoryQuery(query="", kind_filter="lesson"))
    assert not entry_matches_filters(e, MemoryQuery(query="", kind_filter="skill"))


def test_entry_matches_tags_filter_superset() -> None:
    e = MemoryEntry(content="x", tags=["a", "b", "c"])
    assert entry_matches_filters(e, MemoryQuery(query="", tags_filter=["a"]))
    assert entry_matches_filters(e, MemoryQuery(query="", tags_filter=["a", "b"]))
    assert not entry_matches_filters(e, MemoryQuery(query="", tags_filter=["a", "z"]))


def test_entry_matches_min_confidence() -> None:
    e = MemoryEntry(content="x", confidence=0.4)
    assert entry_matches_filters(e, MemoryQuery(query="", min_confidence=0.3))
    assert not entry_matches_filters(e, MemoryQuery(query="", min_confidence=0.5))
