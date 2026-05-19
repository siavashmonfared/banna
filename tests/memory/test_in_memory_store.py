"""Unit tests for InMemoryStore."""
from __future__ import annotations


from banna_agent.memory.base import MemoryEntry, MemoryQuery
from banna_agent.memory.in_memory_store import InMemoryStore


def test_write_then_read_by_id() -> None:
    s = InMemoryStore()
    eid = s.write(MemoryEntry(content="hello"))
    got = s.read_by_id(eid)
    assert got is not None
    assert got.content == "hello"


def test_read_missing_returns_none() -> None:
    s = InMemoryStore()
    assert s.read_by_id("nope") is None


def test_search_substring() -> None:
    s = InMemoryStore()
    s.write(MemoryEntry(content="Netflix ARPU is $11.64"))
    s.write(MemoryEntry(content="Apple TV revenue"))
    hits = s.search(MemoryQuery(query="netflix"))
    assert len(hits) == 1
    assert "Netflix" in hits[0][0].content


def test_search_kind_filter() -> None:
    s = InMemoryStore()
    s.write(MemoryEntry(content="compute stuff", kind="skill"))
    s.write(MemoryEntry(content="compute stuff", kind="lesson"))
    hits = s.search(MemoryQuery(query="compute", kind_filter="skill"))
    assert len(hits) == 1
    assert hits[0][0].kind == "skill"


def test_search_tag_filter() -> None:
    s = InMemoryStore()
    s.write(MemoryEntry(content="x", tags=["netflix"]))
    s.write(MemoryEntry(content="x", tags=["apple"]))
    hits = s.search(MemoryQuery(query="x", tags_filter=["netflix"]))
    assert len(hits) == 1


def test_search_min_confidence() -> None:
    s = InMemoryStore()
    s.write(MemoryEntry(content="sure thing", confidence=0.9))
    s.write(MemoryEntry(content="sure thing", confidence=0.2))
    hits = s.search(MemoryQuery(query="sure", min_confidence=0.5))
    assert len(hits) == 1
    assert hits[0][0].confidence == 0.9


def test_search_respects_k() -> None:
    s = InMemoryStore()
    for i in range(10):
        s.write(MemoryEntry(content=f"target {i}"))
    hits = s.search(MemoryQuery(query="target", k=3))
    assert len(hits) == 3


def test_delete_returns_true_if_present() -> None:
    s = InMemoryStore()
    eid = s.write(MemoryEntry(content="x"))
    assert s.delete(eid) is True
    assert s.delete(eid) is False
    assert s.read_by_id(eid) is None


def test_all_returns_ordered_entries() -> None:
    s = InMemoryStore()
    s.write(MemoryEntry(content="a"))
    s.write(MemoryEntry(content="b"))
    s.write(MemoryEntry(content="c", kind="skill"))
    all_entries = s.all()
    assert len(all_entries) == 3
    skills = s.all(kind="skill")
    assert len(skills) == 1
    assert skills[0].kind == "skill"


def test_clear_empties_the_store() -> None:
    s = InMemoryStore()
    s.write(MemoryEntry(content="x"))
    s.write(MemoryEntry(content="y"))
    assert len(s) == 2
    s.clear()
    assert len(s) == 0
    assert s.all() == []


def test_store_satisfies_memory_protocol() -> None:
    from banna_agent.memory.base import Memory
    s = InMemoryStore()
    assert isinstance(s, Memory)


def test_search_score_ranking() -> None:
    s = InMemoryStore()
    s.write(MemoryEntry(content="netflix arpu is 11.64"))
    s.write(MemoryEntry(content="netflix revenue numbers"))
    hits = s.search(MemoryQuery(query="netflix arpu"))
    # Exact / contained match beats partial token overlap.
    assert hits[0][0].content.startswith("netflix arpu")
