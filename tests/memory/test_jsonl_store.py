"""Unit tests for JSONLStore."""
from __future__ import annotations

import json
from pathlib import Path

from banna_agent.memory.base import MemoryEntry, MemoryQuery
from banna_agent.memory.jsonl_store import JSONLStore


def test_jsonl_write_persists_to_disk(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    s = JSONLStore(p)
    eid = s.write(MemoryEntry(content="hello"))
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["content"] == "hello"
    assert rec["id"] == eid


def test_jsonl_reload_rebuilds_cache(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    s = JSONLStore(p)
    eid1 = s.write(MemoryEntry(content="a"))
    eid2 = s.write(MemoryEntry(content="b"))

    s2 = JSONLStore(p)
    assert s2.read_by_id(eid1) is not None
    assert s2.read_by_id(eid2) is not None
    assert len(s2) == 2


def test_jsonl_delete_writes_tombstone_and_reload_respects_it(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    s = JSONLStore(p)
    eid = s.write(MemoryEntry(content="gone"))
    assert s.delete(eid) is True
    # Tombstone line present
    assert "__tombstone__" in p.read_text()
    # Reload should not resurrect it
    s2 = JSONLStore(p)
    assert s2.read_by_id(eid) is None
    assert len(s2) == 0


def test_jsonl_search_uses_cache(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    s = JSONLStore(p)
    s.write(MemoryEntry(content="Netflix ARPU is 11.64", kind="fact"))
    s.write(MemoryEntry(content="Apple TV plans", kind="fact"))
    hits = s.search(MemoryQuery(query="netflix"))
    assert len(hits) == 1


def test_jsonl_clear_truncates_file(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    s = JSONLStore(p)
    s.write(MemoryEntry(content="x"))
    s.write(MemoryEntry(content="y"))
    s.clear()
    assert p.read_text() == ""
    assert len(s) == 0


def test_jsonl_ignores_malformed_lines_on_load(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    s = JSONLStore(p)
    eid = s.write(MemoryEntry(content="ok"))
    # Append a malformed line.
    with p.open("a", encoding="utf-8") as f:
        f.write("not json\n")
        f.write("\n")
    s2 = JSONLStore(p)
    assert s2.read_by_id(eid) is not None
    assert len(s2) == 1


def test_jsonl_kind_and_tags_persist(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    s = JSONLStore(p)
    s.write(MemoryEntry(content="x", kind="skill", tags=["math"]))
    s2 = JSONLStore(p)
    entries = s2.all()
    assert entries[0].kind == "skill"
    assert entries[0].tags == ["math"]


def test_jsonl_creates_parent_dirs(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "deep" / "m.jsonl"
    s = JSONLStore(p)
    s.write(MemoryEntry(content="x"))
    assert p.exists()
    assert p.parent.exists()


def test_jsonl_satisfies_memory_protocol(tmp_path: Path) -> None:
    from banna_agent.memory.base import Memory
    s = JSONLStore(tmp_path / "m.jsonl")
    assert isinstance(s, Memory)
