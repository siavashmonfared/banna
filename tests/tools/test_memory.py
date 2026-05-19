"""Unit tests for the memory JsonTool."""
from __future__ import annotations

import pytest

from banna_agent.memory.in_memory_store import InMemoryStore
from banna_agent.tools.memory import make_memory_tool


@pytest.fixture
def tool_and_store():
    store = InMemoryStore()
    tool = make_memory_tool(store)
    return tool, store


def test_write_then_list(tool_and_store) -> None:
    tool, store = tool_and_store
    r = tool.handler({"op": "write", "content": "Paris is the capital of France"})
    assert "id" in r
    r2 = tool.handler({"op": "list"})
    assert r2["count"] == 1
    assert r2["entries"][0]["content"].startswith("Paris")


def test_write_with_tags_and_kind(tool_and_store) -> None:
    tool, _ = tool_and_store
    r = tool.handler({
        "op": "write",
        "content": "Netflix ARPU 2023 = 11.64",
        "kind": "fact",
        "tags": ["netflix", "arpu"],
        "confidence": 0.9,
    })
    eid = r["id"]
    read = tool.handler({"op": "read", "id": eid})
    assert read["entry"]["tags"] == ["netflix", "arpu"]
    assert read["entry"]["confidence"] == 0.9


def test_write_rejects_empty_content(tool_and_store) -> None:
    tool, _ = tool_and_store
    with pytest.raises(ValueError, match="non-empty"):
        tool.handler({"op": "write", "content": ""})


def test_write_rejects_bad_kind(tool_and_store) -> None:
    tool, _ = tool_and_store
    with pytest.raises(ValueError, match="kind must be"):
        tool.handler({"op": "write", "content": "x", "kind": "wrong"})


def test_search_returns_sorted_hits(tool_and_store) -> None:
    tool, _ = tool_and_store
    tool.handler({"op": "write", "content": "Netflix has high ARPU"})
    tool.handler({"op": "write", "content": "Apple TV subscriber counts"})
    tool.handler({"op": "write", "content": "Amazon Prime shipping"})
    r = tool.handler({"op": "search", "query": "netflix"})
    assert r["count"] == 1
    assert "Netflix" in r["hits"][0]["content"]
    assert r["hits"][0]["score"] > 0


def test_search_kind_and_tags_filters(tool_and_store) -> None:
    tool, _ = tool_and_store
    tool.handler({"op": "write", "content": "x", "kind": "skill", "tags": ["math"]})
    tool.handler({"op": "write", "content": "x", "kind": "lesson", "tags": ["math"]})
    r = tool.handler({"op": "search", "query": "x", "kind": "skill", "tags": ["math"]})
    assert r["count"] == 1
    assert r["hits"][0]["kind"] == "skill"


def test_delete_removes_entry(tool_and_store) -> None:
    tool, _ = tool_and_store
    eid = tool.handler({"op": "write", "content": "bye"})["id"]
    r = tool.handler({"op": "delete", "id": eid})
    assert r["ok"] is True
    r2 = tool.handler({"op": "read", "id": eid})
    assert r2["entry"] is None


def test_list_kind_filter(tool_and_store) -> None:
    tool, _ = tool_and_store
    tool.handler({"op": "write", "content": "a", "kind": "fact"})
    tool.handler({"op": "write", "content": "b", "kind": "lesson"})
    r = tool.handler({"op": "list", "kind": "lesson"})
    assert r["count"] == 1
    assert r["entries"][0]["kind"] == "lesson"


def test_unknown_op_raises(tool_and_store) -> None:
    tool, _ = tool_and_store
    with pytest.raises(ValueError, match="unknown op"):
        tool.handler({"op": "wiggle"})


def test_tool_metadata() -> None:
    store = InMemoryStore()
    tool = make_memory_tool(store)
    assert tool.name == "memory"
    assert tool.capabilities == frozenset({"state", "memory"})
    assert tool.input_schema["required"] == ["op"]


def test_write_passes_metadata_through(tool_and_store) -> None:
    tool, _ = tool_and_store
    eid = tool.handler({
        "op": "write",
        "content": "x",
        "metadata": {"source_task_id": "gaia-t1", "year": 2023},
    })["id"]
    r = tool.handler({"op": "read", "id": eid})
    assert r["entry"]["metadata"]["source_task_id"] == "gaia-t1"
    assert r["entry"]["metadata"]["year"] == 2023
