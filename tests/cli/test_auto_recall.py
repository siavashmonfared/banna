"""Auto-recall: relevant memories surface in compose_question without the
model invoking the memory tool, and irrelevant ones don't."""
from __future__ import annotations

import pytest

from banna_agent.cli.session import Session
from banna_agent.memory.base import MemoryEntry
from banna_agent.memory.embeddings import HashEmbedder
from banna_agent.memory.in_memory_store import InMemoryStore


@pytest.fixture()
def session() -> Session:
    s = Session(memory_store=InMemoryStore(embedder=HashEmbedder(dim=256)))
    s.memory_store.write(MemoryEntry(content="The user prefers metric units and lives in Toronto."))
    s.memory_store.write(MemoryEntry(content="Project deadline is December 2026."))
    return s


def test_relevant_query_recalls_matching_entry(session):
    out = session.recall_preamble("What units should I use for the report?")
    assert "metric units" in out
    assert "Relevant from memory" in out


def test_recall_requires_lexical_overlap(session):
    # Pure hash-collision noise must not surface a memory.
    assert session.recall_preamble("asdfqwer zxcv unrelated gibberish") == ""


def test_recall_targets_the_right_entry(session):
    out = session.recall_preamble("When is the project deadline?")
    assert "December 2026" in out
    assert "metric units" not in out


def test_compose_question_includes_recall_block(session):
    q = session.compose_question("What units should I use?")
    assert "metric units" in q
    assert q.rstrip().endswith("New question: What units should I use?")


def test_auto_recall_can_be_disabled(session):
    q = session.compose_question("What units should I use?", auto_recall=False)
    assert "metric units" not in q


def test_empty_store_no_recall():
    s = Session(memory_store=InMemoryStore(embedder=HashEmbedder(dim=256)))
    assert s.recall_preamble("anything at all") == ""
