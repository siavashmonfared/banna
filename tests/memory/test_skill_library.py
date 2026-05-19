"""Unit tests for SkillLibrary."""
from __future__ import annotations

import pytest

from banna_agent.memory.in_memory_store import InMemoryStore
from banna_agent.memory.skill_library import Skill, SkillLibrary


@pytest.fixture
def lib() -> SkillLibrary:
    return SkillLibrary(InMemoryStore())


def _skill(name: str = "compute_cagr") -> Skill:
    return Skill(
        name=name,
        signature=f"{name}(end, start, years) -> float",
        description="compound annual growth rate",
        code=(
            f"def {name}(end, start, years):\n"
            f"    return (end / start) ** (1 / years) - 1\n"
        ),
        source_task_id="task1",
        verifier_name="arithmetic",
    )


def test_register_then_load_by_name(lib: SkillLibrary) -> None:
    lib.register(_skill())
    s = lib.load("compute_cagr")
    assert s is not None
    assert s.signature.startswith("compute_cagr")
    assert "** (1 / years)" in s.code


def test_register_rejects_empty_name() -> None:
    lib = SkillLibrary(InMemoryStore())
    bad = Skill(name="", signature="x", description="x", code="def f(): return 1")
    with pytest.raises(ValueError):
        lib.register(bad)


def test_register_rejects_empty_code() -> None:
    lib = SkillLibrary(InMemoryStore())
    bad = Skill(name="x", signature="x", description="x", code="")
    with pytest.raises(ValueError):
        lib.register(bad)


def test_search_returns_skills_only(lib: SkillLibrary) -> None:
    # Register one skill (description = "compound annual growth rate")
    # and one unrelated memory entry with overlapping words.
    lib.register(_skill("skill_alpha"))
    from banna_agent.memory.base import MemoryEntry
    lib.memory.write(MemoryEntry(content="compound annual growth is unrelated",
                                 kind="fact"))
    hits = lib.search("compound annual growth", k=5)
    # Only the skill is returned; the fact is filtered by kind.
    assert len(hits) == 1
    assert hits[0].name == "skill_alpha"


def test_all_returns_every_skill(lib: SkillLibrary) -> None:
    lib.register(_skill("a"))
    lib.register(_skill("b"))
    all_skills = lib.all()
    names = {s.name for s in all_skills}
    assert names == {"a", "b"}
    assert len(lib) == 2


def test_as_python_header_renders_code_and_provenance(lib: SkillLibrary) -> None:
    lib.register(_skill("compute_cagr"))
    skills = lib.all()
    header = lib.as_python_header(skills)
    assert "compute_cagr" in header
    assert "Pre-verified skills" in header
    assert "provenance" in header
    assert "def compute_cagr" in header


def test_as_python_header_empty_list_returns_empty_string(lib: SkillLibrary) -> None:
    assert lib.as_python_header([]) == ""


def test_load_missing_returns_none(lib: SkillLibrary) -> None:
    assert lib.load("nonexistent") is None
