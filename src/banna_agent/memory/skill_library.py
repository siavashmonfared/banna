"""Voyager-style skill library — code-valued memory.

A `Skill` is a named, callable Python snippet that solved a sub-problem
in a past (verifier-confirmed) task. It lives as a `MemoryEntry` with
`kind="skill"` and a structured metadata block.

Usage at inference:
    1. At task start, `SkillLibrary.search(description, k=3)` returns
       the most relevant skills.
    2. `as_python_header(skills)` renders them as importable function
       definitions — inject into the system prompt OR prepend to the
       python_sandbox tool description.
    3. Harvested from successful runs at task end (see `skill_harvester`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import Memory, MemoryEntry, MemoryQuery


@dataclass
class Skill:
    """A callable Python artifact worth remembering across tasks."""

    name: str                               # e.g. "compute_cagr"
    signature: str                          # e.g. "cagr(end_value, start_value, years) -> float"
    description: str                        # one-line purpose
    code: str                               # full function body, importable
    examples: list[dict[str, Any]] = field(default_factory=list)
    # Provenance — populated by the harvester.
    source_task_id: str | None = None
    verifier_name: str | None = None
    created_at: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signature": self.signature,
            "code": self.code,
            "examples": list(self.examples),
            "source_task_id": self.source_task_id,
            "verifier_name": self.verifier_name,
        }

    @classmethod
    def from_entry(cls, entry: MemoryEntry) -> "Skill":
        md = entry.metadata
        return cls(
            name=md.get("name", ""),
            signature=md.get("signature", ""),
            description=entry.content,
            code=md.get("code", ""),
            examples=list(md.get("examples") or []),
            source_task_id=md.get("source_task_id") or entry.source_task_id,
            verifier_name=md.get("verifier_name"),
            created_at=entry.created_at,
        )


class SkillLibrary:
    """Wrapper over a Memory that enforces the `kind=skill` shape."""

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def register(self, skill: Skill) -> str:
        if not skill.name or not skill.code:
            raise ValueError("Skill must have a non-empty name and code")
        entry = MemoryEntry(
            content=skill.description or skill.signature,
            kind="skill",
            metadata=skill.to_metadata(),
            source_task_id=skill.source_task_id,
            verified_by=[skill.verifier_name] if skill.verifier_name else [],
            tags=["skill"],
        )
        return self.memory.write(entry)

    def search(self, description: str, k: int = 3) -> list[Skill]:
        hits = self.memory.search(MemoryQuery(query=description, k=k, kind_filter="skill"))
        return [Skill.from_entry(e) for e, _score in hits]

    def load(self, name: str) -> Skill | None:
        for entry in self.memory.all(kind="skill"):
            if entry.metadata.get("name") == name:
                return Skill.from_entry(entry)
        return None

    def all(self) -> list[Skill]:
        return [Skill.from_entry(e) for e in self.memory.all(kind="skill")]

    def as_python_header(self, skills: list[Skill]) -> str:
        """Render skills as a Python source blob the model can reference.

        Comments the provenance; the actual callable bodies are the
        `code` field. The string is meant to be concatenated at the top
        of a python_sandbox script by the agent.
        """
        if not skills:
            return ""
        parts: list[str] = [
            "# --- Pre-verified skills from past tasks ---",
            "# These are callable Python functions that previously produced",
            "# verifier-confirmed answers. Reuse them when relevant.",
            "",
        ]
        for s in skills:
            parts.append(f"# {s.name}: {s.description}")
            parts.append(f"# signature: {s.signature}")
            if s.source_task_id:
                parts.append(f"# provenance: task={s.source_task_id} verifier={s.verifier_name}")
            parts.append(s.code.rstrip())
            parts.append("")
        return "\n".join(parts)

    def __len__(self) -> int:
        return len(self.memory.all(kind="skill"))
