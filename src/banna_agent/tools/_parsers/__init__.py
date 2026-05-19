"""Output parsers for command-runner tools (pytest, mypy, ruff).

Each parser is a pure function `(stdout, stderr, returncode) -> list[Failure]`.
A `Failure` is the minimum the agent needs to act on: kind, a stable name
(so re-runs can show "this one is now fixed"), one-line detail, and an
optional source location.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Failure:
    """One actionable failure extracted from a tool's output."""

    kind: str         # "test_failure" | "type_error" | "lint_error" | "build_error" | "tool_error"
    name: str         # stable id, e.g. "tests/test_x.py::test_y" or "src/a.py:12:F401"
    detail: str       # short human-readable message
    location: str = ""  # "path:line[:col]" when known

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "name": self.name, "detail": self.detail, "location": self.location}
