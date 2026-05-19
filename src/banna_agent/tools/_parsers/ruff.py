"""Ruff output parser.

Preferred invocation: `ruff check --output-format=json`. If JSON parses,
we use it. If not (older ruff or `--output-format=text`), fall back to
the text format `path:line:col: CODE message`.
"""
from __future__ import annotations

import json
import re

from . import Failure


_TEXT_LINE = re.compile(
    r"^(?P<path>[^:\n]+):(?P<line>\d+):(?P<col>\d+):\s+(?P<code>[A-Z]+\d+)\s+(?P<msg>.+)$",
    re.MULTILINE,
)


def parse(stdout: str, stderr: str, returncode: int) -> list[Failure]:
    if returncode == 0:
        return []
    stdout = stdout or ""
    fails: list[Failure] = []
    # JSON form first.
    stripped = stdout.strip()
    if stripped.startswith("["):
        try:
            items = json.loads(stripped)
            for it in items:
                path = it.get("filename") or it.get("file") or "?"
                loc = it.get("location") or {}
                line = loc.get("row") or loc.get("line") or 0
                col = loc.get("column") or loc.get("col") or 0
                code = it.get("code") or "E"
                msg = it.get("message") or "lint error"
                fails.append(Failure(
                    kind="lint_error",
                    name=f"{path}:{line}:{code}",
                    detail=f"{code}: {msg}"[:240],
                    location=f"{path}:{line}:{col}",
                ))
            return fails
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    text = stdout + "\n" + (stderr or "")
    for m in _TEXT_LINE.finditer(text):
        path, line, col = m.group("path"), m.group("line"), m.group("col")
        code, msg = m.group("code"), m.group("msg").strip()
        fails.append(Failure(
            kind="lint_error",
            name=f"{path}:{line}:{code}",
            detail=f"{code}: {msg}"[:240],
            location=f"{path}:{line}:{col}",
        ))

    if not fails and returncode != 0:
        fails.append(Failure(
            kind="lint_error",
            name="ruff:nonzero_exit",
            detail=f"ruff exited {returncode}; no parseable lint lines",
        ))
    return fails
