"""Mypy output parser.

Mypy's default format is `path:line: severity: message  [error-code]`.
We treat severity in {error} as a failure; `note` lines are folded into
the preceding error's detail when adjacent (best-effort).
"""
from __future__ import annotations

import re

from . import Failure


_LINE = re.compile(
    r"^(?P<path>[^:\n]+):(?P<line>\d+)(?::(?P<col>\d+))?:\s+"
    r"(?P<sev>error|note|warning):\s+(?P<msg>.+?)(?:\s+\[(?P<code>[a-z0-9-]+)\])?$",
    re.MULTILINE,
)


def parse(stdout: str, stderr: str, returncode: int) -> list[Failure]:
    text = (stdout or "") + "\n" + (stderr or "")
    if returncode == 0:
        return []
    fails: list[Failure] = []
    last_error_idx: int | None = None
    for m in _LINE.finditer(text):
        sev = m.group("sev")
        path = m.group("path")
        line = m.group("line")
        col = m.group("col")
        msg = m.group("msg").strip()
        code = m.group("code") or "error"
        loc = f"{path}:{line}" + (f":{col}" if col else "")
        if sev == "error":
            name = f"{path}:{line}:{code}"
            fails.append(Failure(
                kind="type_error",
                name=name,
                detail=msg[:240],
                location=loc,
            ))
            last_error_idx = len(fails) - 1
        elif sev == "note" and last_error_idx is not None:
            f = fails[last_error_idx]
            extra = f" | note: {msg}"
            fails[last_error_idx] = Failure(
                kind=f.kind,
                name=f.name,
                detail=(f.detail + extra)[:240],
                location=f.location,
            )

    if not fails and returncode != 0:
        fails.append(Failure(
            kind="type_error",
            name="mypy:nonzero_exit",
            detail=f"mypy exited {returncode}; no parseable error lines",
        ))
    return fails
