"""Write a text file to disk — the write-side sibling of `read_file`.

Gives the agent a way to produce deliverables the user asked to save
(a rewritten .tex resume, a report, generated code) without shelling
out. `run_shell` is allowlisted and `python_sandbox` may be running in
a network-less container with no host filesystem, so before this tool
existed there was no reliable write path at all.

Guardrails:
  - `~` expands; relative paths resolve against the CWD.
  - Parent directories are created.
  - System prefixes (/etc, /usr, …) are refused outright.
  - The tool is permission-gated under `react+` (see core/agent.py
    `_GATED_TOOLS`), so interactive runs confirm every write.

Handlers raise on failure; the driver converts exceptions into an
error `tool_result` (see base.invoke_tool).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import JsonTool


# Refused even when the permission gate would allow the call — nothing
# the agent writes belongs under these.
_FORBIDDEN_PREFIXES = (
    "/etc", "/usr", "/bin", "/sbin", "/lib", "/boot", "/sys", "/proc",
    "/dev", "/var", "/opt", "/srv", "/System", "/Library", "/Windows",
)

DEFAULT_MAX_BYTES = 5_000_000  # refuse absurd payloads (5 MB of text)


WRITE_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Destination file path. `~` expands; parent "
                           "directories are created automatically.",
        },
        "content": {
            "type": "string",
            "description": "Full text content to write.",
        },
        "mode": {
            "type": "string",
            "enum": ["write", "append"],
            "description": "'write' creates or overwrites (default); "
                           "'append' adds to the end of an existing file.",
        },
    },
    "required": ["path", "content"],
}


def _resolve(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.normpath(str(path)))


def _handler(args: dict[str, Any]) -> dict[str, Any]:
    raw = str(args.get("path") or "").strip()
    if not raw:
        raise ValueError("path is required")
    content = args.get("content")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    payload = content.encode("utf-8")
    if len(payload) > DEFAULT_MAX_BYTES:
        raise ValueError(
            f"content too large ({len(payload)} bytes > {DEFAULT_MAX_BYTES})")
    mode = str(args.get("mode") or "write")
    if mode not in ("write", "append"):
        raise ValueError(f"unknown mode {mode!r} (use 'write' or 'append')")

    path = _resolve(raw)
    for pre in _FORBIDDEN_PREFIXES:
        if str(path) == pre or str(path).startswith(pre + os.sep):
            raise ValueError(f"refusing to write under {pre} (system path)")

    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if mode == "append" else "w", encoding="utf-8") as f:
        f.write(content)
    return {
        "path": str(path),
        "bytes_written": len(payload),
        "mode": mode,
        "created": not existed,
    }


def make_file_writer_tool() -> JsonTool:
    return JsonTool(
        name="write_file",
        description=(
            "Write text content to a file on disk ('write' creates or "
            "overwrites; 'append' adds to the end). Expands ~ and creates "
            "parent directories. Use this to save deliverables the user "
            "asked for (rewritten documents, reports, code, .tex/.md files)."
        ),
        input_schema=WRITE_FILE_SCHEMA,
        handler=_handler,
        capabilities=frozenset({"write", "filesystem"}),
    )
