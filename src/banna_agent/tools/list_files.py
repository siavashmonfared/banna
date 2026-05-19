"""Glob-based directory listing.

Pairs with `read_file` so the agent can navigate workspaces (GAIA
attachments, repo contents) without a shell.

Returns entries sorted by path for deterministic output. Hidden files
(leading dot) are excluded by default — set include_hidden=true to see
them. Symlinks are followed; we don't resolve symlink cycles.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import JsonTool


DEFAULT_LIMIT = 500


def list_files(
    root: str | Path,
    *,
    pattern: str = "**/*",
    limit: int = DEFAULT_LIMIT,
    include_hidden: bool = False,
    files_only: bool = False,
) -> dict[str, Any]:
    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"no such path: {base}")
    if not base.is_dir():
        raise ValueError(f"not a directory: {base}")

    entries: list[dict[str, Any]] = []
    for p in sorted(base.glob(pattern)):
        rel = p.relative_to(base).as_posix()
        if not include_hidden and any(part.startswith(".") for part in rel.split("/")):
            continue
        kind = "dir" if p.is_dir() else "file"
        if files_only and kind == "dir":
            continue
        try:
            size = p.stat().st_size if p.is_file() else 0
        except OSError:
            size = 0
        entries.append({"path": rel, "abs": str(p), "kind": kind, "size_bytes": size})
        if len(entries) >= limit:
            break

    return {
        "root": str(base),
        "pattern": pattern,
        "count": len(entries),
        "truncated": len(entries) >= limit,
        "entries": entries,
    }


def _handler(args: dict[str, Any]) -> dict[str, Any]:
    root = args.get("root", "")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("'root' must be a non-empty string")
    return list_files(
        root,
        pattern=str(args.get("pattern", "**/*")),
        limit=int(args.get("limit", DEFAULT_LIMIT)),
        include_hidden=bool(args.get("include_hidden", False)),
        files_only=bool(args.get("files_only", False)),
    )


LIST_FILES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "root": {"type": "string", "description": "Directory to search under."},
        "pattern": {
            "type": "string",
            "description": "Glob pattern relative to root (default '**/*' = recursive).",
            "default": "**/*",
        },
        "limit": {
            "type": "integer",
            "description": f"Max entries to return (default {DEFAULT_LIMIT}).",
            "default": DEFAULT_LIMIT,
        },
        "include_hidden": {
            "type": "boolean",
            "description": "Include dotfiles and dot-directories (default false).",
            "default": False,
        },
        "files_only": {
            "type": "boolean",
            "description": "Exclude directories, return files only (default false).",
            "default": False,
        },
    },
    "required": ["root"],
    "additionalProperties": False,
}


def make_list_files_tool() -> JsonTool:
    return JsonTool(
        name="list_files",
        description=(
            "List files and directories under a root, matching an optional glob "
            "pattern. Use '**/*' for a full recursive listing."
        ),
        input_schema=LIST_FILES_SCHEMA,
        handler=_handler,
        capabilities=frozenset({"read", "filesystem"}),
    )
