"""Regex grep across files.

A lightweight reimplementation of `grep -n -r` scoped to what the agent
needs: a regex, a root directory, an optional glob filter, a byte cap
per file, and a match cap overall.

Binary files are skipped (we attempt a UTF-8 decode with strict errors
on the first 4KB to decide). Hidden files are excluded unless
include_hidden=true.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import JsonTool


DEFAULT_MAX_MATCHES = 200
DEFAULT_MAX_BYTES_PER_FILE = 5_000_000  # 5 MB
BINARY_SNIFF_BYTES = 4_096


def _looks_text(path: Path) -> bool:
    """Classic 'binary file' heuristic: NUL byte in the sniff range is
    treated as binary. UTF-8 permits NUL bytes technically, but real
    text files almost never contain them."""
    try:
        with path.open("rb") as f:
            head = f.read(BINARY_SNIFF_BYTES)
    except OSError:
        return False
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def grep_text(
    root: str | Path,
    pattern: str,
    *,
    glob: str = "**/*",
    flags: str = "",
    max_matches: int = DEFAULT_MAX_MATCHES,
    max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE,
    include_hidden: bool = False,
) -> dict[str, Any]:
    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"no such path: {base}")

    re_flags = 0
    for ch in flags:
        if ch == "i":
            re_flags |= re.IGNORECASE
        elif ch == "m":
            re_flags |= re.MULTILINE
        elif ch == "s":
            re_flags |= re.DOTALL
        else:
            raise ValueError(f"unknown regex flag {ch!r}; use i/m/s")

    try:
        rx = re.compile(pattern, re_flags)
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from exc

    matches: list[dict[str, Any]] = []
    files_scanned = 0
    if base.is_file():
        candidates = [base]
    else:
        candidates = [p for p in sorted(base.glob(glob)) if p.is_file()]

    for p in candidates:
        rel = p.relative_to(base).as_posix() if base.is_dir() else p.name
        if not include_hidden and any(part.startswith(".") for part in rel.split("/")):
            continue
        try:
            if p.stat().st_size > max_bytes_per_file:
                continue
        except OSError:
            continue
        if not _looks_text(p):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files_scanned += 1
        for lineno, line in enumerate(text.splitlines(), start=1):
            m = rx.search(line)
            if m is None:
                continue
            matches.append({
                "path": rel,
                "line": lineno,
                "text": line[:500],
                "match": m.group(0)[:500],
            })
            if len(matches) >= max_matches:
                break
        if len(matches) >= max_matches:
            break

    return {
        "root": str(base),
        "pattern": pattern,
        "glob": glob,
        "files_scanned": files_scanned,
        "count": len(matches),
        "truncated": len(matches) >= max_matches,
        "matches": matches,
    }


def _handler(args: dict[str, Any]) -> dict[str, Any]:
    root = args.get("root", "")
    pattern = args.get("pattern", "")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("'root' must be a non-empty string")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("'pattern' must be a non-empty regex string")
    return grep_text(
        root,
        pattern,
        glob=str(args.get("glob", "**/*")),
        flags=str(args.get("flags", "")),
        max_matches=int(args.get("max_matches", DEFAULT_MAX_MATCHES)),
        include_hidden=bool(args.get("include_hidden", False)),
    )


GREP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "root": {"type": "string", "description": "Directory (or file) to search under."},
        "pattern": {"type": "string", "description": "Python regex to match."},
        "glob": {
            "type": "string",
            "description": "Glob filter relative to root (default '**/*').",
            "default": "**/*",
        },
        "flags": {
            "type": "string",
            "description": "Regex flag characters: 'i' ignore-case, 'm' multiline, 's' dotall.",
            "default": "",
        },
        "max_matches": {
            "type": "integer",
            "description": f"Max matches to return (default {DEFAULT_MAX_MATCHES}).",
            "default": DEFAULT_MAX_MATCHES,
        },
        "include_hidden": {
            "type": "boolean",
            "description": "Search dotfiles (default false).",
            "default": False,
        },
    },
    "required": ["root", "pattern"],
    "additionalProperties": False,
}


def make_grep_tool() -> JsonTool:
    return JsonTool(
        name="grep_text",
        description=(
            "Regex-search files under a directory. Returns matching lines "
            "with {path, line, text, match}. Binary files are skipped."
        ),
        input_schema=GREP_SCHEMA,
        handler=_handler,
        capabilities=frozenset({"read", "filesystem"}),
    )
