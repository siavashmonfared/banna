"""Auto-saved session store for resume.

Every conversation is mirrored to `~/.config/banna/sessions/<id>.jsonl`
as turns happen, so a session can be resumed after the CLI exits — no
manual `/save` needed. `<id>` is the session's `started_at` timestamp
compacted to a sortable token (e.g. `20260601T142233`).

This sits *next to* the explicit `/save` / `/load` commands (which write
to a user-chosen path); those are unchanged. This module only adds the
automatic, discoverable copy plus a listing for the resume picker.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config_store import config_dir
from .session import Session


def sessions_dir() -> Path:
    return config_dir() / "sessions"


def _id_from_started_at(started_at: str) -> str:
    """`2026-06-01T14:22:33+00:00` -> `20260601T142233`. Falls back to a
    sanitized form if the timestamp isn't the expected shape."""
    digits = re.sub(r"[^0-9T]", "", started_at.replace("+00:00", ""))
    return digits or re.sub(r"[^0-9A-Za-z]", "", started_at) or "session"


def session_path(session: Session) -> Path:
    return sessions_dir() / f"{_id_from_started_at(session.started_at)}.jsonl"


def autosave(session: Session) -> Path | None:
    """Write the session to its auto-save path. Best-effort: returns the
    path on success, None on failure (auto-save must never crash a turn)."""
    if not session.turns:
        return None
    try:
        d = sessions_dir()
        d.mkdir(parents=True, exist_ok=True)
        return session.save_jsonl(session_path(session))
    except OSError:
        return None


@dataclass
class SessionInfo:
    id: str
    path: Path
    started_at: str
    n_turns: int
    first_question: str
    last_question: str


def _peek(path: Path) -> SessionInfo | None:
    """Read a session file's header + first/last question cheaply."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    started_at = ""
    questions: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("__session__"):
            started_at = row.get("started_at", "")
            continue
        q = row.get("question")
        if q:
            questions.append(q)
    if not questions:
        return None
    return SessionInfo(
        id=path.stem,
        path=path,
        started_at=started_at or path.stem,
        n_turns=len(questions),
        first_question=questions[0],
        last_question=questions[-1],
    )


def list_sessions(limit: int = 20) -> list[SessionInfo]:
    """Most-recent-first list of resumable sessions."""
    d = sessions_dir()
    if not d.is_dir():
        return []
    infos: list[SessionInfo] = []
    for p in d.glob("*.jsonl"):
        info = _peek(p)
        if info is not None:
            infos.append(info)
    infos.sort(key=lambda i: i.id, reverse=True)
    return infos[:limit]


def latest_session() -> SessionInfo | None:
    items = list_sessions(limit=1)
    return items[0] if items else None


def load_session(session_id: str) -> Session:
    """Load a session by id (filename stem) or by full path."""
    p = Path(session_id).expanduser()
    if not p.is_file():
        p = sessions_dir() / f"{session_id}.jsonl"
    if not p.is_file():
        raise FileNotFoundError(f"no such session: {session_id}")
    return Session.load_jsonl(p)
