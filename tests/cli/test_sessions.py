"""Auto-save + resume: sessions persist to ~/.config/banna/sessions and
can be listed and reloaded."""
from __future__ import annotations

import importlib

import pytest

from banna_agent.cli.session import Session, Turn


@pytest.fixture()
def sessions_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from banna_agent.cli import config_store, sessions
    importlib.reload(config_store)
    importlib.reload(sessions)
    return sessions


def _session_with(started_at: str, questions: list[str]) -> Session:
    s = Session()
    s.started_at = started_at
    for q in questions:
        s.turns.append(Turn(question=q, answer=f"ans:{q}"))
    return s


def test_autosave_writes_and_lists(sessions_mod):
    s = _session_with("2026-06-01T10:00:00+00:00", ["first q", "second q"])
    path = sessions_mod.autosave(s)
    assert path is not None and path.is_file()

    infos = sessions_mod.list_sessions()
    assert len(infos) == 1
    info = infos[0]
    assert info.id == "20260601T100000"
    assert info.n_turns == 2
    assert info.first_question == "first q"
    assert info.last_question == "second q"


def test_autosave_empty_session_is_noop(sessions_mod):
    s = Session()
    assert sessions_mod.autosave(s) is None
    assert sessions_mod.list_sessions() == []


def test_load_by_id_roundtrip(sessions_mod):
    s = _session_with("2026-06-01T11:30:00+00:00", ["alpha", "beta"])
    sessions_mod.autosave(s)
    loaded = sessions_mod.load_session("20260601T113000")
    assert len(loaded.turns) == 2
    assert loaded.turns[0].question == "alpha"
    assert loaded.turns[1].answer == "ans:beta"


def test_list_is_recent_first(sessions_mod):
    sessions_mod.autosave(_session_with("2026-06-01T09:00:00+00:00", ["old"]))
    sessions_mod.autosave(_session_with("2026-06-02T09:00:00+00:00", ["new"]))
    infos = sessions_mod.list_sessions()
    assert [i.first_question for i in infos] == ["new", "old"]
    assert sessions_mod.latest_session().first_question == "new"


def test_load_missing_raises(sessions_mod):
    with pytest.raises(FileNotFoundError):
        sessions_mod.load_session("nonexistent-id")


def test_autosave_updates_same_file(sessions_mod):
    s = _session_with("2026-06-01T12:00:00+00:00", ["one"])
    p1 = sessions_mod.autosave(s)
    s.turns.append(Turn(question="two", answer="ans:two"))
    p2 = sessions_mod.autosave(s)
    assert p1 == p2  # same id -> same file, updated in place
    assert len(sessions_mod.list_sessions()) == 1
    assert sessions_mod.load_session(s.started_at[:0] or "20260601T120000").turns[-1].question == "two"
