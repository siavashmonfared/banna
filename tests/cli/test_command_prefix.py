"""Prefix dispatch + completion for /commands.

`/pol` (unique prefix) runs /policy; `/s` (ambiguous) lists candidates
instead of erroring; unknown prefixes still error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


from banna_agent.cli import commands as C


@dataclass
class _Console:
    lines: list[str] = field(default_factory=list)
    def print(self, *args: Any, **kw: Any) -> None:
        import re
        msg = " ".join(str(a) for a in args)
        self.lines.append(re.sub(r"\[/?[^\]]+\]", "", msg))


@dataclass
class _App:
    console: _Console = field(default_factory=_Console)


def test_unique_prefix_runs_command(monkeypatch):
    called = {}
    monkeypatch.setitem(C.COMMANDS, "policy",
                        lambda app, args: called.setdefault("policy", args) or False)
    app = _App()
    # `/polic` is a unique prefix of `/policy`.
    C.dispatch(app, "/polic foo")
    assert called.get("policy") == ["foo"]


def test_ambiguous_prefix_lists_candidates():
    app = _App()
    # Several commands start with 's' (save, sessions, show, skills, status…).
    C.dispatch(app, "/s")
    out = "\n".join(app.console.lines)
    assert "ambiguous" in out
    # lists at least two real candidates
    assert out.count("/s") >= 2


def test_exact_name_still_wins_over_prefix(monkeypatch):
    # An exact match must dispatch to that command, not prefix-scan.
    hit = {}
    monkeypatch.setitem(C.COMMANDS, "show",
                        lambda app, args: hit.setdefault("show", True) or False)
    app = _App()
    C.dispatch(app, "/show last")
    assert hit.get("show") is True


def test_unknown_prefix_errors():
    app = _App()
    C.dispatch(app, "/zzzznope")
    out = "\n".join(app.console.lines)
    assert "unknown command" in out


def test_command_names_listed():
    names = C.command_names()
    assert "policy" in names and "resume" in names
    assert names == sorted(names)


def test_install_completer_matches_prefix(monkeypatch):
    # Drive the completer the way readline would, with a stubbed buffer.
    import readline
    C.install_completer()
    monkeypatch.setattr(readline, "get_line_buffer", lambda: "/re")
    comp = readline.get_completer()
    assert comp is not None
    # Every yielded option must start with /re (resume, etc.)
    got = []
    i = 0
    while True:
        v = comp("/re", i)
        if v is None:
            break
        got.append(v); i += 1
    assert got, "completer returned nothing for /re"
    assert all(g.startswith("/re") for g in got)
    assert "/resume" in got
