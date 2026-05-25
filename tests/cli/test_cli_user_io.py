"""Tests for `_CliUserIO` — the REPL's interactive UserIO.

Regression focus: the permission prompt and ask() must pause the
thinking-spinner (a Rich live renderer) before reading stdin. Without
the pause the spinner repaints over the prompt and `input()` reads
nothing, so the box appears frozen / "won't let me pick an option".
"""
from __future__ import annotations

import io
from unittest import mock

from rich.console import Console

from banna_agent.cli.app import _CliUserIO


class _FakeStatus:
    def __init__(self) -> None:
        self.events: list[str] = []

    def stop(self) -> None:
        self.events.append("stop")

    def start(self) -> None:
        self.events.append("start")


def _io() -> tuple[_CliUserIO, _FakeStatus]:
    cli = _CliUserIO(console=Console(file=io.StringIO()))
    status = _FakeStatus()
    cli.bind_status(status)
    return cli, status


def test_confirm_pauses_spinner_around_input() -> None:
    cli, status = _io()
    with mock.patch("builtins.input", return_value="1"):
        decision = cli.confirm(tool_name="run_shell", args={"command": ["x"]}, risk="exec")
    assert decision == "allow_once"
    # Spinner must be stopped before the read and restarted after.
    assert status.events == ["stop", "start"]


def test_confirm_resumes_spinner_even_on_eof() -> None:
    cli, status = _io()
    with mock.patch("builtins.input", side_effect=EOFError):
        decision = cli.confirm(tool_name="run_shell", args={}, risk="exec")
    assert decision == "deny"
    assert status.events == ["stop", "start"]


def test_confirm_decision_mapping() -> None:
    for raw, expected in [("1", "allow_once"), ("2", "allow_always"),
                          ("3", "deny"), ("", "deny")]:
        cli, _ = _io()
        with mock.patch("builtins.input", return_value=raw):
            assert cli.confirm(tool_name="run_shell", args={}, risk="exec") == expected


def test_ask_pauses_and_resumes_spinner() -> None:
    cli, status = _io()
    with mock.patch("builtins.input", return_value="  /tmp/data.csv  "):
        reply = cli.ask("Which path?")
    assert reply == "/tmp/data.csv"
    assert status.events == ["stop", "start"]


def test_user_io_without_status_still_works() -> None:
    """No spinner bound (e.g. non-REPL surface): pause/resume are no-ops."""
    cli = _CliUserIO(console=Console(file=io.StringIO()))
    with mock.patch("builtins.input", return_value="2"):
        assert cli.confirm(tool_name="run_shell", args={}, risk="exec") == "allow_always"
