"""is_command: slash commands vs absolute paths pasted as task text."""
from __future__ import annotations

from banna_agent.cli.commands import COMMANDS, is_command


def test_known_commands_are_commands() -> None:
    for name in COMMANDS:
        assert is_command(f"/{name}"), name
        assert is_command(f"/{name} some args"), name


def test_command_typos_still_dispatch() -> None:
    # A misspelled command should hit the dispatcher (and get the
    # unknown-command hint), not be silently sent to the LLM.
    assert is_command("/modle")
    assert is_command("/hlep me")


def test_absolute_paths_route_to_agent() -> None:
    assert not is_command("/Users/me/resume.tex - take a look at my resume")
    assert not is_command("/home/me/notes.md summarize this")
    assert not is_command("/tmp/data.csv")
    assert not is_command("/Users/me/Documents/file with spaces.pdf review")


def test_non_slash_lines_route_to_agent() -> None:
    assert not is_command("what is 2+2?")
    assert not is_command("read /etc/hostname and tell me the host")
    assert not is_command("")
    assert not is_command("   ")


def test_lone_slash_routes_to_agent() -> None:
    assert not is_command("/")
    assert not is_command("/ Users/me/file.txt")
