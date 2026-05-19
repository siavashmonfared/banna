"""Unit tests for the run_shell tool."""
from __future__ import annotations

import sys

import pytest

from banna_agent.tools.run_shell import make_run_shell_tool, run_shell


def test_run_shell_basic_ok() -> None:
    out = run_shell(["echo", "hello"], shell=False)
    assert out["ok"] is True
    assert out["returncode"] == 0
    assert "hello" in out["stdout"]
    assert out["timeout"] is False


def test_run_shell_with_shell_true() -> None:
    out = run_shell("echo hi && echo bye", shell=True)
    assert out["ok"] is True
    assert "hi" in out["stdout"]
    assert "bye" in out["stdout"]


def test_run_shell_captures_stderr() -> None:
    code = "import sys; sys.stderr.write('broken\\n'); sys.exit(2)"
    out = run_shell([sys.executable, "-c", code], shell=False)
    assert out["ok"] is False
    assert out["returncode"] == 2
    assert "broken" in out["stderr"]


def test_run_shell_times_out() -> None:
    out = run_shell([sys.executable, "-c", "import time; time.sleep(5)"], shell=False, timeout_s=0.3)
    assert out["ok"] is False
    assert out["timeout"] is True
    assert out["returncode"] == -1


def test_run_shell_missing_binary() -> None:
    out = run_shell(["/nonexistent/bin/thing"], shell=False)
    assert out["ok"] is False
    assert "FileNotFoundError" in out["stderr"] or out["returncode"] != 0


def test_run_shell_truncates_output() -> None:
    code = "print('x' * 50000)"
    out = run_shell([sys.executable, "-c", code], shell=False, max_output_chars=100)
    assert out["truncated_stdout"] is True
    assert len(out["stdout"]) == 100


def test_handler_rejects_empty_command() -> None:
    tool = make_run_shell_tool()
    with pytest.raises(ValueError, match="non-empty"):
        tool.handler({"command": ""})


def test_tool_metadata() -> None:
    tool = make_run_shell_tool()
    assert tool.name == "run_shell"
    assert "shell" in tool.capabilities
