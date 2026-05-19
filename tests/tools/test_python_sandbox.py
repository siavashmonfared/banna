"""Unit tests for the Python sandbox tool."""
from __future__ import annotations

import pytest

from banna_agent.tools.python_sandbox import make_python_sandbox_tool, run_python


def test_run_python_prints_stdout() -> None:
    out = run_python("print(2 + 2)")
    assert out["ok"] is True
    assert out["returncode"] == 0
    assert "4" in out["stdout"]
    assert out["stderr"] == ""
    assert out["timeout"] is False


def test_run_python_captures_stderr_on_exception() -> None:
    out = run_python("raise ValueError('boom')")
    assert out["ok"] is False
    assert out["returncode"] != 0
    assert "ValueError" in out["stderr"]
    assert "boom" in out["stderr"]


def test_run_python_times_out() -> None:
    code = "import time\nwhile True: time.sleep(0.1)"
    out = run_python(code, timeout_s=0.5)
    assert out["ok"] is False
    assert out["timeout"] is True
    assert out["returncode"] == -1


def test_run_python_truncates_stdout() -> None:
    code = "print('x' * 50000)"
    out = run_python(code, max_output_chars=1000)
    assert out["truncated_stdout"] is True
    assert len(out["stdout"]) == 1000


def test_handler_rejects_empty_code() -> None:
    tool = make_python_sandbox_tool()
    with pytest.raises(ValueError, match="non-empty"):
        tool.handler({"code": ""})


def test_tool_metadata() -> None:
    tool = make_python_sandbox_tool()
    assert tool.name == "run_python"
    assert "sandbox" in tool.capabilities
    assert tool.input_schema["required"] == ["code"]


def test_handler_roundtrip() -> None:
    tool = make_python_sandbox_tool()
    out = tool.handler({"code": "import sys; print('hello', file=sys.stdout)"})
    assert out["ok"] is True
    assert "hello" in out["stdout"]
