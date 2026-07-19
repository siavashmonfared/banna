"""write_file tool tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from banna_agent.tools.file_writer import make_file_writer_tool


@pytest.fixture()
def tool():
    return make_file_writer_tool()


def test_write_creates_file_and_parents(tool, tmp_path: Path) -> None:
    dest = tmp_path / "sub" / "dir" / "out.tex"
    r = tool.handler({"path": str(dest), "content": "\\documentclass{article}"})
    assert dest.read_text() == "\\documentclass{article}"
    assert r["path"] == str(dest)
    assert r["bytes_written"] == len("\\documentclass{article}")
    assert r["created"] is True
    assert r["mode"] == "write"


def test_write_overwrites(tool, tmp_path: Path) -> None:
    dest = tmp_path / "f.txt"
    dest.write_text("old")
    r = tool.handler({"path": str(dest), "content": "new"})
    assert dest.read_text() == "new"
    assert r["created"] is False


def test_append_mode(tool, tmp_path: Path) -> None:
    dest = tmp_path / "f.txt"
    tool.handler({"path": str(dest), "content": "a"})
    r = tool.handler({"path": str(dest), "content": "b", "mode": "append"})
    assert dest.read_text() == "ab"
    assert r["mode"] == "append"


def test_tilde_expansion(tool, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    r = tool.handler({"path": "~/note.md", "content": "hi"})
    assert (tmp_path / "note.md").read_text() == "hi"
    assert r["path"] == str(tmp_path / "note.md")


def test_refuses_system_paths(tool) -> None:
    with pytest.raises(ValueError, match="system path"):
        tool.handler({"path": "/etc/evil.conf", "content": "x"})
    with pytest.raises(ValueError, match="system path"):
        # normpath collapses the traversal back under /etc
        tool.handler({"path": "/tmp/../etc/evil.conf", "content": "x"})


def test_rejects_bad_args(tool) -> None:
    with pytest.raises(ValueError, match="path is required"):
        tool.handler({"content": "x"})
    with pytest.raises(ValueError, match="content must be a string"):
        tool.handler({"path": "/tmp/x", "content": None})
    with pytest.raises(ValueError, match="unknown mode"):
        tool.handler({"path": "/tmp/x", "content": "x", "mode": "rewrite"})


def test_gated_and_capabilities(tool) -> None:
    from banna_agent.core.agent import _GATED_TOOLS
    assert _GATED_TOOLS.get("write_file") == "write"
    assert "write" in tool.capabilities
