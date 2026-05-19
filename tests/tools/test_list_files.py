"""Unit tests for the list_files tool."""
from __future__ import annotations

from pathlib import Path

import pytest

from banna_agent.tools.list_files import list_files, make_list_files_tool


def _seed(tmp: Path) -> None:
    (tmp / "a.txt").write_text("a")
    (tmp / "b.txt").write_text("b")
    sub = tmp / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")
    hidden = tmp / ".hidden"
    hidden.mkdir()
    (hidden / "secret").write_text("x")


def test_list_files_recursive(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = list_files(tmp_path)
    # Should include a.txt, b.txt, sub, sub/c.txt — but NOT .hidden or .hidden/secret
    paths = [e["path"] for e in out["entries"]]
    assert "a.txt" in paths
    assert "b.txt" in paths
    assert "sub" in paths
    assert "sub/c.txt" in paths
    assert not any(p.startswith(".hidden") for p in paths)


def test_list_files_files_only(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = list_files(tmp_path, files_only=True)
    kinds = {e["kind"] for e in out["entries"]}
    assert kinds == {"file"}


def test_list_files_include_hidden(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = list_files(tmp_path, include_hidden=True)
    paths = [e["path"] for e in out["entries"]]
    assert any(p.startswith(".hidden") for p in paths)


def test_list_files_pattern(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = list_files(tmp_path, pattern="*.txt")
    paths = [e["path"] for e in out["entries"]]
    assert set(paths) == {"a.txt", "b.txt"}


def test_list_files_limit(tmp_path: Path) -> None:
    _seed(tmp_path)
    out = list_files(tmp_path, limit=2)
    assert len(out["entries"]) == 2
    assert out["truncated"] is True


def test_list_files_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_files(tmp_path / "missing")


def test_list_files_rejects_file_root(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        list_files(f)


def test_tool_handler_requires_root() -> None:
    tool = make_list_files_tool()
    with pytest.raises(ValueError, match="non-empty"):
        tool.handler({"root": ""})


def test_tool_metadata() -> None:
    tool = make_list_files_tool()
    assert tool.name == "list_files"
    assert tool.capabilities == frozenset({"read", "filesystem"})
