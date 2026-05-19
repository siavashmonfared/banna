"""Unit tests for the grep_text tool."""
from __future__ import annotations

from pathlib import Path

import pytest

from banna_agent.tools.grep import grep_text, make_grep_tool


def test_grep_basic_match(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello world\ngoodbye world\n")
    (tmp_path / "b.txt").write_text("nothing here\n")
    out = grep_text(tmp_path, pattern="world")
    paths = sorted({m["path"] for m in out["matches"]})
    assert paths == ["a.txt"]
    assert out["count"] == 2
    assert out["matches"][0]["line"] == 1


def test_grep_case_insensitive_flag(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("Hello\nHELLO\nhello\n")
    out = grep_text(tmp_path, pattern="hello", flags="i")
    assert out["count"] == 3


def test_grep_regex(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("foo1\nbar2\nfoo3\n")
    out = grep_text(tmp_path, pattern=r"foo\d")
    matches = [m["match"] for m in out["matches"]]
    assert matches == ["foo1", "foo3"]


def test_grep_invalid_regex_raises(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("anything")
    with pytest.raises(ValueError, match="invalid regex"):
        grep_text(tmp_path, pattern="(")


def test_grep_rejects_unknown_flag(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("anything")
    with pytest.raises(ValueError, match="unknown regex flag"):
        grep_text(tmp_path, pattern="x", flags="z")


def test_grep_skips_binary_files(tmp_path: Path) -> None:
    (tmp_path / "t.txt").write_text("target line")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02target\x00\x00\x00")
    out = grep_text(tmp_path, pattern="target")
    paths = {m["path"] for m in out["matches"]}
    assert "t.txt" in paths
    assert "b.bin" not in paths


def test_grep_glob_filter(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("target")
    (tmp_path / "a.md").write_text("target")
    out = grep_text(tmp_path, pattern="target", glob="*.py")
    paths = {m["path"] for m in out["matches"]}
    assert paths == {"a.py"}


def test_grep_max_matches(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("\n".join(["match"] * 50))
    out = grep_text(tmp_path, pattern="match", max_matches=10)
    assert len(out["matches"]) == 10
    assert out["truncated"] is True


def test_grep_tool_handler() -> None:
    tool = make_grep_tool()
    with pytest.raises(ValueError, match="'root'"):
        tool.handler({"root": "", "pattern": "x"})
    with pytest.raises(ValueError, match="'pattern'"):
        tool.handler({"root": "/tmp", "pattern": ""})


def test_tool_metadata() -> None:
    tool = make_grep_tool()
    assert tool.name == "grep_text"
    assert tool.capabilities == frozenset({"read", "filesystem"})
