"""Unit tests for the file reader tool."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from banna_agent.tools.file_reader import make_file_reader_tool, read_file


def test_read_text_file(tmp_path: Path) -> None:
    p = tmp_path / "sample.txt"
    p.write_text("hello world\nline two")
    out = read_file(p)
    assert out["kind"] == "text"
    assert out["ext"] == ".txt"
    assert "hello world" in out["text"]
    assert out["truncated"] is False


def test_read_text_file_truncates(tmp_path: Path) -> None:
    p = tmp_path / "big.txt"
    p.write_text("x" * 1000)
    out = read_file(p, max_chars=50)
    assert out["truncated"] is True
    assert len(out["text"]) == 50


def test_read_json_file_as_text(tmp_path: Path) -> None:
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"a": 1, "b": [1, 2, 3]}))
    out = read_file(p)
    assert out["kind"] == "text"
    assert "[1, 2, 3]" in out["text"]


def test_read_csv(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    p = tmp_path / "t.csv"
    p.write_text("a,b,c\n1,2,3\n4,5,6\n")
    out = read_file(p)
    assert out["kind"] == "table"
    assert out["columns"] == ["a", "b", "c"]
    assert out["n_rows_shown"] == 2
    assert "preview_markdown" in out


def test_read_tsv(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    p = tmp_path / "t.tsv"
    p.write_text("a\tb\n1\t2\n3\t4\n")
    out = read_file(p)
    assert out["columns"] == ["a", "b"]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_file(tmp_path / "nope.txt")


def test_handler_rejects_empty_path() -> None:
    tool = make_file_reader_tool()
    with pytest.raises(ValueError, match="non-empty"):
        tool.handler({"path": ""})


def test_read_unknown_extension_falls_back_to_text(tmp_path: Path) -> None:
    p = tmp_path / "x.unknown"
    p.write_text("this is text")
    out = read_file(p)
    assert out["kind"] in ("text", "binary")


def test_tool_metadata() -> None:
    tool = make_file_reader_tool()
    assert tool.name == "read_file"
    assert tool.capabilities == frozenset({"read", "filesystem"})
