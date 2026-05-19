"""Unit tests for the GAIA loader (offline via JSONL)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from banna_agent.benchmarks.gaia.loader import (
    GAIATask,
    load_gaia_from_jsonl,
)


def test_gaiatask_has_attachment_flag() -> None:
    t1 = GAIATask(task_id="a", question="q", level=1, answer="x")
    t2 = GAIATask(task_id="b", question="q", level=2, answer="y", file_name="data.xlsx")
    assert t1.has_attachment is False
    assert t2.has_attachment is True


def test_load_gaia_from_jsonl_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "mini.jsonl"
    rows = [
        {"task_id": "t1", "question": "What is 2+2?", "level": 1, "answer": "4"},
        {
            "task_id": "t2",
            "question": "How many rows in spreadsheet?",
            "level": 2,
            "answer": "17",
            "file_name": "data.xlsx",
            "file_path": "/tmp/data.xlsx",
            "metadata": {"steps": 3},
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    tasks = load_gaia_from_jsonl(p)
    assert len(tasks) == 2
    assert tasks[0].task_id == "t1"
    assert tasks[0].level == 1
    assert tasks[0].answer == "4"
    assert tasks[1].has_attachment
    assert tasks[1].metadata == {"steps": 3}


def test_load_gaia_from_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "with_blanks.jsonl"
    p.write_text(
        '\n'
        + json.dumps({"task_id": "t1", "question": "q", "level": 1, "answer": "a"})
        + '\n\n'
        + json.dumps({"task_id": "t2", "question": "q", "level": 2, "answer": "b"})
        + '\n'
    )
    tasks = load_gaia_from_jsonl(p)
    assert [t.task_id for t in tasks] == ["t1", "t2"]


def test_load_gaia_from_missing_jsonl_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gaia_from_jsonl(tmp_path / "nope.jsonl")
