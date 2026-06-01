"""Trace HTML renderer tests: well-formedness, escaping, and that each
event kind (think / tool / parallel batch / ask_user / final) surfaces."""
from __future__ import annotations

import json
from pathlib import Path

from banna_agent.trace import render_file, render_html


def _events() -> list[dict]:
    rid = "run-1"
    return [
        {"run_id": rid, "step": -1, "kind": "run_start",
         "payload": {"policy": "react+", "question": "What is 2+2 & <b>more</b>?"}},
        {"run_id": rid, "step": 0, "kind": "propose",
         "payload": {"kind_of_action": "think", "action_text": "Let me reason about this."}},
        {"run_id": rid, "step": 1, "kind": "propose",
         "payload": {"kind_of_action": "tool_call", "tool_name": "search"}},
        {"run_id": rid, "step": 1, "kind": "tool_call",
         "payload": {"tool_name": "search", "arguments": {"query": "2+2"}}},
        {"run_id": rid, "step": 1, "kind": "tool_result",
         "payload": {"ok": True, "preview": "found: 4", "wall_s": 0.5}},
        {"run_id": rid, "step": 1, "kind": "observation",
         "payload": {"cumulative_tokens_in": 100, "cumulative_tokens_out": 20,
                     "cumulative_wall_s": 1.2, "evidence_count": 1}},
        {"run_id": rid, "step": 2, "kind": "propose",
         "payload": {"kind_of_action": "tool_batch"}},
        {"run_id": rid, "step": 2, "kind": "tool_batch",
         "payload": {"tool_names": ["search", "read_url"], "n": 2}},
        {"run_id": rid, "step": 3, "kind": "ask_user",
         "payload": {"question": "Which source do you prefer?"}},
        {"run_id": rid, "step": 4, "kind": "run_end",
         "payload": {"final_answer": "4", "budget_reason": "ok", "steps_used": 4}},
    ]


def test_renders_full_document():
    doc = render_html(_events())
    assert doc.startswith("<!doctype html>")
    assert doc.rstrip().endswith("</html>")
    assert "react+" in doc


def test_each_kind_surfaces():
    doc = render_html(_events())
    assert "THINK" in doc
    assert "Let me reason about this." in doc
    assert "search" in doc
    assert "found: 4" in doc
    assert "PARALLEL BATCH" in doc
    assert "ASK_USER" in doc
    assert "Which source do you prefer?" in doc
    assert "final answer" in doc
    assert ">4<" in doc or "4" in doc


def test_html_is_escaped():
    doc = render_html(_events())
    # The question has & and <b>; both must be escaped, not rendered.
    assert "&amp;" in doc
    assert "&lt;b&gt;" in doc
    assert "<b>more</b>" not in doc


def test_totals_rendered():
    doc = render_html(_events())
    assert "tokens in" in doc
    assert "100" in doc
    assert "evidence" in doc


def test_render_file_roundtrip(tmp_path: Path):
    log = tmp_path / "run.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in _events()))
    out = render_file(log)
    assert out == log.with_suffix(".html")
    assert out.is_file()
    assert "<!doctype html>" in out.read_text()


def test_render_file_explicit_out(tmp_path: Path):
    log = tmp_path / "run.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in _events()))
    dst = tmp_path / "custom.html"
    out = render_file(log, dst)
    assert out == dst
    assert dst.is_file()


def test_missing_file_raises(tmp_path: Path):
    import pytest
    with pytest.raises(FileNotFoundError):
        render_file(tmp_path / "nope.jsonl")


def test_tolerates_malformed_lines(tmp_path: Path):
    log = tmp_path / "run.jsonl"
    log.write_text('{"run_id":"r","kind":"run_start","payload":{"question":"hi"}}\n'
                   'not json at all\n'
                   '\n'
                   '{"run_id":"r","kind":"run_end","payload":{"final_answer":"bye"}}\n')
    out = render_file(log)
    doc = out.read_text()
    assert "hi" in doc and "bye" in doc
