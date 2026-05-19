"""End-to-end test of GAIA runner with a scripted fake LLM.

No network, no real dataset — we build a 3-task GAIATask list and run it
through `run_gaia` with a fake LLM that emits pre-canned replies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from banna_agent.benchmarks.gaia.loader import GAIATask
from banna_agent.benchmarks.gaia.runner import run_gaia
from banna_agent.llm.base import ContentBlock, LLMReply, Usage
from banna_agent.policies.react import ReActPolicy
from banna_agent.tools.base import ToolRegistry
from banna_agent.tools.calculator import make_calculator_tool


@dataclass
class _ScriptedLLM:
    replies: list[LLMReply]
    provider: str = "scripted"
    model: str = "s"

    def chat(self, **_: Any) -> LLMReply:
        if not self.replies:
            return LLMReply(provider="scripted", model="s", content=[], stop_reason="end_turn")
        return self.replies.pop(0)


def _text(t: str) -> LLMReply:
    return LLMReply(
        provider="scripted", model="s",
        content=[ContentBlock(kind="text", text=t)],
        stop_reason="end_turn",
        usage=Usage(tokens_in=5, tokens_out=2),
    )


def _tools() -> ToolRegistry:
    return ToolRegistry([make_calculator_tool()])


def test_final_answer_is_submitted_literally(tmp_path: Path) -> None:
    """Post-Phase-2: the driver no longer rewrites the model's answer.
    We submit the literal string ('$12') and let the GAIA scorer apply
    its own normalization for the comparison — which still scores it
    correct because parse_number('$12') == 12."""
    tasks = [GAIATask(task_id="t1", question="How much did it cost?", level=1, answer="12")]
    llm = _ScriptedLLM([_text("$12")])
    run_gaia(tasks, llm=llm, tools=_tools(), policy=ReActPolicy(),
             out_dir=tmp_path, verbose=False)
    import json
    rec = json.loads((tmp_path / "results.jsonl").read_text().splitlines()[0])
    assert rec["pred_answer"] == "$12"
    assert rec["is_correct"] is True


def test_tool_call_counts_use_real_event_schema(tmp_path: Path) -> None:
    """Regression: events emit ``tool_name`` (not ``name``), so the metric
    collector must read that field. Without this, every call gets
    bucketed as '?'."""
    import json
    from banna_agent.llm.base import ContentBlock, Usage
    # Reply that triggers a calculator tool_use, then a final answer.
    tool_use_reply = LLMReply(
        provider="scripted", model="s",
        content=[ContentBlock(
            kind="tool_use", id="t1", name="calculator",
            arguments={"expression": "2+2"},
        )],
        stop_reason="tool_use",
        usage=Usage(tokens_in=5, tokens_out=5),
    )
    final_reply = _text("4")
    tasks = [GAIATask(task_id="t1", question="2+2?", level=1, answer="4")]
    llm = _ScriptedLLM([tool_use_reply, final_reply])
    run_gaia(tasks, llm=llm, tools=_tools(), policy=ReActPolicy(),
             out_dir=tmp_path, verbose=False)
    rec = json.loads((tmp_path / "results.jsonl").read_text().splitlines()[0])
    # The call must be bucketed under its real name, not '?'.
    assert rec["tool_calls_total"] >= 1
    assert "calculator" in rec["tool_calls_by_name"]
    assert rec["tool_calls_by_name"]["calculator"] >= 1
    assert "?" not in rec["tool_calls_by_name"]


def test_jsonl_includes_phase1_metric_fields(tmp_path: Path) -> None:
    """Phase 1: per-task JSONL must carry the ablation fields used by
    benchmarks/gaia/report.py — policy_name, model_name, finished_reason,
    timeout flag, tool-call counters, cache stats."""
    import json
    tasks = [GAIATask(task_id="t1", question="2+2?", level=1, answer="4")]
    llm = _ScriptedLLM([_text("4")])
    run_gaia(tasks, llm=llm, tools=_tools(), policy=ReActPolicy(),
             out_dir=tmp_path, verbose=False)
    lines = (tmp_path / "results.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    for fld in (
        "policy_name", "model_name", "timeout", "finished_reason",
        "tool_calls_total", "tool_calls_by_name",
        "tool_errors_total", "tool_errors_by_name",
        "cache_hits", "cache_misses",
    ):
        assert fld in rec, f"missing metric field: {fld}"
    assert rec["policy_name"] == "ReActPolicy"
    assert rec["model_name"] == "s"
    assert rec["finished_reason"] == "final"
    assert rec["timeout"] is False


# ---------------------------------------------------------------------------
# Minimal pass/fail scoring path
# ---------------------------------------------------------------------------


def test_run_gaia_scores_mixed_results(tmp_path: Path) -> None:
    tasks = [
        GAIATask(task_id="t1", question="2+2?", level=1, answer="4"),
        GAIATask(task_id="t2", question="3+3?", level=1, answer="6"),
        GAIATask(task_id="t3", question="What is 4+4?", level=1, answer="8"),
    ]
    # Replies: correct, wrong, correct.
    llm = _ScriptedLLM([_text("4"), _text("7"), _text("8")])
    out = run_gaia(
        tasks,
        llm=llm,
        tools=_tools(),
        policy=ReActPolicy(),
        out_dir=tmp_path,
        verbose=False,
    )
    assert out.n_total == 3
    assert out.n_correct == 2
    assert out.accuracy == pytest.approx(2 / 3)
    # Per-level aggregation
    assert out.by_level[1]["n_correct"] == 2
    # JSONL + summary written
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    # Per-task logs written
    log_files = list((tmp_path / "logs").glob("*.jsonl"))
    assert len(log_files) == 3


def test_run_gaia_handles_budget_trip(tmp_path: Path) -> None:
    """LLM keeps calling the calculator tool; budget=1 step trips immediately."""
    from banna_agent.core.types import Budget

    tasks = [GAIATask(task_id="t", question="?", level=1, answer="ok")]
    # Two replies — policy budget cap forces only one step.
    llm = _ScriptedLLM([
        LLMReply(
            provider="s", model="s",
            content=[ContentBlock(kind="tool_use", id="c1", name="calculator",
                                  arguments={"expression": "1+1"})],
            stop_reason="tool_use",
        ),
        _text("2"),
    ])
    out = run_gaia(
        tasks,
        llm=llm,
        tools=_tools(),
        policy=ReActPolicy(),
        budget_factory=lambda task: Budget(max_steps=1, max_wall_s=5.0),
        out_dir=tmp_path,
        verbose=False,
    )
    assert out.n_total == 1
    # Pred was never produced -> incorrect, match_kind="empty"
    assert out.n_correct == 0
    r = out.results[0]
    assert r.match_kind == "empty"
    assert r.steps_used == 1


def test_run_gaia_by_match_kind_breakdown(tmp_path: Path) -> None:
    tasks = [
        GAIATask(task_id="a", question="2+2?", level=1, answer="4"),
        GAIATask(task_id="b", question="capital of France?", level=1, answer="Paris"),
        GAIATask(task_id="c", question="yes or no?", level=1, answer="yes"),
    ]
    llm = _ScriptedLLM([_text("4"), _text("Paris"), _text("yes")])
    out = run_gaia(
        tasks, llm=llm, tools=_tools(), policy=ReActPolicy(),
        out_dir=tmp_path, verbose=False,
    )
    assert out.accuracy == 1.0
    kinds = {k: v["n_total"] for k, v in out.by_match_kind.items()}
    assert kinds == {"numeric": 1, "string": 1, "yes_no": 1}


def test_run_gaia_without_out_dir_still_scores() -> None:
    """Runner must work without an output directory (for tests / quick REPL use)."""
    tasks = [GAIATask(task_id="t", question="?", level=1, answer="x")]
    llm = _ScriptedLLM([_text("x")])
    out = run_gaia(tasks, llm=llm, tools=_tools(), policy=ReActPolicy(), verbose=False)
    assert out.n_correct == 1


def test_format_question_pdf_hint_mentions_pdf_tools(tmp_path: Path) -> None:
    from banna_agent.benchmarks.gaia.runner import _format_question
    pdf = tmp_path / "spec.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")  # marker only; summary may or may not parse
    task = GAIATask(task_id="t", question="What is in the file?", level=1,
                    answer="x", file_name="spec.pdf", file_path=str(pdf))
    out = _format_question(task)
    assert "pdf_open" in out
    assert "pdf_read_page" in out
    assert "An attached file is available at" in out


def test_format_question_xlsx_hint_mentions_xlsx_tools(tmp_path: Path) -> None:
    from banna_agent.benchmarks.gaia.runner import _format_question
    f = tmp_path / "data.xlsx"
    f.write_bytes(b"PK\x03\x04")  # bogus content; introspection will silently drop
    task = GAIATask(task_id="t", question="What's the total?", level=1,
                    answer="x", file_name="data.xlsx", file_path=str(f))
    out = _format_question(task)
    assert "xlsx_list_sheets" in out
    assert "xlsx_read_range" in out


def test_format_question_csv_keeps_read_file(tmp_path: Path) -> None:
    from banna_agent.benchmarks.gaia.runner import _format_question
    f = tmp_path / "data.csv"
    f.write_text("name,score\nalice,42\n")
    task = GAIATask(task_id="t", question="Top scorer?", level=1,
                    answer="x", file_name="data.csv", file_path=str(f))
    out = _format_question(task)
    assert "read_file" in out
    # Cheap header pre-introspection lands in the prompt.
    assert "header line" in out
    assert "name,score" in out


def test_format_question_no_attachment_omits_file_block() -> None:
    from banna_agent.benchmarks.gaia.runner import _format_question
    task = GAIATask(task_id="t", question="2+2?", level=1, answer="4")
    out = _format_question(task)
    assert "attached file" not in out
    assert "Answer with just the short answer" in out


def test_format_question_omits_summary_when_introspection_fails(tmp_path: Path) -> None:
    """Garbage PDF magic bytes shouldn't blow up `_format_question`."""
    from banna_agent.benchmarks.gaia.runner import _format_question
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"not a real pdf")
    task = GAIATask(task_id="t", question="?", level=1, answer="x",
                    file_name="bad.pdf", file_path=str(f))
    out = _format_question(task)
    # Hint still present, but no "File summary:" line for a broken file.
    assert "pdf_open" in out
    # We don't strictly forbid a summary — pypdf might still succeed on
    # some byte patterns — but the call must not raise.


def test_default_budget_step_caps_match_post_c4() -> None:
    """C4 bumped the L1/L2/L3 step caps after the 05-16 nano run showed
    19/27 budget_steps failures on L1. Pin the new defaults so a casual
    refactor doesn't silently regress them."""
    from banna_agent.benchmarks.gaia.runner import _default_budget
    expected = {1: 12, 2: 18, 3: 24}
    for level, want in expected.items():
        b = _default_budget(GAIATask(task_id="t", question="?", level=level, answer="x"))
        assert b.max_steps == want, f"L{level} default step cap regressed"


def test_run_gaia_captures_policy_exceptions(tmp_path: Path) -> None:
    """If the policy itself crashes, the task is marked incorrect but the run continues."""

    class _BoomPolicy:
        name = "boom"
        def propose(self, *a, **kw):
            raise RuntimeError("unexpected")

    tasks = [
        GAIATask(task_id="t1", question="?", level=1, answer="a"),
        GAIATask(task_id="t2", question="?", level=1, answer="b"),
    ]
    llm = _ScriptedLLM([])
    out = run_gaia(
        tasks, llm=llm, tools=_tools(), policy=_BoomPolicy(),
        out_dir=tmp_path, verbose=False,
    )
    assert out.n_correct == 0
    # Runs both even though policy crashes each time.
    assert out.n_total == 2
