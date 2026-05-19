"""Report module tests for Phase 1.

We synthesize a small results.jsonl on disk and confirm the aggregator
produces the expected resume-table cells and failure taxonomy.
"""
from __future__ import annotations

import json
from pathlib import Path

from banna_agent.benchmarks.gaia.report import aggregate


def _make_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_aggregate_groups_by_policy_model_level(tmp_path: Path) -> None:
    rows = [
        {"policy_name": "ReAct", "model_name": "claude-haiku", "level": 1,
         "is_correct": True, "cost_usd": 0.01, "steps_used": 5,
         "tokens_in": 1000, "tokens_out": 200, "wall_s": 4.0,
         "timeout": False, "finished_reason": "final",
         "cache_hits": 0, "cache_misses": 3},
        {"policy_name": "ReAct", "model_name": "claude-haiku", "level": 1,
         "is_correct": False, "cost_usd": 0.02, "steps_used": 8,
         "tokens_in": 2000, "tokens_out": 300, "wall_s": 12.0,
         "timeout": True, "finished_reason": "budget_wall",
         "cache_hits": 1, "cache_misses": 2},
        {"policy_name": "Planner", "model_name": "claude-haiku", "level": 1,
         "is_correct": True, "cost_usd": 0.03, "steps_used": 6,
         "tokens_in": 1500, "tokens_out": 250, "wall_s": 5.0,
         "timeout": False, "finished_reason": "final",
         "cache_hits": 2, "cache_misses": 1},
    ]
    p = tmp_path / "results.jsonl"
    _make_jsonl(p, rows)

    report = aggregate([p])
    cells = {(c.policy, c.model, c.level): c for c in report.cells}
    assert set(cells) == {
        ("ReAct", "claude-haiku", 1),
        ("Planner", "claude-haiku", 1),
    }

    react = cells[("ReAct", "claude-haiku", 1)]
    assert react.n == 2
    assert react.n_correct == 1
    assert react.accuracy == 0.5
    assert react.cost_per_task == 0.015
    assert react.steps_per_task == 6.5
    assert react.tokens_per_task == (1000 + 200 + 2000 + 300) / 2
    assert react.n_timeout == 1
    assert react.timeout_pct == 50.0
    assert react.cache_hits == 1
    assert react.cache_misses == 5
    assert react.finished_reasons["final"] == 1
    assert react.finished_reasons["budget_wall"] == 1


def test_render_table_has_header_and_rows(tmp_path: Path) -> None:
    p = tmp_path / "r.jsonl"
    _make_jsonl(p, [
        {"policy_name": "ReAct", "model_name": "m", "level": 1,
         "is_correct": True, "cost_usd": 0.01, "steps_used": 3,
         "tokens_in": 100, "tokens_out": 20, "wall_s": 1.0,
         "timeout": False, "finished_reason": "final"},
    ])
    out = aggregate([p]).render_table()
    assert "policy" in out and "acc%" in out
    assert "ReAct" in out


def test_failure_taxonomy_rendered(tmp_path: Path) -> None:
    p = tmp_path / "r.jsonl"
    _make_jsonl(p, [
        {"policy_name": "ReAct", "model_name": "m", "level": 1, "is_correct": False,
         "finished_reason": "budget_steps"},
        {"policy_name": "ReAct", "model_name": "m", "level": 1, "is_correct": False,
         "finished_reason": "budget_steps"},
        {"policy_name": "ReAct", "model_name": "m", "level": 1, "is_correct": True,
         "finished_reason": "final"},
    ])
    out = aggregate([p]).render_failure_taxonomy()
    assert "budget_steps" in out
    assert "final" in out


def test_missing_fields_use_question_marks(tmp_path: Path) -> None:
    # Older JSONL without policy_name/model_name should still aggregate.
    p = tmp_path / "r.jsonl"
    _make_jsonl(p, [
        {"level": 1, "is_correct": True, "cost_usd": 0.01,
         "steps_used": 3, "tokens_in": 100, "tokens_out": 10, "wall_s": 1.0},
    ])
    report = aggregate([p])
    assert report.cells[0].policy == "?"
    assert report.cells[0].model == "?"
