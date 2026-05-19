"""Unit tests for the JSONL event log."""
from __future__ import annotations

import json
from pathlib import Path

from banna_agent.core.events import AgentEvent, EventKind, EventLog, emit


def test_event_log_in_memory() -> None:
    log = EventLog()
    log.emit(AgentEvent(run_id="r1", step=0, kind=EventKind.RUN_START))
    assert len(log.events) == 1
    assert log.events[0].kind == EventKind.RUN_START


def test_event_log_writes_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    log = EventLog(p)
    log.emit(AgentEvent(run_id="r1", step=0, kind=EventKind.RUN_START))
    log.emit(AgentEvent(run_id="r1", step=1, kind=EventKind.OBSERVATION, payload={"ok": True}))
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["kind"] == EventKind.RUN_START
    assert first["run_id"] == "r1"


def test_event_log_truncates_on_new_instance(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    EventLog(p).emit(AgentEvent(run_id="r1", step=0, kind=EventKind.RUN_START))
    # fresh run should clear the file
    EventLog(p).emit(AgentEvent(run_id="r2", step=0, kind=EventKind.RUN_START))
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["run_id"] == "r2"


def test_filter_by_kind() -> None:
    log = EventLog()
    log.emit(AgentEvent(run_id="r", step=0, kind=EventKind.RUN_START))
    log.emit(AgentEvent(run_id="r", step=1, kind=EventKind.PROPOSE))
    log.emit(AgentEvent(run_id="r", step=1, kind=EventKind.OBSERVATION))
    log.emit(AgentEvent(run_id="r", step=2, kind=EventKind.PROPOSE))
    proposals = log.filter(EventKind.PROPOSE)
    assert len(proposals) == 2


def test_emit_helper_is_null_safe() -> None:
    # When log is None, it's a no-op and doesn't raise.
    emit(None, run_id="r", step=0, kind=EventKind.RUN_START)


def test_emit_helper_adds_to_log() -> None:
    log = EventLog()
    emit(log, run_id="r", step=3, kind=EventKind.PROPOSE, tool_name="search")
    assert log.events[0].payload["tool_name"] == "search"
    assert log.events[0].step == 3
