"""Unit tests for the recall_evidence tool (B1 safety net)."""
from __future__ import annotations

import pytest

from banna_agent.core.run_context import reset_current_state, set_current_state
from banna_agent.core.state import AgentState
from banna_agent.core.types import Action, ActionKind, Observation
from banna_agent.tools.recall import make_recall_tool


def _state_with_page(text: str, evidence_id: str = "ev1") -> AgentState:
    state = AgentState(question="q")
    state.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="read_url",
               tool_args={"url": "http://x"}),
        Observation(ok=True, data={"url": "http://x", "title": "T",
                                   "text": text, "evidence_id": evidence_id}),
    )
    # Mirror the driver's auto-registration: evidence content is snippet-capped.
    state.add_evidence(source="http://x", content=text[:1000], origin_step=0,
                       meta={"title": "T"})
    state.evidence[-1].evidence_id = evidence_id
    return state


def test_recall_returns_full_text_from_trace() -> None:
    big = "Z" * 9000
    state = _state_with_page(big, "ev1")
    tool = make_recall_tool()
    token = set_current_state(state)
    try:
        out = tool.handler({"evidence_id": "ev1"})
    finally:
        reset_current_state(token)
    assert out["evidence_id"] == "ev1"
    assert out["source"] == "http://x"
    assert out["text"] == big          # full text, not the 1000-char snippet
    assert out["chars"] == 9000


def test_recall_unknown_id_reports_error_field() -> None:
    state = _state_with_page("hello", "ev1")
    tool = make_recall_tool()
    token = set_current_state(state)
    try:
        out = tool.handler({"evidence_id": "nope"})
    finally:
        reset_current_state(token)
    assert "error" in out


def test_recall_multiple_ids() -> None:
    state = _state_with_page("first", "ev1")
    state.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="read_url",
               tool_args={"url": "http://y"}),
        Observation(ok=True, data={"url": "http://y", "text": "second",
                                   "evidence_id": "ev2"}),
    )
    state.add_evidence(source="http://y", content="second", origin_step=1)
    state.evidence[-1].evidence_id = "ev2"

    tool = make_recall_tool()
    token = set_current_state(state)
    try:
        out = tool.handler({"evidence_ids": ["ev1", "ev2"]})
    finally:
        reset_current_state(token)
    assert out["count"] == 2
    assert {r["text"] for r in out["results"]} == {"first", "second"}


def test_recall_outside_run_raises() -> None:
    tool = make_recall_tool()
    # No current state published.
    with pytest.raises(ValueError):
        tool.handler({"evidence_id": "ev1"})


def test_recall_requires_an_id() -> None:
    state = _state_with_page("x", "ev1")
    tool = make_recall_tool()
    token = set_current_state(state)
    try:
        with pytest.raises(ValueError):
            tool.handler({})
    finally:
        reset_current_state(token)
