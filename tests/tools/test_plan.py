"""Unit tests for the plan/todo scratchpad tool."""
from __future__ import annotations

import pytest

from banna_agent.tools.plan import make_plan_tool, reset_all


@pytest.fixture(autouse=True)
def _clean() -> None:
    reset_all()


def _h():
    return make_plan_tool().handler


def test_add_then_list_returns_item() -> None:
    h = _h()
    r = h({"op": "add", "step": "find ARPU"})
    assert r["item"]["step"] == "find ARPU"
    assert r["item"]["status"] == "todo"
    assert len(r["plan"]) == 1

    listed = h({"op": "list"})
    assert len(listed["plan"]) == 1
    assert listed["plan"][0]["step"] == "find ARPU"


def test_update_status_and_notes() -> None:
    h = _h()
    item = h({"op": "add", "step": "compute 17*23"})["item"]
    r = h({"op": "update", "id": item["id"], "status": "done", "notes": "= 391"})
    assert r["item"]["status"] == "done"
    assert r["item"]["notes"] == "= 391"


def test_update_rejects_invalid_status() -> None:
    h = _h()
    item = h({"op": "add", "step": "x"})["item"]
    with pytest.raises(ValueError, match="invalid status"):
        h({"op": "update", "id": item["id"], "status": "bogus"})


def test_update_missing_id_raises() -> None:
    h = _h()
    with pytest.raises(KeyError):
        h({"op": "update", "id": "nonexistent", "status": "done"})


def test_clear_wipes_plan() -> None:
    h = _h()
    h({"op": "add", "step": "a"})
    h({"op": "add", "step": "b"})
    r = h({"op": "clear"})
    assert r["plan"] == []
    assert h({"op": "list"})["plan"] == []


def test_plans_are_isolated_by_plan_id() -> None:
    h = _h()
    h({"op": "add", "step": "only in plan A", "plan_id": "A"})
    h({"op": "add", "step": "only in plan B", "plan_id": "B"})
    plan_a = h({"op": "list", "plan_id": "A"})["plan"]
    plan_b = h({"op": "list", "plan_id": "B"})["plan"]
    assert len(plan_a) == 1 and plan_a[0]["step"] == "only in plan A"
    assert len(plan_b) == 1 and plan_b[0]["step"] == "only in plan B"


def test_add_rejects_empty_step() -> None:
    h = _h()
    with pytest.raises(ValueError, match="non-empty"):
        h({"op": "add", "step": ""})


def test_unknown_op_raises() -> None:
    h = _h()
    with pytest.raises(ValueError, match="unknown op"):
        h({"op": "wiggle"})


def test_tool_metadata() -> None:
    tool = make_plan_tool()
    assert tool.name == "plan"
    assert tool.capabilities == frozenset({"state", "scratchpad"})
    assert tool.input_schema["required"] == ["op"]
