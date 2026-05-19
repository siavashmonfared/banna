"""Unit tests for the tool base contract."""
from __future__ import annotations

import pytest

from banna_agent.tools.base import JsonTool, ToolRegistry, invoke_tool


def _echo_handler(args: dict) -> dict:
    return {"got": args}


def _raising_handler(args: dict) -> dict:
    raise RuntimeError("boom")


def _make_tool(name: str = "echo") -> JsonTool:
    return JsonTool(
        name=name,
        description="echoes args",
        input_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
        handler=_echo_handler,
    )


# ---------------------------------------------------------------------------
# JsonTool projection
# ---------------------------------------------------------------------------


def test_to_tool_spec_drops_handler_keeps_schema() -> None:
    tool = _make_tool()
    spec = tool.to_tool_spec()
    assert spec.name == "echo"
    assert spec.description == "echoes args"
    assert spec.input_schema == tool.input_schema
    # The LLM layer must not see the handler.
    assert not hasattr(spec, "handler")


def test_jsontool_is_frozen() -> None:
    tool = _make_tool()
    with pytest.raises(Exception):
        tool.name = "renamed"  # type: ignore[misc]


def test_capabilities_default_empty() -> None:
    tool = _make_tool()
    assert tool.capabilities == frozenset()


# ---------------------------------------------------------------------------
# invoke_tool — success and failure paths
# ---------------------------------------------------------------------------


def test_invoke_tool_success_returns_ok() -> None:
    tool = _make_tool()
    inv = invoke_tool(tool, {"msg": "hi"})
    assert inv.ok is True
    assert inv.result == {"got": {"msg": "hi"}}
    assert inv.error is None
    assert inv.wall_s >= 0.0


def test_invoke_tool_exception_becomes_error_invocation() -> None:
    tool = JsonTool(
        name="bad",
        description="always fails",
        input_schema={"type": "object"},
        handler=_raising_handler,
    )
    inv = invoke_tool(tool, {})
    assert inv.ok is False
    assert inv.result == {"error": "boom"}
    assert inv.error is not None
    assert "RuntimeError" in inv.error


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def test_registry_register_and_get() -> None:
    reg = ToolRegistry([_make_tool("a"), _make_tool("b")])
    assert reg.names() == ["a", "b"]
    assert reg.get("a").name == "a"
    assert reg.get("missing") is None


def test_registry_rejects_duplicates() -> None:
    reg = ToolRegistry([_make_tool("a")])
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(_make_tool("a"))


def test_registry_to_tool_specs_preserves_order_and_drops_handlers() -> None:
    reg = ToolRegistry([_make_tool("a"), _make_tool("b")])
    specs = reg.to_tool_specs()
    assert [s.name for s in specs] == ["a", "b"]
    for s in specs:
        assert not hasattr(s, "handler")
