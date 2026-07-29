"""Tool names must be portable across every provider.

The registry is the single choke point between all tool sources and all
providers, so the rule is enforced there rather than per-adapter or
per-source. These tests pin the contract, not any one provider's syntax.
"""
from __future__ import annotations

import re

from banna_agent.tools.base import JsonTool, ToolRegistry, portable_tool_name

# The intersection of the providers' documented constraints.
_PORTABLE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{0,63}$")


def _tool(name: str) -> JsonTool:
    return JsonTool(
        name=name,
        description="d",
        input_schema={"type": "object", "properties": {}},
        handler=lambda args: {"ok": True},
    )


def test_dotted_names_are_rewritten() -> None:
    """A dot is rejected outright by OpenAI and Anthropic, and collides with
    gpt-oss's own `functions.*` namespacing."""
    assert "." not in portable_tool_name("collab.collab_start")


def test_portable_names_pass_through_unchanged() -> None:
    for name in ("read_file", "collab__collab_ask", "web-search", "_private"):
        assert portable_tool_name(name) == name


def test_hostile_names_are_coerced() -> None:
    """Names arrive from MCP servers we don't control, so nothing about
    their shape can be assumed."""
    for name in ("a b/c", "tool!@#", "…unicode", "9leading_digit", "", "."):
        assert _PORTABLE.match(portable_tool_name(name)), name


def test_overlong_names_stay_distinct_after_truncation() -> None:
    """Two names sharing a long prefix must not collapse into one tool."""
    a = portable_tool_name("server__" + "x" * 100 + "_alpha")
    b = portable_tool_name("server__" + "x" * 100 + "_beta")
    assert len(a) <= 64 and len(b) <= 64
    assert a != b


def test_normalization_is_stable() -> None:
    """Re-normalizing an already-normalized name is a no-op, so a name that
    round-trips through the registry twice doesn't drift."""
    for name in ("collab.collab_start", "a b/c", "x" * 200):
        once = portable_tool_name(name)
        assert portable_tool_name(once) == once


def test_registry_stores_and_advertises_the_portable_name() -> None:
    """The stored key, the advertised spec, and what the model calls back
    all have to be the same string, or dispatch breaks."""
    reg = ToolRegistry([_tool("collab.collab_ask")])
    name = reg.names()[0]
    assert _PORTABLE.match(name)
    assert [s.name for s in reg.to_tool_specs()] == [name]
    assert reg.get(name) is not None
    assert reg.get("collab.collab_ask") is None   # the old name is gone


def test_registry_normalizes_every_source_not_just_mcp() -> None:
    """The rule is a property of the registry, so it covers built-ins and
    any tool source added later, not only the MCP bridge."""
    reg = ToolRegistry([_tool("weird name!"), _tool("fine_name")])
    assert all(_PORTABLE.match(n) for n in reg.names())


def test_registry_still_rejects_genuine_duplicates() -> None:
    reg = ToolRegistry([_tool("read_file")])
    try:
        reg.register(_tool("read_file"))
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected duplicate registration to raise")


def test_registry_dispatch_survives_normalization() -> None:
    """The handler must still be reachable under the rewritten name."""
    reg = ToolRegistry([_tool("srv.do/thing")])
    tool = reg.get(reg.names()[0])
    assert tool.handler({}) == {"ok": True}
