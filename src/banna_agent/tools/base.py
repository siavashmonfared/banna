"""Tool contract.

`JsonTool` is an exact mirror of Banna's
`engine/agents_lib/protocol.py::JsonTool`. This is deliberate — the
week-3 `AgentRunner` adapter needs to accept Banna `JsonTool` objects
and hand them to our agent loop as if they were ours. Identical shape
means zero translation layer.

The schema is JSON Schema (the same format every provider accepts). A
handler is a synchronous `Callable[[dict], dict]`. Handlers may raise;
the driver catches and converts to a `tool_result` block with
`is_error=True`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class JsonTool:
    """An executable tool declaration.

    Fields mirror Banna's `JsonTool` exactly:
      name         - unique identifier, matches what the model calls
      description  - one-line capability summary shown to the model
      input_schema - JSON Schema dict; providers consume directly
      handler      - sync `dict -> dict`; may raise on failure

    Optional extension:
      capabilities - frozenset like {"network", "read", "sandbox"}. Used
                     by the driver to categorize tool risk. Empty by
                     default, so legacy Banna tools interop cleanly.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    capabilities: frozenset[str] = field(default_factory=frozenset)

    def to_tool_spec(self):
        """Project to the LLM layer's ToolSpec (declaration-only).

        The LLM client never sees the `handler` — it only advertises the
        schema to the model. The handler is invoked by the driver when a
        tool_use block comes back.
        """
        from ..llm.base import ToolSpec  # local import to avoid cycles
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )


@dataclass
class ToolInvocation:
    """Result of one handler call. The driver wraps this into a
    `tool_result` ContentBlock. Kept separate from `ContentBlock` because
    it carries telemetry the policy doesn't need (wall time)."""

    ok: bool
    result: Any
    wall_s: float = 0.0
    error: str | None = None


def invoke_tool(tool: JsonTool, arguments: dict[str, Any]) -> ToolInvocation:
    """Call a tool safely. Exceptions become error ToolInvocations; the
    tool itself never crashes the agent loop."""
    t0 = time.monotonic()
    try:
        result = tool.handler(arguments)
    except Exception as exc:
        return ToolInvocation(
            ok=False,
            result={"error": str(exc)},
            wall_s=time.monotonic() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )
    return ToolInvocation(ok=True, result=result, wall_s=time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Lightweight name -> JsonTool map. Policies receive a registry and
    look up tools by name when dispatching `tool_use` blocks."""

    def __init__(self, tools: list[JsonTool] | None = None) -> None:
        self._tools: dict[str, JsonTool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: JsonTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> JsonTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def as_list(self) -> list[JsonTool]:
        return list(self._tools.values())

    def to_tool_specs(self):
        """Project the whole registry to LLM-layer ToolSpecs."""
        return [t.to_tool_spec() for t in self._tools.values()]
