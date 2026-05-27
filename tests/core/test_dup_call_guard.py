"""Anti-loop: identical repeat tool calls are skipped, not re-executed.

Regression coverage for the GAIA L3 search-loopers — tasks that issued
the same query 12+ times and timed out without committing. A byte-
identical repeat of a dedup-eligible (read-only) tool returns the prior
result with a nudge instead of paying the latency again; side-effecting /
stateful / terminal tools always run.
"""
from __future__ import annotations

from banna_agent.core.agent import _execute, _NO_DEDUP_TOOLS
from banna_agent.core.state import AgentState
from banna_agent.core.types import Action, ActionKind
from banna_agent.tools.base import JsonTool, ToolRegistry


def _counting_tool(name: str, calls: list[str]) -> JsonTool:
    def handler(args: dict) -> dict:
        calls.append(args.get("q", ""))
        return {"value": f"result for {args.get('q')}"}
    return JsonTool(
        name=name,
        description="x",
        input_schema={"type": "object",
                      "properties": {"q": {"type": "string"}}},
        handler=handler,
    )


def _call(name: str, **args) -> Action:
    return Action(kind=ActionKind.TOOL_CALL, tool_name=name, tool_args=args,
                  meta={})


def test_identical_read_call_runs_once() -> None:
    calls: list[str] = []
    tools = ToolRegistry([_counting_tool("search", calls)])
    state = AgentState(question="?")

    o1 = _execute(state, _call("search", q="abc"), tools, log=None)
    o2 = _execute(state, _call("search", q="abc"), tools, log=None)

    assert calls == ["abc"], "second identical search must not re-execute"
    assert o1.ok and o2.ok
    assert o2.data.get("duplicate_skipped") is True
    assert "DUPLICATE CALL SKIPPED" in o2.data["note"]


def test_different_args_still_run() -> None:
    calls: list[str] = []
    tools = ToolRegistry([_counting_tool("search", calls)])
    state = AgentState(question="?")

    _execute(state, _call("search", q="abc"), tools, log=None)
    _execute(state, _call("search", q="def"), tools, log=None)

    assert calls == ["abc", "def"], "distinct queries must both run"


def test_side_effecting_tool_is_never_deduped() -> None:
    # run_python is on the no-dedup list: identical code must run every time.
    assert "run_python" in _NO_DEDUP_TOOLS
    calls: list[str] = []
    tools = ToolRegistry([_counting_tool("run_python", calls)])
    state = AgentState(question="?")

    _execute(state, _call("run_python", q="print(1)"), tools, log=None)
    _execute(state, _call("run_python", q="print(1)"), tools, log=None)

    assert calls == ["print(1)", "print(1)"], "run_python must always execute"
