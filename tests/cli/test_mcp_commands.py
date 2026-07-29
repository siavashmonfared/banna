"""MCP tools as slash commands."""
from __future__ import annotations

from typing import Any

import pytest

from banna_agent.cli import mcp_commands
from banna_agent.cli.commands import dispatch
from banna_agent.tools.base import JsonTool, ToolRegistry


class _Console:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args: Any, **kwargs: Any) -> None:
        self.lines.append(" ".join(str(a) for a in args))

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class _Manager:
    def __init__(self, statuses: list[dict]) -> None:
        self._statuses = statuses

    def statuses(self) -> list[dict]:
        return [dict(s) for s in self._statuses]


class _App:
    """Just the surface mcp_commands touches."""

    def __init__(self, tools: list[JsonTool], statuses: list[dict]) -> None:
        self.console = _Console()
        self.tools = ToolRegistry(tools)
        self._mcp_manager = _Manager(statuses)

    def _mcp_tools(self) -> list:
        return self.tools.as_list()


ASK_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "agent": {"type": "string"},
        "timeout": {"type": "integer"},
    },
    "required": ["prompt"],
}


@pytest.fixture()
def app() -> _App:
    calls: list[dict] = []

    def ask(args):
        calls.append(args)
        return {"text": f"answered: {args.get('prompt')}"}

    def read(args):
        return {"text": "thread contents"}

    tools = [
        JsonTool(name="collab__collab_ask", description="Ask another agent.",
                 input_schema=ASK_SCHEMA, handler=ask),
        JsonTool(name="collab__collab_read", description="Read the thread.",
                 input_schema={"type": "object", "properties": {}}, handler=read),
    ]
    a = _App(tools, [{"name": "collab", "state": "connected",
                      "tools": ["collab__collab_ask", "collab__collab_read"]}])
    a.calls = calls
    return a


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_bare_server_command_lists_its_tools(app) -> None:
    """`/collab` with no arguments has to answer 'what can this thing do'."""
    dispatch(app, "/collab")
    assert "collab__collab_ask" in app.console.text
    assert "Ask another agent." in app.console.text


def test_completion_offers_servers_and_both_name_forms(app) -> None:
    names = mcp_commands.command_names(app)
    assert "collab" in names
    assert "collab__collab_ask" in names
    assert "collab_ask" in names


def test_disconnected_servers_are_not_offered(app) -> None:
    app._mcp_manager = _Manager(
        [{"name": "collab", "state": "failed", "tools": []}])
    assert mcp_commands.command_names(app) == []


# ---------------------------------------------------------------------------
# Calling
# ---------------------------------------------------------------------------


def test_server_plus_tool_form(app) -> None:
    dispatch(app, "/collab collab_ask what should we build")
    assert app.calls == [{"prompt": "what should we build"}]


def test_full_tool_name_form(app) -> None:
    dispatch(app, "/collab__collab_ask hello there")
    assert app.calls == [{"prompt": "hello there"}]


def test_short_tool_name_form(app) -> None:
    dispatch(app, "/collab_ask hello")
    assert app.calls == [{"prompt": "hello"}]


def test_result_text_is_shown(app) -> None:
    dispatch(app, "/collab_ask ping")
    assert "answered: ping" in app.console.text


def test_keyword_args_are_typed(app) -> None:
    """`timeout=30` must arrive as a number, not the string '30', or the
    server's schema validation rejects it."""
    dispatch(app, "/collab_ask agent=codex timeout=30 review this")
    assert app.calls == [
        {"agent": "codex", "timeout": 30, "prompt": "review this"}]


def test_unknown_keys_fall_into_free_text(app) -> None:
    """A bare `x=y` that isn't a real field is prose, not an argument —
    guessing otherwise would send the server a field it never declared."""
    dispatch(app, "/collab_ask is a=b valid syntax")
    assert app.calls == [{"prompt": "is a=b valid syntax"}]


def test_tool_with_no_string_field_takes_no_free_text(app) -> None:
    dispatch(app, "/collab_read")
    assert "thread contents" in app.console.text


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_ambiguous_short_name_asks_instead_of_guessing(app) -> None:
    """Two servers exposing the same tool name must not silently resolve to
    whichever was registered first."""
    app.tools.register(JsonTool(
        name="other__collab_ask", description="d",
        input_schema=ASK_SCHEMA, handler=lambda a: {"text": "wrong one"}))
    app._mcp_manager = _Manager([
        {"name": "collab", "state": "connected", "tools": ["collab__collab_ask"]},
        {"name": "other", "state": "connected", "tools": ["other__collab_ask"]},
    ])
    dispatch(app, "/collab_ask hi")
    assert "ambiguous" in app.console.text
    assert app.calls == []


def test_unknown_tool_on_a_known_server_reports_clearly(app) -> None:
    dispatch(app, "/collab nope")
    assert "no tool named" in app.console.text


def test_handler_errors_are_reported_not_raised(app) -> None:
    app.tools.register(JsonTool(
        name="collab__collab_boom", description="d",
        input_schema={"type": "object", "properties": {}},
        handler=lambda a: (_ for _ in ()).throw(RuntimeError("server died"))))
    app._mcp_manager = _Manager([{
        "name": "collab", "state": "connected",
        "tools": ["collab__collab_ask", "collab__collab_boom"]}])
    dispatch(app, "/collab_boom")
    assert "server died" in app.console.text


def test_unrelated_commands_still_reach_the_normal_error(app) -> None:
    """MCP dispatch must not swallow genuine typos."""
    dispatch(app, "/definitelynotacommand")
    assert "unknown command" in app.console.text


def test_builtin_commands_are_unaffected(app) -> None:
    """A server named like a builtin must not shadow it."""
    app._mcp_manager = _Manager(
        [{"name": "tools", "state": "connected", "tools": ["tools__x"]}])
    dispatch(app, "/tools")
    assert "unknown command" not in app.console.text


def test_no_servers_connected_is_a_clean_passthrough() -> None:
    a = _App([], [])
    dispatch(a, "/collab")
    assert "unknown command" in a.console.text


def test_lazy_connect_only_for_configured_server_names(monkeypatch) -> None:
    """Servers connect on the first agent turn, so `/collab` typed right
    after launch has to bring them up — but a typo must not pay the cost of
    starting every subprocess."""
    connected: list[str] = []

    class _Lazy(_App):
        def __init__(self):
            super().__init__([], [])
            self._mcp_manager = None

        def _mcp_tools(self):
            connected.append("yes")
            self._mcp_manager = _Manager([])
            return []

    monkeypatch.setattr(
        "banna_agent.cli.mcp_config.read_mcp_servers", lambda: {"collab": {}})

    a = _Lazy()
    dispatch(a, "/typo_not_a_server")
    assert connected == []

    dispatch(a, "/collab")
    assert connected == ["yes"]
