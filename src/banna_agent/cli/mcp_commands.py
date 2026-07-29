"""Call MCP tools directly as slash commands.

Bridged MCP tools are registered under portable names like
`collab__collab_ask`. This module lets the user invoke them from the REPL
the same way built-in commands work:

    /collab                        list what the collab server offers
    /collab collab_ask <text>      call one of its tools
    /collab__collab_ask <text>     the full name works too
    /collab_ask <text>             so does the short name, when unambiguous

Everything here is resolved live from the connected servers, so a server
installed mid-session becomes callable immediately — there is no static
command table to keep in sync.

Typing the command *is* the authorization, so these calls skip the
permission gate the agent's own tool calls go through.
"""
from __future__ import annotations

import json
from typing import Any

from ..tools.base import invoke_tool

_SEP = "__"


def _manager(app: Any) -> Any:
    return getattr(app, "_mcp_manager", None)


def server_tool_map(app: Any) -> dict[str, list[str]]:
    """`{server: [local tool name, ...]}` for the connected servers.

    Read from the manager's own status records rather than by splitting
    tool names, so a server whose name contains the separator still maps
    correctly.
    """
    mgr = _manager(app)
    if mgr is None:
        return {}
    out: dict[str, list[str]] = {}
    for status in mgr.statuses():
        if status.get("state") == "connected" and status.get("tools"):
            out[status["name"]] = list(status["tools"])
    return out


def _remote_name(server: str, local: str) -> str:
    """The tool's name as the server itself knows it."""
    prefix = f"{server}{_SEP}"
    return local[len(prefix):] if local.startswith(prefix) else local


def command_names(app: Any) -> list[str]:
    """Every slash form that resolves to an MCP tool — for tab completion."""
    names: list[str] = []
    for server, tools in server_tool_map(app).items():
        names.append(server)
        names.extend(tools)
        names.extend(_remote_name(server, t) for t in tools)
    return sorted(set(names))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _coerce(value: str) -> Any:
    """`key=value` values are JSON when they parse as JSON, else strings.

    So `count=3` and `deep=true` arrive typed, while `name=alice` stays a
    string instead of failing to parse.
    """
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def _free_text_field(schema: dict[str, Any]) -> str | None:
    """The property that bare, unkeyed text should fill.

    Prefers the single required string; falls back to the first string
    property. This is what makes `/collab_ask what should we build?` work
    instead of forcing `prompt="what should we build?"`.
    """
    props = (schema or {}).get("properties") or {}
    required = [r for r in (schema or {}).get("required") or []
                if (props.get(r) or {}).get("type") in (None, "string")]
    if len(required) == 1:
        return required[0]
    for name, spec in props.items():
        if (spec or {}).get("type") == "string":
            return name
    return None


def parse_tool_args(schema: dict[str, Any], tokens: list[str]) -> dict[str, Any]:
    """Turn command-line tokens into a tool argument dict.

    `key=value` tokens become typed fields. Anything left over is joined
    and dropped into the tool's free-text field.
    """
    args: dict[str, Any] = {}
    loose: list[str] = []
    props = (schema or {}).get("properties") or {}
    for tok in tokens:
        key, sep, value = tok.partition("=")
        if sep and key in props:
            args[key] = _coerce(value)
        else:
            loose.append(tok)
    if loose:
        field = _free_text_field(schema)
        if field and field not in args:
            args[field] = " ".join(loose)
    return args


# ---------------------------------------------------------------------------
# Resolution + dispatch
# ---------------------------------------------------------------------------


def _resolve_tool_name(app: Any, name: str) -> tuple[str | None, str | None]:
    """Map a typed command to a registered tool name.

    Returns `(tool_name, error)`. An ambiguous short name resolves to no
    tool and an explanatory error rather than an arbitrary pick.
    """
    mapping = server_tool_map(app)
    for tools in mapping.values():
        if name in tools:
            return name, None
    matches = [
        t for server, tools in mapping.items() for t in tools
        if _remote_name(server, t) == name
    ]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        shown = "  ".join(f"/{m}" for m in matches)
        return None, f"/{name} is ambiguous across servers — use: {shown}"
    return None, None


def _print_server_tools(app: Any, server: str, tools: list[str]) -> None:
    p = app.console.print
    registry = getattr(app, "tools", None)
    p()
    p(f"[scout.agent]● {server}[/scout.agent]  [scout.muted]"
      f"{len(tools)} tool(s)[/scout.muted]")
    for local in tools:
        tool = registry.get(local) if registry else None
        desc = (tool.description if tool else "").strip().splitlines()
        summary = desc[0] if desc else ""
        p(f"  [scout.text]/{local}[/scout.text]  [scout.muted]{summary}[/scout.muted]")
    p()
    p(f"  [scout.muted]call with[/scout.muted] /{server} <tool> <text>   "
      f"[scout.muted]or[/scout.muted] /{server} <tool> key=value …")
    p()


def _run_tool(app: Any, tool_name: str, tokens: list[str]) -> bool:
    registry = getattr(app, "tools", None)
    tool = registry.get(tool_name) if registry else None
    if tool is None:
        app.console.print(f"[red]tool not registered:[/red] {tool_name}")
        return False

    args = parse_tool_args(tool.input_schema, tokens)
    app.console.print(f"[dim]▸ {tool_name}({_format_args(args)})[/dim]")
    result = invoke_tool(tool, args)
    if not result.ok:
        app.console.print(f"[red]✗ {result.error}[/red]")
        return False
    app.console.print(_render(result.result))
    app.console.print(f"[dim]  ({result.wall_s:.1f}s)[/dim]")
    return False


def _format_args(args: dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        text = v if isinstance(v, str) else json.dumps(v)
        if len(text) > 60:
            text = text[:57] + "…"
        parts.append(f"{k}={text!r}" if isinstance(v, str) else f"{k}={text}")
    return ", ".join(parts)


def _render(result: Any) -> str:
    """MCP results are usually `{"text": ...}`; show that plainly and fall
    back to pretty JSON for anything else."""
    if isinstance(result, dict):
        text = result.get("text")
        if isinstance(text, str) and len(result) <= 2:
            return text
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, indent=2, default=str)
    except Exception:
        return str(result)


def _connect_if_configured(app: Any, name: str) -> None:
    """Bring servers up if `name` looks like one we're configured for.

    Servers connect lazily on the first agent turn, so `/collab` typed
    straight after launch would otherwise find nothing. Gated on the name
    matching a configured server so an ordinary typo doesn't pay the cost
    of starting every subprocess.
    """
    if _manager(app) is not None:
        return
    try:
        from .mcp_config import read_mcp_servers
        configured = read_mcp_servers()
    except Exception:
        return
    if name not in configured:
        return
    try:
        app._mcp_tools()
    except Exception:
        pass


def try_dispatch(app: Any, name: str, args: list[str]) -> bool | None:
    """Handle `name` if it refers to an MCP server or tool.

    Returns None when it doesn't, so the caller can fall through to its
    normal unknown-command handling.
    """
    _connect_if_configured(app, name)
    mapping = server_tool_map(app)
    if not mapping:
        return None

    if name in mapping:
        tools = mapping[name]
        if not args:
            _print_server_tools(app, name, tools)
            return False
        sub = args[0]
        target, err = _resolve_tool_name(app, sub)
        if target is None and f"{name}{_SEP}{sub}" in tools:
            target = f"{name}{_SEP}{sub}"
        if target is None:
            app.console.print(
                err or f"[red]{name} has no tool named[/red] {sub}   "
                f"(try /{name})")
            return False
        return _run_tool(app, target, args[1:])

    target, err = _resolve_tool_name(app, name)
    if err:
        app.console.print(f"[yellow]{err}[/yellow]")
        return False
    if target is not None:
        return _run_tool(app, target, args)
    return None
