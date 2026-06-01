"""Bridge MCP server tools into the agent's `JsonTool` registry.

`bridge_session` turns each tool an MCP server advertises into a
`JsonTool` whose handler proxies to the server. `McpManager` owns the
sessions for one CLI run: it connects every configured server, collects
the bridged tools, and shuts the servers down at exit. A server that
fails to start logs a warning and is skipped — it never crashes the loop.
"""
from __future__ import annotations

from typing import Any, Callable

from ..base import JsonTool
from .client import McpError, McpServerConfig, McpSession, connect


def _bridge_name(server: str, tool: str, *, prefix: bool) -> str:
    """Namespaced tool name. With `prefix`, `collab_start` on server
    `collab` becomes `collab.collab_start`, avoiding collisions with
    built-ins or other servers."""
    return f"{server}.{tool}" if prefix else tool


def bridge_session(
    session: McpSession,
    *,
    prefix: bool = True,
    capabilities: frozenset[str] = frozenset({"mcp", "network"}),
) -> list[JsonTool]:
    """Return one `JsonTool` per tool the server advertises.

    The handler captures `session` and the *remote* tool name, calling
    `tools/call` over the live transport. MCP tools are external code, so
    they carry the `mcp` + `network` capabilities — the CLI routes those
    through the same permission gate as `run_shell`.
    """
    tools: list[JsonTool] = []
    for decl in session.list_tools():
        remote_name = decl.name
        local_name = _bridge_name(session.name, remote_name, prefix=prefix)

        def _make_handler(remote: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
            def _handler(args: dict[str, Any]) -> dict[str, Any]:
                return session.call_tool(remote, args or {})
            return _handler

        desc = decl.description or f"{remote_name} (via MCP server {session.name})"
        tools.append(JsonTool(
            name=local_name,
            description=desc,
            input_schema=decl.input_schema,
            handler=_make_handler(remote_name),
            capabilities=capabilities,
        ))
    return tools


class McpManager:
    """Owns the MCP sessions for one CLI run.

    Usage:
        mgr = McpManager(configs, warn=console.print)
        mgr.start_all()
        registry_tools = mgr.tools()
        ...
        mgr.close_all()
    """

    def __init__(
        self,
        configs: list[McpServerConfig],
        *,
        prefix: bool = True,
        warn: Callable[[str], None] | None = None,
    ) -> None:
        self._configs = configs
        self._prefix = prefix
        self._warn = warn or (lambda _msg: None)
        self._sessions: list[McpSession] = []
        self._tools: list[JsonTool] = []

    def start_all(self) -> None:
        """Connect every configured server. A failure on one server is
        logged and skipped — the others (and the agent) keep working."""
        for cfg in self._configs:
            try:
                sess = connect(cfg)
            except McpError as exc:
                self._warn(f"[mcp] skipping {cfg.name!r}: {exc}")
                continue
            self._sessions.append(sess)
            try:
                bridged = bridge_session(sess, prefix=self._prefix)
            except McpError as exc:
                self._warn(f"[mcp] {cfg.name!r}: tools/list failed: {exc}")
                continue
            self._tools.extend(bridged)

    def tools(self) -> list[JsonTool]:
        return list(self._tools)

    def server_count(self) -> int:
        return len(self._sessions)

    def close_all(self) -> None:
        for sess in self._sessions:
            try:
                sess.close()
            except Exception:
                pass
        self._sessions.clear()
        self._tools.clear()
