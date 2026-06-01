"""MCP (Model Context Protocol) client + tool bridge.

Lets the agent use tools served by an external MCP server as if they were
native `JsonTool`s. Two transports are supported:

  * **stdio** — the server is launched as a subprocess; JSON-RPC messages
    are newline-delimited over its stdin/stdout. This is what most local
    MCP servers (e.g. the collab server) use.
  * **http** — the server is reached over Streamable HTTP: JSON-RPC is
    POSTed to a URL; the response is either a JSON body or an SSE stream.

The public surface is small:

  * `McpServerConfig` — how to reach one server (transport + launch info).
  * `connect(config)` -> `McpSession` — handshake and return a live session.
  * `bridge_session(session, *, prefix, gated)` -> `list[JsonTool]` — turn
    the server's advertised tools into `JsonTool`s the registry accepts.
  * `McpManager` — owns several sessions for a CLI run; `start_all()` /
    `tools()` / `close_all()`.

Nothing here imports a third-party MCP SDK: the protocol is small and we
speak it directly over stdio (stdlib `subprocess`) or HTTP (`requests`,
already a dependency).
"""
from __future__ import annotations

from .client import (
    McpError,
    McpServerConfig,
    McpSession,
    connect,
)
from .bridge import McpManager, bridge_session

__all__ = [
    "McpError",
    "McpServerConfig",
    "McpSession",
    "connect",
    "bridge_session",
    "McpManager",
]
