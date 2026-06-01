"""Persistent registry of MCP servers for the `banna` CLI.

Stored as JSON at `~/.config/banna/mcp.json` (the hand-rolled TOML writer
in `config_store` can't represent the nested arrays/objects an MCP server
entry needs, so MCP config gets its own file). Shape:

    {
      "servers": {
        "collab": {
          "transport": "stdio",
          "command": "python3",
          "args": ["/path/to/server.py"],
          "env": {}
        },
        "remote": {
          "transport": "http",
          "url": "https://example.com/mcp",
          "headers": {"Authorization": "Bearer ..."}
        }
      }
    }
"""
from __future__ import annotations

import json
from pathlib import Path

from ..tools.mcp import McpServerConfig
from .config_store import config_dir


def mcp_config_path() -> Path:
    return config_dir() / "mcp.json"


def _read_raw() -> dict:
    p = mcp_config_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_raw(data: dict) -> Path:
    p = mcp_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return p


def read_mcp_servers() -> dict[str, dict]:
    """Return `{name: entry_dict}` for every configured server."""
    servers = _read_raw().get("servers") or {}
    return {k: v for k, v in servers.items() if isinstance(v, dict)}


def load_mcp_configs() -> list[McpServerConfig]:
    """Materialize the stored entries into `McpServerConfig` objects."""
    out: list[McpServerConfig] = []
    for name, entry in read_mcp_servers().items():
        out.append(McpServerConfig(
            name=name,
            transport=entry.get("transport", "stdio"),
            command=entry.get("command"),
            args=list(entry.get("args") or []),
            env=dict(entry.get("env") or {}),
            url=entry.get("url"),
            headers=dict(entry.get("headers") or {}),
            timeout_s=float(entry.get("timeout_s", 30.0)),
        ))
    return out


def add_stdio_server(name: str, command: str, args: list[str],
                     *, env: dict[str, str] | None = None) -> Path:
    data = _read_raw()
    servers = dict(data.get("servers") or {})
    servers[name] = {
        "transport": "stdio",
        "command": command,
        "args": list(args),
        "env": dict(env or {}),
    }
    data["servers"] = servers
    return _write_raw(data)


def add_http_server(name: str, url: str,
                    *, headers: dict[str, str] | None = None) -> Path:
    data = _read_raw()
    servers = dict(data.get("servers") or {})
    servers[name] = {
        "transport": "http",
        "url": url,
        "headers": dict(headers or {}),
    }
    data["servers"] = servers
    return _write_raw(data)


def remove_server(name: str) -> bool:
    data = _read_raw()
    servers = dict(data.get("servers") or {})
    if name not in servers:
        return False
    del servers[name]
    data["servers"] = servers
    _write_raw(data)
    return True
