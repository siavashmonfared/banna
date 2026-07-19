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
import subprocess
import sys
from pathlib import Path

from ..tools.mcp import McpServerConfig
from .config_store import config_dir


# ---------------------------------------------------------------------------
# Standard servers — curated MCP servers banna knows how to fetch and
# register on demand (`/mcp install <name>`). Cloned under
# `~/.config/banna/mcp-servers/<repo>` and registered in mcp.json like
# any hand-added server, so `banna config mcp remove <name>` works on
# them too.
# ---------------------------------------------------------------------------

STANDARD_SERVERS: dict[str, dict] = {
    "collab": {
        "repo": "https://github.com/siavashmonfared/agentic-tools.git",
        "repo_dir": "agentic-tools",
        "server_rel": "collab-mcp/server.py",
        "blurb": "shared brainstorm thread for multi-agent workflows "
                 "(agentic-tools)",
        "requires": ("mcp",),   # import names the server needs at runtime
    },
}


def standard_install_dir() -> Path:
    return config_dir() / "mcp-servers"


def standard_server_path(name: str) -> Path | None:
    """Path to the installed server script, or None if not cloned yet."""
    spec = STANDARD_SERVERS.get(name)
    if not spec:
        return None
    p = standard_install_dir() / spec["repo_dir"] / spec["server_rel"]
    return p if p.is_file() else None


def install_standard_server(name: str, *, say=print) -> tuple[bool, str]:
    """Clone (or update) a standard server, check its deps, register it.

    Returns (ok, message). Registration uses the running interpreter so
    the dependency check and the server subprocess agree on environment.
    """
    spec = STANDARD_SERVERS.get(name)
    if not spec:
        return False, f"unknown standard server {name!r} (have: {', '.join(STANDARD_SERVERS)})"

    dest = standard_install_dir() / spec["repo_dir"]
    try:
        if (dest / ".git").is_dir():
            say(f"updating {dest} …")
            subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"],
                           check=True, capture_output=True, text=True, timeout=120)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            say(f"cloning {spec['repo']} → {dest} …")
            subprocess.run(["git", "clone", "--depth", "1", spec["repo"], str(dest)],
                           check=True, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        return False, "git is not installed — install git and retry"
    except subprocess.TimeoutExpired:
        return False, "git timed out — check network and retry"
    except subprocess.CalledProcessError as exc:
        return False, f"git failed: {(exc.stderr or exc.stdout or '').strip()[-300:]}"

    server_py = dest / spec["server_rel"]
    if not server_py.is_file():
        return False, f"clone succeeded but {spec['server_rel']} not found in {dest}"

    # Standard servers run out of a dedicated venv so their deps (the
    # `mcp` SDK etc.) never touch the user's environment — and PEP 668
    # "externally managed" system Pythons can't be pip-installed into
    # anyway.
    venv_dir = standard_install_dir() / ".venv"
    venv_py = venv_dir / "bin" / "python"
    if not venv_py.exists():
        say(f"creating venv {venv_dir} …")
        r = subprocess.run([sys.executable, "-m", "venv", str(venv_dir)],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return False, (f"could not create venv: "
                           f"{(r.stderr or r.stdout).strip()[-300:]}")
    missing = [m for m in spec.get("requires", ())
               if subprocess.run([str(venv_py), "-c", f"import {m}"],
                                 capture_output=True).returncode != 0]
    if missing:
        say(f"installing server deps into venv: {', '.join(missing)} …")
        r = subprocess.run(
            [str(venv_py), "-m", "pip", "install", "--quiet", *missing],
            capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return False, (f"could not pip install {' '.join(missing)}: "
                           f"{(r.stderr or r.stdout).strip()[-300:]}")

    add_stdio_server(name, str(venv_py), [str(server_py)])
    return True, f"{name} installed and registered ({server_py})"


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
