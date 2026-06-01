"""Tests for MCP server config persistence (~/.config/banna/mcp.json)."""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Reload modules so config_dir() picks up the patched env at import time
    # of any cached path (config_dir reads the env live, so just clear file).
    from banna_agent.cli import config_store, mcp_config
    importlib.reload(config_store)
    importlib.reload(mcp_config)
    return mcp_config


def test_add_and_load_stdio(isolated_config):
    m = isolated_config
    m.add_stdio_server("collab", "python3", ["/path/server.py"])
    servers = m.read_mcp_servers()
    assert "collab" in servers
    assert servers["collab"]["transport"] == "stdio"
    assert servers["collab"]["command"] == "python3"
    assert servers["collab"]["args"] == ["/path/server.py"]

    configs = m.load_mcp_configs()
    assert len(configs) == 1
    assert configs[0].name == "collab"
    assert configs[0].command == "python3"
    assert configs[0].args == ["/path/server.py"]


def test_add_http(isolated_config):
    m = isolated_config
    m.add_http_server("remote", "https://example.com/mcp",
                      headers={"Authorization": "Bearer x"})
    cfgs = {c.name: c for c in m.load_mcp_configs()}
    assert cfgs["remote"].transport == "http"
    assert cfgs["remote"].url == "https://example.com/mcp"
    assert cfgs["remote"].headers["Authorization"] == "Bearer x"


def test_remove(isolated_config):
    m = isolated_config
    m.add_stdio_server("a", "cmd", [])
    m.add_stdio_server("b", "cmd", [])
    assert m.remove_server("a") is True
    assert set(m.read_mcp_servers()) == {"b"}
    assert m.remove_server("nope") is False


def test_empty_when_unset(isolated_config):
    assert isolated_config.read_mcp_servers() == {}
    assert isolated_config.load_mcp_configs() == []
