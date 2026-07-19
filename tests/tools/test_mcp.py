"""MCP client + bridge tests.

The stdio path is exercised end-to-end against a tiny in-repo fake MCP
server (a Python script speaking the JSON-RPC subset over stdin/stdout),
so we test the real subprocess + framing, not a mock. The bridge and
manager are tested against that live session.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from banna_agent.tools.mcp import (
    McpError,
    McpManager,
    McpServerConfig,
    bridge_session,
    connect,
)

# A minimal MCP server: initialize -> tools/list -> tools/call (echo).
_FAKE_SERVER = textwrap.dedent('''
    import json, sys
    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n"); sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        mid = msg.get("id"); method = msg.get("method")
        if method == "initialize":
            send({"jsonrpc":"2.0","id":mid,"result":{
                "protocolVersion":"2025-06-18",
                "serverInfo":{"name":"fake","version":"0.0.1"},
                "capabilities":{"tools":{}}}})
        elif method == "notifications/initialized":
            pass  # notification, no reply
        elif method == "tools/list":
            send({"jsonrpc":"2.0","id":mid,"result":{"tools":[
                {"name":"echo","description":"Echo back text.",
                 "inputSchema":{"type":"object",
                    "properties":{"text":{"type":"string"}},
                    "required":["text"]}}]}})
        elif method == "tools/call":
            params = msg.get("params") or {}
            args = params.get("arguments") or {}
            if params.get("name") == "echo":
                send({"jsonrpc":"2.0","id":mid,"result":{
                    "content":[{"type":"text","text":"echo: "+str(args.get("text",""))}],
                    "isError":False}})
            else:
                send({"jsonrpc":"2.0","id":mid,"error":{"code":-32601,"message":"no such tool"}})
        else:
            send({"jsonrpc":"2.0","id":mid,"error":{"code":-32601,"message":"unknown method"}})
''')


@pytest.fixture()
def fake_server(tmp_path: Path) -> Path:
    p = tmp_path / "fake_mcp_server.py"
    p.write_text(_FAKE_SERVER)
    return p


def _cfg(fake_server: Path) -> McpServerConfig:
    return McpServerConfig(
        name="fake", transport="stdio",
        command=sys.executable, args=[str(fake_server)], timeout_s=10.0,
    )


def test_connect_and_list_tools(fake_server):
    sess = connect(_cfg(fake_server))
    try:
        assert sess.server_info.get("name") == "fake"
        tools = sess.list_tools()
        assert [t.name for t in tools] == ["echo"]
        assert tools[0].input_schema["required"] == ["text"]
    finally:
        sess.close()


def test_call_tool_flattens_text(fake_server):
    sess = connect(_cfg(fake_server))
    try:
        out = sess.call_tool("echo", {"text": "hi"})
        assert out["text"] == "echo: hi"
        assert "isError" not in out
    finally:
        sess.close()


def test_bridge_produces_gated_jsontools(fake_server):
    sess = connect(_cfg(fake_server))
    try:
        tools = bridge_session(sess, prefix=True)
        assert len(tools) == 1
        t = tools[0]
        assert t.name == "fake.echo"           # namespaced
        assert "mcp" in t.capabilities
        # The handler proxies to the live server.
        result = t.handler({"text": "world"})
        assert result["text"] == "echo: world"
    finally:
        sess.close()


def test_bridge_no_prefix(fake_server):
    sess = connect(_cfg(fake_server))
    try:
        tools = bridge_session(sess, prefix=False)
        assert tools[0].name == "echo"
    finally:
        sess.close()


def test_manager_start_collect_close(fake_server):
    mgr = McpManager([_cfg(fake_server)], prefix=True)
    mgr.start_all()
    try:
        assert mgr.server_count() == 1
        names = [t.name for t in mgr.tools()]
        assert names == ["fake.echo"]
    finally:
        mgr.close_all()
    assert mgr.server_count() == 0


def test_manager_skips_broken_server(fake_server):
    warnings: list[str] = []
    good = _cfg(fake_server)
    bad = McpServerConfig(name="bad", transport="stdio",
                          command="/nonexistent/binary_xyz", args=[], timeout_s=5.0)
    mgr = McpManager([bad, good], warn=warnings.append)
    mgr.start_all()
    try:
        # Bad server skipped with a warning; good server still works.
        assert mgr.server_count() == 1
        assert any("bad" in w for w in warnings)
        assert [t.name for t in mgr.tools()] == ["fake.echo"]
    finally:
        mgr.close_all()


def test_manager_statuses(fake_server):
    good = _cfg(fake_server)
    bad = McpServerConfig(name="bad", transport="stdio",
                          command="/nonexistent/binary_xyz", args=[], timeout_s=5.0)
    mgr = McpManager([bad, good])
    mgr.start_all()
    try:
        by_name = {s["name"]: s for s in mgr.statuses()}
        assert by_name["bad"]["state"] == "failed"
        assert by_name["bad"]["error"]
        assert by_name["bad"]["tools"] == []
        assert by_name["fake"]["state"] == "connected"
        assert by_name["fake"]["error"] is None
        assert by_name["fake"]["tools"] == ["fake.echo"]
        assert by_name["fake"]["transport"] == "stdio"
    finally:
        mgr.close_all()


def test_config_validation():
    with pytest.raises(McpError):
        connect(McpServerConfig(name="x", transport="stdio", command=None))
    with pytest.raises(McpError):
        connect(McpServerConfig(name="x", transport="http", url=None))
    with pytest.raises(McpError):
        connect(McpServerConfig(name="x", transport="carrier-pigeon"))
