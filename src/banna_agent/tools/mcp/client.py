"""Minimal MCP (Model Context Protocol) client: stdio + HTTP transports.

We speak JSON-RPC 2.0 directly. The handshake is:

    -> initialize          (client capabilities + protocol version)
    <- initialize result   (server info + capabilities)
    -> notifications/initialized
    ... then tools/list and tools/call as needed.

Only the subset the bridge needs is implemented: `initialize`,
`tools/list`, `tools/call`. Everything is synchronous — one request,
block for its response — which matches the agent's synchronous tool
handler contract (`dict -> dict`).
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# Protocol version we advertise. Servers negotiate down if needed; we
# don't hard-fail on a mismatch (the spec allows the server to pick).
PROTOCOL_VERSION = "2025-06-18"
DEFAULT_TIMEOUT_S = 30.0


class McpError(RuntimeError):
    """Any MCP transport / protocol / server-side failure."""


@dataclass
class McpServerConfig:
    """How to reach one MCP server.

    transport == "stdio": `command` + `args` launch the server subprocess;
      `env` is merged into the child environment.
    transport == "http":  `url` is the JSON-RPC endpoint; `headers` carries
      auth (e.g. {"Authorization": "Bearer ..."}).
    """

    name: str
    transport: str = "stdio"            # "stdio" | "http"
    # stdio
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # http
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    # shared
    timeout_s: float = DEFAULT_TIMEOUT_S

    def validate(self) -> None:
        if self.transport == "stdio":
            if not self.command:
                raise McpError(f"mcp server {self.name!r}: stdio transport needs a command")
        elif self.transport == "http":
            if not self.url:
                raise McpError(f"mcp server {self.name!r}: http transport needs a url")
        else:
            raise McpError(
                f"mcp server {self.name!r}: unknown transport {self.transport!r} "
                "(use 'stdio' or 'http')"
            )


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class _Transport:
    """Send one JSON-RPC request, return the parsed result (or raise)."""

    def request(self, method: str, params: dict[str, Any] | None,
                *, timeout_s: float) -> Any:
        raise NotImplementedError

    def notify(self, method: str, params: dict[str, Any] | None) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _StdioTransport(_Transport):
    """Newline-delimited JSON-RPC over a subprocess's stdin/stdout.

    A background reader thread pumps stdout lines into a dict keyed by
    request id, so concurrent batch tool-calls (the agent runs tool
    batches in a thread pool) don't interleave reads on the pipe.
    """

    def __init__(self, cfg: McpServerConfig) -> None:
        import os
        self._cfg = cfg
        child_env = {**os.environ, **cfg.env}
        try:
            self._proc = subprocess.Popen(
                [cfg.command, *cfg.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=child_env,
                text=True,
                bufsize=1,  # line-buffered
            )
        except (OSError, ValueError) as exc:
            raise McpError(
                f"mcp server {cfg.name!r}: failed to launch "
                f"{cfg.command!r}: {exc}"
            ) from exc
        self._next_id = 1
        self._lock = threading.Lock()
        self._pending: dict[int, Any] = {}
        self._cond = threading.Condition()
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # non-JSON server chatter; ignore
            mid = msg.get("id")
            if mid is None:
                continue  # a notification from the server; we don't consume
            with self._cond:
                self._pending[mid] = msg
                self._cond.notify_all()
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _send(self, payload: dict[str, Any]) -> None:
        assert self._proc.stdin is not None
        if self._proc.poll() is not None:
            raise McpError(
                f"mcp server {self._cfg.name!r}: process exited "
                f"(code {self._proc.returncode})"
            )
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise McpError(
                f"mcp server {self._cfg.name!r}: write failed: {exc}"
            ) from exc

    def request(self, method, params, *, timeout_s):
        with self._lock:
            mid = self._next_id
            self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": mid, "method": method,
                    "params": params or {}})
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while mid not in self._pending:
                if self._closed:
                    raise McpError(
                        f"mcp server {self._cfg.name!r}: connection closed "
                        f"awaiting {method!r}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise McpError(
                        f"mcp server {self._cfg.name!r}: timed out after "
                        f"{timeout_s:.0f}s awaiting {method!r}"
                    )
                self._cond.wait(timeout=remaining)
            msg = self._pending.pop(mid)
        if "error" in msg and msg["error"] is not None:
            err = msg["error"]
            raise McpError(
                f"mcp server {self._cfg.name!r}: {method} -> "
                f"{err.get('message', err)}"
            )
        return msg.get("result")

    def notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self):
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            try:
                self._proc.kill()
            except OSError:
                pass


class _HttpTransport(_Transport):
    """JSON-RPC over Streamable HTTP. The server may answer a POST with a
    plain JSON body or an SSE stream (`text/event-stream`); we handle both
    and return the first JSON-RPC response object matching our id."""

    def __init__(self, cfg: McpServerConfig) -> None:
        import requests  # already a dependency
        self._cfg = cfg
        self._requests = requests
        self._session = requests.Session()
        self._next_id = 1
        self._lock = threading.Lock()
        self._mcp_session_id: str | None = None

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self._cfg.headers,
        }
        if self._mcp_session_id:
            h["Mcp-Session-Id"] = self._mcp_session_id
        return h

    def _post(self, payload: dict[str, Any], *, timeout_s: float):
        try:
            resp = self._session.post(
                self._cfg.url, json=payload, headers=self._headers(),
                timeout=timeout_s, stream=True,
            )
        except self._requests.RequestException as exc:
            raise McpError(
                f"mcp server {self._cfg.name!r}: HTTP request failed: {exc}"
            ) from exc
        # Capture a server-assigned session id from the initialize response.
        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self._mcp_session_id = sid
        return resp

    def _extract(self, resp, mid: int) -> Any:
        ctype = resp.headers.get("Content-Type", "")
        if "text/event-stream" in ctype:
            # Parse SSE: lines beginning "data: " carry JSON-RPC frames.
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                body = raw[len("data:"):].strip()
                try:
                    msg = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == mid:
                    return msg
            raise McpError(
                f"mcp server {self._cfg.name!r}: SSE stream ended with no "
                f"response for id {mid}"
            )
        # Plain JSON body.
        if resp.status_code >= 400:
            raise McpError(
                f"mcp server {self._cfg.name!r}: HTTP {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise McpError(
                f"mcp server {self._cfg.name!r}: non-JSON response: "
                f"{resp.text[:200]}"
            ) from exc

    def request(self, method, params, *, timeout_s):
        with self._lock:
            mid = self._next_id
            self._next_id += 1
        resp = self._post(
            {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}},
            timeout_s=timeout_s,
        )
        msg = self._extract(resp, mid)
        if "error" in msg and msg["error"] is not None:
            err = msg["error"]
            raise McpError(
                f"mcp server {self._cfg.name!r}: {method} -> "
                f"{err.get('message', err)}"
            )
        return msg.get("result")

    def notify(self, method, params):
        # Notifications carry no id and expect no body; fire and forget.
        try:
            self._post(
                {"jsonrpc": "2.0", "method": method, "params": params or {}},
                timeout_s=self._cfg.timeout_s,
            )
        except McpError:
            pass

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class McpToolDecl:
    """A tool advertised by an MCP server (declaration only)."""

    name: str
    description: str
    input_schema: dict[str, Any]


class McpSession:
    """A live, initialized connection to one MCP server."""

    def __init__(self, cfg: McpServerConfig, transport: _Transport,
                 server_info: dict[str, Any]) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self._t = transport
        self.server_info = server_info

    def list_tools(self) -> list[McpToolDecl]:
        result = self._t.request("tools/list", {}, timeout_s=self.cfg.timeout_s) or {}
        out: list[McpToolDecl] = []
        for t in result.get("tools", []) or []:
            name = t.get("name")
            if not name:
                continue
            out.append(McpToolDecl(
                name=name,
                description=t.get("description") or "",
                input_schema=t.get("inputSchema") or {"type": "object", "properties": {}},
            ))
        return out

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke one tool. Returns a normalized dict the agent can read.

        MCP returns `{content: [...], isError: bool}`. We flatten text
        content blocks into a single `text` field, keep any structured
        content, and surface `isError` so the driver marks failures.
        """
        result = self._t.request(
            "tools/call", {"name": name, "arguments": arguments or {}},
            timeout_s=self.cfg.timeout_s,
        ) or {}
        blocks = result.get("content", []) or []
        texts: list[str] = []
        structured: list[Any] = []
        for b in blocks:
            btype = b.get("type")
            if btype == "text":
                texts.append(b.get("text", ""))
            elif btype in ("resource", "resource_link"):
                structured.append(b)
            else:
                structured.append(b)
        out: dict[str, Any] = {}
        if texts:
            out["text"] = "\n".join(texts)
        if structured:
            out["content"] = structured
        if result.get("structuredContent") is not None:
            out["structured"] = result["structuredContent"]
        if result.get("isError"):
            out["isError"] = True
        if not out:
            out["text"] = ""
        return out

    def close(self) -> None:
        self._t.close()


def connect(cfg: McpServerConfig) -> McpSession:
    """Launch/reach a server, run the initialize handshake, return a session."""
    cfg.validate()
    if cfg.transport == "stdio":
        transport: _Transport = _StdioTransport(cfg)
    else:
        transport = _HttpTransport(cfg)
    try:
        init = transport.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "banna", "version": "0.2.0"},
            },
            timeout_s=cfg.timeout_s,
        ) or {}
        transport.notify("notifications/initialized", {})
    except McpError:
        transport.close()
        raise
    return McpSession(cfg, transport, server_info=init.get("serverInfo") or {})
