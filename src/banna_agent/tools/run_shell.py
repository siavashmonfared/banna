"""Shell execution tool.

Same philosophy as `tools/python_sandbox.py` — not a security sandbox.
Subprocess with timeout + captured stdio. Security is the deployer's
problem (firejail, bubblewrap, Docker, gVisor, etc.).

Defaults prefer safety:
  - shell=False, so `command` is an argv list, not a shell string.
  - If shell=True is explicitly passed, `command` may be a string and
    the subprocess uses the default system shell. The caller takes
    responsibility; this is how models ordinarily phrase commands.

Permission gate (interactive use):
  - The factory accepts an optional `confirm: (command, matched) -> bool`
    callback. When the command matches a risky pattern (pip install,
    apt install, sudo, rm -rf, curl|sh, etc.), the handler calls
    `confirm` and refuses if it returns False. When `confirm` is None
    (e.g. in headless GAIA runs), the tool runs without prompting —
    same behavior as before.

Workspace:
  - If `cwd` is passed, the subprocess runs there. Otherwise, current cwd.
  - Environment variables aren't sanitized; the subprocess inherits the
    agent's env. Callers should scrub secrets from env before running
    an untrusted workspace.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import time
from typing import Any, Callable

from .base import JsonTool


DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_OUTPUT_CHARS = 20_000


# ---------------------------------------------------------------------------
# Permission gate
# ---------------------------------------------------------------------------

# Patterns that warrant a prompt before execution. Tuned to catch the common
# install / sudo / destructive-delete cases without false-positive-ing on
# everyday commands. `re.search` semantics — anchor with \b where needed.
RISKY_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\bpip[0-9]?\s+(install|uninstall)\b"),
    re.compile(r"\b(?:python[0-9.]*\s+-m\s+)?pip\s+(install|uninstall)\b"),
    re.compile(r"\bapt(?:-get)?\s+(install|remove|purge|upgrade)\b"),
    re.compile(r"\b(?:dnf|yum)\s+(install|remove|update)\b"),
    re.compile(r"\bbrew\s+(install|uninstall|upgrade)\b"),
    re.compile(r"\bnpm\s+(install|uninstall|i\b)"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\brm\s+(?:-[a-zA-Z]*r[a-zA-Z]*|--recursive)\b"),
    re.compile(r"\b(?:chmod|chown)\s+(?:-R|[0-9])"),
    re.compile(r"\b(?:curl|wget)\b.*\|\s*(?:sh|bash|zsh|python[0-9.]*)"),
    re.compile(r"\bgit\s+push\b.*(?:--force|-f\b)"),
    re.compile(r">\s*/dev/(?:sd|nvme|hd)"),
    re.compile(r"\bmkfs\."),
    re.compile(r"\bdd\s+.*\bof=/dev/"),
)


def is_risky(cmd: str) -> tuple[bool, str]:
    """Return (True, matched_text) if `cmd` matches any risky pattern."""
    for p in RISKY_PATTERNS:
        m = p.search(cmd)
        if m:
            return True, m.group(0)
    return False, ""


# Type alias for the confirm callback the factory accepts.
ConfirmFn = Callable[[str, str], bool]


def run_shell(
    command: str | list[str],
    *,
    cwd: str | None = None,
    shell: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> dict[str, Any]:
    """Run a shell command. Returns {ok, returncode, stdout, stderr, timeout, wall_s}."""
    if shell:
        cmd: Any = command if isinstance(command, str) else " ".join(shlex.quote(a) for a in command)
    else:
        cmd = command if isinstance(command, list) else shlex.split(command)

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            shell=shell,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        wall = time.monotonic() - t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        trunc_out = len(stdout) > max_output_chars
        trunc_err = len(stderr) > max_output_chars
        if trunc_out:
            stdout = stdout[:max_output_chars]
        if trunc_err:
            stderr = stderr[:max_output_chars]
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timeout": False,
            "truncated_stdout": trunc_out,
            "truncated_stderr": trunc_err,
            "wall_s": wall,
        }
    except subprocess.TimeoutExpired as e:
        wall = time.monotonic() - t0
        out = e.stdout or ""
        err = e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", errors="replace")
        return {
            "ok": False,
            "returncode": -1,
            "stdout": out[:max_output_chars],
            "stderr": err[:max_output_chars],
            "timeout": True,
            "truncated_stdout": len(out) > max_output_chars,
            "truncated_stderr": len(err) > max_output_chars,
            "wall_s": wall,
        }
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "returncode": -2,
            "stdout": "",
            "stderr": f"FileNotFoundError: {exc}",
            "timeout": False,
            "truncated_stdout": False,
            "truncated_stderr": False,
            "wall_s": time.monotonic() - t0,
        }


def _make_handler(confirm: ConfirmFn | None):
    def _handler(args: dict[str, Any]) -> dict[str, Any]:
        command = args.get("command")
        if not command or not isinstance(command, (str, list)):
            raise ValueError("'command' must be a non-empty string or list")

        # Render to a single string for risk inspection regardless of form.
        if isinstance(command, list):
            cmd_str = " ".join(shlex.quote(a) for a in command)
        else:
            cmd_str = command

        if confirm is not None:
            risky, matched = is_risky(cmd_str)
            if risky and not confirm(cmd_str, matched):
                return {
                    "ok": False,
                    "returncode": -3,
                    "stdout": "",
                    "stderr": (
                        f"command denied by user (matched risky pattern: {matched!r}). "
                        "Re-run with explicit user approval, or pick a less invasive approach."
                    ),
                    "timeout": False,
                    "truncated_stdout": False,
                    "truncated_stderr": False,
                    "wall_s": 0.0,
                    "denied": True,
                }

        return run_shell(
            command,
            cwd=args.get("cwd") or None,
            shell=bool(args.get("shell", True)),
            timeout_s=float(args.get("timeout_s", DEFAULT_TIMEOUT_S)),
        )

    return _handler


RUN_SHELL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": ["string", "array"],
            "items": {"type": "string"},
            "description": (
                "Command to run. Either a shell string (shell=true) or an "
                "argv list (shell=false). Prefer argv lists when the model "
                "knows the binary path."
            ),
        },
        "cwd": {
            "type": "string",
            "description": "Working directory for the subprocess.",
        },
        "shell": {
            "type": "boolean",
            "description": "Run through a shell (default true). Set false to pass argv directly.",
            "default": True,
        },
        "timeout_s": {
            "type": "number",
            "description": f"Wall-time limit in seconds (default {DEFAULT_TIMEOUT_S}).",
            "default": DEFAULT_TIMEOUT_S,
        },
    },
    "required": ["command"],
    "additionalProperties": False,
}


def make_run_shell_tool(confirm: ConfirmFn | None = None) -> JsonTool:
    """Build the run_shell tool.

    `confirm`, when provided, is a `(command, matched_pattern) -> bool`
    callback invoked before any *risky* command (pip install, sudo, rm
    -rf, …) runs. Return True to allow, False to deny. None means no
    gate — fine for headless GAIA / batch use, dangerous for
    interactive REPLs.
    """
    return JsonTool(
        name="run_shell",
        description=(
            "Execute a shell command in a subprocess with a wall-time limit. "
            "Returns stdout, stderr, returncode, and timeout flag. Not a "
            "security sandbox — don't run untrusted input without OS-level isolation. "
            "Privileged operations (pip install, sudo, rm -rf, etc.) require "
            "explicit user approval in interactive use."
        ),
        input_schema=RUN_SHELL_SCHEMA,
        handler=_make_handler(confirm),
        capabilities=frozenset({"sandbox", "shell", "write"}),
    )
