"""Python sandbox tool — execute untrusted code in a subprocess with a timeout.

Subprocess isolation (not `exec`) because:
1. GAIA asks the agent to compute on attached XLSX/CSV/PDF — pandas,
   regex, numeric work. A subprocess gives us real timeout + memory
   separation, which `exec` cannot.
2. Any exception / infinite loop / OOM in the sandboxed code dies
   cleanly without taking down the agent loop.

We do **not** try to be a true security sandbox. The subprocess inherits
the user's network and filesystem. The threat model is: "the LLM might
write runaway or wrong code," not "malicious code trying to escape."
For a production deployment, wrap this with firejail, bubblewrap,
nsjail, Docker, or gVisor. We note that in the docstring and move on.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .base import JsonTool


DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_OUTPUT_CHARS = 20_000


def run_python(code: str, *, timeout_s: float = DEFAULT_TIMEOUT_S,
               max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
               workspace: str | None = None) -> dict[str, Any]:
    """Run `code` as a subprocess, return stdout/stderr/returncode.

    Returned dict shape (JSON-serializable):
        {
          "ok": bool,
          "returncode": int,
          "stdout": str,
          "stderr": str,
          "timeout": bool,
          "truncated_stdout": bool,
          "truncated_stderr": bool,
          "wall_s": float,
        }
    """
    import time
    workspace_path = Path(workspace) if workspace else None
    if workspace_path:
        workspace_path.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        dir=str(workspace_path) if workspace_path else None,
    ) as f:
        f.write(code)
        script_path = f.name

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(workspace_path) if workspace_path else None,
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
        ok = proc.returncode == 0
        # When the script failed, prepend a single-line error summary so
        # the agent can't gloss over a silent failure (the cffe0e32 bug:
        # the model fabricated an answer claiming to have parsed a docx
        # via `import docx`, when in reality python-docx was missing and
        # the only signal was a buried ImportError in stderr).
        err_summary: str | None = None
        if not ok:
            first_stderr_line = stderr.strip().splitlines()[-1] if stderr.strip() else ""
            err_summary = (
                f"Python exited with code {proc.returncode}"
                + (f": {first_stderr_line}" if first_stderr_line else " (no stderr)")
            )
        result = {
            "ok": ok,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timeout": False,
            "truncated_stdout": trunc_out,
            "truncated_stderr": trunc_err,
            "wall_s": wall,
        }
        if err_summary:
            result["error"] = err_summary
        return result
    except subprocess.TimeoutExpired as e:
        wall = time.monotonic() - t0
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "ok": False,
            "returncode": -1,
            "stdout": stdout[:max_output_chars],
            "stderr": stderr[:max_output_chars],
            "timeout": True,
            "truncated_stdout": len(stdout) > max_output_chars,
            "truncated_stderr": len(stderr) > max_output_chars,
            "wall_s": wall,
        }
    finally:
        try:
            Path(script_path).unlink(missing_ok=True)
        except OSError:
            pass


def _handler(args: dict[str, Any]) -> dict[str, Any]:
    code = args.get("code", "")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("'code' must be a non-empty string")
    timeout_s = float(args.get("timeout_s", DEFAULT_TIMEOUT_S))
    return run_python(code, timeout_s=timeout_s)


PYTHON_SANDBOX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": (
                "Python source to execute in a subprocess. The script runs "
                "with the project's interpreter. stdout and stderr are "
                "captured and returned. Files you create stay on disk only "
                "if a workspace is configured."
            ),
        },
        "timeout_s": {
            "type": "number",
            "description": f"Wall-time limit in seconds (default {DEFAULT_TIMEOUT_S}).",
            "default": DEFAULT_TIMEOUT_S,
        },
    },
    "required": ["code"],
    "additionalProperties": False,
}


def make_python_sandbox_tool() -> JsonTool:
    return JsonTool(
        name="run_python",
        description=(
            "Execute Python code in a subprocess with a wall-time limit. "
            "Returns stdout, stderr, returncode, and timeout flag."
        ),
        input_schema=PYTHON_SANDBOX_SCHEMA,
        handler=_handler,
        capabilities=frozenset({"sandbox", "write", "compute"}),
    )
