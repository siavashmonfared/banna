"""Python sandbox tool — execute model-emitted code with a timeout.

Execution is delegated to a `SandboxBackend` (see `tools/sandbox.py`):

  * `process` (default) — a host subprocess. Real timeout + memory
    separation, but the subprocess inherits the user's filesystem,
    network, and credentials. This is the behavior the GAIA numbers were
    measured under, kept byte-for-byte.
  * `docker` — a throwaway container with no network, a read-only
    rootfs, dropped capabilities, and resource limits, for untrusted
    input or shared infrastructure.

The threat model for the default backend is "the LLM might write runaway
or wrong code," not "malicious code trying to escape." Pick `docker`
(via --sandbox=docker or BANNA_SANDBOX=docker) when the stronger model
is needed.
"""
from __future__ import annotations

from typing import Any, Callable

from .base import JsonTool
from .package_policy import PackagePolicy
from .sandbox import (
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_TIMEOUT_S,
    DockerBackend,
    SandboxBackend,
    resolve_sandbox_backend,
)


def run_python(code: str, *, timeout_s: float = DEFAULT_TIMEOUT_S,
               max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
               workspace: str | None = None,
               backend: "str | SandboxBackend | None" = None) -> dict[str, Any]:
    """Run `code` and return stdout/stderr/returncode.

    Returned dict shape (JSON-serializable):
        {ok, returncode, stdout, stderr, timeout,
         truncated_stdout, truncated_stderr, wall_s, [error]}

    `backend` selects the isolation policy; None resolves from
    BANNA_SANDBOX (default "process").
    """
    return resolve_sandbox_backend(backend).run_python(
        code, timeout_s=timeout_s, max_output_chars=max_output_chars,
        workspace=workspace,
    )


def _make_handler(backend: "str | SandboxBackend | None"):
    def _handler(args: dict[str, Any]) -> dict[str, Any]:
        code = args.get("code", "")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("'code' must be a non-empty string")
        timeout_s = float(args.get("timeout_s", DEFAULT_TIMEOUT_S))
        return run_python(code, timeout_s=timeout_s, backend=backend)
    return _handler


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


def make_python_sandbox_tool(
    sandbox: "str | SandboxBackend | None" = None,
    *,
    approve_install: "Callable[[str, str], bool] | None" = None,
    package_policy: PackagePolicy | None = None,
    base_image: str | None = None,
) -> JsonTool:
    """Build the run_python tool. `sandbox` selects the isolation backend
    ("process" / "docker" / a backend instance); None resolves from
    BANNA_SANDBOX, defaulting to "process".

    When the resolved backend is a Docker backend, an optional on-demand
    install policy can be attached: `package_policy` (the trusted allowlist),
    `approve_install` (callback for non-allowlisted packages), and a custom
    `base_image`. These are ignored for the process backend, so the default /
    GAIA path (`make_python_sandbox_tool()`) is unchanged.
    """
    backend = resolve_sandbox_backend(sandbox)
    if isinstance(backend, DockerBackend) and (
        package_policy is not None or base_image
    ):
        backend = DockerBackend(
            image=base_image or backend.image,
            memory=backend.memory,
            cpus=backend.cpus,
            pids=backend.pids,
            docker_bin=backend.docker_bin,
            package_policy=package_policy,
            on_unlisted=approve_install,
        )
    return JsonTool(
        name="run_python",
        description=(
            "Execute Python code in a subprocess with a wall-time limit. "
            "Returns stdout, stderr, returncode, and timeout flag."
        ),
        input_schema=PYTHON_SANDBOX_SCHEMA,
        handler=_make_handler(backend),
        capabilities=frozenset({"sandbox", "write", "compute"}),
    )
