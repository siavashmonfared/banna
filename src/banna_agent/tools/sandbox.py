"""Execution-isolation backends for the code-running tools.

`run_python` and `run_shell` both reduce to "execute something with a
wall-time limit and capture stdio". *How* that something is isolated is a
policy decision the deployer should control, so it lives behind a
`SandboxBackend` interface with two implementations:

  * `ProcessBackend` (default) — a plain subprocess on the host. Real
    timeout + memory separation from the agent loop, but the subprocess
    inherits the user's filesystem, network, and credentials. This is the
    behavior the GAIA validation numbers were measured under; it is kept
    byte-for-byte so those numbers don't move.
  * `DockerBackend` — runs the same code/command inside a throwaway
    container with no network, a read-only root filesystem, dropped
    capabilities, and cpu/memory/pid limits. Suitable for untrusted input
    or shared infrastructure.

Select with `--sandbox {process,docker}` or the `BANNA_SANDBOX` env var.
Adding a third backend (bubblewrap, nsjail, gVisor) is a new subclass and
a new branch in `resolve_sandbox_backend` — no caller changes.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .package_policy import PackagePolicy

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_OUTPUT_CHARS = 20_000

# Docker resource defaults — conservative caps for a single tool call.
DEFAULT_DOCKER_IMAGE = "python:3.12-slim"
DEFAULT_DOCKER_MEMORY = "512m"
DEFAULT_DOCKER_CPUS = "1.0"
DEFAULT_DOCKER_PIDS = 256

# Cap install/rebuild attempts within one run_python call so a script that
# imports a new missing module on every line can't loop forever.
MAX_INSTALL_RETRIES = 5

# `ModuleNotFoundError: No module named 'foo'` / `'foo.bar'`.
_MNFE_RE = re.compile(r"No module named ['\"]([\w][\w.]*)['\"]")


def parse_missing_module(stderr: str) -> str | None:
    """Return the top-level import name from a ModuleNotFoundError, else None.

    `import a.b.c` reports `'a.b.c'` (or `'a.b'`); pip installs distributions,
    so we collapse to the top-level package and let the allowlist map it to a
    pip spec.
    """
    if not stderr:
        return None
    m = _MNFE_RE.search(stderr)
    if not m:
        return None
    return m.group(1).split(".")[0]


# ---------------------------------------------------------------------------
# Result shaping (identical across backends, so the agent sees one schema)
# ---------------------------------------------------------------------------

def _shape(
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    timed_out: bool,
    wall_s: float,
    max_output_chars: int,
    python_err_summary: bool,
) -> dict[str, Any]:
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    stdout = stdout or ""
    stderr = stderr or ""
    trunc_out = len(stdout) > max_output_chars
    trunc_err = len(stderr) > max_output_chars
    ok = (returncode == 0) and not timed_out
    result: dict[str, Any] = {
        "ok": ok,
        "returncode": returncode,
        "stdout": stdout[:max_output_chars],
        "stderr": stderr[:max_output_chars],
        "timeout": timed_out,
        "truncated_stdout": trunc_out,
        "truncated_stderr": trunc_err,
        "wall_s": wall_s,
    }
    # Surface a one-line failure summary so the agent can't gloss over a
    # silent error (the cffe0e32 bug: model fabricated an answer claiming
    # to have parsed a docx when python-docx was actually missing and the
    # only signal was a buried ImportError in stderr).
    if python_err_summary and not ok and not timed_out:
        last = stderr.strip().splitlines()[-1] if stderr.strip() else ""
        result["error"] = (
            f"Python exited with code {returncode}"
            + (f": {last}" if last else " (no stderr)")
        )
    return result


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class SandboxBackend(ABC):
    """Executes model-emitted code/commands under some isolation policy."""

    name: str = "base"

    @abstractmethod
    def run_python(
        self,
        code: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        workspace: str | None = None,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    def run_shell(
        self,
        command: str | list[str],
        *,
        cwd: str | None = None,
        shell: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# ProcessBackend — host subprocess (default; behavior frozen for GAIA parity)
# ---------------------------------------------------------------------------

class ProcessBackend(SandboxBackend):
    """A plain host subprocess. No OS-level isolation beyond process
    separation + a wall-time kill. This is the default and the behavior
    the public GAIA numbers were measured under."""

    name = "process"

    def run_python(self, code, *, timeout_s=DEFAULT_TIMEOUT_S,
                   max_output_chars=DEFAULT_MAX_OUTPUT_CHARS, workspace=None):
        workspace_path = Path(workspace) if workspace else None
        if workspace_path:
            workspace_path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir=str(workspace_path) if workspace_path else None,
        ) as f:
            f.write(code)
            script_path = f.name
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=timeout_s,
                cwd=str(workspace_path) if workspace_path else None,
            )
            return _shape(
                returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
                timed_out=False, wall_s=time.monotonic() - t0,
                max_output_chars=max_output_chars, python_err_summary=True,
            )
        except subprocess.TimeoutExpired as e:
            return _shape(
                returncode=-1, stdout=e.stdout or "", stderr=e.stderr or "",
                timed_out=True, wall_s=time.monotonic() - t0,
                max_output_chars=max_output_chars, python_err_summary=False,
            )
        finally:
            try:
                Path(script_path).unlink(missing_ok=True)
            except OSError:
                pass

    def run_shell(self, command, *, cwd=None, shell=True,
                  timeout_s=DEFAULT_TIMEOUT_S, max_output_chars=DEFAULT_MAX_OUTPUT_CHARS):
        import shlex
        if shell:
            cmd: Any = command if isinstance(command, str) else " ".join(
                shlex.quote(a) for a in command)
        else:
            cmd = command if isinstance(command, list) else shlex.split(command)
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, shell=shell, cwd=cwd, capture_output=True,
                text=True, timeout=timeout_s,
            )
            return _shape(
                returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
                timed_out=False, wall_s=time.monotonic() - t0,
                max_output_chars=max_output_chars, python_err_summary=False,
            )
        except subprocess.TimeoutExpired as e:
            return _shape(
                returncode=-1, stdout=e.stdout or "", stderr=e.stderr or "",
                timed_out=True, wall_s=time.monotonic() - t0,
                max_output_chars=max_output_chars, python_err_summary=False,
            )
        except FileNotFoundError as exc:
            return _shape(
                returncode=-2, stdout="", stderr=f"FileNotFoundError: {exc}",
                timed_out=False, wall_s=time.monotonic() - t0,
                max_output_chars=max_output_chars, python_err_summary=False,
            )


# ---------------------------------------------------------------------------
# DockerBackend — throwaway container with no network + locked-down rootfs
# ---------------------------------------------------------------------------

class DockerBackend(SandboxBackend):
    """Runs each call in a fresh `docker run --rm` container.

    Hardening applied to every invocation:
      --network none          no outbound/inbound network
      --read-only             read-only root filesystem
      --tmpfs /tmp            writable scratch that vanishes with the container
      --cap-drop ALL          drop all Linux capabilities
      --security-opt no-new-privileges
      --memory / --cpus / --pids-limit   resource caps
      --user 1000:1000        non-root

    A workspace, when supplied, is bind-mounted read-write at /work so the
    code can read task attachments and write intermediate files; nothing
    else on the host is visible.
    """

    name = "docker"

    def __init__(
        self,
        *,
        image: str = DEFAULT_DOCKER_IMAGE,
        memory: str = DEFAULT_DOCKER_MEMORY,
        cpus: str = DEFAULT_DOCKER_CPUS,
        pids: int = DEFAULT_DOCKER_PIDS,
        docker_bin: str = "docker",
        package_policy: "PackagePolicy | None" = None,
        on_unlisted: "Callable[[str, str], bool] | None" = None,
        build_timeout_s: float = 300.0,
    ) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids = pids
        self.docker_bin = docker_bin
        # On-demand install policy. When `package_policy is None`, run_python
        # behaves exactly as a bare container run (the GAIA/default path).
        self.package_policy = package_policy
        self.on_unlisted = on_unlisted
        self.build_timeout_s = build_timeout_s
        # The base image stays fixed; `image` rolls forward to the latest
        # derived tag as packages are installed this session. Builds always
        # layer the full accumulated pin set onto the base, so the cache tag
        # is deterministic for a given (base, pin set).
        self._base_image = image
        self._installed_pins: set[str] = set()

    def _base_argv(self, *, workspace: str | None, interactive: bool) -> list[str]:
        argv = [
            self.docker_bin, "run", "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:rw,exec,size=256m",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", str(self.pids),
            "--user", "1000:1000",
        ]
        if interactive:
            argv.append("-i")  # keep stdin open for `python -`
        if workspace:
            ws = Path(workspace)
            ws.mkdir(parents=True, exist_ok=True)
            argv += ["-v", f"{ws.resolve()}:/work:rw", "-w", "/work"]
        return argv

    def _execute(self, argv, *, stdin, timeout_s, max_output_chars, python_err_summary):
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                argv, input=stdin, capture_output=True, text=True,
                timeout=timeout_s,
            )
            return _shape(
                returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
                timed_out=False, wall_s=time.monotonic() - t0,
                max_output_chars=max_output_chars, python_err_summary=python_err_summary,
            )
        except subprocess.TimeoutExpired as e:
            return _shape(
                returncode=-1, stdout=e.stdout or "", stderr=e.stderr or "",
                timed_out=True, wall_s=time.monotonic() - t0,
                max_output_chars=max_output_chars, python_err_summary=False,
            )
        except FileNotFoundError:
            # docker binary not on PATH — fail loudly rather than silently
            # falling back to the unisolated host.
            return _shape(
                returncode=-2, stdout="",
                stderr=(
                    f"docker not found ({self.docker_bin!r} is not on PATH). "
                    "Install Docker, or run with --sandbox=process."
                ),
                timed_out=False, wall_s=time.monotonic() - t0,
                max_output_chars=max_output_chars, python_err_summary=False,
            )

    def _run_python_once(self, code, *, timeout_s, max_output_chars, workspace):
        # Feed the script over stdin (`python -`) so we never have to write
        # it onto the read-only rootfs.
        argv = self._base_argv(workspace=workspace, interactive=True)
        argv += [self.image, "python", "-"]
        return self._execute(argv, stdin=code, timeout_s=timeout_s,
                             max_output_chars=max_output_chars, python_err_summary=True)

    def run_python(self, code, *, timeout_s=DEFAULT_TIMEOUT_S,
                   max_output_chars=DEFAULT_MAX_OUTPUT_CHARS, workspace=None):
        result = self._run_python_once(
            code, timeout_s=timeout_s, max_output_chars=max_output_chars,
            workspace=workspace)
        # No install policy → behave exactly like a bare container run.
        if self.package_policy is None:
            return result

        from .docker_images import build_derived_image

        retries = 0
        while not result["ok"] and retries < MAX_INSTALL_RETRIES:
            missing = parse_missing_module(result.get("stderr", ""))
            if missing is None:
                return result

            spec = self.package_policy.resolve(missing)
            if spec is None:
                # Not allowlisted. Ask a human if one is attached; otherwise
                # (GAIA/batch, or a deny) surface the import error unchanged.
                pip_spec = missing  # import name == pip name, unpinned (v1)
                if self.on_unlisted is None or not self.on_unlisted(missing, pip_spec):
                    return result
                self.package_policy.approve_session(missing, pip_spec)
                spec = pip_spec

            # Phase 1: build a derived image (network ON, no model code) with
            # the full accumulated pin set layered onto the base image.
            self._installed_pins.add(spec)
            ok, tag, log = build_derived_image(
                self._base_image, sorted(self._installed_pins),
                docker_bin=self.docker_bin, timeout_s=self.build_timeout_s,
            )
            if not ok:
                self._installed_pins.discard(spec)
                extra = f"\n[banna] package install failed for {spec!r}:\n{log}"
                result["stderr"] = (result.get("stderr", "") + extra)[:max_output_chars]
                return result

            # Phase 2: re-run the untrusted code in the derived image, still
            # `--network none`.
            self.image = tag
            retries += 1
            result = self._run_python_once(
                code, timeout_s=timeout_s, max_output_chars=max_output_chars,
                workspace=workspace)
        return result

    def run_shell(self, command, *, cwd=None, shell=True,
                  timeout_s=DEFAULT_TIMEOUT_S, max_output_chars=DEFAULT_MAX_OUTPUT_CHARS):
        import shlex
        if isinstance(command, list):
            cmd_str = " ".join(shlex.quote(a) for a in command)
        else:
            cmd_str = command
        # `cwd` maps to the mounted workspace inside the container.
        argv = self._base_argv(workspace=cwd, interactive=False)
        argv += [self.image, "sh", "-c", cmd_str]
        return self._execute(argv, stdin=None, timeout_s=timeout_s,
                             max_output_chars=max_output_chars, python_err_summary=False)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

_PROCESS_SINGLETON = ProcessBackend()


def resolve_sandbox_backend(
    mode: "str | SandboxBackend | None" = None,
) -> SandboxBackend:
    """Return a backend for `mode`.

    `mode` may be a backend instance (returned as-is), a string
    ("process" / "docker"), or None — in which case the `BANNA_SANDBOX`
    env var is consulted, defaulting to "process".
    """
    if isinstance(mode, SandboxBackend):
        return mode
    name = (mode or os.environ.get("BANNA_SANDBOX") or "process").strip().lower()
    if name in ("", "process", "host", "none"):
        return _PROCESS_SINGLETON
    if name == "docker":
        image = os.environ.get("BANNA_SANDBOX_IMAGE")
        return DockerBackend(image=image) if image else DockerBackend()
    raise ValueError(
        f"unknown sandbox mode: {name!r} (expected 'process' or 'docker')"
    )
