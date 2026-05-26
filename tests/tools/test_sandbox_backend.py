"""Sandbox backend selection + Docker command construction.

The Docker tests don't require Docker — they monkeypatch subprocess.run to
capture the argv the backend builds, and assert the hardening flags are
present. One opt-in integration test actually runs a container if Docker
is available.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from banna_agent.tools import sandbox
from banna_agent.tools.sandbox import (
    DockerBackend,
    ProcessBackend,
    resolve_sandbox_backend,
)


# --- resolution -------------------------------------------------------------

def test_resolve_defaults_to_process(monkeypatch) -> None:
    monkeypatch.delenv("BANNA_SANDBOX", raising=False)
    assert resolve_sandbox_backend(None).name == "process"


def test_resolve_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("BANNA_SANDBOX", "docker")
    assert resolve_sandbox_backend(None).name == "docker"


def test_resolve_explicit_arg_beats_env(monkeypatch) -> None:
    monkeypatch.setenv("BANNA_SANDBOX", "docker")
    assert resolve_sandbox_backend("process").name == "process"


def test_resolve_passthrough_instance() -> None:
    b = ProcessBackend()
    assert resolve_sandbox_backend(b) is b


def test_resolve_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        resolve_sandbox_backend("firejail")


# --- process backend (default behavior) ------------------------------------

def test_process_backend_runs_python() -> None:
    r = ProcessBackend().run_python("print(6*7)")
    assert r["ok"] is True
    assert r["stdout"].strip() == "42"


def test_process_backend_python_error_summary() -> None:
    r = ProcessBackend().run_python("import nonexistent_module_xyz")
    assert r["ok"] is False
    assert "error" in r and "Python exited" in r["error"]


# --- docker backend: command construction (no Docker needed) ---------------

def _capture_argv(monkeypatch):
    captured = {}

    class _Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_docker_python_command_is_hardened(monkeypatch, tmp_path) -> None:
    cap = _capture_argv(monkeypatch)
    DockerBackend(image="python:3.12-slim", memory="256m").run_python(
        "print(1)", workspace=str(tmp_path))
    argv = cap["argv"]
    joined = " ".join(argv)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert "--read-only" in argv
    assert "--cap-drop" in argv and argv[argv.index("--cap-drop") + 1] == "ALL"
    assert "--security-opt" in argv and "no-new-privileges" in argv
    assert "--memory" in argv and argv[argv.index("--memory") + 1] == "256m"
    assert "--pids-limit" in argv
    assert "python:3.12-slim" in argv
    # Script is fed over stdin, not written to the read-only rootfs.
    assert argv[-2:] == ["python", "-"]
    assert cap["kwargs"]["input"] == "print(1)"
    # Workspace bind-mounted read-write at /work.
    assert f"{tmp_path.resolve()}:/work:rw" in joined


def test_docker_shell_command_is_hardened(monkeypatch) -> None:
    cap = _capture_argv(monkeypatch)
    DockerBackend().run_shell("ls -la", shell=True)
    argv = cap["argv"]
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    # Command handed to a shell inside the container.
    assert argv[-3:] == ["sh", "-c", "ls -la"]


def test_docker_missing_binary_fails_loudly(monkeypatch) -> None:
    def boom(*a, **k):
        raise FileNotFoundError("docker")
    monkeypatch.setattr(subprocess, "run", boom)
    r = DockerBackend().run_python("print(1)")
    assert r["ok"] is False
    assert "docker not found" in r["stderr"]
    assert "--sandbox=process" in r["stderr"]


# --- opt-in real-container integration -------------------------------------

@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_docker_backend_real_execution_no_network() -> None:
    backend = DockerBackend()
    r = backend.run_python("print('hi from container')", timeout_s=60)
    assert r["ok"] is True
    assert "hi from container" in r["stdout"]
    # Network is disabled: a socket connect must fail.
    net = backend.run_python(
        "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)",
        timeout_s=60,
    )
    assert net["ok"] is False
