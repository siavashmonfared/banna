"""On-demand package install for the Docker sandbox (two-phase, allowlisted).

These tests don't require Docker: `subprocess.run` is monkeypatched to script
the container runs, and `build_derived_image` is monkeypatched to capture the
pins that would be installed (no real `docker build`). One opt-in integration
test actually builds + runs a container if Docker is available.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from banna_agent.tools import docker_images
from banna_agent.tools.package_policy import PackagePolicy
from banna_agent.tools.sandbox import (
    DEFAULT_DOCKER_IMAGE,
    MAX_INSTALL_RETRIES,
    DockerBackend,
    parse_missing_module,
)


# --- pure helpers -----------------------------------------------------------

def test_parse_missing_module_basic() -> None:
    assert parse_missing_module("ModuleNotFoundError: No module named 'pandas'") == "pandas"


def test_parse_missing_module_dotted_takes_toplevel() -> None:
    assert parse_missing_module("No module named 'google.protobuf'") == "google"


def test_parse_missing_module_none_on_clean() -> None:
    assert parse_missing_module("") is None
    assert parse_missing_module("Traceback: ValueError: nope") is None


def test_derived_image_tag_sorted_and_stable() -> None:
    a = docker_images.derived_image_tag("python:3.12-slim", ["a==1", "b==2"])
    b = docker_images.derived_image_tag("python:3.12-slim", ["b==2", "a==1", "a==1"])
    assert a == b
    assert a.startswith("banna-sbx:")
    # Base image folds into the key.
    assert a != docker_images.derived_image_tag("python:3.13-slim", ["a==1", "b==2"])


def test_render_dockerfile_pins_sorted() -> None:
    df = docker_images.render_dockerfile("python:3.12-slim", ["b==2", "a==1"])
    assert df.startswith("FROM python:3.12-slim\n")
    assert "RUN pip install --no-cache-dir a==1 b==2" in df


# --- build argv (no Docker needed) ------------------------------------------

class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_build_uses_network_run_does_not(monkeypatch) -> None:
    captured: list[list[str]] = []

    def fake(argv, **kwargs):
        captured.append(argv)
        if "inspect" in argv:
            return _Proc(returncode=1)  # not cached → build
        return _Proc(returncode=0, stdout="built")

    monkeypatch.setattr(docker_images.subprocess, "run", fake)
    ok, tag, _log = docker_images.build_derived_image(
        "python:3.12-slim", ["six==1.16.0"], docker_bin="docker")
    assert ok is True
    build_argv = next(a for a in captured if "build" in a)
    # The *build* gets network; only the run container is `--network none`.
    assert "--network" not in build_argv
    assert "-t" in build_argv and tag in build_argv


# --- install loop in DockerBackend.run_python -------------------------------

def _script_runs(monkeypatch, results: list[_Proc]):
    """Monkeypatch subprocess.run (the per-call container run) to yield
    `results` in order, repeating the last. Returns the captured run argvs."""
    captured: list[list[str]] = []
    it = iter(results)
    last = results[-1]

    def fake(argv, **kwargs):
        captured.append(argv)
        try:
            return next(it)
        except StopIteration:
            return last

    monkeypatch.setattr(subprocess, "run", fake)
    return captured


def _fake_build(calls: list[dict], *, ok=True, tag="banna-sbx:deadbeef"):
    def fake_build(base_image, pins, *, docker_bin="docker", timeout_s=300.0):
        calls.append({"base": base_image, "pins": list(pins)})
        return (ok, tag, "" if ok else "build log: boom")
    return fake_build


def test_allowlist_hit_auto_installs_no_userio(monkeypatch) -> None:
    runs = _script_runs(monkeypatch, [
        _Proc(returncode=1, stderr="ModuleNotFoundError: No module named 'cv2'"),
        _Proc(returncode=0, stdout="ok"),
    ])
    builds: list[dict] = []
    monkeypatch.setattr(docker_images, "build_derived_image", _fake_build(builds))

    backend = DockerBackend(
        package_policy=PackagePolicy(allowlist={"cv2": "opencv-python==4.10.0.84"}),
        on_unlisted=None,  # no human, but allowlisted → still installs
    )
    r = backend.run_python("import cv2", timeout_s=5)

    assert r["ok"] is True
    assert builds and "opencv-python==4.10.0.84" in builds[0]["pins"]
    # Every container run stays network-less.
    assert all("--network" in a and a[a.index("--network") + 1] == "none"
               for a in runs)
    assert backend.image == "banna-sbx:deadbeef"


def test_unlisted_without_userio_denies(monkeypatch) -> None:
    _script_runs(monkeypatch, [
        _Proc(returncode=1, stderr="ModuleNotFoundError: No module named 'foo'"),
    ])
    builds: list[dict] = []
    monkeypatch.setattr(docker_images, "build_derived_image", _fake_build(builds))

    backend = DockerBackend(package_policy=PackagePolicy(), on_unlisted=None)
    r = backend.run_python("import foo", timeout_s=5)

    assert r["ok"] is False
    assert not builds  # never tried to install
    assert "error" in r  # the silent-failure summary is preserved


def test_unlisted_with_userio_approves(monkeypatch) -> None:
    _script_runs(monkeypatch, [
        _Proc(returncode=1, stderr="ModuleNotFoundError: No module named 'foo'"),
        _Proc(returncode=0, stdout="ok"),
    ])
    builds: list[dict] = []
    monkeypatch.setattr(docker_images, "build_derived_image", _fake_build(builds))
    asked: list[tuple[str, str]] = []

    def approve(import_name, pip_spec):
        asked.append((import_name, pip_spec))
        return True

    backend = DockerBackend(package_policy=PackagePolicy(), on_unlisted=approve)
    r = backend.run_python("import foo", timeout_s=5)

    assert r["ok"] is True
    assert asked == [("foo", "foo")]
    assert builds and "foo" in builds[0]["pins"]


def test_unlisted_with_userio_denies(monkeypatch) -> None:
    _script_runs(monkeypatch, [
        _Proc(returncode=1, stderr="ModuleNotFoundError: No module named 'foo'"),
    ])
    builds: list[dict] = []
    monkeypatch.setattr(docker_images, "build_derived_image", _fake_build(builds))

    backend = DockerBackend(package_policy=PackagePolicy(),
                            on_unlisted=lambda i, s: False)
    r = backend.run_python("import foo", timeout_s=5)
    assert r["ok"] is False
    assert not builds


def test_import_name_to_pip_name_mapping(monkeypatch) -> None:
    _script_runs(monkeypatch, [
        _Proc(returncode=1, stderr="No module named 'sklearn'"),
        _Proc(returncode=0, stdout="ok"),
    ])
    builds: list[dict] = []
    monkeypatch.setattr(docker_images, "build_derived_image", _fake_build(builds))

    backend = DockerBackend(
        package_policy=PackagePolicy(allowlist={"sklearn": "scikit-learn==1.5.2"}))
    backend.run_python("import sklearn", timeout_s=5)
    assert "scikit-learn==1.5.2" in builds[0]["pins"]


def test_session_pin_accumulation(monkeypatch) -> None:
    _script_runs(monkeypatch, [
        _Proc(returncode=1, stderr="No module named 'cv2'"),
        _Proc(returncode=1, stderr="No module named 'numpy'"),
        _Proc(returncode=0, stdout="ok"),
    ])
    builds: list[dict] = []
    monkeypatch.setattr(docker_images, "build_derived_image", _fake_build(builds))

    backend = DockerBackend(package_policy=PackagePolicy(allowlist={
        "cv2": "opencv-python==4.10.0.84", "numpy": "numpy==2.1.1"}))
    r = backend.run_python("import cv2; import numpy", timeout_s=5)

    assert r["ok"] is True
    assert len(builds) == 2
    # Second build layers BOTH pins onto the base.
    assert "opencv-python==4.10.0.84" in builds[1]["pins"]
    assert "numpy==2.1.1" in builds[1]["pins"]
    # Always built FROM the fixed base image, never a derived tag.
    assert all(b["base"] == DEFAULT_DOCKER_IMAGE for b in builds)


def test_build_failure_returns_original_error(monkeypatch) -> None:
    _script_runs(monkeypatch, [
        _Proc(returncode=1, stderr="No module named 'cv2'"),
    ])
    builds: list[dict] = []
    monkeypatch.setattr(docker_images, "build_derived_image",
                        _fake_build(builds, ok=False))

    backend = DockerBackend(
        package_policy=PackagePolicy(allowlist={"cv2": "opencv-python==4.10.0.84"}))
    r = backend.run_python("import cv2", timeout_s=5)

    assert r["ok"] is False
    assert "No module named 'cv2'" in r["stderr"]
    assert "boom" in r["stderr"]  # build log appended
    assert backend.image == DEFAULT_DOCKER_IMAGE  # no roll-forward on failure


def test_retry_loop_is_bounded(monkeypatch) -> None:
    # A script that imports a *new* missing module forever.
    counter = {"n": 0}

    def fake(argv, **kwargs):
        n = counter["n"]
        counter["n"] += 1
        return _Proc(returncode=1, stderr=f"No module named 'mod{n}'")

    monkeypatch.setattr(subprocess, "run", fake)
    builds: list[dict] = []
    monkeypatch.setattr(docker_images, "build_derived_image", _fake_build(builds))

    backend = DockerBackend(package_policy=PackagePolicy(),
                            on_unlisted=lambda i, s: True)
    r = backend.run_python("import mod0", timeout_s=5)
    assert r["ok"] is False
    assert len(builds) == MAX_INSTALL_RETRIES


def test_no_policy_behaves_like_bare_run(monkeypatch) -> None:
    # package_policy=None → never installs, returns the first result unchanged.
    _script_runs(monkeypatch, [
        _Proc(returncode=1, stderr="No module named 'cv2'"),
    ])
    builds: list[dict] = []
    monkeypatch.setattr(docker_images, "build_derived_image", _fake_build(builds))

    backend = DockerBackend()  # no policy
    r = backend.run_python("import cv2", timeout_s=5)
    assert r["ok"] is False
    assert not builds


# --- opt-in real-container integration --------------------------------------

@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_real_install_and_run_no_network() -> None:
    # cowsay is a tiny pure-Python package not in the slim base image.
    backend = DockerBackend(
        package_policy=PackagePolicy(allowlist={"cowsay": "cowsay"}),
        build_timeout_s=600.0,
    )
    r = backend.run_python("import cowsay; print('imported ok')", timeout_s=120)
    assert r["ok"] is True, r
    assert "imported ok" in r["stdout"]
    # The derived run container still has no network.
    net = backend.run_python(
        "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)",
        timeout_s=120,
    )
    assert net["ok"] is False
