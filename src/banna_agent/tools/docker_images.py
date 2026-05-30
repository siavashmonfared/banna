"""Build + cache derived Docker images for the sandbox's on-demand installs.

Installing packages and running untrusted model code must **never** happen in
the same container. The two-phase split here is the whole safety story:

  * ``docker build`` runs with network access but executes only the Dockerfile
    we author — a ``FROM <base>`` plus a ``pip install`` of vetted pins. No
    model-emitted code runs during the build.
  * the resulting image then runs the model's code with ``--network none`` (see
    ``DockerBackend`` in ``sandbox.py``), so the untrusted code never has the
    network the installer needed.

Derived images are cached by a content hash of ``(base image, sorted pins)``,
so a given package set is built at most once per machine.
"""
from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

DEFAULT_BUILD_TIMEOUT_S = 300.0


def derived_image_tag(base_image: str, pins: list[str]) -> str:
    """A stable ``banna-sbx:<digest>`` tag for a (base image, pin set).

    Pins are sorted + de-duplicated before hashing so ordering doesn't matter;
    the base image is folded in so changing ``--sandbox-image`` invalidates the
    cache correctly.
    """
    norm = sorted(set(pins))
    blob = base_image + "\n" + "\n".join(norm)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return f"banna-sbx:{digest}"


def render_dockerfile(base_image: str, pins: list[str]) -> str:
    """A minimal Dockerfile: base image + a single pinned ``pip install``."""
    norm = sorted(set(pins))
    pkgs = " ".join(norm)
    return f"FROM {base_image}\nRUN pip install --no-cache-dir {pkgs}\n"


def image_exists(tag: str, *, docker_bin: str = "docker") -> bool:
    """True iff ``docker image inspect <tag>`` succeeds."""
    try:
        proc = subprocess.run(
            [docker_bin, "image", "inspect", tag],
            capture_output=True, text=True,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        return False


def build_derived_image(
    base_image: str,
    pins: list[str],
    *,
    docker_bin: str = "docker",
    timeout_s: float = DEFAULT_BUILD_TIMEOUT_S,
) -> tuple[bool, str, str]:
    """Build (or reuse, if cached) an image with ``pins`` baked in.

    Returns ``(ok, tag, log)``. The build runs with the network enabled but no
    model code — only the generated Dockerfile is in the build context.
    """
    tag = derived_image_tag(base_image, pins)
    if image_exists(tag, docker_bin=docker_bin):
        return True, tag, ""
    dockerfile = render_dockerfile(base_image, pins)
    try:
        with tempfile.TemporaryDirectory(prefix="banna-build-") as ctx:
            (Path(ctx) / "Dockerfile").write_text(dockerfile, encoding="utf-8")
            proc = subprocess.run(
                [docker_bin, "build", "-t", tag, "-f",
                 str(Path(ctx) / "Dockerfile"), ctx],
                capture_output=True, text=True, timeout=timeout_s,
            )
            log = (proc.stdout or "") + (proc.stderr or "")
            return proc.returncode == 0, tag, log
    except subprocess.TimeoutExpired:
        return False, tag, f"docker build timed out after {timeout_s}s"
    except FileNotFoundError:
        return False, tag, (
            f"docker not found ({docker_bin!r} is not on PATH). "
            "Install Docker, or run with --sandbox=process."
        )
