"""Shared engine for command-driven feedback (tests, types, lint, build).

Two surfaces consume this engine:

* `tools/run_tests.py` — JsonTool factories the policy can call mid-trace.
  The model invokes `run_pytest` / `run_mypy` / `run_ruff` and gets back
  a structured observation (rc, parsed failures, output tails).

* `verifiers/command.py` — `CommandVerifier`. At FINAL_ANSWER time (or
  per-tick when configured cheap), runs the same command set and turns
  each Failure into a `ClaimCheck` so the existing rejection-deposit
  machinery repels broken-build / failing-test plan regions.

Caching: keyed by (cmd-string, cwd, watch_globs_hash). When file state
under `watch_globs` is unchanged since the last invocation, the cached
result is returned and `cached=True` is set on the result. The hash uses
(relpath, mtime_ns, size) tuples — fast, correct under normal edits.

Allow-list: each Runner carries a tuple of allowed argv-0 binaries. A
call whose first token isn't in the allow-list returns rc=-4 with a
tool_error Failure rather than executing. The allow-list is the
deployer's seatbelt; it is NOT a sandbox.
"""
from __future__ import annotations

import hashlib
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ._parsers import Failure
from ._parsers import pytest as _pytest_parser
from ._parsers import mypy as _mypy_parser
from ._parsers import ruff as _ruff_parser
from .run_shell import run_shell


# Parser registry: maps a "kind" to (parser_fn, default_argv0_allowlist).
ParserFn = Callable[[str, str, int], list[Failure]]

PARSERS: dict[str, ParserFn] = {
    "pytest": _pytest_parser.parse,
    "mypy": _mypy_parser.parse,
    "ruff": _ruff_parser.parse,
}


@dataclass(frozen=True)
class CommandResult:
    """One command run's structured outcome."""

    kind: str
    cmd: str
    rc: int
    stdout_tail: str
    stderr_tail: str
    failures: tuple[Failure, ...]
    elapsed_s: float
    cached: bool
    timeout: bool = False

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "cmd": self.cmd,
            "rc": self.rc,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "failures": [f.to_dict() for f in self.failures],
            "elapsed_s": self.elapsed_s,
            "cached": self.cached,
            "timeout": self.timeout,
            "ok": self.rc == 0 and not self.failures,
        }


@dataclass
class CommandRunner:
    """Runs allow-listed commands with file-hash-keyed result caching.

    Process-local cache (a dict) — no persistence. That keeps Phase 0
    self-contained; the Phase 1 HTTP cache is a separate concern.

    ``cwd`` is required to be an existing directory: the cache
    fingerprint walks files under it, and "I forgot to set cwd and we
    fingerprinted my home dir" is a worse failure mode than a clear
    error at construction. Pass a pathlib.Path or string.
    """

    cwd: str
    allowed_bins: tuple[str, ...] = ("pytest", "python", "python3", "mypy", "ruff")
    watch_globs: tuple[str, ...] = ("**/*.py", "**/*.pyi", "pyproject.toml")
    timeout_s: float = 120.0
    max_output_chars: int = 8_000
    _cache: dict[str, CommandResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        p = Path(self.cwd).expanduser()
        if not p.is_dir():
            raise ValueError(
                f"CommandRunner.cwd must be an existing directory; got {self.cwd!r}"
            )
        # Store the resolved absolute form so subprocess.cwd and the
        # workspace fingerprint agree even if the caller chdirs later.
        self.cwd = str(p.resolve())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, kind: str, cmd: str | Sequence[str]) -> CommandResult:
        """Run a command of the given parser kind. Use the cache when possible."""
        cmd_str = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
        argv0 = shlex.split(cmd_str)[0] if cmd_str else ""

        if argv0 not in self.allowed_bins:
            return CommandResult(
                kind=kind,
                cmd=cmd_str,
                rc=-4,
                stdout_tail="",
                stderr_tail=f"command not in allow-list: {argv0!r}",
                failures=(Failure(
                    kind="tool_error",
                    name=f"allowlist:{argv0}",
                    detail=f"{argv0!r} is not in CommandRunner.allowed_bins",
                ),),
                elapsed_s=0.0,
                cached=False,
            )

        key = self._cache_key(kind, cmd_str)
        hit = self._cache.get(key)
        if hit is not None:
            return CommandResult(
                kind=hit.kind, cmd=hit.cmd, rc=hit.rc,
                stdout_tail=hit.stdout_tail, stderr_tail=hit.stderr_tail,
                failures=hit.failures, elapsed_s=hit.elapsed_s,
                cached=True, timeout=hit.timeout,
            )

        t0 = time.monotonic()
        raw = run_shell(
            cmd_str,
            cwd=self.cwd,
            shell=True,
            timeout_s=self.timeout_s,
            max_output_chars=self.max_output_chars,
        )
        elapsed = time.monotonic() - t0
        parser = PARSERS.get(kind)
        if parser is None:
            failures: tuple[Failure, ...] = ()
        else:
            failures = tuple(parser(raw.get("stdout", ""), raw.get("stderr", ""), raw.get("returncode", -1)))

        result = CommandResult(
            kind=kind,
            cmd=cmd_str,
            rc=int(raw.get("returncode", -1)),
            stdout_tail=_tail(raw.get("stdout", ""), self.max_output_chars),
            stderr_tail=_tail(raw.get("stderr", ""), self.max_output_chars),
            failures=failures,
            elapsed_s=elapsed,
            cached=False,
            timeout=bool(raw.get("timeout", False)),
        )
        self._cache[key] = result
        return result

    def invalidate(self) -> None:
        """Drop all cached results. Call after a workspace reset."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cache_key(self, kind: str, cmd_str: str) -> str:
        return hashlib.sha256(
            f"{kind}\0{cmd_str}\0{self.cwd}\0{self._workspace_fingerprint()}".encode()
        ).hexdigest()

    def _workspace_fingerprint(self) -> str:
        """Hash of (relpath, mtime_ns, size) over files matching watch_globs."""
        root = Path(self.cwd).resolve()
        entries: list[tuple[str, int, int]] = []
        seen: set[Path] = set()
        for pat in self.watch_globs:
            for p in root.glob(pat):
                if not p.is_file() or p in seen:
                    continue
                seen.add(p)
                try:
                    st = p.stat()
                except OSError:
                    continue
                entries.append((str(p.relative_to(root)), st.st_mtime_ns, st.st_size))
        entries.sort()
        h = hashlib.sha256()
        for rel, mtime, size in entries:
            h.update(f"{rel}\0{mtime}\0{size}\n".encode())
        return h.hexdigest()


def _tail(s: str, max_chars: int) -> str:
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return "…[truncated]…\n" + s[-max_chars:]


def runner_for_workspace(cwd: str, **overrides) -> CommandRunner:
    """Convenience factory used by both the tool and verifier surfaces.

    ``cwd`` is required (no default). The runner validates it exists
    in __post_init__.
    """
    return CommandRunner(cwd=cwd, **overrides)
