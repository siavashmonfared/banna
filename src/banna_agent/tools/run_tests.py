"""JsonTool factories for command-driven feedback.

These are the *inner-loop* surface: the policy emits `run_pytest` /
`run_mypy` / `run_ruff` mid-trace, and the structured failure list lands
in `state.trace` as an observation the LLM can read on the next tick.

The matching *outer-loop* surface — running these at FINAL_ANSWER and
emitting ClaimChecks — is `verifiers/command.py`. Both share a
`CommandRunner` instance so the cache is hit across surfaces (running
the same pytest as a tool then as a verifier costs one subprocess).
"""
from __future__ import annotations

from typing import Any

from .base import JsonTool
from ._command_runner import CommandRunner


_PYTEST_DESC = (
    "Run pytest in the workspace. Returns a structured result with the "
    "returncode and a list of parsed test failures (node id, assertion "
    "message, source location). Output is cached: re-running with no "
    "file changes returns immediately. Prefer narrow `-k`/path arguments "
    "to keep runs cheap; the full suite is fine when you genuinely want "
    "a final check."
)

_MYPY_DESC = (
    "Run mypy in the workspace. Returns parsed type errors "
    "(path:line:code, message). Cached across calls when files haven't "
    "changed. Useful as a fast check before running tests."
)

_RUFF_DESC = (
    "Run ruff in the workspace. Returns parsed lint diagnostics. "
    "Cheap (<1s typical); safe to run after every edit."
)


def _result_to_observation(result) -> dict[str, Any]:
    """Project a CommandResult into the LLM-visible dict shape."""
    d = result.to_dict()
    # Keep the LLM payload terse: drop redundant `cmd` and rename tails.
    return {
        "ok": d["ok"],
        "rc": d["rc"],
        "kind": d["kind"],
        "cached": d["cached"],
        "timeout": d["timeout"],
        "elapsed_s": round(d["elapsed_s"], 3),
        "failures": d["failures"],
        "stdout_tail": d["stdout_tail"],
        "stderr_tail": d["stderr_tail"],
    }


def make_run_pytest_tool(runner: CommandRunner) -> JsonTool:
    rn = runner

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        target = (args.get("target") or "").strip()
        k_expr = (args.get("k") or "").strip()
        extra = " ".join(args.get("extra_args") or [])
        cmd = "pytest --tb=short -q"
        if k_expr:
            cmd += f" -k {k_expr!r}"
        if target:
            cmd += f" {target}"
        if extra:
            cmd += f" {extra}"
        return _result_to_observation(rn.run("pytest", cmd))

    schema = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Path or nodeid (optional). Empty = whole suite."},
            "k": {"type": "string", "description": "Pytest -k expression (optional)."},
            "extra_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra pytest CLI args (advanced).",
            },
        },
        "additionalProperties": False,
    }
    return JsonTool(
        name="run_pytest",
        description=_PYTEST_DESC,
        input_schema=schema,
        handler=handler,
        capabilities=frozenset({"shell", "test"}),
    )


def make_run_mypy_tool(runner: CommandRunner) -> JsonTool:
    rn = runner

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        target = (args.get("target") or "").strip()
        extra = " ".join(args.get("extra_args") or [])
        cmd = "mypy"
        if target:
            cmd += f" {target}"
        if extra:
            cmd += f" {extra}"
        return _result_to_observation(rn.run("mypy", cmd))

    schema = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Path to type-check (default: project config)."},
            "extra_args": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }
    return JsonTool(
        name="run_mypy",
        description=_MYPY_DESC,
        input_schema=schema,
        handler=handler,
        capabilities=frozenset({"shell", "typecheck"}),
    )


def make_run_ruff_tool(runner: CommandRunner) -> JsonTool:
    rn = runner

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        target = (args.get("target") or ".").strip()
        cmd = f"ruff check --output-format=json {target}"
        return _result_to_observation(rn.run("ruff", cmd))

    schema = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Path to lint (default '.')."},
        },
        "additionalProperties": False,
    }
    return JsonTool(
        name="run_ruff",
        description=_RUFF_DESC,
        input_schema=schema,
        handler=handler,
        capabilities=frozenset({"shell", "lint"}),
    )
