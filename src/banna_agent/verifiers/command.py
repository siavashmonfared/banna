"""CommandVerifier — turn build/test/lint failures into ClaimChecks.

Pairs with `tools/run_tests.py`. Both surfaces share a `CommandRunner`
so the cache is hit across them: if the policy ran pytest mid-trace and
the workspace hasn't changed by FINAL_ANSWER, this verifier's pytest
call is free.

Each parsed Failure becomes one `ClaimCheck` with:
  * claim_id      = f"cmd:{kind}:{failure.name}"  (stable across runs)
  * verdict       = "fail"
  * verifier_name = "command"
  * meta.error_class = failure.kind ("test_failure" | "type_error" | ...)
  * meta.location    = failure.location

This shape plugs straight into `verdicts_to_rejection_deposits`, which
already has table entries keyed on error_class (`tool_error`, etc.). New
classes (test_failure, type_error, lint_error, build_error) fall
through to `_FAIL_DEFAULT` — fine for v1; tune the deposit table later
once we see how the policies behave.

Cost model: each configured command carries `cost` in {"cheap",
"expensive"}. The verifier only runs expensive commands when
`proposed_answer` is not None (i.e. at FINAL_ANSWER). Cheap commands
run on every `check()` call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from ..core.state import AgentState
from ..tools._command_runner import CommandRunner, runner_for_workspace
from .base import ANSWER_CLAIM_ID, ClaimCheck


Cost = Literal["cheap", "expensive"]


@dataclass(frozen=True)
class CommandSpec:
    """One command this verifier should run.

    kind  — parser kind: "pytest" | "mypy" | "ruff".
    cmd   — full shell command, e.g. "pytest --tb=short -q tests/".
    cost  — "cheap" (run every tick) | "expensive" (run only at FINAL).
    """

    kind: str
    cmd: str
    cost: Cost = "expensive"


@dataclass
class CommandVerifier:
    """Run a fixed set of commands; emit one ClaimCheck per failure."""

    commands: Sequence[CommandSpec] = field(default_factory=tuple)
    runner: CommandRunner | None = None
    name: str = "command"

    def __post_init__(self) -> None:
        if self.runner is None:
            raise ValueError(
                "CommandVerifier requires an explicit CommandRunner; "
                "build one via runner_for_workspace(cwd) so the workspace "
                "directory is pinned per-task."
            )

    def check(
        self,
        state: AgentState,
        proposed_answer: str | None = None,
    ) -> list[ClaimCheck]:
        is_final = proposed_answer is not None
        checks: list[ClaimCheck] = []
        for spec in self.commands:
            if spec.cost == "expensive" and not is_final:
                continue
            result = self.runner.run(spec.kind, spec.cmd)
            if not result.failures and result.rc == 0:
                checks.append(ClaimCheck(
                    claim_id=ANSWER_CLAIM_ID,
                    verifier_name=self.name,
                    verdict="ok",
                    detail=f"{spec.kind}: passed ({'cached' if result.cached else 'fresh'})",
                    meta={"kind": spec.kind, "cmd": spec.cmd, "cached": result.cached},
                ))
                continue
            for f in result.failures:
                checks.append(ClaimCheck(
                    claim_id=f"cmd:{spec.kind}:{f.name}",
                    verifier_name=self.name,
                    verdict="fail",
                    detail=f"{f.kind}: {f.detail}",
                    meta={
                        "error_class": f.kind,
                        "location": f.location,
                        "kind": spec.kind,
                        "cached": result.cached,
                    },
                ))
            # If rc!=0 but no failures parsed, emit a generic tool_error.
            if result.rc != 0 and not result.failures:
                checks.append(ClaimCheck(
                    claim_id=f"cmd:{spec.kind}:rc{result.rc}",
                    verifier_name=self.name,
                    verdict="fail",
                    detail=f"{spec.kind} exited {result.rc} with no parseable failures",
                    meta={"error_class": "tool_error", "kind": spec.kind},
                ))
        return checks


def default_command_verifier(
    *,
    cwd: str,
    pytest_target: str = "",
    enable_mypy: bool = False,
    enable_ruff: bool = False,
) -> CommandVerifier:
    """Build a sensible default: expensive pytest at FINAL only, cheap ruff each tick."""
    runner = runner_for_workspace(cwd)
    cmds: list[CommandSpec] = []
    pytest_cmd = "pytest --tb=short -q"
    if pytest_target:
        pytest_cmd += f" {pytest_target}"
    cmds.append(CommandSpec(kind="pytest", cmd=pytest_cmd, cost="expensive"))
    if enable_ruff:
        cmds.append(CommandSpec(kind="ruff", cmd="ruff check --output-format=json .", cost="cheap"))
    if enable_mypy:
        cmds.append(CommandSpec(kind="mypy", cmd="mypy", cost="expensive"))
    return CommandVerifier(commands=tuple(cmds), runner=runner)
