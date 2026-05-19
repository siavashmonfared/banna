"""Harvest skills from a verifier-confirmed successful run.

Called AFTER a task completes and a verifier has signed off on the final
answer. The harvester scans `state.trace.steps` for `run_python` tool
calls that (a) succeeded and (b) contain callable definitions; it names
them (via LLM or heuristic), attaches provenance, and registers them
into the `SkillLibrary`.

Gating — per design decision (c) in the plan:
    A skill is only harvested when `verifier_name` is provided AND the
    final answer was correct. The caller is responsible for both
    conditions; this module doesn't run the verifier itself.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from ..core.state import AgentState
from ..core.types import ActionKind
from .skill_library import Skill, SkillLibrary


@dataclass
class HarvestConfig:
    verifier_name: str = "arithmetic"       # required to accept a skill
    min_function_lines: int = 2             # drop trivial one-liners
    max_skills_per_run: int = 3


def _extract_function_defs(code: str) -> list[tuple[str, str, str]]:
    """Return list of (name, signature, body) for top-level `def` statements.

    Signature is a string like 'foo(a, b=1) -> int' when annotations allow.
    Body is the source slice including the def line.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lines = code.splitlines()
    out: list[tuple[str, str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        # Slice source lines: node.lineno is 1-indexed; end_lineno similar.
        start = node.lineno - 1
        end = node.end_lineno or (start + 1)
        body = "\n".join(lines[start:end])
        try:
            sig = _format_signature(node)
        except Exception:
            sig = f"{node.name}(...)"
        out.append((node.name, sig, body))
    return out


def _format_signature(node: ast.FunctionDef) -> str:
    import ast as _a
    args = node.args
    parts: list[str] = []
    for arg in args.args:
        a = arg.arg
        if arg.annotation is not None:
            a += f": {_a.unparse(arg.annotation)}"
        parts.append(a)
    ret = ""
    if node.returns is not None:
        ret = f" -> {_a.unparse(node.returns)}"
    return f"{node.name}({', '.join(parts)}){ret}"


def harvest_from_run(
    state: AgentState,
    library: SkillLibrary,
    *,
    config: HarvestConfig | None = None,
    correct_final_answer: bool,
    verifier_name: str | None = None,
) -> list[Skill]:
    """Scan a completed `AgentState` for harvestable skills.

    Parameters
    ----------
    state                 : the completed agent state
    library               : the SkillLibrary to register into
    config                : harvest config; `HarvestConfig()` default
    correct_final_answer  : must be True to harvest anything (design choice c)
    verifier_name         : overrides config.verifier_name; must be truthy
                            to harvest anything
    """
    cfg = config or HarvestConfig()
    v = verifier_name or cfg.verifier_name
    if not correct_final_answer or not v:
        return []

    harvested: list[Skill] = []
    task_id = state.metadata.get("task_id") or state.state_id

    for step in state.trace.steps:
        if step.action.kind != ActionKind.TOOL_CALL:
            continue
        if step.action.tool_name != "run_python":
            continue
        if not step.observation.ok:
            continue
        code = step.action.tool_args.get("code", "")
        if not isinstance(code, str) or not code.strip():
            continue

        for name, signature, body in _extract_function_defs(code):
            if len(body.splitlines()) < cfg.min_function_lines:
                continue
            # Skip "private" helpers.
            if name.startswith("_"):
                continue
            # Skip duplicates.
            if library.load(name) is not None:
                continue
            skill = Skill(
                name=name,
                signature=signature,
                description=_describe_skill(name, signature),
                code=body,
                source_task_id=task_id,
                verifier_name=v,
            )
            library.register(skill)
            harvested.append(skill)
            if len(harvested) >= cfg.max_skills_per_run:
                return harvested
    return harvested


def _describe_skill(name: str, signature: str) -> str:
    """Derive a short human-readable description from the function name.

    Heuristic only — the LLM-assisted describer is future work.
    """
    # snake_case -> "snake case"
    words = re.sub(r"[_\-]+", " ", name).strip()
    return f"{words} — {signature}"
