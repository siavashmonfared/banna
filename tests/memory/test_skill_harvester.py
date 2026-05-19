"""Unit tests for skill_harvester — the verifier-gated skill extractor."""
from __future__ import annotations


from banna_agent.core.state import AgentState
from banna_agent.core.types import Action, ActionKind, Observation
from banna_agent.memory.in_memory_store import InMemoryStore
from banna_agent.memory.skill_harvester import (
    HarvestConfig,
    harvest_from_run,
)
from banna_agent.memory.skill_library import SkillLibrary


def _mk_state(with_good_code: bool = True, correct: bool = True) -> AgentState:
    state = AgentState(question="compute 17*23")
    if with_good_code:
        code = (
            "def compute_cagr(end, start, years):\n"
            "    return (end / start) ** (1 / years) - 1\n"
            "print(compute_cagr(200, 100, 3))\n"
        )
        state.append_step(
            Action(kind=ActionKind.TOOL_CALL, tool_name="run_python",
                   tool_args={"code": code}),
            Observation(ok=True, data={"stdout": "0.2599", "returncode": 0}),
        )
    if correct:
        state.append_step(
            Action(kind=ActionKind.FINAL_ANSWER, answer="0.26"),
            Observation(ok=True, text="0.26"),
        )
    state.metadata["task_id"] = "gaia-42"
    return state


# ---------------------------------------------------------------------------
# Gating — correct_final_answer and verifier_name both required
# ---------------------------------------------------------------------------


def test_harvest_skipped_when_answer_wrong() -> None:
    lib = SkillLibrary(InMemoryStore())
    state = _mk_state()
    out = harvest_from_run(state, lib, correct_final_answer=False,
                           verifier_name="arithmetic")
    assert out == []
    assert len(lib) == 0


def test_harvest_skipped_when_no_verifier() -> None:
    lib = SkillLibrary(InMemoryStore())
    state = _mk_state()
    out = harvest_from_run(state, lib, correct_final_answer=True,
                           verifier_name=None,
                           config=HarvestConfig(verifier_name=""))
    assert out == []
    assert len(lib) == 0


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_harvest_extracts_function_from_run_python() -> None:
    lib = SkillLibrary(InMemoryStore())
    state = _mk_state()
    out = harvest_from_run(state, lib, correct_final_answer=True,
                           verifier_name="arithmetic")
    assert len(out) == 1
    assert out[0].name == "compute_cagr"
    assert "end, start, years" in out[0].signature
    # Attached provenance.
    assert out[0].source_task_id == "gaia-42"
    assert out[0].verifier_name == "arithmetic"
    # Registered into library.
    stored = lib.load("compute_cagr")
    assert stored is not None
    assert stored.code.startswith("def compute_cagr")


def test_harvest_skips_private_functions() -> None:
    lib = SkillLibrary(InMemoryStore())
    state = AgentState(question="?")
    state.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="run_python",
               tool_args={"code": "def _helper(x):\n    return x+1\n"}),
        Observation(ok=True, data={"stdout": "", "returncode": 0}),
    )
    out = harvest_from_run(state, lib, correct_final_answer=True,
                           verifier_name="arithmetic")
    assert out == []


def test_harvest_skips_failed_tool_calls() -> None:
    lib = SkillLibrary(InMemoryStore())
    state = AgentState(question="?")
    state.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="run_python",
               tool_args={"code": "def good(x):\n    return x\n"}),
        Observation(ok=False, data={"returncode": 1},
                    error="RuntimeError: boom"),
    )
    out = harvest_from_run(state, lib, correct_final_answer=True,
                           verifier_name="arithmetic")
    assert out == []


def test_harvest_skips_duplicate_names() -> None:
    lib = SkillLibrary(InMemoryStore())
    state = _mk_state()
    first = harvest_from_run(state, lib, correct_final_answer=True,
                             verifier_name="arithmetic")
    second = harvest_from_run(state, lib, correct_final_answer=True,
                              verifier_name="arithmetic")
    assert len(first) == 1
    assert len(second) == 0
    assert len(lib) == 1


def test_harvest_skips_trivial_one_liners() -> None:
    lib = SkillLibrary(InMemoryStore())
    state = AgentState(question="?")
    state.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="run_python",
               tool_args={"code": "def tiny(): return 1\n"}),
        Observation(ok=True, data={"stdout": "", "returncode": 0}),
    )
    out = harvest_from_run(
        state, lib, correct_final_answer=True, verifier_name="arithmetic",
        config=HarvestConfig(min_function_lines=2),
    )
    assert out == []


def test_harvest_respects_max_skills_per_run() -> None:
    lib = SkillLibrary(InMemoryStore())
    state = AgentState(question="?")
    code = "\n".join(
        f"def fn_{i}(x):\n    return x + {i}\n" for i in range(5)
    )
    state.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="run_python",
               tool_args={"code": code}),
        Observation(ok=True, data={"stdout": "", "returncode": 0}),
    )
    out = harvest_from_run(
        state, lib, correct_final_answer=True, verifier_name="arithmetic",
        config=HarvestConfig(max_skills_per_run=2, min_function_lines=2),
    )
    assert len(out) == 2


def test_harvest_ignores_non_python_tool_calls() -> None:
    lib = SkillLibrary(InMemoryStore())
    state = AgentState(question="?")
    state.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="search",
               tool_args={"query": "netflix"}),
        Observation(ok=True, data={"hits": []}),
    )
    out = harvest_from_run(state, lib, correct_final_answer=True,
                           verifier_name="arithmetic")
    assert out == []


def test_harvest_handles_syntactically_broken_code() -> None:
    lib = SkillLibrary(InMemoryStore())
    state = AgentState(question="?")
    state.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="run_python",
               tool_args={"code": "def broken(:\n  pass"}),
        Observation(ok=True, data={"stdout": ""}),
    )
    out = harvest_from_run(state, lib, correct_final_answer=True,
                           verifier_name="arithmetic")
    assert out == []  # ast.parse failed; harvester returns nothing
