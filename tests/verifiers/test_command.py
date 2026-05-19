"""CommandVerifier + CommandRunner gate tests for Phase 0.

Gate (per the plan):
  1. A failing pytest run produces one fail ClaimCheck per failing test
     with the correct meta.error_class.
  2. Re-running with no file changes is cached (no second subprocess).
  3. Allow-list rejects unlisted commands with a tool_error.

We don't need to actually invoke pytest from the test — that would
make tests slow and circular. Instead we monkeypatch `run_shell` in
the runner module and assert behavior end-to-end through
CommandVerifier.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from banna_agent.core.state import AgentState
from banna_agent.tools import _command_runner as cr
from banna_agent.verifiers.command import (
    CommandSpec,
    CommandVerifier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


PYTEST_FAILING_OUTPUT = """\
============================= test session starts ==============================
collected 2 items

tests/test_x.py .F                                                       [100%]

=================================== FAILURES ===================================
______________________________ test_does_a_thing _______________________________

    def test_does_a_thing():
>       assert 1 + 1 == 3
E       assert 2 == 3

tests/test_x.py:4: AssertionError
=========================== short test summary info ============================
FAILED tests/test_x.py::test_does_a_thing - assert 2 == 3
========================= 1 failed, 1 passed in 0.02s ==========================
"""


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    # The runner hashes files under watch_globs; create one so the
    # fingerprint is non-empty (otherwise cache is still correct, just
    # less interesting).
    (tmp_path / "a.py").write_text("x = 1\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Gate 1: parsed failures land as one ClaimCheck each.
# ---------------------------------------------------------------------------


def test_failing_pytest_emits_one_check_per_failure(monkeypatch, workspace) -> None:
    calls = {"n": 0}

    def fake_run_shell(cmd, *, cwd=None, shell=True, timeout_s=120.0, max_output_chars=8000):
        calls["n"] += 1
        return {
            "ok": False,
            "returncode": 1,
            "stdout": PYTEST_FAILING_OUTPUT,
            "stderr": "",
            "timeout": False,
            "truncated_stdout": False,
            "truncated_stderr": False,
            "wall_s": 0.01,
        }

    monkeypatch.setattr(cr, "run_shell", fake_run_shell)

    runner = cr.CommandRunner(cwd=str(workspace))
    verifier = CommandVerifier(
        commands=(CommandSpec(kind="pytest", cmd="pytest --tb=short -q", cost="expensive"),),
        runner=runner,
    )

    state = AgentState(question="irrelevant for this test")
    checks = verifier.check(state, proposed_answer="42")

    fails = [c for c in checks if c.verdict == "fail"]
    assert len(fails) == 1
    c = fails[0]
    assert c.verifier_name == "command"
    assert c.meta["error_class"] == "test_failure"
    assert "test_does_a_thing" in c.claim_id
    assert "assert 2 == 3" in c.detail
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Gate 2: cache is hit on re-run with no file changes.
# ---------------------------------------------------------------------------


def test_repeated_run_uses_cache(monkeypatch, workspace) -> None:
    calls = {"n": 0}

    def fake_run_shell(cmd, *, cwd=None, shell=True, timeout_s=120.0, max_output_chars=8000):
        calls["n"] += 1
        return {
            "ok": True, "returncode": 0,
            "stdout": "1 passed", "stderr": "",
            "timeout": False, "truncated_stdout": False, "truncated_stderr": False,
            "wall_s": 0.01,
        }

    monkeypatch.setattr(cr, "run_shell", fake_run_shell)

    runner = cr.CommandRunner(cwd=str(workspace))
    r1 = runner.run("pytest", "pytest --tb=short -q")
    r2 = runner.run("pytest", "pytest --tb=short -q")
    assert r1.cached is False
    assert r2.cached is True
    assert calls["n"] == 1


def test_cache_invalidates_on_file_change(monkeypatch, workspace) -> None:
    calls = {"n": 0}

    def fake_run_shell(cmd, *, cwd=None, shell=True, timeout_s=120.0, max_output_chars=8000):
        calls["n"] += 1
        return {
            "ok": True, "returncode": 0,
            "stdout": "1 passed", "stderr": "",
            "timeout": False, "truncated_stdout": False, "truncated_stderr": False,
            "wall_s": 0.01,
        }

    monkeypatch.setattr(cr, "run_shell", fake_run_shell)

    runner = cr.CommandRunner(cwd=str(workspace))
    runner.run("pytest", "pytest --tb=short -q")
    # Mutate the watched workspace.
    (workspace / "a.py").write_text("x = 2\n")
    import os, time
    # Force a distinct mtime even on coarse-grained filesystems.
    future = time.time() + 1
    os.utime(workspace / "a.py", (future, future))

    r2 = runner.run("pytest", "pytest --tb=short -q")
    assert r2.cached is False
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Gate 3: allow-list rejection.
# ---------------------------------------------------------------------------


def test_allowlist_rejects_unknown_binary(monkeypatch, workspace) -> None:
    # Even if run_shell would succeed, the runner must refuse.
    def boom(*a, **kw):
        raise AssertionError("run_shell should not be called for disallowed bin")

    monkeypatch.setattr(cr, "run_shell", boom)
    runner = cr.CommandRunner(cwd=str(workspace), allowed_bins=("pytest",))
    result = runner.run("pytest", "rm -rf /")
    assert result.rc == -4
    assert len(result.failures) == 1
    assert result.failures[0].kind == "tool_error"
    assert "allowlist" in result.failures[0].name


# ---------------------------------------------------------------------------
# Cost: cheap runs every tick, expensive only at FINAL.
# ---------------------------------------------------------------------------


def test_expensive_skipped_when_no_proposed_answer(monkeypatch, workspace) -> None:
    def fake_run_shell(*a, **kw):
        return {"ok": True, "returncode": 0, "stdout": "[]", "stderr": "",
                "timeout": False, "truncated_stdout": False, "truncated_stderr": False, "wall_s": 0.0}

    monkeypatch.setattr(cr, "run_shell", fake_run_shell)
    runner = cr.CommandRunner(cwd=str(workspace))
    verifier = CommandVerifier(
        commands=(
            CommandSpec(kind="ruff", cmd="ruff check --output-format=json .", cost="cheap"),
            CommandSpec(kind="pytest", cmd="pytest --tb=short -q", cost="expensive"),
        ),
        runner=runner,
    )
    state = AgentState(question="q")
    mid_trace = verifier.check(state, proposed_answer=None)
    kinds = {c.meta.get("kind") for c in mid_trace}
    assert kinds == {"ruff"}  # pytest skipped, ruff ran (and passed)

    final = verifier.check(state, proposed_answer="42")
    kinds_final = {c.meta.get("kind") for c in final}
    assert kinds_final == {"ruff", "pytest"}


# ---------------------------------------------------------------------------
# default_verifiers integration: command verifier is opt-in.
# ---------------------------------------------------------------------------


def test_default_verifiers_excludes_command_unless_passed() -> None:
    from banna_agent.verifiers import default_verifiers
    names = [getattr(v, "name", type(v).__name__) for v in default_verifiers()]
    assert "command" not in names

    # CommandVerifier now requires an explicit runner with a real cwd.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cv = CommandVerifier(commands=(), runner=cr.CommandRunner(cwd=td))
        names2 = [getattr(v, "name", type(v).__name__) for v in default_verifiers(command_verifier=cv)]
        assert "command" in names2
        assert names2[-1] == "command"  # appended at end


# ---------------------------------------------------------------------------
# Deposit table picks up new error classes.
# ---------------------------------------------------------------------------


