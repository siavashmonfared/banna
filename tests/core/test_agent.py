"""Integration-style unit tests for the driver `run_policy`.

Policies here are *hand-rolled test doubles* — we don't need ReAct for
these tests, we need deterministic sequences of Actions to verify the
driver's state-transition contract.
"""
from __future__ import annotations


from banna_agent.core.agent import run_policy
from banna_agent.core.events import EventKind, EventLog
from banna_agent.core.state import AgentState
from banna_agent.core.types import Action, ActionKind, Budget
from banna_agent.tools.base import JsonTool, ToolRegistry
from banna_agent.tools.calculator import make_calculator_tool


class _ScriptedPolicy:
    """Replays a canned list of Actions; the driver still enforces budget."""
    name = "scripted"

    def __init__(self, actions: list[Action]) -> None:
        self.actions = list(actions)
        self.calls = 0

    def propose(self, state, *, llm, tools) -> Action:
        self.calls += 1
        if not self.actions:
            return Action(kind=ActionKind.FINAL_ANSWER, answer="END")
        return self.actions.pop(0)


class _DummyLLM:
    """Fulfills the LLMClient protocol (even though this test doesn't use it)."""
    provider = "dummy"

    def chat(self, **_):
        raise AssertionError("policy should not hit the LLM in this test")


def _calc_registry() -> ToolRegistry:
    return ToolRegistry([make_calculator_tool()])


# ---------------------------------------------------------------------------
# Happy path: a scripted think → tool_call → final_answer
# ---------------------------------------------------------------------------


def test_driver_runs_think_tool_and_finalizes() -> None:
    policy = _ScriptedPolicy([
        Action(kind=ActionKind.THINK, text="I'll use the calculator."),
        Action(kind=ActionKind.TOOL_CALL, tool_name="calculator",
               tool_args={"expression": "17 * 23"}),
        Action(kind=ActionKind.FINAL_ANSWER, answer="391"),
    ])
    state = AgentState(question="17*23?", budget=Budget(max_steps=10, max_wall_s=5.0))
    log = EventLog()

    state = run_policy(state, policy, llm=_DummyLLM(), tools=_calc_registry(), log=log)

    assert state.is_done
    assert state.trace.final_answer == "391"
    # 3 steps, one of which was a successful calculator call.
    assert len(state.trace.steps) == 3
    tool_step = state.trace.steps[1]
    assert tool_step.action.tool_name == "calculator"
    assert tool_step.observation.ok is True
    assert tool_step.observation.data["value"] == 391.0

    # Events: run_start, 3× propose, 2× observation-like (tool has call+result+obs)
    kinds = [e.kind for e in log.events]
    assert kinds[0] == EventKind.RUN_START
    assert kinds[-1] == EventKind.RUN_END
    assert EventKind.TOOL_CALL in kinds
    assert EventKind.TOOL_RESULT in kinds


# ---------------------------------------------------------------------------
# Budget trip: steps exhausted mid-run
# ---------------------------------------------------------------------------


def test_driver_stops_on_step_budget() -> None:
    policy = _ScriptedPolicy([
        Action(kind=ActionKind.THINK, text="t1"),
        Action(kind=ActionKind.THINK, text="t2"),
        Action(kind=ActionKind.THINK, text="t3"),  # shouldn't reach this
    ])
    state = AgentState(question="?", budget=Budget(max_steps=2, max_wall_s=5.0))
    state = run_policy(state, policy, llm=_DummyLLM(), tools=_calc_registry())
    assert not state.is_done
    assert len(state.trace.steps) == 2


# ---------------------------------------------------------------------------
# Unknown tool becomes a non-fatal error observation
# ---------------------------------------------------------------------------


def test_driver_handles_unknown_tool() -> None:
    policy = _ScriptedPolicy([
        Action(kind=ActionKind.TOOL_CALL, tool_name="nonexistent", tool_args={}),
        Action(kind=ActionKind.FINAL_ANSWER, answer="gave up"),
    ])
    state = AgentState(question="?", budget=Budget(max_steps=5))
    state = run_policy(state, policy, llm=_DummyLLM(), tools=_calc_registry())
    assert state.is_done
    assert state.trace.final_answer == "gave up"
    tool_step = state.trace.steps[0]
    assert tool_step.observation.ok is False
    assert "unknown tool" in tool_step.observation.error


# ---------------------------------------------------------------------------
# Tool raises → still captured as error observation
# ---------------------------------------------------------------------------


def test_driver_captures_tool_exceptions() -> None:
    def _boom(_: dict) -> dict:
        raise RuntimeError("crash")

    bad = JsonTool(
        name="bad",
        description="always fails",
        input_schema={"type": "object"},
        handler=_boom,
    )
    policy = _ScriptedPolicy([
        Action(kind=ActionKind.TOOL_CALL, tool_name="bad", tool_args={}),
        Action(kind=ActionKind.FINAL_ANSWER, answer="ok"),
    ])
    state = AgentState(question="?", budget=Budget(max_steps=5))
    registry = ToolRegistry([bad])
    state = run_policy(state, policy, llm=_DummyLLM(), tools=registry)
    err_step = state.trace.steps[0]
    assert err_step.observation.ok is False
    assert "RuntimeError" in err_step.observation.error


# ---------------------------------------------------------------------------
# Search-style tool auto-registers evidence from hits
# ---------------------------------------------------------------------------


def test_driver_auto_registers_evidence_from_search_hits() -> None:
    def _fake_search(_: dict) -> dict:
        return {
            "query": "netflix",
            "hits": [
                {"title": "Investing", "url": "https://investing.com/netflix", "snippet": "ARPU was $11.64"},
                {"title": "SEC", "url": "https://sec.gov/10k", "snippet": "fiscal 2023"},
            ],
        }

    search_tool = JsonTool(
        name="search",
        description="fake search",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        handler=_fake_search,
    )
    policy = _ScriptedPolicy([
        Action(kind=ActionKind.TOOL_CALL, tool_name="search", tool_args={"q": "netflix"}),
        Action(kind=ActionKind.FINAL_ANSWER, answer="$11.64"),
    ])
    state = AgentState(question="Netflix ARPU?", budget=Budget(max_steps=5))
    state = run_policy(state, policy, llm=_DummyLLM(), tools=ToolRegistry([search_tool]))
    assert state.is_done
    # Both hits should have been registered as evidence.
    urls = sorted(ev.source for ev in state.evidence)
    assert urls == sorted(["https://investing.com/netflix", "https://sec.gov/10k"])


# ---------------------------------------------------------------------------
# Policy exceptions also don't crash the loop
# ---------------------------------------------------------------------------


class _CrashingPolicy:
    name = "crashing"
    def propose(self, state, *, llm, tools):
        raise RuntimeError("nope")


# ---------------------------------------------------------------------------
# Parallel tool dispatch (TOOL_BATCH)
# ---------------------------------------------------------------------------


def _slow_tool(name: str, delay_s: float, returns: dict) -> JsonTool:
    """A JsonTool whose handler sleeps `delay_s` then returns `returns`."""
    import time

    def _h(_: dict) -> dict:
        time.sleep(delay_s)
        return dict(returns)

    return JsonTool(
        name=name,
        description=f"slow tool {name}",
        input_schema={"type": "object", "properties": {}},
        handler=_h,
    )


def test_tool_batch_runs_calls_concurrently() -> None:
    """Two 0.15s tools dispatched as a batch should finish in ~0.15s, not 0.3s."""
    import time

    a = _slow_tool("a", 0.15, {"value": 1})
    b = _slow_tool("b", 0.15, {"value": 2})
    policy = _ScriptedPolicy([
        Action(
            kind=ActionKind.TOOL_BATCH,
            meta={"batch_calls": [
                {"name": "a", "args": {}},
                {"name": "b", "args": {}},
            ]},
        ),
        Action(kind=ActionKind.FINAL_ANSWER, answer="done"),
    ])
    state = AgentState(question="?", budget=Budget(max_steps=5, max_wall_s=5.0))
    log = EventLog()
    t0 = time.monotonic()
    state = run_policy(state, policy, llm=_DummyLLM(),
                       tools=ToolRegistry([a, b]), log=log)
    elapsed = time.monotonic() - t0

    assert state.is_done
    # If dispatched serially, this would take ≥0.30s. Concurrent dispatch
    # should land near 0.15s — give ourselves headroom for thread spinup.
    assert elapsed < 0.27, f"batch ran serially ({elapsed:.3f}s)"

    # One step recorded with kind=TOOL_BATCH; observation carries the
    # per-call summary on `data["batch"]`.
    batch_step = state.trace.steps[0]
    assert batch_step.action.kind == ActionKind.TOOL_BATCH
    assert batch_step.observation.ok is True
    assert batch_step.observation.data["n"] == 2
    names_out = sorted(r["name"] for r in batch_step.observation.data["batch"])
    assert names_out == ["a", "b"]

    # Events: a TOOL_BATCH boundary marker plus per-call TOOL_CALL +
    # TOOL_RESULT with `in_batch=True`.
    kinds = [e.kind for e in log.events]
    assert EventKind.TOOL_BATCH in kinds
    batch_evs = [e for e in log.events if e.kind == EventKind.TOOL_BATCH]
    assert batch_evs[0].payload["tool_names"] == ["a", "b"]
    in_batch_calls = [
        e for e in log.events
        if e.kind == EventKind.TOOL_CALL and e.payload.get("in_batch")
    ]
    assert len(in_batch_calls) == 2


def test_tool_batch_with_one_failing_sub_call_records_combined_error() -> None:
    """If one sub-call errors, batch ok=False but the other still ran."""
    good = _slow_tool("good", 0.0, {"value": 1})

    def _bad_h(_: dict) -> dict:
        raise RuntimeError("boom")

    bad = JsonTool(
        name="bad", description="bad",
        input_schema={"type": "object", "properties": {}},
        handler=_bad_h,
    )
    policy = _ScriptedPolicy([
        Action(
            kind=ActionKind.TOOL_BATCH,
            meta={"batch_calls": [
                {"name": "good", "args": {}},
                {"name": "bad", "args": {}},
            ]},
        ),
        Action(kind=ActionKind.FINAL_ANSWER, answer="done"),
    ])
    state = AgentState(question="?", budget=Budget(max_steps=5))
    state = run_policy(state, policy, llm=_DummyLLM(),
                       tools=ToolRegistry([good, bad]))
    batch_step = state.trace.steps[0]
    assert batch_step.observation.ok is False
    assert "bad" in (batch_step.observation.error or "")
    by_name = {r["name"]: r for r in batch_step.observation.data["batch"]}
    assert by_name["good"]["ok"] is True
    assert by_name["bad"]["ok"] is False


def test_tool_batch_auto_registers_evidence_per_sub_call() -> None:
    """Each sub-call's hits should land in state.evidence, same as single tools."""
    def _h(payload: dict) -> dict:
        return {"hits": [{"title": payload["q"], "url": f"https://x/{payload['q']}",
                          "snippet": payload["q"]}]}

    search = JsonTool(
        name="search", description="fake",
        input_schema={"type": "object",
                      "properties": {"q": {"type": "string"}}},
        handler=_h,
    )
    policy = _ScriptedPolicy([
        Action(
            kind=ActionKind.TOOL_BATCH,
            meta={"batch_calls": [
                {"name": "search", "args": {"q": "alpha"}},
                {"name": "search", "args": {"q": "beta"}},
            ]},
        ),
        Action(kind=ActionKind.FINAL_ANSWER, answer="done"),
    ])
    state = AgentState(question="?", budget=Budget(max_steps=5))
    state = run_policy(state, policy, llm=_DummyLLM(),
                       tools=ToolRegistry([search]))
    sources = sorted(ev.source for ev in state.evidence)
    assert sources == ["https://x/alpha", "https://x/beta"]


def test_driver_survives_policy_exception() -> None:
    state = AgentState(question="?", budget=Budget(max_steps=3))
    log = EventLog()
    state = run_policy(state, _CrashingPolicy(), llm=_DummyLLM(),
                       tools=_calc_registry(), log=log)
    assert not state.is_done
    errs = log.filter(EventKind.ERROR)
    assert len(errs) == 1
    assert "RuntimeError" in errs[0].payload["error"]
