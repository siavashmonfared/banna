"""Tests for the `react+` policy: ASK_USER, permission gate, batch fallback."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from banna_agent.core.agent import _execute, run_policy
from banna_agent.core.budget import Budget
from banna_agent.core.state import AgentState
from banna_agent.core.types import Action, ActionKind, Observation, Trace
from banna_agent.core.user_io import UserIO
from banna_agent.llm.base import (
    ContentBlock,
    LLMReply,
    Usage,
)
from banna_agent.policies.react_plus import ReActPlusPolicy
from banna_agent.tools.base import JsonTool, ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class _RecordingUserIO:
    """UserIO double that records calls and returns scripted answers."""
    reply: str = "yes use /tmp instead"
    decision: str = "allow_once"
    ask_calls: list[str] = field(default_factory=list)
    confirm_calls: list[dict] = field(default_factory=list)

    def ask(self, question: str) -> str:
        self.ask_calls.append(question)
        return self.reply

    def confirm(self, *, tool_name: str, args: dict, risk: str) -> str:
        self.confirm_calls.append({"tool_name": tool_name, "args": args, "risk": risk})
        return self.decision


def _state(question: str = "test") -> AgentState:
    return AgentState(
        question=question,
        trace=Trace(question=question, run_id="r"),
        budget=Budget(max_steps=10, max_wall_s=60.0),
    )


# ---------------------------------------------------------------------------
# ASK_USER plumbing (loop side)
# ---------------------------------------------------------------------------


def test_ask_user_action_calls_user_io_and_returns_reply() -> None:
    """An ASK_USER action should block on UserIO.ask and surface the reply."""
    state = _state()
    user_io = _RecordingUserIO(reply="actually /tmp/data.csv")
    action = Action(kind=ActionKind.ASK_USER, text="Which path?", meta={})

    obs = _execute(state, action, ToolRegistry(), log=None, user_io=user_io)

    assert user_io.ask_calls == ["Which path?"]
    assert obs.ok is True
    assert obs.text == "actually /tmp/data.csv"
    # Reply stashed for the policy to fold into the next prompt.
    assert state.metadata["user_replies"][-1]["reply"] == "actually /tmp/data.csv"


def test_ask_user_in_batch_mode_degrades_to_marker_think() -> None:
    """With no UserIO (GAIA / CI), ASK_USER must not crash. It degrades
    to a synthetic THINK so the trace is comparable to a non-interactive
    run."""
    state = _state()
    action = Action(kind=ActionKind.ASK_USER, text="Which path?", meta={})

    obs = _execute(state, action, ToolRegistry(), log=None, user_io=None)

    assert obs.ok is True
    assert "no user available" in (obs.text or "").lower()
    assert "user_replies" not in state.metadata


def test_ask_user_empty_reply_observation_marked_not_ok() -> None:
    """An empty reply (user hit Enter / EOF) is still recorded but
    surfaces ok=False so the policy knows it didn't get information."""
    state = _state()
    user_io = _RecordingUserIO(reply="")
    action = Action(kind=ActionKind.ASK_USER, text="?", meta={})

    obs = _execute(state, action, ToolRegistry(), log=None, user_io=user_io)

    assert obs.ok is False


def test_ask_user_block_time_excluded_from_wall_budget() -> None:
    """Human think-time at an ask_user prompt must not count against the
    agent's wall budget. Regression for the gpt-oss:20b session where a
    ~310s pause at the prompt tripped budget_wall (200s)."""
    import time

    from banna_agent.core.budget import BudgetTracker

    @dataclass
    class _SlowUserIO:
        delay: float = 0.3

        def ask(self, question: str) -> str:
            time.sleep(self.delay)  # simulate a human deliberating
            return "ok"

        def confirm(self, *, tool_name, args, risk) -> str:  # pragma: no cover
            return "deny"

    state = _state()
    state.budget = Budget(max_steps=10, max_wall_s=0.1)  # tiny wall cap
    tracker = BudgetTracker(state.budget)
    tracker.start()
    action = Action(kind=ActionKind.ASK_USER, text="?", meta={})

    obs = _execute(state, action, ToolRegistry(), log=None,
                   user_io=_SlowUserIO(delay=0.3), budget=tracker)

    assert obs.text == "ok"
    # The 0.3s human pause is 3x the wall cap, yet the budget must be OK
    # because pause()/resume() shifted the timer past it.
    assert tracker.check().value == "ok"


def test_synthesis_does_not_echo_ask_user_question() -> None:
    """When the budget trips on an unanswered ASK_USER, the cheap
    synthesis fallback must not return the clarifying question as the
    final answer. Regression for the gpt-oss:20b session that ended
    FINAL_ANSWER='What would you like me to help you with?'."""
    state = _state(question="top 5 Guardian US headlines today")
    q = "What would you like me to help you with?"
    state.append_step(
        Action(kind=ActionKind.ASK_USER, text=q, meta={}),
        Observation(ok=False, text="(no reply)"),
    )
    policy = ReActPlusPolicy(model="gpt-oss:20b")

    # llm=None forces the cheap branch (the one that walked the trace).
    synth = policy.synthesize_on_exhaustion(state, llm=None, tools=None)

    # Either None (no answer available) or some best guess — but never
    # the clarifying question parroted back.
    if synth is not None:
        assert synth.answer != q


# ---------------------------------------------------------------------------
# Permission gate
# ---------------------------------------------------------------------------


def test_run_shell_deny_returns_synthetic_error_observation() -> None:
    """Denying the gate should produce an error Observation and NOT
    invoke the tool. Policy will see this as a tool failure and can
    pick another approach."""
    state = _state()
    user_io = _RecordingUserIO(decision="deny")

    # Register a real run_shell tool that we DON'T want invoked.
    invoked = {"count": 0}

    def _handler(args: dict) -> dict:
        invoked["count"] += 1
        return {"ok": True}

    tools = ToolRegistry([JsonTool(
        name="run_shell", description="shell",
        input_schema={"type": "object", "properties": {}},
        handler=_handler,
    )])
    action = Action(
        kind=ActionKind.TOOL_CALL, tool_name="run_shell",
        tool_args={"command": "rm -rf /"}, meta={},
    )

    obs = _execute(state, action, tools, log=None, user_io=user_io)

    assert invoked["count"] == 0, "denied tool must not be invoked"
    assert obs.ok is False
    assert "user denied" in (obs.error or "")
    assert user_io.confirm_calls[0]["tool_name"] == "run_shell"


def test_run_shell_allow_always_memoizes_signature() -> None:
    """A second identical call should NOT re-prompt after allow_always."""
    state = _state()
    user_io = _RecordingUserIO(decision="allow_always")

    tools = ToolRegistry([JsonTool(
        name="run_shell", description="shell",
        input_schema={"type": "object", "properties": {}},
        handler=lambda args: {"ok": True, "stdout": "hello"},
    )])
    action = Action(
        kind=ActionKind.TOOL_CALL, tool_name="run_shell",
        tool_args={"command": "ls /tmp"}, meta={},
    )

    _execute(state, action, tools, log=None, user_io=user_io)
    _execute(state, action, tools, log=None, user_io=user_io)

    # Only one prompt — the second call hit the allowlist.
    assert len(user_io.confirm_calls) == 1


def test_non_gated_tools_skip_the_permission_prompt() -> None:
    """Calculator / search / read_file / etc. should never prompt."""
    state = _state()
    user_io = _RecordingUserIO(decision="deny")

    tools = ToolRegistry([JsonTool(
        name="calculator", description="calc",
        input_schema={"type": "object", "properties": {}},
        handler=lambda args: {"value": 42},
    )])
    action = Action(
        kind=ActionKind.TOOL_CALL, tool_name="calculator",
        tool_args={"expr": "6*7"}, meta={},
    )

    _execute(state, action, tools, log=None, user_io=user_io)

    assert user_io.confirm_calls == []


def test_batch_mode_bypasses_permission_gate() -> None:
    """With user_io=None (GAIA), gated tools must run without prompting."""
    state = _state()

    tools = ToolRegistry([JsonTool(
        name="run_shell", description="shell",
        input_schema={"type": "object", "properties": {}},
        handler=lambda args: {"ok": True, "stdout": ""},
    )])
    action = Action(
        kind=ActionKind.TOOL_CALL, tool_name="run_shell",
        tool_args={"command": "echo ok"}, meta={},
    )

    obs = _execute(state, action, tools, log=None, user_io=None)

    assert obs.ok is True


# ---------------------------------------------------------------------------
# ReActPlusPolicy
# ---------------------------------------------------------------------------


def test_react_plus_intercepts_ask_user_tool_call() -> None:
    """When the LLM emits a tool_call for `ask_user`, the policy must
    convert it to ActionKind.ASK_USER (mirroring how `final_answer` is
    intercepted)."""
    policy = ReActPlusPolicy()

    class _LLM:
        provider = "openai"

        def chat(self, **_kwargs: Any) -> LLMReply:
            return LLMReply(
                provider="openai",
                model="gpt-5-nano",
                content=[ContentBlock(
                    kind="tool_use", id="x", name="ask_user",
                    arguments={"question": "Which file did you mean?"},
                )],
                stop_reason="tool_use",
                usage=Usage(tokens_in=10, tokens_out=5),
            )

    state = _state(question="please process my file")
    action = policy.propose(state, llm=_LLM(), tools=ToolRegistry())

    assert action.kind == ActionKind.ASK_USER
    assert action.text == "Which file did you mean?"
    assert action.meta.get("via_ask_user_tool") is True


def test_ask_user_reply_is_folded_into_conversation_history() -> None:
    """After an answered ASK_USER step, the built message history must
    contain both the question (assistant) and the reply (user). Regression
    for the session where the model re-asked the same clarifying question
    3x because the answers never reached its context."""
    policy = ReActPlusPolicy()
    state = _state(question="top 5 restaurants in Highland Park")
    state.append_step(
        Action(kind=ActionKind.ASK_USER,
               text="Which rating source — Google, Yelp, or critics?", meta={}),
        Observation(ok=True, text="use google ratings",
                    data={"question": "Which rating source — Google, Yelp, or critics?",
                          "reply": "use google ratings"}),
    )

    msgs = policy._history(state)
    flat = " ".join(
        b.text or ""
        for m in msgs for b in m.content
        if getattr(b, "kind", None) == "text"
    )
    assert "Which rating source" in flat   # the question is replayed
    assert "use google ratings" in flat    # the answer is replayed
    # The reply must be a user-role turn (so the model treats it as input,
    # not its own thought).
    assert any(
        m.role == "user" and any(
            (b.text or "") == "use google ratings" for b in m.content
        )
        for m in msgs
    )


def test_react_plus_advertises_ask_user_in_tool_specs() -> None:
    """The model should see `ask_user` in the tools list — it's the
    only way it can emit the call. The augmentation must not mutate
    the underlying registry (shared with other policies)."""
    policy = ReActPlusPolicy()
    inner = ToolRegistry()
    captured: list[Any] = []

    class _LLM:
        provider = "openai"

        def chat(self, *, tools=(), **_kwargs: Any) -> LLMReply:
            captured.append(list(tools))
            return LLMReply(
                provider="openai", model="m", content=[],
                stop_reason="stop", usage=Usage(),
            )

    policy.propose(_state(), llm=_LLM(), tools=inner)

    tool_names = [t.name for t in captured[0]]
    assert "ask_user" in tool_names
    # Underlying registry untouched.
    assert "ask_user" not in inner.names()


def test_react_plus_prompt_contains_error_scoping_guidance() -> None:
    """The augmented system prompt must include the two guardrails."""
    policy = ReActPlusPolicy()
    assert "specific call" in policy.system_prompt.lower()
    assert "ask_user" in policy.system_prompt
    assert "2 consecutive" in policy.system_prompt.lower()


def test_react_plus_name_is_literal_react_plus() -> None:
    """The CLI surfaces this name in events / ablation tables."""
    assert ReActPlusPolicy().name == "react+"
