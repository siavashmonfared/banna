"""Tests for the ReActPolicy action-intent guard.

The guard catches the "model returned code-as-text instead of using a
tool" failure mode that produced the empty-file bug in the user's
real-estate-estimator transcript.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from banna_agent.core.state import AgentState
from banna_agent.core.types import (
    Action,
    ActionKind,
    Budget,
    Observation,
)
from banna_agent.llm.base import ContentBlock, LLMReply, Usage
from banna_agent.policies.react import (
    ReActPolicy,
    _looks_like_action_request,
    _reply_looks_like_code_explanation,
    _trace_has_successful_action,
)
from banna_agent.tools.base import ToolRegistry
from banna_agent.tools.calculator import make_calculator_tool


@dataclass
class _ScriptedLLM:
    replies: list[LLMReply]
    calls: list[dict[str, Any]] = field(default_factory=list)
    provider: str = "scripted"
    def chat(self, **kwargs: Any) -> LLMReply:
        self.calls.append(kwargs)
        if not self.replies:
            return LLMReply(provider="scripted", model="s", content=[], stop_reason="end_turn")
        return self.replies.pop(0)


def _text_reply(t: str) -> LLMReply:
    return LLMReply(
        provider="scripted", model="s",
        content=[ContentBlock(kind="text", text=t)],
        stop_reason="end_turn",
        usage=Usage(tokens_in=10, tokens_out=3),
    )


def _tool_reply(name: str, args: dict) -> LLMReply:
    return LLMReply(
        provider="scripted", model="s",
        content=[ContentBlock(kind="tool_use", id="t1", name=name, arguments=args)],
        stop_reason="tool_use",
        usage=Usage(tokens_in=20, tokens_out=5),
    )


def _calc_tools() -> ToolRegistry:
    return ToolRegistry([make_calculator_tool()])


# ---------------------------------------------------------------------------
# _looks_like_action_request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "develop a simple code to estimate price and save to ~/Desktop/tmp",
    "write a python script to /tmp/x.py",
    "create a file at ./output.csv",
    "run the build and save logs to ~/logs",
    "save it to /home/me/out.json",
    "generate a config.yaml in ~/Desktop",
])
def test_action_request_detection_true_positives(question: str) -> None:
    assert _looks_like_action_request(question) is True


@pytest.mark.parametrize("question", [
    "what is the capital of France?",
    "explain how python decorators work",
    "compare numpy and pandas",
    "what is 2+2?",
    "write a poem about clouds",      # verb but no FS hint
    "the path to ~/Desktop is what?",  # FS hint but no action verb
    "",
])
def test_action_request_detection_true_negatives(question: str) -> None:
    assert _looks_like_action_request(question) is False


# ---------------------------------------------------------------------------
# _reply_looks_like_code_explanation
# ---------------------------------------------------------------------------


def test_reply_with_triple_backticks_is_code_blob() -> None:
    text = "Here's the script:\n```python\nimport os\nprint(1)\n```\nSave it."
    assert _reply_looks_like_code_explanation(text) is True


def test_reply_with_multiple_imports_is_code_blob() -> None:
    text = "import os\nimport sys\nfrom pathlib import Path\n\ndef main():\n    pass"
    assert _reply_looks_like_code_explanation(text) is True


def test_plain_text_reply_is_not_code_blob() -> None:
    text = "The answer is 42. The capital is Paris. No code here."
    assert _reply_looks_like_code_explanation(text) is False


def test_empty_text_is_not_code_blob() -> None:
    assert _reply_looks_like_code_explanation("") is False


# ---------------------------------------------------------------------------
# _trace_has_successful_action
# ---------------------------------------------------------------------------


def test_empty_trace_has_no_successful_action() -> None:
    s = AgentState(question="q")
    assert _trace_has_successful_action(s) is False


def test_trace_with_successful_tool_call() -> None:
    s = AgentState(question="q")
    s.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="calculator", tool_args={"expression": "1+1"}),
        Observation(ok=True, text="2"),
    )
    assert _trace_has_successful_action(s) is True


def test_trace_with_failed_tool_call_only() -> None:
    s = AgentState(question="q")
    s.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="x", tool_args={}),
        Observation(ok=False, error="boom"),
    )
    assert _trace_has_successful_action(s) is False


# ---------------------------------------------------------------------------
# Policy integration: the guard fires
# ---------------------------------------------------------------------------


_CODE_BLOB_REPLY = """Here's a Python script you can save to ~/Desktop/tmp/estimate.py:

```python
import os
import math
from pathlib import Path

def estimate(address):
    return 0

if __name__ == "__main__":
    estimate("foo")
```

Save the above and run it with `python3 estimate.py`.
"""


def test_guard_fires_on_action_question_with_code_blob_reply() -> None:
    """The exact failure mode from the user's real-estate transcript."""
    llm = _ScriptedLLM([_text_reply(_CODE_BLOB_REPLY)])
    policy = ReActPolicy(action_intent_guard=True)
    state = AgentState(
        question="develop a simple code to estimate my real estate, save to ~/Desktop/tmp",
        budget=Budget(max_steps=5, max_wall_s=5.0),
    )
    action = policy.propose(state, llm=llm, tools=_calc_tools())
    assert action.kind == ActionKind.THINK
    assert action.meta.get("guard") == "action_intent"
    assert "actually perform" in action.text.lower() or "python_sandbox" in action.text


def test_guard_disabled_returns_legacy_final_answer() -> None:
    """`action_intent_guard=False` restores pre-patch behavior."""
    llm = _ScriptedLLM([_text_reply(_CODE_BLOB_REPLY)])
    policy = ReActPolicy(action_intent_guard=False)
    state = AgentState(
        question="develop a simple code to estimate my real estate, save to ~/Desktop/tmp",
        budget=Budget(max_steps=5, max_wall_s=5.0),
    )
    action = policy.propose(state, llm=llm, tools=_calc_tools())
    assert action.kind == ActionKind.FINAL_ANSWER


def test_guard_does_not_fire_on_qa_question() -> None:
    """A QA-shaped question + text reply still becomes FINAL_ANSWER."""
    llm = _ScriptedLLM([_text_reply("The answer is 42.")])
    policy = ReActPolicy(action_intent_guard=True)
    state = AgentState(
        question="What is 2+40?",
        budget=Budget(max_steps=5, max_wall_s=5.0),
    )
    action = policy.propose(state, llm=llm, tools=_calc_tools())
    assert action.kind == ActionKind.FINAL_ANSWER


def test_guard_does_not_fire_when_tool_call_present() -> None:
    """An action question with a tool_call reply → TOOL_CALL (regression)."""
    llm = _ScriptedLLM([_tool_reply("calculator", {"expression": "1+1"})])
    policy = ReActPolicy(action_intent_guard=True)
    state = AgentState(
        question="write the result of 1+1 to /tmp/x.txt",
        budget=Budget(max_steps=5, max_wall_s=5.0),
    )
    action = policy.propose(state, llm=llm, tools=_calc_tools())
    assert action.kind == ActionKind.TOOL_CALL


def test_guard_does_not_loop_after_successful_action() -> None:
    """Once a tool_call succeeded, a follow-up text reply is a real
    FINAL_ANSWER even if it contains code (e.g. summarizing what ran)."""
    llm = _ScriptedLLM([_text_reply(_CODE_BLOB_REPLY)])
    policy = ReActPolicy(action_intent_guard=True)
    state = AgentState(
        question="develop a simple code and save to ~/Desktop/tmp",
        budget=Budget(max_steps=5, max_wall_s=5.0),
    )
    # Pre-seed the trace with a successful tool_call.
    state.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="calculator", tool_args={"expression": "1"}),
        Observation(ok=True, text="1"),
    )
    action = policy.propose(state, llm=llm, tools=_calc_tools())
    assert action.kind == ActionKind.FINAL_ANSWER


def test_guard_does_not_fire_on_plain_text_action_reply() -> None:
    """If the model replied to an action question with plain text (no
    code blob), we still let it through. The guard is conservative on
    purpose — it only catches the high-confidence failure mode."""
    llm = _ScriptedLLM([_text_reply("Done. I wrote the file as requested.")])
    policy = ReActPolicy(action_intent_guard=True)
    state = AgentState(
        question="create a file at ~/Desktop/x.py",
        budget=Budget(max_steps=5, max_wall_s=5.0),
    )
    action = policy.propose(state, llm=llm, tools=_calc_tools())
    # Plain text → falls through to FINAL_ANSWER. (False-negative is
    # the right trade-off vs. nagging the user on borderline cases.)
    assert action.kind == ActionKind.FINAL_ANSWER
