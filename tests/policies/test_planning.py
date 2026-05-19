"""Tests for the shared planning helpers (`policies/_planning.py`)."""
from __future__ import annotations

from dataclasses import dataclass, field

from banna_agent.llm.base import ContentBlock, LLMReply, Usage
from banna_agent.policies._planning import (
    Plan,
    _extract_json,
    propose_candidate_plans,
    propose_plan,
)


@dataclass
class _FakeLLM:
    replies: list[LLMReply] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    provider: str = "fake"

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0) if self.replies else LLMReply(
            provider="fake", model="m", content=[], stop_reason="end_turn"
        )


def _reply(text: str) -> LLMReply:
    return LLMReply(
        provider="fake", model="m",
        content=[ContentBlock(kind="text", text=text)],
        stop_reason="end_turn",
        usage=Usage(tokens_in=5, tokens_out=3),
    )


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


def test_extract_json_bare_object() -> None:
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_in_fence() -> None:
    s = 'here you go: ```json\n{"a": 2}\n```'
    assert _extract_json(s) == {"a": 2}


def test_extract_json_first_of_many() -> None:
    s = 'text {"a": 1, "b": [1,2]} more text'
    assert _extract_json(s) == {"a": 1, "b": [1, 2]}


def test_extract_json_malformed_returns_none() -> None:
    assert _extract_json("no json here") is None
    assert _extract_json("{not valid}") is None


def test_extract_json_empty() -> None:
    assert _extract_json("") is None


# ---------------------------------------------------------------------------
# propose_plan
# ---------------------------------------------------------------------------


def test_propose_plan_happy_path() -> None:
    llm = _FakeLLM([_reply('{"plan": ["find X", "compute Y", "summarize"]}')])
    p = propose_plan(llm, "What is X?")
    assert p.steps == ["find X", "compute Y", "summarize"]
    assert p.depth == 3
    assert p.meta["tokens_in"] == 5


def test_propose_plan_handles_fenced_json() -> None:
    llm = _FakeLLM([_reply('```json\n{"plan": ["step a", "step b"]}\n```')])
    p = propose_plan(llm, "q")
    assert p.steps == ["step a", "step b"]


def test_propose_plan_parse_fail_returns_empty() -> None:
    llm = _FakeLLM([_reply("nope, no json")])
    p = propose_plan(llm, "q")
    assert p.steps == []
    assert p.meta.get("parse_fail") is True


def test_propose_plan_llm_exception_returns_empty() -> None:
    class _BrokenLLM:
        provider = "broken"
        def chat(self, **_): raise RuntimeError("boom")

    p = propose_plan(_BrokenLLM(), "q")
    assert p.steps == []
    assert "error" in p.meta
    assert "RuntimeError" in p.meta["error"]


def test_propose_plan_strips_empty_steps() -> None:
    llm = _FakeLLM([_reply('{"plan": ["step a", "", "  ", "step b"]}')])
    p = propose_plan(llm, "q")
    assert p.steps == ["step a", "step b"]


# ---------------------------------------------------------------------------
# propose_candidate_plans
# ---------------------------------------------------------------------------


def test_propose_candidate_plans_returns_list_of_plans() -> None:
    llm = _FakeLLM([_reply(
        '{"plans": ['
        '["a1", "a2"], '
        '["b1", "b2", "b3"]'
        ']}'
    )])
    plans = propose_candidate_plans(llm, "q", n_candidates=2)
    assert len(plans) == 2
    assert plans[0].steps == ["a1", "a2"]
    assert plans[1].steps == ["b1", "b2", "b3"]
    assert plans[0].meta["branch_id"] == 0
    assert plans[1].meta["branch_id"] == 1


def test_propose_candidate_plans_skips_empty_and_non_list() -> None:
    llm = _FakeLLM([_reply(
        '{"plans": [["a1"], "not a list", [], ["b1", "b2"]]}'
    )])
    plans = propose_candidate_plans(llm, "q", n_candidates=4)
    assert len(plans) == 2
    assert plans[0].steps == ["a1"]
    assert plans[1].steps == ["b1", "b2"]


def test_propose_candidate_plans_parse_fail_returns_empty() -> None:
    llm = _FakeLLM([_reply("not valid")])
    assert propose_candidate_plans(llm, "q") == []


def test_plan_with_step_result_pads() -> None:
    p = Plan(steps=["a", "b", "c"])
    p.with_step_result(1, {"x": 1})
    assert p.step_results[0] == {}
    assert p.step_results[1] == {"x": 1}
    assert p.depth == 3


def test_plan_is_done_when_final_answer_set() -> None:
    p = Plan(steps=["a"])
    assert not p.is_done
    p.final_answer = "42"
    assert p.is_done
