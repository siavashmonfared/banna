"""final_answer — the structured commit channel.

Most ReAct-style agents conflate "the model emitted text" with "the
model is done", which forces the model to choose between thinking out
loud and producing a terse committed answer in the same turn. On
benchmarks scored with exact-match (GAIA, MMLU-style), that conflation
costs accuracy: the model writes 'Looking at the data: ... so 17' and
the scorer rejects the whole string even though the literal answer is
right there.

This tool separates the two slots:

  * `reasoning`  — free-form summary of how the answer was derived.
                   The scorer ignores this field. The model can put
                   whatever preamble it wants here.
  * `answer`     — ONLY the literal answer string. The scorer reads
                   exactly this. A `maxLength` is set high enough to
                   cover any legitimate GAIA answer (long paper titles,
                   comma-lists, reversed-sentence translations) while
                   still preventing the model from dumping a 2 KB
                   reasoning paragraph into the field.

The tool *handler* is a no-op pass-through: returning the args is
enough. The ReAct policy intercepts calls to this tool and converts
them into a FINAL_ANSWER action; the handler exists so that a registry
lookup for "final_answer" succeeds (the registry insists every
declared tool has a callable).
"""
from __future__ import annotations

from typing import Any

from .base import JsonTool


def _handler(args: dict[str, Any]) -> dict[str, Any]:
    answer = str(args.get("answer", "")).strip()
    reasoning = str(args.get("reasoning", "")).strip()
    return {"answer": answer, "reasoning": reasoning}


FINAL_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": (
                "Optional brief summary of how you arrived at the answer. "
                "The scorer ignores this field — put any preamble, framing, "
                "or working-out here so it doesn't pollute `answer`."
            ),
            "maxLength": 1000,
        },
        "answer": {
            "type": "string",
            "description": (
                "ONLY the literal answer string. No 'The answer is...', "
                "no 'Based on...', no quotes around it, no markdown, no "
                "trailing period unless the answer literally ends with one. "
                "Examples (good): '17', '0.1777', 'Guatemala', 'b, e', "
                "'Mapping Human Oriented Information to Software Agents for "
                "Online Systems Usage'. Examples (bad): 'The answer is 17.', "
                "'Looking at the data: 17', '\"17\"'."
            ),
            "maxLength": 500,
        },
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional list of evidence_id strings that you used to derive "
                "the answer. These IDs appear in tool results (search hits, "
                "fetched URLs, file reads) under the `evidence_id` field. "
                "Cite the specific pieces of evidence whose content supports "
                "your answer. The citation verifier will check that your "
                "answer is actually supported by the cited evidence; "
                "fabricated answers with no grounding will be rejected."
            ),
        },
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def make_final_answer_tool() -> JsonTool:
    return JsonTool(
        name="final_answer",
        description=(
            "Commit your final answer. Call this exactly once per task, "
            "when you are ready to commit. The `answer` field must contain "
            "ONLY the literal answer the scorer will read; put any "
            "explanation in the optional `reasoning` field. After this "
            "tool is called the task terminates."
        ),
        input_schema=FINAL_ANSWER_SCHEMA,
        handler=_handler,
        capabilities=frozenset({"pure", "terminal"}),
    )
