"""`recall_evidence` — re-read the full text of evidence the agent already saw.

B1 ("bounded history projection") trims bulky tool observations (a 10k-char
page read, a long file) down to a short snippet in the replayed message
history, so the agent stops re-paying for every past page on every turn.
The full text is not lost: it still lives verbatim in the trace
(`state.trace.steps[i].observation.data`). This tool is the safety net —
when the snippet isn't enough, the model calls `recall_evidence` with the
`evidence_id` shown in the truncation marker and gets the full text back
*for free* (no network re-fetch, no re-billing of unrelated history).

The tool reads the live run via :mod:`core.run_context` rather than closing
over a state, because the tool registry is built once and shared across
every run (see that module's docstring).
"""
from __future__ import annotations

from typing import Any

from ..core.run_context import get_current_state
from .base import JsonTool

# Keys under which tools stash their bulky text payload, in priority order.
_TEXT_KEYS: tuple[str, ...] = ("text", "content", "markdown", "body")


def _full_text_for(state: Any, evidence_id: str) -> dict[str, Any]:
    """Resolve one evidence id to its full stored payload.

    Looks the Evidence up by id, follows `origin_step` back to the trace
    observation that produced it, and returns the bulky text field. Falls
    back to the (capped) `Evidence.content` if the originating observation
    is gone or carried no text body.
    """
    ev = next((e for e in state.evidence if e.evidence_id == evidence_id), None)
    if ev is None:
        return {"evidence_id": evidence_id, "error": "no such evidence_id"}

    out: dict[str, Any] = {
        "evidence_id": evidence_id,
        "source": ev.source,
        "title": str((ev.meta or {}).get("title") or ""),
    }

    # Pull the full text back out of the originating trace observation.
    text: str | None = None
    if ev.origin_step is not None:
        step = next(
            (s for s in state.trace.steps if s.idx == ev.origin_step), None
        )
        data = getattr(step.observation, "data", None) if step else None
        if isinstance(data, dict):
            for k in _TEXT_KEYS:
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    text = v
                    break

    # Fall back to the registered (snippet-capped) content.
    if text is None:
        text = ev.content or ""

    out["text"] = text
    out["chars"] = len(text)
    return out


def _handle(args: dict[str, Any]) -> dict[str, Any]:
    state = get_current_state()
    if state is None:
        raise ValueError(
            "recall_evidence is only available inside an agent run"
        )

    ids: list[str] = []
    single = args.get("evidence_id")
    if isinstance(single, str) and single.strip():
        ids.append(single.strip())
    many = args.get("evidence_ids")
    if isinstance(many, list):
        ids.extend(str(x).strip() for x in many if str(x).strip())

    # De-dupe, preserve order.
    seen: set[str] = set()
    ids = [i for i in ids if not (i in seen or seen.add(i))]

    if not ids:
        raise ValueError("provide 'evidence_id' (string) or 'evidence_ids' (list)")

    results = [_full_text_for(state, eid) for eid in ids]
    if len(results) == 1:
        return results[0]
    return {"count": len(results), "results": results}


RECALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "evidence_id": {
            "type": "string",
            "description": (
                "The evidence_id to re-read in full (as shown in the "
                "'… truncated — recall_evidence(...)' marker on a trimmed "
                "tool result)."
            ),
        },
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Recall several pieces of evidence at once.",
        },
    },
    "additionalProperties": False,
}


def make_recall_tool() -> JsonTool:
    """Build the `recall_evidence` tool. It reads the live run via the
    current-state contextvar, so it needs no per-run construction."""
    return JsonTool(
        name="recall_evidence",
        description=(
            "Re-read the FULL text of a page/file/result you already fetched "
            "earlier, by its evidence_id. Older results are shown trimmed to a "
            "snippet to save context; call this to get the complete text back "
            "(no re-download). Use the evidence_id from the truncation marker."
        ),
        input_schema=RECALL_SCHEMA,
        handler=_handle,
        capabilities=frozenset({"state"}),
    )
