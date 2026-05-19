"""Deterministic JSONL event log.

Every tick of the driver emits one or more events. The log is the canonical
replayable record of a run — if tests fail, the log tells us exactly what
the agent saw, decided, and did. Keep the schema flat and stable.

Event schema:
    {
      "run_id":   str,     # AgentState.trace.run_id
      "ts":       str,     # ISO 8601 UTC, second resolution
      "step":     int,     # trace.steps index this event belongs to, or -1
      "kind":     str,     # one of EventKind values
      "payload":  dict,    # kind-specific data
    }

EventKinds are plain strings (not Enum) so logs remain readable as JSON
without coercion and so new kinds can be added without breaking replay.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .types import _now_iso


# ---------------------------------------------------------------------------
# Event kinds
# ---------------------------------------------------------------------------

class EventKind:
    RUN_START = "run_start"
    RUN_END = "run_end"
    PROPOSE = "propose"             # a Policy decided on an Action
    LLM_CALL = "llm_call"           # an LLMReply was observed
    TOOL_CALL = "tool_call"         # a tool was invoked
    TOOL_RESULT = "tool_result"     # a tool returned
    OBSERVATION = "observation"     # a Step was appended
    BUDGET = "budget"               # budget state changed
    VERIFIER = "verifier"           # a verifier ran
    COMPACT = "compact"             # trace compaction occurred
    ERROR = "error"                 # driver-level error


# ---------------------------------------------------------------------------
# Event container
# ---------------------------------------------------------------------------


@dataclass
class AgentEvent:
    run_id: str
    step: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Event log (in-memory + optional JSONL file)
# ---------------------------------------------------------------------------


class EventLog:
    """Thin append-only buffer plus optional on-disk JSONL mirror.

    Hand one of these to the driver. Use `events[]` for in-process
    assertions in tests, `log.path` for disk inspection in real runs.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self.events: list[AgentEvent] = []
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate on new run so the JSONL is one-run-per-file.
            self._path.write_text("")

    def emit(self, event: AgentEvent) -> None:
        self.events.append(event)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")

    @property
    def path(self) -> Path | None:
        return self._path

    def filter(self, kind: str) -> list[AgentEvent]:
        return [e for e in self.events if e.kind == kind]


# ---------------------------------------------------------------------------
# Convenience emitters
# ---------------------------------------------------------------------------


def emit(
    log: EventLog | None,
    *,
    run_id: str,
    step: int,
    kind: str,
    **payload: Any,
) -> None:
    """Shortcut. Null-safe: `log=None` is a no-op (useful in tests)."""
    if log is None:
        return
    log.emit(AgentEvent(run_id=run_id, step=step, kind=kind, payload=payload))
