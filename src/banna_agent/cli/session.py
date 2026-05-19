"""Conversational session state.

Holds:
  * `turns`            — chronological list of (question, answer, AgentState).
  * `memory_store`     — the InMemoryStore the `memory` tool persists into.
                         Persists across turns within a single session, so
                         "remember that X" on turn N is recallable on turn N+1.
  * helpers to format a recent-context preamble for follow-up questions and
    to save / load the transcript as JSONL.

Each new question gets a fresh `AgentState` with a fresh `Budget`. The
*conversational* feel is achieved purely by injecting a short text
preamble of recent Q/A pairs into the new question — we never grow a
single AgentState across turns. That keeps the substrate semantics
clean and matches how the GAIA harness uses the agent.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.state import AgentState
from ..memory.embeddings import HashEmbedder
from ..memory.in_memory_store import InMemoryStore
from ..memory.jsonl_store import JSONLStore
from ..memory.skill_library import SkillLibrary


# Default location for the persistent memory file. Honors XDG conventions
# but defaults to the simpler ~/.config/myagent/memory.jsonl.
DEFAULT_MEMORY_PATH = Path(
    os.environ.get(
        "MYAGENT_MEMORY_PATH",
        str(Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
            / "myagent" / "memory.jsonl"),
    )
).expanduser()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Turn:
    """One Q/A pair from the session."""

    question: str
    answer: str
    ts: str = field(default_factory=_now_iso)
    wall_s: float = 0.0
    steps_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    provider: str = ""
    model: str = ""
    policy: str = ""
    budget_reason: str = "ok"
    error: str | None = None
    # Reserved: pointer to the saved trace path on disk for replay.
    trace_path: str | None = None


@dataclass
class Session:
    """Conversation state shared across turns.

    Memory persistence:
      - `memory_store`, when None, defaults to a `JSONLStore` at
        DEFAULT_MEMORY_PATH (or `MYAGENT_MEMORY_PATH` env var). That
        means anything the `memory` tool writes survives quitting and
        re-launching the CLI.
      - Pass `memory_store=InMemoryStore(...)` to opt into volatile
        memory (e.g. for tests).
      - `clear()` defaults to clearing only the *transcript*, not the
        memory store. Pass `wipe_memory=True` to also drop persistent
        memory; the JSONL file is truncated.
    """

    turns: list[Turn] = field(default_factory=list)
    memory_store: Any | None = None  # JSONLStore | InMemoryStore
    memory_path: Path | None = None
    last_state: AgentState | None = None
    started_at: str = field(default_factory=_now_iso)
    # SkillLibrary is built over the same memory store; lazily set in
    # __post_init__. Tools that consume the skill library read it from
    # here.
    skill_library: SkillLibrary | None = None

    def __post_init__(self) -> None:
        if self.memory_store is None:
            self.memory_path = self.memory_path or DEFAULT_MEMORY_PATH
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_store = JSONLStore(
                self.memory_path,
                embedder=HashEmbedder(dim=256),
            )
        if self.skill_library is None:
            self.skill_library = SkillLibrary(self.memory_store)

    # --- recording --------------------------------------------------------

    def add_turn(self, turn: Turn, state: AgentState) -> None:
        self.turns.append(turn)
        self.last_state = state

    # --- preamble for follow-ups -----------------------------------------

    def format_preamble(self, max_turns: int = 4, max_chars: int = 1500) -> str:
        """Render the recent Q/A history as a context preamble.

        Injected into each new question on subsequent turns so the model
        can resolve references like "and what about France?" without us
        carrying a single AgentState across turns.
        """
        if not self.turns:
            return ""
        recent = self.turns[-max_turns:]
        lines = ["Recent conversation (most recent last):"]
        for t in recent:
            q = _truncate(t.question, 200)
            a = _truncate(t.answer, 200)
            lines.append(f"  Q: {q}")
            lines.append(f"  A: {a}")
        out = "\n".join(lines)
        if len(out) > max_chars:
            out = out[-max_chars:]
            cut = out.find("\n")
            if cut >= 0:
                out = out[cut + 1 :]
            out = "Recent conversation (truncated):\n" + out
        return out

    def compose_question(
        self,
        user_text: str,
        *,
        skill_header: str = "",
    ) -> str:
        """Wrap the user's text with the recent-conversation preamble + optional skill header."""
        parts: list[str] = []
        preamble = self.format_preamble()
        if preamble:
            parts.append(preamble)
        if skill_header:
            parts.append(skill_header)
        if preamble or skill_header:
            parts.append(f"New question: {user_text}")
        else:
            parts.append(user_text)
        return "\n\n".join(parts)

    # --- persistence ------------------------------------------------------

    def save_jsonl(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "__session__": True,
                "started_at": self.started_at,
                "n_turns": len(self.turns),
            }) + "\n")
            for t in self.turns:
                f.write(json.dumps(asdict(t), default=str) + "\n")
        return p

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "Session":
        p = Path(path).expanduser().resolve()
        sess = cls()
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("__session__"):
                sess.started_at = row.get("started_at", sess.started_at)
                continue
            sess.turns.append(Turn(**row))
        return sess

    # --- maintenance ------------------------------------------------------

    def clear(self, *, wipe_memory: bool = False) -> None:
        """Drop transcript. Memory store is preserved unless wipe_memory=True.

        Started-at is preserved either way; the session is the same
        thing semantically, just with a fresh transcript window.
        """
        self.turns.clear()
        self.last_state = None
        if wipe_memory:
            if self.memory_path is not None and self.memory_path.exists():
                # Truncate the JSONL file and rebuild an empty store.
                self.memory_path.write_text("")
                self.memory_store = JSONLStore(
                    self.memory_path, embedder=HashEmbedder(dim=256),
                )
            else:
                # In-memory only — just rebuild it.
                self.memory_store = InMemoryStore(embedder=HashEmbedder(dim=256))
            self.skill_library = SkillLibrary(self.memory_store)


def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"
