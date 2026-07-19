"""ReAct policy — the baseline tool-use loop, implemented as a Policy.

Design: this is a *Policy*, not the loop. The loop lives in
`core/agent.py::run_policy`. A Policy is a function `(state) -> Action`.
ReAct's decision rule is:

  1. Convert the current AgentState into a prompt (question + trace
     summary + tool results).
  2. Call the LLM, declaring the available tools.
  3. If the LLM returned tool_calls: emit a single TOOL_CALL when there's
     one, or a TOOL_BATCH when the model emitted ≥2 in the same reply (B7 —
     parallel tool-use, dispatched concurrently by the driver and replayed
     as parallel tool_use/tool_result blocks in `_history`).
  4. Otherwise, if the LLM returned text, emit FINAL_ANSWER with the
     text.
  5. On any parse failure, emit a THINK action with the error text and
     let the driver retry next tick.

The *transition* (tool dispatch, observation, state update) happens in
the driver. ReAct never touches tools or state mutation directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.state import AgentState
from ..core.types import Action, ActionKind
from ..llm.base import ContentBlock, LLMClient, Message, ToolSpec
from ..tools.base import ToolRegistry


def _force_tool_extra(provider: str, tool_name: str | None) -> dict[str, Any]:
    """Return provider-specific ``extra`` kwargs that force the next LLM
    reply to call a tool.

    When ``tool_name`` is given, pin the reply to that specific tool.
    When ``tool_name`` is None, force the reply to call *some* tool
    (any-tool mode) — useful when the model has been silent for several
    turns and we have no evidence yet so committing `final_answer`
    would just commit garbage.

    Ollama support for forced tools varies wildly across local models —
    we return an empty dict for ollama and rely on the consecutive-
    commit-required hard cap below to escape the loop.
    """
    p = (provider or "").lower()
    if p in ("anthropic", "bedrock"):
        if tool_name is None:
            return {"tool_choice": {"type": "any"}}
        return {"tool_choice": {"type": "tool", "name": tool_name}}
    if p == "openai":
        if tool_name is None:
            return {"tool_choice": "required"}
        return {
            "tool_choice": {
                "type": "function",
                "function": {"name": tool_name},
            }
        }
    if p == "gemini":
        cfg: dict[str, Any] = {"mode": "ANY"}
        if tool_name is not None:
            cfg["allowed_function_names"] = [tool_name]
        return {"tool_config": {"function_calling_config": cfg}}
    return {}


def _count_consecutive_empty_replies(state: AgentState) -> int:
    """How many `[empty_reply]` THINKs sit at the trace tail.

    Separate from `_count_consecutive_commit_required` so we can branch
    on the difference: empty replies with no evidence mean the model
    has stopped responding *and* has nothing to commit; the right move
    is to force any tool call, not force `final_answer`.
    """
    n = 0
    for step in reversed(state.trace.steps):
        m = step.action.meta or {}
        if m.get("empty_reply"):
            n += 1
        else:
            break
    return n


def _trace_has_evidence(state: AgentState) -> bool:
    """True if the trace contains any successful tool call.

    Used to choose between `tool_choice=required` (still gathering) vs
    `tool_choice=final_answer` (have evidence, just need to commit) when
    escaping the empty-reply loop.
    """
    for step in state.trace.steps:
        if step.action.kind == ActionKind.TOOL_CALL and step.observation.ok:
            return True
    return False


def _count_consecutive_commit_required(state: AgentState) -> int:
    """How many commit_required / empty_reply THINKs sit at the trace tail.

    Used to (a) escalate to forced tool_choice after one nudge, and
    (b) hard-cap the loop at two so we never burn the step budget on
    identical nudges (the ec09fa32 ping-pong-riddle failure).
    """
    n = 0
    for step in reversed(state.trace.steps):
        m = step.action.meta or {}
        text = step.action.text or ""
        if m.get("commit_required") or text.startswith("[empty_reply]"):
            n += 1
        else:
            break
    return n


def _last_preceding_text(state: AgentState) -> str:
    """Recover the most recent plain-text reply captured in a
    commit_required THINK's meta. Used as the bailout answer when the
    loop hits its hard cap."""
    for step in reversed(state.trace.steps):
        m = step.action.meta or {}
        txt = m.get("preceding_text") or m.get("commit_required_text") or ""
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
    return ""


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful agent. Two modes:\n"
    "  (a) RESEARCH/QA: use tools to gather evidence, then COMMIT to a "
    "concise final answer.\n"
    "  (b) ACTION: if the user asks you to create, modify, save, or "
    "delete a file; run, execute, install, or build something; write "
    "code to disk; take any side effect — you MUST use the appropriate "
    "tool to perform it. The tools available include `python_sandbox` "
    "(run Python code, including file writes via pathlib), `run_shell` "
    "(execute shell commands), `read_file` / `list_files` / `grep` "
    "(inspect the filesystem). Returning the code or command as text "
    "DOES NOT COUNT as completing the task. The user expects bytes on "
    "disk or output from execution, not an explanation.\n\n"
    "How to operate:\n"
    "  • Use tools when you don't know an answer with confidence, OR "
    "when the user wants you to do something rather than answer.\n"
    "  • For conversational questions (greetings, basic facts you "
    "already know), answer directly with no tools.\n"
    "  • After 2-4 tool calls on a research topic you usually have what "
    "you need. Stop searching and commit. Do NOT keep hunting for a "
    "perfect-match \"official\" source once you have a defensible "
    "answer from the data already gathered.\n"
    "  • When the question asks for a number: COMPUTE it from the "
    "intermediate values yourself (use calculator if needed) and give "
    "the number with units.\n"
    "  • When the question asks for a name, year, or list: just give "
    "that, terse, no preamble.\n"
    "  • For ACTION tasks: after the tool actually succeeds (you see "
    "the observation), then emit a short FINAL_ANSWER confirming what "
    "was done — e.g. \"Wrote 47 lines to /path/file.py and ran it; "
    "output: ...\".\n"
    "  • Once you have enough evidence (research) OR the side-effect "
    "is confirmed (action): return PLAIN TEXT with the answer and no "
    "further tool calls.\n\n"
    "Common failure modes to AVOID:\n"
    "  • RETURNING CODE OR COMMANDS AS TEXT WHEN THE USER ASKED YOU TO "
    "RUN OR WRITE THEM. This is the single most common failure on "
    "action tasks. If you catch yourself about to say \"save this to "
    "X\" or \"run this command\", instead use a tool to do it.\n"
    "  • Looping on additional searches when you already have a "
    "defensible answer (the most common research-task mistake).\n"
    "  • Hedging with multiple variants instead of picking one and "
    "committing.\n"
    "  • Stalling on minor precision when the question only needs an "
    "order-of-magnitude or an approximate value.\n\n"
    "Context note: to save space, the FULL text of a page or file you "
    "fetched is kept only for your MOST RECENT tool result; older results "
    "are shown trimmed to a snippet ending in a '… trimmed' marker. If you "
    "need the full text of an earlier result back, call "
    "`recall_evidence` with the evidence_id shown in that marker — it "
    "returns the complete text with no re-download."
)


# ---------------------------------------------------------------------------
# Action-intent guard
# ---------------------------------------------------------------------------


# Imperative verbs that signal the user wants a side effect, not an
# explanation. Match case-insensitively as standalone words.
_ACTION_VERBS = frozenset({
    "create", "make", "write", "save", "generate", "produce",
    "run", "execute", "launch", "start",
    "install", "build", "compile", "deploy", "setup", "set",
    "modify", "edit", "patch", "rename", "delete", "remove",
    "download", "fetch", "scrape",
    "develop", "implement", "code",
})

# Hints that the user is pointing at a filesystem destination.
_FS_HINTS = frozenset({
    ".py", ".sh", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml",
    ".html", ".css", ".csv", ".sql",
    "~/", "./", "../",
    "to disk", "to file", "to a file",
    "in ~/", "in ./", "in /",
    "save to", "save it to", "write to", "write it to",
    "save as",
})


def _looks_like_action_request(question: str) -> bool:
    """Cheap heuristic: does the user's question want a side effect?

    Two signals must combine to qualify:
      1. At least one imperative verb token from `_ACTION_VERBS`
         appears as a word boundary match.
      2. At least one filesystem-destination hint from `_FS_HINTS`
         appears anywhere in the question.

    Both are required to reduce false positives — "write a poem
    about clouds" hits (1) but not (2); "the path to ~/Desktop is…"
    hits (2) but not (1).
    """
    if not question:
        return False
    q = question.lower()
    # (1) action verb as a word
    import re
    tokens = set(re.findall(r"[a-z]+", q))
    if not (tokens & _ACTION_VERBS):
        return False
    # (2) filesystem hint anywhere
    return any(h in q for h in _FS_HINTS)


def _trace_has_successful_action(state: AgentState) -> bool:
    """Has any tool_call already succeeded in this run?

    If yes, the model is free to wrap up with a text final answer
    summarizing what it did — the guard should not loop here. Only
    fire the guard when we haven't yet performed a side effect.
    """
    for step in state.trace.steps:
        if step.action.kind == ActionKind.TOOL_CALL and step.observation.ok:
            return True
    return False


def _reply_looks_like_code_explanation(text: str) -> bool:
    """Does the LLM's text reply contain a code block?

    Triple-backtick fence is the strongest signal; we also accept a
    long indented block of obvious code (def/class/import at start of
    line) as a softer signal.
    """
    if not text:
        return False
    if "```" in text:
        return True
    # Soft signal: multiple python-shaped statements at the start of lines.
    import re
    code_lines = re.findall(
        r"(?m)^\s*(?:def |class |import |from \S+ import |if __name__)",
        text,
    )
    return len(code_lines) >= 2


# ---------------------------------------------------------------------------
# B6: enumerable-target detection for coverage-aware commit pressure
# ---------------------------------------------------------------------------
#
# The blunt "commit now" nudge (see _step_pressure_message) systematically
# truncates MULTI-PART questions — "revenue for 2018 through 2023" (6 numbers),
# "name the three largest" (3 items) — because it fires before all parts are
# gathered. We can't reuse CoverageVerifier (it grades claim→evidence support
# AFTER commit, not question coverage), so we detect enumerable targets from
# the question text and reshape the nudge to preserve completeness.

import re as _re

# A year token, 1900–2099. Bare \d{4} would match page counts / IDs.
_YEAR = r"(?:19|20)\d{2}"
# An explicit year RANGE: "2018-2023", "from 2018 to 2023", "between 2018 and
# 2023", "2018 through 2023". Endpoints captured; expanded to the full set.
_YEAR_RANGE_RE = _re.compile(
    rf"\b({_YEAR})\s*(?:-|–|—|to|through|thru|until|and)\s*({_YEAR})\b",
    _re.IGNORECASE,
)
_YEAR_RE = _re.compile(rf"\b{_YEAR}\b")
# Cues that a bare list of years is meant enumerably (answer each one).
_PER_YEAR_CUE = _re.compile(
    r"\b(each|every|per|all|respectively|for the years?)\b", _re.IGNORECASE)

# Number words → int, for explicit-count questions ("list three", "top 5").
_NUM_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_COUNT_RE = _re.compile(
    r"\b(?:list|name|give|find|identify|provide|top|first|"
    r"the)\s+(\d{1,2}|two|three|four|five|six|seven|eight|nine|ten)\b",
    _re.IGNORECASE,
)


def _enumerable_targets(question: str) -> dict[str, Any]:
    """Detect enumerable sub-parts in a question.

    Returns ``{"years": [..], "count": int|None}``. ``years`` is a sorted
    list of the individual year strings the answer must cover (from an
    explicit range, or a bare list of ≥3 years, or ≥2 years plus a
    per-year cue). ``count`` is N when the question explicitly asks for N
    items (and N ≥ 2). Both empty/None when nothing enumerable is found.
    Conservative by design: a false negative just keeps today's behavior.
    """
    if not question:
        return {"years": [], "count": None}
    q = question

    years: list[str] = []
    m = _YEAR_RANGE_RE.search(q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = min(a, b), max(a, b)
        # Guard against absurd spans (a stray pair of years far apart).
        if 0 < hi - lo <= 30:
            years = [str(y) for y in range(lo, hi + 1)]
    if not years:
        found = sorted(set(_YEAR_RE.findall(q)))
        # A bare list is "enumerable" only with a strong signal: ≥3 distinct
        # years, or ≥2 plus an explicit per-year cue.
        if len(found) >= 3 or (len(found) >= 2 and _PER_YEAR_CUE.search(q)):
            years = found

    count: int | None = None
    cm = _COUNT_RE.search(q)
    if cm:
        tok = cm.group(1).lower()
        n = int(tok) if tok.isdigit() else _NUM_WORDS.get(tok)
        if n is not None and 2 <= n <= 20:
            count = n

    return {"years": years, "count": count}


@dataclass
class ReActPolicy:
    """The baseline ReAct Policy. Provider-agnostic.

    Fields:
      name               - identifier used in events / ablation tables
      system_prompt      - prepended to every chat call
      model              - overrides LLMClient.model if set
      max_tokens         - per-call ceiling
      temperature        - per-call temperature
    """

    name: str = "react"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model: str | None = None
    # Bumped from 1024 to give reasoning-capable models (gpt-5,
    # o-series, …) enough headroom that internal CoT doesn't consume
    # the entire budget before any visible text is emitted. Plain
    # models just see slightly more output room and rarely use it.
    max_tokens: int = 2048
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    # When `steps_used / max_steps` ≥ this fraction, append a "commit
    # now" nudge to the message history so the model is forced to
    # finalize even if the system prompt didn't already convince it.
    # 0.6 = nudge appears once you've used 60% of your step budget.
    # Set to 0 or > 1 to disable.
    commit_pressure_threshold: float = 0.6

    # --- B6: coverage-aware commit pressure --------------------------------
    # When True (default), the commit-pressure nudge becomes completeness-
    # aware for ENUMERABLE questions (a year range/list, or an explicit count
    # like "name the three…"). Instead of the blunt "stop now, commit", it
    # reminds the model to cover every part and to gather only the still-
    # missing ones first. This fixes premature commitment on multi-part
    # questions (revenue across 6 years, 3 companies) without changing
    # single-answer behavior. Set False for the blunt nudge always (ablation).
    coverage_aware_pressure: bool = True
    # Hard ceiling: once this fraction of the step budget is spent, fall back
    # to the blunt "commit now" nudge regardless of coverage, so the run
    # ALWAYS commits eventually and the heuristic can never stall it.
    coverage_hard_ceiling: float = 0.9

    # If True (default), guard against the failure mode where the user
    # asks for a side effect (write a file / run a script) and the
    # model replies with code-as-text instead of invoking a tool. When
    # the user's question looks action-shaped AND the LLM reply has
    # no tool_calls AND the text contains a code block, emit a THINK
    # with a re-prompt instead of FINAL_ANSWER. Set False to restore
    # pre-patch behavior.
    action_intent_guard: bool = True

    # --- B1: bounded history projection ------------------------------------
    # When True (default), historical tool observations are replayed into the
    # LLM message history as a COMPACT projection — long text fields are
    # trimmed to `history_snippet_chars` with a marker pointing at the
    # evidence_id the model can pass to `recall_evidence` for the full text.
    # The MOST RECENT tool observation is always kept full (the model usually
    # needs the last result verbatim to act on it). This kills the quadratic
    # token blow-up where step N re-sends every prior page in full. The full
    # text is never lost — it stays in the trace and is reachable via
    # recall_evidence. Set False to restore verbatim full replay (ablation /
    # rollback lever; doubles as a telemetry field).
    project_history: bool = True
    # Per-field char cap for trimmed historical text. ~1.5k chars (~375
    # tokens) keeps a usable snippet while shedding the boilerplate bulk.
    history_snippet_chars: int = 1500

    # --- internal: message history reconstruction --------------------------

    def _history(self, state: AgentState) -> list[Message]:
        """Project AgentState.trace back into an LLM message list.

        First turn: user with the question.
        Subsequent turns replay prior (assistant tool_use, user tool_result)
        pairs in order. Text-only THINK actions attach their text to the
        assistant turn so the model sees its own reasoning.
        """
        msgs: list[Message] = [
            Message(role="user", content=[ContentBlock(kind="text", text=state.question)])
        ]
        # B1: index of the last tool step — its observation is kept full
        # (the model usually needs the latest result verbatim to act on it);
        # every earlier tool observation is projected down to a snippet.
        last_tool_idx = -1
        if self.project_history:
            for step in state.trace.steps:
                if step.action.kind in (ActionKind.TOOL_CALL, ActionKind.TOOL_BATCH):
                    last_tool_idx = step.idx
        for step in state.trace.steps:
            act = step.action
            obs = step.observation
            if act.kind == ActionKind.THINK:
                # Empty-reply markers are internal bookkeeping, not model
                # output. Replaying them as assistant turns teaches the
                # model to emit the marker string — N identical turns in a
                # row is a repetition attractor, and a forced final_answer
                # after a run of empties parrots them into the answer.
                if (act.meta or {}).get("empty_reply"):
                    continue
                # Default: represent thinking as an assistant text turn
                # (the model's own scratchpad).
                # Exception: feedback injected by external machinery (e.g.
                # the verifier_retry wrapper) must appear as a USER turn,
                # otherwise the model reads its own directive as its own
                # thought and ignores it.
                is_external = bool((act.meta or {}).get("verifier_retry"))
                role = "user" if is_external else "assistant"
                msgs.append(Message(
                    role=role,
                    content=[ContentBlock(kind="text", text=act.text or "")],
                ))
            elif act.kind == ActionKind.TOOL_CALL:
                # assistant's tool_use turn. If the model wrote chain-of-
                # thought text alongside the tool call (captured in
                # meta.preceding_text), include it as a text block so the
                # model's reasoning is part of the conversation history.
                preceding = (act.meta or {}).get("preceding_text") or ""
                blocks: list[ContentBlock] = []
                if preceding.strip():
                    blocks.append(ContentBlock(kind="text", text=preceding))
                blocks.append(ContentBlock(
                    kind="tool_use",
                    id=f"step_{step.idx}",
                    name=act.tool_name or "",
                    arguments=dict(act.tool_args),
                ))
                msgs.append(Message(role="assistant", content=blocks))
                # user's tool_result turn
                if obs.ok:
                    payload = obs.data
                    # B1: project historical observations to a snippet, but
                    # keep the most recent tool result full.
                    if self.project_history and step.idx != last_tool_idx:
                        payload = self._project_payload(payload)
                else:
                    payload = {"error": obs.error or "tool failed"}
                msgs.append(Message(
                    role="user",
                    content=[ContentBlock(
                        kind="tool_result",
                        id=f"step_{step.idx}",
                        name=act.tool_name or "",
                        result=payload,
                        is_error=not obs.ok,
                    )],
                ))
            elif act.kind == ActionKind.TOOL_BATCH:
                # B7: replay parallel tool calls. The model emitted ≥2
                # tool_use blocks in one turn; we dispatched them as a batch.
                # Reconstruct that as ONE assistant turn with N tool_use
                # blocks and ONE user turn with N matching tool_result blocks
                # (parallel-tool-use shape both providers understand). Without
                # this branch the whole batch — calls AND results — is dropped
                # from history and the model re-requests what it already ran.
                meta = act.meta or {}
                batch_calls = meta.get("batch_calls") or []
                data = obs.data if isinstance(obs.data, dict) else {}
                summary = data.get("batch") or []
                n = max(len(batch_calls), len(summary))
                project = self.project_history and step.idx != last_tool_idx

                use_blocks: list[ContentBlock] = []
                preceding = meta.get("preceding_text") or ""
                if preceding.strip():
                    use_blocks.append(ContentBlock(kind="text", text=preceding))
                result_blocks: list[ContentBlock] = []
                for i in range(n):
                    bc = batch_calls[i] if i < len(batch_calls) else {}
                    sub = summary[i] if i < len(summary) else {}
                    name = bc.get("name") or sub.get("name") or ""
                    tu_id = f"step_{step.idx}_{i}"
                    use_blocks.append(ContentBlock(
                        kind="tool_use",
                        id=tu_id,
                        name=name,
                        arguments=dict(bc.get("args") or {}),
                    ))
                    ok = bool(sub.get("ok", True))
                    if ok:
                        payload = sub.get("result")
                        if project:
                            payload = self._project_payload(payload)
                    else:
                        payload = {"error": sub.get("error") or "tool failed"}
                    result_blocks.append(ContentBlock(
                        kind="tool_result",
                        id=tu_id,
                        name=name,
                        result=payload,
                        is_error=not ok,
                    ))
                # Defensive: a tool_use with no matching tool_result (or vice
                # versa) is a provider-level error. Only emit when balanced.
                if use_blocks and result_blocks:
                    msgs.append(Message(role="assistant", content=use_blocks))
                    msgs.append(Message(role="user", content=result_blocks))
            elif act.kind == ActionKind.ASK_USER:
                # Replay the clarifying exchange so the model SEES its own
                # question and the user's answer. Without this the Q&A is
                # dropped from history and the model re-asks the same
                # question every turn (observed: 3x re-ask on a single
                # task). react+ only — bare react never emits ASK_USER, so
                # this branch is dead there and the 42.4% trace is unchanged.
                question = (act.text or "").strip()
                reply = ""
                if isinstance(obs.data, dict):
                    reply = str(obs.data.get("reply") or "").strip()
                if not reply:
                    reply = (obs.text or "").strip()
                if question:
                    msgs.append(Message(
                        role="assistant",
                        content=[ContentBlock(kind="text", text=question)],
                    ))
                msgs.append(Message(
                    role="user",
                    content=[ContentBlock(
                        kind="text",
                        text=reply or "(no answer provided)",
                    )],
                ))
            # FINAL_ANSWER should terminate the loop, so we don't project it.

        # Step-pressure nudge — once we've used ≥ commit_pressure_threshold
        # of the step budget without finalizing, append an explicit
        # "commit now" message. Reasoning models (gpt-5*, o-series)
        # tend to keep verifying past the point of having a defensible
        # answer; this gives the budget tracker a way to force them to
        # commit before steps trip.
        nudge = self._step_pressure_message(state)
        if nudge is not None:
            msgs.append(nudge)
        return msgs

    def _project_payload(self, payload: Any) -> Any:
        """B1: shrink a historical tool result for replay.

        Walks the result dict and trims any string longer than
        `history_snippet_chars` down to a snippet, appending a marker that
        tells the model how many chars were dropped and how to get them back
        (`recall_evidence` keyed on the result's evidence_id). Non-dict
        payloads and short fields pass through untouched. The trace itself is
        never mutated — this only affects the copy sent to the LLM.
        """
        cap = self.history_snippet_chars
        if cap <= 0:
            return payload
        if not isinstance(payload, dict):
            if isinstance(payload, str) and len(payload) > cap:
                return payload[:cap] + f"\n… [+{len(payload) - cap} chars trimmed]"
            return payload

        eid = payload.get("evidence_id")
        recall_hint = (
            f"; call recall_evidence(evidence_id='{eid}') for the full text"
            if eid else ""
        )
        out: dict[str, Any] = {}
        for k, v in payload.items():
            if isinstance(v, str) and len(v) > cap:
                dropped = len(v) - cap
                out[k] = (
                    v[:cap]
                    + f"\n… [+{dropped} chars trimmed to save context{recall_hint}]"
                )
            elif isinstance(v, list) and len(v) > 30:
                out[k] = v[:30] + [f"… [+{len(v) - 30} more items trimmed]"]
            else:
                out[k] = v
        return out

    def _step_pressure_message(self, state: AgentState) -> Message | None:
        max_steps = state.budget.max_steps
        thr = self.commit_pressure_threshold
        if not max_steps or thr <= 0 or thr > 1:
            return None
        # Count *productive tool calls*, not just steps. A reply that emits
        # several parallel calls is one TOOL_BATCH step, so a model can fire
        # 2–3× as many calls as steps (the L3 search-loopers ran 24 searches
        # across 12 steps) — keying on steps alone lets that volume slip
        # under the threshold. But we must NOT count duplicate-skipped calls:
        # the dedup guard already returns those from cache without doing
        # work, and counting them would fire "commit now" pressure on a loop
        # the harness already neutralized — and, worse, cut off a model just
        # as it pivots to a genuinely new tool. So: max(steps, distinct
        # productive calls).
        n_calls = 0
        for s in state.trace.steps:
            data = s.observation.data if isinstance(s.observation.data, dict) else {}
            if s.action.kind == ActionKind.TOOL_CALL:
                if not data.get("duplicate_skipped"):
                    n_calls += 1
            elif s.action.kind == ActionKind.TOOL_BATCH:
                batch = data.get("batch")
                if batch:
                    for sub in batch:
                        res = sub.get("result") if isinstance(sub, dict) else None
                        if not (isinstance(res, dict) and res.get("duplicate_skipped")):
                            n_calls += 1
                else:
                    n_calls += len((s.action.meta or {}).get("batch_calls") or [])
        used = max(state.budget.steps_used, n_calls)
        frac = used / max_steps
        if frac < thr:
            return None
        remaining = max(0, max_steps - used)
        text = self._commit_nudge_text(state, used, max_steps, remaining, frac)
        return Message(role="user", content=[ContentBlock(kind="text", text=text)])

    # The blunt "stop now, commit" nudge — used for single-answer questions
    # and as the hard-ceiling fallback for enumerable ones.
    _BLUNT_NUDGE = (
        "Stop calling tools and commit to the best answer you can "
        "produce from the data already gathered. Do not search for "
        "additional confirmation — return PLAIN TEXT with the answer."
    )

    def _commit_nudge_text(
        self, state: AgentState, used: int, max_steps: int,
        remaining: int, frac: float,
    ) -> str:
        prefix = (
            f"NOTE FROM THE RUNTIME: you've already used {used} of "
            f"{max_steps} step budget ({remaining} remaining). "
        )
        # B6: below the hard ceiling, reshape the nudge for enumerable
        # (multi-part) questions so pressure doesn't truncate the answer.
        if self.coverage_aware_pressure and frac < self.coverage_hard_ceiling:
            body = self._coverage_nudge_body(state)
            if body is not None:
                return prefix + body
        return prefix + self._BLUNT_NUDGE

    def _coverage_nudge_body(self, state: AgentState) -> str | None:
        """Completeness-preserving nudge body for enumerable questions, or
        None if the question has no detectable enumerable targets."""
        targets = _enumerable_targets(state.question)
        years = targets["years"]
        count = targets["count"]
        if not years and not count:
            return None

        parts: list[str] = []
        if years:
            # Name the years still absent from everything gathered so far, so
            # the model spends its last steps on the gaps, not re-confirmation.
            corpus = self._gathered_corpus(state)
            missing = [y for y in years if y not in corpus]
            span = f"{years[0]}–{years[-1]}" if len(years) > 1 else years[0]
            if missing:
                parts.append(
                    f"This question is MULTI-PART: it needs every year in "
                    f"{span}. You have not yet gathered data for: "
                    f"{', '.join(missing)}. Get ONLY those, then commit with "
                    f"ALL {len(years)} years in one answer."
                )
            else:
                parts.append(
                    f"This question needs every year in {span} "
                    f"({len(years)} values). Commit now with ALL of them in "
                    f"one answer — do not stop after a subset."
                )
        if count:
            parts.append(
                f"This question asks for {count} items. Make sure your answer "
                f"lists all {count} — do not commit a partial list."
            )
        parts.append(
            "Do not run further confirmation searches; spend any remaining "
            "steps only on missing parts, then return PLAIN TEXT."
        )
        return " ".join(parts)

    def _gathered_corpus(self, state: AgentState) -> str:
        """Concatenate the text the run has collected (evidence + tool
        observation string fields) for cheap substring coverage checks."""
        chunks: list[str] = []
        for ev in state.evidence:
            if ev.content:
                chunks.append(ev.content)
        for step in state.trace.steps:
            data = step.observation.data
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, str):
                        chunks.append(v)
            if step.observation.text:
                chunks.append(step.observation.text)
        return "\n".join(chunks)

    # --- Policy.propose ----------------------------------------------------

    def propose(
        self,
        state: AgentState,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
    ) -> Action:
        specs: list[ToolSpec] = tools.to_tool_specs()
        kwargs: dict[str, Any] = {
            "messages": self._history(state),
            "tools": specs,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": self.system_prompt,
        }
        if self.model:
            kwargs["model"] = self.model
        if self.extra:
            kwargs["extra"] = self.extra

        # Repair-step handling: at the trace tail we may have a run of
        # empty_reply / commit_required THINKs. Branch on what kind of
        # stuck-ness we're in:
        #
        #   (a) ≥2 commit_required with a recoverable plain-text reply →
        #       bail directly with that text as the answer.
        #   (b) ≥2 empty_reply *and* no evidence yet → force any tool
        #       call so the model starts gathering (rather than forcing
        #       it to commit `final_answer` on an empty trace, which
        #       would commit garbage).
        #   (c) ≥1 nudge of any kind with `final_answer` registered →
        #       force `final_answer` so the model commits what it has.
        tool_names = tools.names() if tools is not None else ()
        cr_run = _count_consecutive_commit_required(state)
        empty_run = _count_consecutive_empty_replies(state)
        preceding = _last_preceding_text(state)

        if cr_run >= 2 and preceding and "final_answer" in tool_names:
            return Action(
                kind=ActionKind.FINAL_ANSWER,
                answer=preceding,
                meta={
                    "policy": self.name,
                    "commit_required_bailout": True,
                    "consecutive_nudges": cr_run,
                },
            )

        if empty_run >= 2 and not _trace_has_evidence(state):
            forced = _force_tool_extra(getattr(llm, "provider", ""), None)
            if forced:
                merged = dict(kwargs.get("extra") or {})
                merged.update(forced)
                kwargs["extra"] = merged
        elif cr_run >= 1 and "final_answer" in tool_names:
            forced = _force_tool_extra(
                getattr(llm, "provider", ""), "final_answer",
            )
            if forced:
                merged = dict(kwargs.get("extra") or {})
                merged.update(forced)
                kwargs["extra"] = merged

        try:
            reply = llm.chat(**kwargs)
        except Exception as exc:
            # Provider errors are handled by the driver, which owns the
            # retry/backoff loop and pauses the wall clock while it waits:
            #   - retryable (rate limit / transient 5xx) → driver retries
            #     this same propose() with backoff off the task budget.
            #   - non-retryable (missing API key, model can't tool-call) →
            #     driver terminates instead of burning steps on a
            #     deterministic failure.
            # Either way we re-raise rather than swallowing into a THINK,
            # so the .retryable hint isn't lost. Non-provider exceptions
            # (e.g. a malformed reply we can't parse) still get the THINK
            # fallback so a one-off blip self-recovers on the next tick.
            from ..llm.base import ProviderError
            if isinstance(exc, ProviderError):
                raise
            return Action(
                kind=ActionKind.THINK,
                text=f"[llm_error] {type(exc).__name__}: {exc}",
                meta={"policy": self.name, "error": True},
            )

        # Prefer tool calls over text. If the model emitted ≥2 tool_calls
        # in the same reply (and none of them is final_answer, which is
        # the terminal commit), dispatch them as a TOOL_BATCH so the
        # driver can run them concurrently. A single tool_call still
        # follows the original path.
        if reply.has_tool_calls:
            preceding_text = (reply.text or "").strip()
            call_names = [c.name for c in reply.tool_calls if c.name]
            has_final = any(n == "final_answer" for n in call_names)
            if len(reply.tool_calls) >= 2 and not has_final:
                # Deduplicate identical (name, args) pairs — providers
                # occasionally echo the same call. Cap the batch to 5 to
                # bound provider-side rate-limit blast.
                seen: set[tuple[str, str]] = set()
                batch: list[dict[str, Any]] = []
                for c in reply.tool_calls:
                    if not c.name:
                        continue
                    args = dict(c.arguments or {})
                    key = (c.name, str(sorted(args.items())))
                    if key in seen:
                        continue
                    seen.add(key)
                    batch.append({"name": c.name, "args": args})
                    if len(batch) >= 5:
                        break
                if len(batch) >= 2:
                    return Action(
                        kind=ActionKind.TOOL_BATCH,
                        meta={
                            "policy": self.name,
                            "provider": reply.provider,
                            "model": reply.model,
                            "stop_reason": reply.stop_reason,
                            "tokens_in": reply.usage.tokens_in,
                            "tokens_out": reply.usage.tokens_out,
                            "preceding_text": preceding_text,
                            "batch_calls": batch,
                            "batch_names": [c["name"] for c in batch],
                        },
                    )
                # Fell through (dedup collapsed to 1) — fall into the
                # single-call path below using the first call.
            call = reply.tool_calls[0]
            # The model may emit reasoning text alongside the tool call in
            # the same response (chain-of-thought before commit). Preserve
            # it so future turns see the model's working — both as
            # context for its own next decision and so it can use it as
            # CoT scaffolding when writing the answer.
            # Intercept `final_answer` calls: they're a structured commit
            # channel, not a real side-effect tool. Convert directly to a
            # FINAL_ANSWER action so the agent loop terminates with the
            # `answer` field (not the model's surrounding text).
            if call.name == "final_answer":
                args = dict(call.arguments or {})
                ans = str(args.get("answer", "")).strip()
                reasoning = str(args.get("reasoning", "")).strip()
                # Phase 7: register the answer as a Claim with the model's
                # cited evidence_ids. The CitationVerifier then grades
                # support (Jaccard overlap + numeric-substring check)
                # and rejects fabricated answers with no grounding.
                raw_ids = args.get("evidence_ids") or []
                if isinstance(raw_ids, str):
                    raw_ids = [raw_ids]
                evidence_ids = [
                    str(x).strip() for x in raw_ids if str(x).strip()
                ]
                # Only register a citation-bearing claim when the model
                # actually cited something — otherwise CoverageVerifier
                # would flag every uncited answer as "no support" which
                # blocks turns that legitimately need no external evidence
                # (e.g. the riddle in ec09fa32).
                if evidence_ids:
                    state.add_claim(
                        text=ans,
                        supports=evidence_ids,
                        origin_step=len(state.trace.steps),
                    )
                return Action(
                    kind=ActionKind.FINAL_ANSWER,
                    answer=ans,
                    text=reasoning or ans,
                    meta={
                        "policy": self.name,
                        "provider": reply.provider,
                        "model": reply.model,
                        "stop_reason": reply.stop_reason,
                        "tokens_in": reply.usage.tokens_in,
                        "tokens_out": reply.usage.tokens_out,
                        "via_final_answer_tool": True,
                        "preceding_text": preceding_text,
                        "evidence_ids": evidence_ids,
                    },
                )
            return Action(
                kind=ActionKind.TOOL_CALL,
                tool_name=call.name,
                tool_args=dict(call.arguments),
                meta={
                    "policy": self.name,
                    "provider": reply.provider,
                    "model": reply.model,
                    "stop_reason": reply.stop_reason,
                    "tokens_in": reply.usage.tokens_in,
                    "tokens_out": reply.usage.tokens_out,
                    "preceding_text": preceding_text,
                },
            )

        # No tools called.
        text = reply.text.strip()
        if not text:
            # The model produced nothing actionable; think and try again.
            # Tagged `repair=True` so it doesn't consume the main step
            # budget — burned-step empties were a top failure mode on
            # gpt-5-nano (see validation_experiment_gpt_nano_05162026.md).
            return Action(
                kind=ActionKind.THINK,
                text="[empty_reply] model returned no text and no tool_calls",
                meta={
                    "policy": self.name,
                    "stop_reason": reply.stop_reason,
                    "repair": True,
                    "empty_reply": True,
                },
            )

        # If the `final_answer` tool is registered, the model is expected
        # to commit via that tool — not via a plain-text turn. Push the
        # model toward the tool via a user-role THINK instead of silently
        # promoting prose to FINAL_ANSWER (which conflates reasoning with
        # the committed answer and breaks exact-match scoring).
        if "final_answer" in (tools.names() if tools is not None else ()):
            return Action(
                kind=ActionKind.THINK,
                text=(
                    "[commit_required] You responded with plain text, but "
                    "this task only accepts answers via the `final_answer` "
                    "tool. Call `final_answer` now with the literal answer "
                    "in the `answer` field. Put any explanation in the "
                    "`reasoning` field — do NOT put it in `answer`."
                ),
                meta={
                    "policy": self.name,
                    "verifier_retry": True,  # render as user-role next tick
                    "commit_required": True,
                    "repair": True,
                    # Stash the plain-text reply so the next-tick bailout
                    # (if the model ignores the forced tool_choice) can
                    # recover the model's answer instead of submitting "".
                    "preceding_text": text,
                    "stop_reason": reply.stop_reason,
                    "tokens_in": reply.usage.tokens_in,
                    "tokens_out": reply.usage.tokens_out,
                },
            )

        # Action-intent guard: if the user wanted a side effect and the
        # model replied with code-as-text, don't declare victory. Emit
        # a THINK with a hint so the next tick re-asks the model to
        # actually invoke a tool. See `_looks_like_action_request` and
        # `_reply_looks_like_code_explanation`.
        if (
            self.action_intent_guard
            and _looks_like_action_request(state.question)
            and _reply_looks_like_code_explanation(text)
            and not _trace_has_successful_action(state)
        ):
            return Action(
                kind=ActionKind.THINK,
                text=(
                    "[action_required] The user asked for a side effect "
                    "(file write, code execution, etc.) but the previous "
                    "reply was code-as-text with no tool call. Re-do this "
                    "by calling `python_sandbox` (to run Python that writes "
                    "the file, e.g. via `pathlib.Path(...).write_text(...)`) "
                    "or `run_shell` (to execute the command). Do NOT return "
                    "the code as text; actually perform the side effect."
                ),
                meta={
                    "policy": self.name,
                    "guard": "action_intent",
                    "stop_reason": reply.stop_reason,
                    "tokens_in": reply.usage.tokens_in,
                    "tokens_out": reply.usage.tokens_out,
                },
            )

        return Action(
            kind=ActionKind.FINAL_ANSWER,
            answer=text,
            text=text,
            meta={
                "policy": self.name,
                "provider": reply.provider,
                "model": reply.model,
                "stop_reason": reply.stop_reason,
                "tokens_in": reply.usage.tokens_in,
                "tokens_out": reply.usage.tokens_out,
            },
        )

    # ------------------------------------------------------------------
    # Budget-exhaustion synthesis
    # ------------------------------------------------------------------

    def synthesize_on_exhaustion(
        self,
        state: AgentState,
        *,
        llm: LLMClient | None = None,
        tools: ToolRegistry | None = None,
        timeout_s: float = 15.0,
    ) -> Action | None:
        """Emit a best-effort FINAL_ANSWER when the budget just tripped.

        Default chain:
          1. One-shot LLM call with `tool_choice=final_answer` and a
             short max_tokens. The model sees the question + a compact
             trace digest and is told "your budget ran out; commit your
             best guess now". Capped by ``timeout_s`` wall.
          2. If the LLM call fails / times out / returns empty → fall
             through to the cheap branch.
          3. Cheap branch: most recent Claim text → most recent short
             plain-text reply on the trace → ``None`` (caller decides).

        Returning ``None`` means "no synthesis available"; the caller
        leaves ``trace.final_answer`` unset.
        """
        # --- Cheap fallback (used as the safety net for branch 1) ---
        #
        # Skip the `[empty_reply] model returned no text and no tool_calls`
        # marker (line ~546) — that string is internal bookkeeping for the
        # empty-reply detector, not an answer the model committed. Earlier
        # versions of this fallback let the marker land in `pred_answer`
        # and score zero on every such task on GAIA. Treat it as no-text.
        def _is_marker(txt: str) -> bool:
            return txt.startswith("[empty_reply]")

        def _cheap() -> Action | None:
            if state.claims:
                ans = (state.claims[-1].text or "").strip()
                if ans and not _is_marker(ans):
                    return Action(
                        kind=ActionKind.FINAL_ANSWER,
                        answer=ans,
                        text=ans,
                        meta={
                            "policy": self.name,
                            "synthesis": "cheap_last_claim",
                        },
                    )
            for step in reversed(state.trace.steps):
                # An ASK_USER step's `text` is the clarifying *question*,
                # never an answer. Skipping it stops synthesis from
                # echoing "What would you like me to help with?" back as
                # the final answer (react+ only; bare react has no
                # ASK_USER steps, so this is a no-op there).
                if step.action.kind == ActionKind.ASK_USER:
                    continue
                m = step.action.meta or {}
                if m.get("empty_reply"):
                    continue
                txt = (m.get("preceding_text") or "").strip()
                if not txt:
                    txt = (step.action.text or "").strip()
                # Short text only — long traces are reasoning, not answers.
                if txt and len(txt) <= 80 and not _is_marker(txt):
                    return Action(
                        kind=ActionKind.FINAL_ANSWER,
                        answer=txt,
                        text=txt,
                        meta={
                            "policy": self.name,
                            "synthesis": "cheap_last_text",
                        },
                    )
            return None

        if llm is None or tools is None:
            return _cheap()

        if "final_answer" not in tools.names():
            return _cheap()

        # --- Branch 1: LLM-driven best-effort commit ---
        try:
            digest = self._exhaustion_digest(state)
        except Exception:
            return _cheap()

        synth_system = (
            "You ran out of budget before committing an answer. "
            "Look at the question and the trace digest below, then "
            "call `final_answer` exactly once with your single best "
            "guess. Do NOT call any other tool. If you truly have no "
            "information, still emit your best guess from the question "
            "itself — empty answers always score zero."
        )
        synth_user = (
            f"Question: {state.question}\n\n"
            f"Trace digest:\n{digest}\n\n"
            "Commit your best answer now via `final_answer`."
        )
        kwargs: dict[str, Any] = {
            "messages": [Message(
                role="user",
                content=[ContentBlock(kind="text", text=synth_user)],
            )],
            "tools": tools.to_tool_specs(),
            "max_tokens": 256,
            "temperature": 0.0,
            "system": synth_system,
        }
        if self.model:
            kwargs["model"] = self.model
        forced = _force_tool_extra(getattr(llm, "provider", ""), "final_answer")
        if forced:
            kwargs["extra"] = forced

        import threading

        result: dict[str, Any] = {}

        def _call() -> None:
            try:
                result["reply"] = llm.chat(**kwargs)
            except Exception as exc:  # pragma: no cover (provider-specific)
                result["error"] = exc

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive() or "reply" not in result:
            return _cheap()

        reply = result["reply"]
        if reply.has_tool_calls:
            call = reply.tool_calls[0]
            if call.name == "final_answer":
                args = dict(call.arguments or {})
                ans = str(args.get("answer", "")).strip()
                if ans:
                    return Action(
                        kind=ActionKind.FINAL_ANSWER,
                        answer=ans,
                        text=str(args.get("reasoning", "")).strip() or ans,
                        meta={
                            "policy": self.name,
                            "synthesis": "llm",
                            "provider": reply.provider,
                            "model": reply.model,
                            "tokens_in": reply.usage.tokens_in,
                            "tokens_out": reply.usage.tokens_out,
                        },
                    )
        # The model returned plain text (or empty) — fall back.
        txt = (reply.text or "").strip()
        if txt and len(txt) <= 240:
            return Action(
                kind=ActionKind.FINAL_ANSWER,
                answer=txt,
                text=txt,
                meta={
                    "policy": self.name,
                    "synthesis": "llm_text",
                    "provider": reply.provider,
                    "model": reply.model,
                    "tokens_in": reply.usage.tokens_in,
                    "tokens_out": reply.usage.tokens_out,
                },
            )
        return _cheap()

    def _exhaustion_digest(self, state: AgentState, *, max_items: int = 8) -> str:
        """Compact, model-readable summary of the trace for synthesis.

        We pick the last `max_items` tool-call outcomes and any claims,
        formatted as one short line each. Keeps prompt cost negligible.
        """
        lines: list[str] = []
        for step in state.trace.steps[-max_items:]:
            a = step.action
            if a.kind == ActionKind.TOOL_CALL:
                ok = "ok" if step.observation.ok else "err"
                preview = (step.observation.text or "")
                if isinstance(step.observation.data, dict):
                    for k in ("summary", "text", "answer", "value"):
                        v = step.observation.data.get(k)
                        if isinstance(v, str) and v.strip():
                            preview = v
                            break
                preview = preview[:120].replace("\n", " ")
                lines.append(f"- tool {a.tool_name} ({ok}): {preview}")
            elif a.kind == ActionKind.THINK and not (a.meta or {}).get("repair"):
                txt = (a.text or "")[:120].replace("\n", " ")
                if txt:
                    lines.append(f"- think: {txt}")
        for cl in state.claims[-3:]:
            lines.append(f"- claim: {cl.text[:120]}")
        return "\n".join(lines) if lines else "(empty trace)"
