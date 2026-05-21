"""`react+` — ReAct augmented with `ask_user` and error-scoping.

Subclasses `ReActPolicy` so every reliability fix that landed on the
base policy (empty-reply detection, forced tool choice, synthesis on
exhaustion, the C1-C6 pass) is inherited verbatim. `react+` only adds:

1. An **`ask_user` tool affordance**. Advertised to the model alongside
   the real tools. The policy intercepts it in `propose()` and emits
   `ActionKind.ASK_USER` — the agent loop then blocks on
   `UserIO.ask(question)` (or degrades to a marker THINK in batch mode).

2. An **augmented system prompt** with two guardrails the original
   ReAct prompt is missing:

   * *Error-scoping.* A `FileNotFoundError` on one path means that path
     doesn't exist, not that file access is unavailable. Try
     alternatives, or `ask_user` to confirm. (This is the failure mode
     we saw on the macOS-path-on-Linux session.)
   * *Retry-budget escape hatch.* After 2 consecutive failed tool calls
     on the same intent, prefer `ask_user` over a third attempt.

The point of keeping this as a separate policy (rather than amending
`react`) is that `react` already has a 42.4 % GAIA validation behind
it. We don't want to retroactively change the policy under the
published number. `react+` will be validated separately before
landing in the public CLI as a default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.state import AgentState
from ..core.types import Action, ActionKind
from ..llm.base import LLMClient, ToolSpec
from ..tools.base import ToolRegistry
from .react import DEFAULT_SYSTEM_PROMPT, ReActPolicy


REACT_PLUS_PROMPT_SUFFIX = (
    "\n\nWhen things go wrong:\n"
    "  • Tool errors describe THIS specific call's outcome, not the "
    "tool family. A `FileNotFoundError` on one path doesn't mean files "
    "are inaccessible — it means *that path* doesn't exist. Try other "
    "paths, switch tools, or use `ask_user` to confirm with the user.\n"
    "  • If 2 consecutive tool calls on the same intent fail, prefer "
    "`ask_user` over a third attempt. The user can clarify a typo, a "
    "wrong assumption, or confirm an alternative.\n"
    "  • The user's premise may be wrong (paths from another OS, "
    "stale URLs, files they thought were uploaded). Surface this with "
    "`ask_user` rather than silently giving up.\n\n"
    "The `ask_user` tool:\n"
    "  • Use it when the request is ambiguous, when a tool error "
    "suggests the user's input was wrong, or when you'd otherwise "
    "have to guess to make progress.\n"
    "  • Phrase the question as a single, concrete ask — \"What "
    "platform are these paths from? They don't exist on this box.\" "
    "is better than \"Can you help me?\".\n"
    "  • If `ask_user` itself returns no answer (batch mode / non-"
    "interactive), commit your best-guess FINAL_ANSWER explaining "
    "the assumption you made."
)


def _ask_user_tool_spec() -> ToolSpec:
    """ToolSpec advertised to the LLM. The policy intercepts calls
    locally; this is never dispatched through the tool registry.
    """
    return ToolSpec(
        name="ask_user",
        description=(
            "Pause the loop and ask the user a clarifying question. "
            "Use when the request is ambiguous, when a tool error "
            "suggests the user's input was wrong, or to confirm a "
            "non-obvious assumption before continuing. In non-"
            "interactive runs this returns a marker observation and "
            "the loop continues with a best-guess strategy."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "A single, concrete question to ask the user. "
                        "Phrase as one question, not a list."
                    ),
                },
            },
            "required": ["question"],
        },
    )


@dataclass
class ReActPlusPolicy(ReActPolicy):
    """ReAct with `ask_user` affordance + error-scoping guardrails.

    Identical wire shape to `ReActPolicy` apart from:
      * `name = "react+"` (so events/ablation tables can distinguish)
      * augmented `system_prompt`
      * `ask_user` ToolSpec injected into every chat call's `tools`
      * `propose()` intercepts `ask_user` calls and emits ASK_USER
    """

    name: str = "react+"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT + REACT_PLUS_PROMPT_SUFFIX

    def propose(
        self,
        state: AgentState,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
    ) -> Action:
        # Augment the registry's tool specs with `ask_user`. The base
        # `propose()` builds `specs = tools.to_tool_specs()` itself, so
        # we can't just pass the augmented list down — we wrap the
        # ToolRegistry with a shim that returns the extended list.
        augmented = _AugmentedRegistry(tools, _ask_user_tool_spec())
        action = super().propose(state, llm=llm, tools=augmented)

        # Intercept ask_user tool calls: convert TOOL_CALL{name=ask_user}
        # into a structured ASK_USER action. Mirrors how the base policy
        # intercepts `final_answer`.
        if (
            action.kind == ActionKind.TOOL_CALL
            and (action.tool_name or "").lower() == "ask_user"
        ):
            question = ""
            if isinstance(action.tool_args, dict):
                question = str(action.tool_args.get("question") or "").strip()
            return Action(
                kind=ActionKind.ASK_USER,
                text=question or "(no question)",
                meta={
                    **(action.meta or {}),
                    "policy": self.name,
                    "via_ask_user_tool": True,
                },
            )
        return action


class _AugmentedRegistry:
    """ToolRegistry shim that adds an extra ToolSpec to the advertised list.

    Delegates everything else to the wrapped registry. The extra
    spec is never invoked (the policy intercepts it before TOOL_CALL
    dispatch reaches the registry).
    """

    def __init__(self, inner: ToolRegistry, extra: ToolSpec) -> None:
        self._inner = inner
        self._extra = extra

    def to_tool_specs(self) -> list[ToolSpec]:
        return list(self._inner.to_tool_specs()) + [self._extra]

    def names(self) -> Any:
        # Some sites use names() membership tests; include the synthetic
        # tool so the model's `ask_user` calls aren't flagged as unknown.
        base = list(self._inner.names())
        if self._extra.name not in base:
            base.append(self._extra.name)
        return base

    def get(self, name: str) -> Any:
        return self._inner.get(name)

    def __getattr__(self, attr: str) -> Any:
        # Fallback for any other attribute the policy might read.
        return getattr(self._inner, attr)
