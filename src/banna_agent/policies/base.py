"""The Policy contract.

A `Policy` is a function: `(AgentState) -> Action`. The driver runs the
loop; the policy only *chooses*.

Policies can access the LLM client and tool registry. Both are passed in
so the policy does not own these resources — the driver does.

Week-2 note: verifier_retry and best_first will be additional Policy
implementations. None of them need to touch the inner loop — the
transition function in `core/agent.py` stays the same.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.state import AgentState
from ..core.types import Action
from ..llm.base import LLMClient
from ..tools.base import ToolRegistry


@runtime_checkable
class Policy(Protocol):
    """Decide the next Action given the current state."""

    name: str

    def propose(
        self,
        state: AgentState,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
    ) -> Action:
        ...
