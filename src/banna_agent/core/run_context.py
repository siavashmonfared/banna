"""Per-run state handle.

Tools are built once and shared across many runs — the GAIA registry is
constructed a single time and reused for every task — so a tool handler
cannot close over the live :class:`AgentState` at build time. Instead
``run_policy`` publishes the in-flight state here for the duration of a
run, and tools that need to read the live trace (e.g. ``recall_evidence``,
which pulls the full text of a previously-fetched page back out of the
trace after B1 projects it down to a snippet in the message history) read
it back via :func:`get_current_state`.

A ``contextvars.ContextVar`` (not a module global) keeps this correct under
nesting: ``best_of_n`` and friends may call ``run_policy`` recursively, and
each call set/reset pair restores the previous run's state on exit.
"""
from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import AgentState

_CURRENT_STATE: contextvars.ContextVar = contextvars.ContextVar(
    "banna_current_state", default=None
)


def set_current_state(state: "AgentState"):
    """Publish ``state`` as the active run. Returns a token for ``reset``."""
    return _CURRENT_STATE.set(state)


def reset_current_state(token) -> None:
    """Restore whatever state was active before the matching ``set``."""
    _CURRENT_STATE.reset(token)


def get_current_state():
    """The :class:`AgentState` of the in-flight run, or ``None`` if no run
    is active (e.g. a tool invoked directly in a unit test)."""
    return _CURRENT_STATE.get()
