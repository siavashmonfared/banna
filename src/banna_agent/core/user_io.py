"""User interaction contract for human-in-the-loop policies.

`UserIO` is the seam between the agent loop and whatever surface
(REPL, web UI, test harness) is talking to a human. Two operations:

* `ask(question)`    — clarifying question (model emits `ASK_USER`).
                       Returns the user's text response.
* `confirm(...)`     — permission gate before a side-effecting tool
                       call. Returns one of `"allow_once"`,
                       `"allow_always"`, `"deny"`.

When `run_policy` is called with `user_io=None` (the default), the
loop is in **batch mode**: `ASK_USER` actions degrade to a synthetic
THINK with the marker "(no user available — proceeding with best
guess)", and the permission gate auto-allows. This keeps GAIA / CI
runs producing identical traces to the old `react` policy.
"""
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable


PermissionDecision = Literal["allow_once", "allow_always", "deny"]
ToolRisk = Literal["read", "exec", "write", "net"]


@runtime_checkable
class UserIO(Protocol):
    """Minimal contract any interactive surface must satisfy."""

    def ask(self, question: str) -> str:
        """Display the question, block on input, return the answer string.

        Implementations should treat EOF / interrupt as the empty
        string (the policy can then decide whether to retry or give
        up). Implementations are also free to record the exchange to
        any logging surface they want — the agent loop will still
        emit its own ASK_USER event for the trace.
        """
        ...

    def confirm(
        self,
        *,
        tool_name: str,
        args: dict,
        risk: ToolRisk,
    ) -> PermissionDecision:
        """Ask the user whether a tool call should be allowed.

        Called by the agent loop just before invoking a tool whose
        risk is gated. Implementations should render the tool name,
        the (truncated) args, and the risk class, then return one of
        the three decisions.

        Note: the *session* allowlist (for `allow_always`) is the
        loop's responsibility, not the UserIO's — implementations
        only have to return the decision; the loop deduplicates.
        """
        ...
