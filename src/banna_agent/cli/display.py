"""Rich rendering for the interactive CLI.

Two responsibilities:

  1. `StreamingEventLog` — a subclass of `core.events.EventLog` that
     pretty-prints each event with rich as it fires, so the user sees
     the agent "thinking → calling search → reading url → answering"
     in real time, not as a wall of text after the fact.

  2. Static helpers: header banner, final-answer panel, tables for
     tools and turn history.

Design choices:
  * Each step's persistent line ("▸ search(query=...)") is printed
    directly via `console.print` so it stays in scrollback.
  * Live spinner activity ("thinking…", "running search…") is
    optional — controlled by an injected `rich.status.Status` handle
    if the caller is inside `console.status(...)`. If `status` is
    None, the EventLog still works and just skips the spinner update.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.events import AgentEvent, EventKind, EventLog
from .theme import render_brand_title


# ---------------------------------------------------------------------------
# StreamingEventLog
# ---------------------------------------------------------------------------


@dataclass
class StreamingEventLog(EventLog):
    """An EventLog that pretty-prints each event as it's emitted.

    Construct with the rich Console and (optionally) a Status handle
    obtained from `console.status(...)`. While inside the status
    context, the spinner text is updated based on event kind.

    `state`, when provided, is used to render step counters as
    [step N/M] where M is `state.budget.max_steps`.
    """

    console: Console = None  # type: ignore[assignment]
    status: Any = None  # rich.status.Status; None disables spinner updates
    state: Any = None    # AgentState — used for step-counter denominators
    show_args: bool = True
    show_text: bool = True

    def __post_init__(self) -> None:
        # Replicate EventLog.__init__ without taking a path arg.
        self._path = None
        self.events = []
        self._max_steps: int = 0
        self._max_wall_s: float = 0.0
        if self.state is not None:
            self._max_steps = int(getattr(self.state.budget, "max_steps", 0) or 0)
            self._max_wall_s = float(getattr(self.state.budget, "max_wall_s", 0.0) or 0.0)

    def emit(self, event: AgentEvent) -> None:  # type: ignore[override]
        # Persist to in-memory list (parent behavior).
        self.events.append(event)
        # Pretty-print.
        try:
            self._render(event)
        except Exception:
            # Never let a render bug break the agent loop.
            pass

    def _render(self, ev: AgentEvent) -> None:
        kind = ev.kind
        p = ev.payload or {}

        if kind == EventKind.RUN_START:
            self._update_status("[dim]thinking…[/dim]")
            return

        if kind == EventKind.PROPOSE:
            action_kind = p.get("kind_of_action", "")
            tool = p.get("tool_name") or ""
            text = p.get("action_text") or ""
            is_err = bool(p.get("is_error"))
            prefix = self._step_prefix(ev.step)
            if action_kind == "tool_call":
                self._update_status(f"[cyan]running[/cyan] [bold]{tool}[/bold]…")
            elif action_kind == "think":
                if is_err:
                    self.console.print(
                        f"{prefix}[red]✗ llm error[/red] [dim]{_short(text, 200)}[/dim]"
                    )
                elif text:
                    self.console.print(
                        f"{prefix}[dim italic]💭 {_short(text, 200)}[/dim italic]"
                    )
                self._update_status("[dim]thinking…[/dim]")
            elif action_kind == "final_answer":
                # Surface the final answer mid-stream so the user sees
                # it the moment the policy commits, not only via the
                # answer panel after the loop returns.
                if text:
                    self.console.print(
                        f"{prefix}[bold green]★ FINAL_ANSWER:[/bold green] "
                        f"[green]{_short(text, 240)}[/green]"
                    )
                self._update_status("[dim]finalizing…[/dim]")
            return

        if kind == EventKind.TOOL_CALL:
            tool = p.get("tool_name") or "?"
            args = p.get("arguments") or {}
            args_s = _short_args(args) if self.show_args else ""
            prefix = self._step_prefix(ev.step)
            self.console.print(
                f"{prefix}[cyan]▸[/cyan] [bold]{tool}[/bold]({args_s})"
            )
            return

        if kind == EventKind.TOOL_RESULT:
            ok = p.get("ok", True)
            wall = float(p.get("wall_s") or 0.0)
            err = p.get("error")
            preview = (p.get("preview") or "").strip()
            if ok:
                preview_part = f"[white]{_short(preview, 100)}[/white] " if preview else ""
                self.console.print(
                    f"  [green]✓[/green] {preview_part}[dim]({wall:.1f}s)[/dim]"
                )
            else:
                self.console.print(
                    f"  [red]✗[/red] [red]{_short(err or 'failed', 80)}[/red]"
                    f" [dim]({wall:.1f}s)[/dim]"
                )
            return

        if kind == EventKind.OBSERVATION:
            # Optional running-totals line: tokens accumulated, wall %.
            # Quiet by default — only fire when there's signal worth
            # showing (a non-trivial token delta or evidence delta).
            tokens_in = int(p.get("tokens_in") or 0)
            tokens_out = int(p.get("tokens_out") or 0)
            cum_t_in = int(p.get("cumulative_tokens_in") or 0)
            cum_t_out = int(p.get("cumulative_tokens_out") or 0)
            cum_wall = float(p.get("cumulative_wall_s") or 0.0)
            if tokens_in or tokens_out:
                # Print on the same indent as the ✓ line above.
                self.console.print(
                    f"    [dim]+{tokens_in}→{tokens_out} tok  ·  "
                    f"total {cum_t_in}→{cum_t_out} tok"
                    + (f"  ·  {cum_wall:.1f}s" if cum_wall else "")
                    + "[/dim]"
                )
            return

        if kind == EventKind.BUDGET:
            reason = p.get("reason", "?")
            # Decode the reason into actuals when state is in scope.
            detail = ""
            if self.state is not None:
                b = self.state.budget
                if reason == "budget_wall":
                    detail = f" ({b.elapsed_wall_s:.1f}s/{b.max_wall_s:.0f}s)"
                elif reason == "budget_steps":
                    detail = f" ({b.steps_used}/{b.max_steps} steps)"
                elif reason == "budget_tokens" and b.max_tokens_total:
                    detail = (f" ({b.tokens_in + b.tokens_out}/"
                              f"{b.max_tokens_total} tok)")
                elif reason == "budget_cost" and b.max_cost_usd:
                    detail = f" (${b.cost_usd:.4f}/${b.max_cost_usd:.4f})"
            self.console.print(
                f"[yellow]⚠ budget tripped: {reason}{detail}[/yellow]"
            )
            return

        if kind == EventKind.ERROR:
            err = p.get("error", "?")
            where = p.get("where", "")
            self.console.print(f"[red]error[/red] [dim]({where})[/dim]: {err}")
            return

        if kind == EventKind.COMPACT:
            dropped = p.get("dropped_steps", "?")
            kept = p.get("kept_steps", "?")
            self.console.print(
                f"[magenta]↺ trace compacted: dropped {dropped} steps, "
                f"kept {kept}[/magenta]"
            )
            return

        if kind == EventKind.RUN_END:
            self._update_status("")
            return

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _step_prefix(self, step_idx: int) -> str:
        """Render a `[N/M] ` prefix when state is known, else ``""``.

        `step_idx` is 0-based (the index the about-to-be-appended step
        will get). Display as 1-based so the first step is `[1/M]`.
        """
        if step_idx < 0:
            return ""
        n = step_idx + 1
        if self._max_steps:
            return f"[dim]\\[{n}/{self._max_steps}][/dim] "
        return f"[dim]\\[{n}][/dim] "

    def _update_status(self, msg: str) -> None:
        if self.status is None:
            return
        try:
            self.status.update(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Static helpers — banner, answer panel, trajectory
# ---------------------------------------------------------------------------


def header_panel(
    *,
    provider: str,
    model: str,
    policy: str,
    tools: list[str],
    budget_steps: int,
    budget_wall_s: float,
) -> Panel:
    import os
    try:
        cwd = "~/" + os.path.relpath(os.getcwd(), os.path.expanduser("~"))
        if cwd.startswith("~/.."):
            cwd = os.getcwd()
    except Exception:
        cwd = ""

    body = Text()
    body.append("provider  ", style="scout.muted")
    body.append(f"{provider}\n", style="scout.text")
    body.append("model     ", style="scout.muted")
    body.append(f"{model or '(provider default)'}\n", style="scout.text")
    body.append("policy    ", style="scout.muted")
    body.append(f"{policy}\n", style="scout.text")
    body.append("tools     ", style="scout.muted")
    body.append(", ".join(tools) if tools else "(none)", style="scout.text")
    body.append("\n")
    body.append("budget    ", style="scout.muted")
    body.append(f"{budget_steps} steps · {budget_wall_s:.0f}s wall",
                style="scout.text")

    return Panel(
        body,
        title=render_brand_title(cwd),
        title_align="left",
        border_style="scout.border",
    )


def final_answer_panel(
    *,
    answer: str,
    steps_used: int,
    wall_s: float,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    budget_reason: str,
) -> Panel:
    """Render the agent's final-answer block."""
    body = Text()
    body.append("● banna\n", style="scout.agent")
    body.append("  ")
    body.append(answer or "(no answer)", style="scout.text")
    body.append("\n  ")
    body.append(
        f"{steps_used} step{'s' if steps_used != 1 else ''} · "
        f"{wall_s:.1f}s · "
        f"{tokens_in}→{tokens_out} tok",
        style="scout.muted",
    )
    if cost_usd > 0:
        body.append(f" · ${cost_usd:.4f}", style="scout.muted")
    if budget_reason and budget_reason != "ok":
        body.append(f" · stopped: {budget_reason}", style="scout.warn")
    return body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _short_args(args: dict) -> str:
    """Render tool args compactly for inline display."""
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        if isinstance(v, str):
            v_s = _short(v, 60)
            parts.append(f'{k}="{v_s}"')
        elif isinstance(v, (int, float, bool)):
            parts.append(f"{k}={v}")
        else:
            try:
                v_s = json.dumps(v, default=str)
            except Exception:
                v_s = str(v)
            parts.append(f"{k}={_short(v_s, 60)}")
    return ", ".join(parts)


def tools_table(specs: list[Any]) -> Table:
    t = Table(title="tools", title_style="cyan", show_lines=False, expand=False)
    t.add_column("name", style="bold")
    t.add_column("description")
    for s in specs:
        t.add_row(s.name, _short(s.description, 80))
    return t


def turns_table(turns: list[Any], *, max_rows: int = 20) -> Table:
    t = Table(title="transcript", title_style="cyan", show_lines=False, expand=True)
    t.add_column("#", style="dim", width=3)
    t.add_column("question", overflow="fold")
    t.add_column("answer", overflow="fold")
    t.add_column("steps", style="dim", justify="right", width=5)
    t.add_column("wall", style="dim", justify="right", width=6)
    for i, turn in enumerate(turns[-max_rows:]):
        t.add_row(
            str(i),
            _short(turn.question, 80),
            _short(turn.answer, 80),
            str(turn.steps_used),
            f"{turn.wall_s:.1f}s",
        )
    return t
