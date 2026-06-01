"""Interactive CLI app — `myagent` / `myAgent`.

A Claude-Code-style terminal UI: a header banner, a `›` prompt, a live
spinner during LLM calls, tool-call lines as the agent works, and a
final-answer panel. Slash commands switch model / provider / policy
on the fly.

Run after `pip install -e .` as `myagent` or `myAgent`. Run without
installing as `python3 -m banna_agent.cli`.

Startup flags mirror the experiment runner so muscle memory carries
over. Once the REPL is up, the same knobs are reachable as slash
commands (`/model`, `/policy`, `/budget`, etc.).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.prompt import Confirm

from ..core.agent import run_policy
from ..core.events import EventKind
from ..core.state import AgentState
from ..core.types import Budget
from ..llm.registry import list_providers, make_client
from ..memory.compactor import CompactionConfig, TraceCompactor
from ..memory.skill_harvester import HarvestConfig, harvest_from_run
from ..tools.base import ToolRegistry
from ..tools.calculator import make_calculator_tool
from ..tools.file_reader import make_file_reader_tool
from ..tools.grep import make_grep_tool
from ..tools.list_files import make_list_files_tool
from ..tools.memory import make_memory_tool
from ..tools.plan import make_plan_tool
from ..tools.python_sandbox import make_python_sandbox_tool
from ..tools.run_shell import make_run_shell_tool
from ..tools.search import make_search_tool
from ..tools.url_reader import make_url_reader_tool
from .commands import POLICY_NAMES, dispatch, is_command
from .display import (
    StreamingEventLog,
    final_answer_panel,
    header_panel,
)
from .session import Session, Turn
from .theme import (
    BRAND,
    animate_hero,
    scout_theme,
)


def _make_console() -> Console:
    return Console(theme=scout_theme())


try:
    import readline  # noqa: F401  (gives input() history + line editing)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@dataclass
class MyAgentApp:
    """Holds runtime config + session, owns the REPL loop."""

    provider: str = "openai"
    model: str | None = "gpt-5-nano"
    policy_name: str = "react+"
    temperature: float = 0.7
    n_candidates: int = 3
    budget_steps: int = 15
    budget_wall_s: float = 300.0
    budget_tokens: int | None = None
    budget_cost_usd: float | None = 5.0

    # Trace compaction — off by default; toggle with /compact.
    compact_enabled: bool = False
    compact_threshold_tokens: int = 30_000
    compact_keep_last_n: int = 4

    # SkillLibrary injection + harvest — off by default; toggle with /skills.
    skills_enabled: bool = False
    skills_top_k: int = 3
    skills_harvest_quality_threshold: float = 0.7

    no_shell: bool = False
    # Isolation backend for run_python / run_shell: "process" (host) or
    # "docker" (network-less container). None resolves from BANNA_SANDBOX.
    sandbox: str | None = None
    # Base image for --sandbox=docker (None → python:3.12-slim). On-demand
    # installs are layered on top of this.
    sandbox_image: str | None = None
    no_plan: bool = False

    console: Console = field(default_factory=_make_console)
    session: Session = field(default_factory=Session)

    # Built lazily — reconstructed on /provider, /model, /policy.
    llm: Any = None
    tools: ToolRegistry | None = None
    policy: Any = None

    # Session-scoped trusted-package allowlist for the docker sandbox; built
    # once on first docker rebuild so session approvals persist across /model
    # etc. The active thinking-spinner, parked here so _approve_install can
    # pause it around a blocking prompt.
    _package_policy: Any = None
    _active_status: Any = None
    # MCP servers: connected once on first tool build, reused across
    # rebuilds, and shut down when the REPL exits.
    _mcp_manager: Any = None

    # ------------------------------------------------------------------
    # build / rebuild
    # ------------------------------------------------------------------

    def rebuild_llm(self) -> None:
        self.llm = make_client(self.provider, model=self.model)

    def rebuild_tools(self) -> None:
        # On-demand package installs are only meaningful for the docker
        # backend. Attach the allowlist + approval callback there; the process
        # backend (default/GAIA) gets the plain factory call, unchanged.
        py_extra: dict[str, Any] = {}
        if self.sandbox == "docker":
            if self._package_policy is None:
                from ..tools.package_policy import PackagePolicy, default_allowlist
                from .config_store import read_package_allowlist
                # Built-in trusted defaults, with the user's config layered on
                # top (so config can override or extend any default pin).
                self._package_policy = PackagePolicy(
                    allowlist={**default_allowlist(), **read_package_allowlist()})
            py_extra = dict(
                approve_install=self._approve_install,
                package_policy=self._package_policy,
                base_image=self.sandbox_image,
            )
        tool_list = [
            make_search_tool(),
            make_url_reader_tool(),
            make_file_reader_tool(),
            make_calculator_tool(),
            make_python_sandbox_tool(sandbox=self.sandbox, **py_extra),
            make_list_files_tool(),
            make_grep_tool(),
        ]
        if not self.no_plan:
            tool_list.append(make_plan_tool())
        if not self.no_shell:
            tool_list.append(make_run_shell_tool(
                confirm=self._confirm_shell, sandbox=self.sandbox))
        # Bind the memory tool to the session's memory store so writes
        # persist across turns within this session.
        if self.session.memory_store is not None:
            tool_list.append(make_memory_tool(self.session.memory_store))
        # MCP server tools. Connect once (first rebuild), then reuse the
        # live sessions across subsequent rebuilds (/model, /policy, …).
        tool_list.extend(self._mcp_tools())
        self.tools = ToolRegistry(tool_list)

    def _mcp_tools(self) -> list:
        """Connect configured MCP servers (once) and return their bridged
        tools, registering each as permission-gated. Failures are warned
        and skipped — a broken server never blocks the REPL."""
        if self._mcp_manager is None:
            from ..tools.mcp import McpManager
            from .mcp_config import load_mcp_configs
            configs = load_mcp_configs()
            if not configs:
                return []
            self._mcp_manager = McpManager(
                configs, prefix=True,
                warn=lambda m: self.console.print(f"[yellow]{m}[/yellow]"),
            )
            self._mcp_manager.start_all()
            from ..core.agent import register_gated_tool
            for t in self._mcp_manager.tools():
                register_gated_tool(t.name, risk="mcp")
            n = self._mcp_manager.server_count()
            ntools = len(self._mcp_manager.tools())
            if ntools:
                self.console.print(
                    f"[dim]mcp: {ntools} tool(s) from {n} server(s)[/dim]")
        return self._mcp_manager.tools()

    def close_mcp(self) -> None:
        """Shut down all MCP server subprocesses/sessions."""
        if self._mcp_manager is not None:
            self._mcp_manager.close_all()
            self._mcp_manager = None

    def _make_user_io(self) -> Any:
        """Build a UserIO for the loop's ask_user + permission gate.

        Returns ``None`` when the active policy doesn't use either
        feature — keeps `react`'s wire-level behavior identical to
        before this change (no extra prompts, identical trace shape).
        """
        if self.policy_name not in ("react+",):
            return None
        return _CliUserIO(console=self.console)

    def _confirm_shell(self, command: str, matched: str) -> bool:
        """Pause the agent and ask the user before running a risky shell command.

        Triggered from inside the run_shell tool. We render a styled
        warning box, show the matched pattern, and read a yes/no from
        stdin. Default is No — accidental Enter never grants permission.
        Ctrl-C is treated as Deny.
        """
        self.console.print()
        self.console.print(
            "[yellow]⚠ agent wants to run a privileged shell command[/yellow]"
        )
        self.console.print(f"  [bold red]$ {command}[/bold red]")
        self.console.print(f"  [dim]matched risky pattern: {matched}[/dim]")
        try:
            ok = Confirm.ask("Allow?", default=False, console=self.console)
        except (KeyboardInterrupt, EOFError):
            self.console.print("[red]denied[/red]")
            return False
        if ok:
            self.console.print("[green]allowed[/green]")
        else:
            self.console.print("[red]denied[/red]")
        return ok

    def _approve_install(self, import_name: str, pip_spec: str) -> bool:
        """Ask whether to install a non-allowlisted package into the docker
        sandbox image. Returns True to proceed (install for this session),
        False to deny. Triggered from inside run_python when an import fails
        and the package isn't on the allowlist; available under every policy.

        [1] install once   [2] add to allowlist (persist)   [3] deny.
        """
        status = self._active_status
        if status is not None:
            try:
                status.stop()
            except Exception:
                pass
        try:
            self.console.print()
            self.console.print(
                "[yellow]⚠ sandbox code needs a package not on the "
                "allowlist[/yellow]"
            )
            self.console.print(
                f"  import [bold]{import_name}[/bold]  →  "
                f"pip install [bold]{pip_spec}[/bold]"
            )
            self.console.print(
                "[1] install once   [2] add to allowlist (persist)   [3] deny"
            )
            try:
                raw = input("> [1/2/3, default 3]: ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("[red]denied[/red]")
                return False
            if raw == "1":
                self.console.print("[green]installing for this session[/green]")
                return True
            if raw == "2":
                from .config_store import write_package_allowlist
                write_package_allowlist({import_name: pip_spec})
                if self._package_policy is not None:
                    self._package_policy.allowlist[import_name] = pip_spec
                self.console.print("[green]added to allowlist[/green]")
                return True
            self.console.print("[red]denied[/red]")
            return False
        finally:
            if status is not None:
                try:
                    status.start()
                except Exception:
                    pass

    def rebuild_policy(self) -> None:
        self.policy = self._build_policy()

    def _build_policy(self) -> Any:
        # `react+` is the only policy exposed by the public CLI. It
        # subclasses `ReActPolicy`, so the entire ReAct engine is
        # inherited; the bare `react` module ships only as that parent.
        name = self.policy_name
        if name == "react+":
            from ..policies.react_plus import ReActPlusPolicy
            return ReActPlusPolicy(model=self.model)
        raise ValueError(f"unknown policy: {name}")

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------

    def print_header(self) -> None:
        # Brief running-hero animation above the panel — the 4-row mascot
        # cycling through leg-shift frames. Settles on the idle pose.
        try:
            animate_hero(
                self.console,
                frames=12,
                fps=8,
                leading_text=f"[scout.agent]{BRAND};[/scout.agent] [scout.muted]under-construction[/scout.muted]",
            )
        except Exception:
            pass
        specs = self.tools.to_tool_specs() if self.tools else []
        self.console.print(header_panel(
            provider=self.provider,
            model=self.model or "",
            policy=self.policy_name,
            tools=[s.name for s in specs],
            budget_steps=self.budget_steps,
            budget_wall_s=self.budget_wall_s,
        ))

    # ------------------------------------------------------------------
    # task loop
    # ------------------------------------------------------------------

    def _run_task(self, user_text: str) -> None:
        # Skill injection: when enabled, find the top-k relevant skills
        # and prepend their callable headers to the question so the
        # model sees them inline. Harmless when the library is empty —
        # `as_python_header([])` returns "".
        skill_header = ""
        injected_skills: list = []
        if self.skills_enabled and self.session.skill_library is not None:
            try:
                injected_skills = self.session.skill_library.search(
                    user_text, k=self.skills_top_k,
                )
            except Exception:
                injected_skills = []
            if injected_skills:
                blob = self.session.skill_library.as_python_header(injected_skills)
                skill_header = (
                    "Relevant skills available — call these inside the "
                    "`run_python` tool when applicable:\n\n" + blob
                )

        question = self.session.compose_question(user_text, skill_header=skill_header)
        budget = Budget(
            max_steps=self.budget_steps,
            max_wall_s=self.budget_wall_s,
            max_tokens_total=self.budget_tokens,
            max_cost_usd=self.budget_cost_usd,
        )
        state = AgentState(question=question, budget=budget)

        # Build a TraceCompactor only when the user has enabled it. Off by
        # default to keep wall-time and tokens predictable for short tasks.
        compactor = None
        if self.compact_enabled:
            compactor = TraceCompactor(
                llm=self.llm,
                config=CompactionConfig(
                    enabled=True,
                    threshold_tokens=self.compact_threshold_tokens,
                    keep_last_n_steps=self.compact_keep_last_n,
                ),
            )

        t0 = time.monotonic()
        err: str | None = None
        with self.console.status("[dim]thinking…[/dim]", spinner="dots") as status:
            # Parked so _approve_install (called from inside run_python, under
            # the spinner) can pause/restart it around its blocking prompt.
            self._active_status = status
            log = StreamingEventLog(
                console=self.console, status=status, state=state,
            )
            user_io = self._make_user_io()
            # The thinking spinner is a live renderer that owns the
            # terminal; a bare input() underneath it gets repainted over
            # and reads nothing. Hand the user_io the spinner handle so
            # it can stop/restart it around each blocking prompt.
            if user_io is not None and hasattr(user_io, "bind_status"):
                user_io.bind_status(status)
            try:
                state = run_policy(state, self.policy, llm=self.llm,
                                   tools=self.tools, log=log,
                                   compactor=compactor,
                                   user_io=user_io)
            except KeyboardInterrupt:
                err = "interrupted by user (Ctrl-C)"
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        self._active_status = None

        # Prefer the budget's elapsed wall: it excludes time the loop sat
        # paused on an ask_user / permission prompt (see BudgetTracker.pause).
        # Falls back to raw monotonic if the loop never ticked the budget
        # (e.g. it errored before the first step).
        wall_s = getattr(state.budget, "elapsed_wall_s", 0.0) or (time.monotonic() - t0)
        answer = state.trace.final_answer or ""
        budget_reason = "ok"
        for ev in reversed(log.events):
            if ev.kind == EventKind.RUN_END:
                budget_reason = ev.payload.get("budget_reason", "ok")
                break
            if ev.kind == EventKind.BUDGET:
                budget_reason = ev.payload.get("reason", "ok")
                break

        if err:
            self.console.print(f"[red]error:[/red] {err}")

        # Post-run hook: verifier-driven segment promotion + (when D2 is on)
        # skill harvest. Only fires when the policy implements it.
        promo_report = None
        post_hook = getattr(self.policy, "on_run_end", None)
        if post_hook is not None and not err:
            try:
                promo_report = post_hook(state)
            except Exception as exc:
                self.console.print(f"[yellow]post-run hook failed: {exc}[/yellow]")
        if promo_report and promo_report.get("ran"):
            verdicts = promo_report.get("verdicts", {})
            quality = promo_report.get("quality", 0.0)
            promoted = promo_report.get("promoted", 0)
            n_segs = promo_report.get("n_segments", 0)
            verdict_str = "  ".join(
                f"[{('green' if k == 'ok' else 'red' if k == 'fail' else 'yellow')}]{k}={v}[/]"
                for k, v in verdicts.items() if v
            )
            self.console.print(
                f"[dim]verifiers:[/dim] {verdict_str or '(none)'}   "
                f"[dim]quality=[/dim]{quality:.2f}   "
                f"[dim]segments→P_K:[/dim] {promoted}/{n_segs}"
            )
            # Show which verifier failed / warned and why.
            for fd in promo_report.get("fail_details", []) or []:
                marker = ("[red]✗[/red]" if fd.get("verdict") == "fail"
                          else "[yellow]⚠[/yellow]")
                vname = fd.get("verifier_name", "?")
                claim = fd.get("claim_id", "")
                detail = (fd.get("detail") or "").strip()
                # Trim long detail strings.
                if len(detail) > 200:
                    detail = detail[:197] + "…"
                claim_part = f" [dim]({claim})[/dim]" if claim and claim != "__answer__" else ""
                self.console.print(
                    f"  {marker} [bold]{vname}[/bold]{claim_part}: "
                    f"[dim]{detail or '(no detail)'}[/dim]"
                )

        # Skill harvest: when enabled and the run cleared the quality
        # threshold (taken from the verifier promo_report when available,
        # else assumed 1.0 if state finalized cleanly), look at the trace
        # for `run_python` calls that defined reusable functions.
        harvested_skills: list = []
        if (
            self.skills_enabled
            and not err
            and state.is_done
            and self.session.skill_library is not None
        ):
            quality = (promo_report or {}).get("quality", 1.0) if promo_report else 1.0
            if quality >= self.skills_harvest_quality_threshold:
                try:
                    harvested_skills = harvest_from_run(
                        state,
                        self.session.skill_library,
                        config=HarvestConfig(verifier_name="run_end_hook"),
                        correct_final_answer=True,
                        verifier_name="run_end_hook",
                    )
                except Exception as exc:
                    self.console.print(f"[yellow]skill harvest failed: {exc}[/yellow]")
        if injected_skills or harvested_skills:
            inj = ", ".join(s.name for s in injected_skills) or "(none)"
            harv = ", ".join(s.name for s in harvested_skills) or "(none)"
            self.console.print(
                f"[dim]skills:[/dim] injected=[bold]{inj}[/bold]   "
                f"harvested=[bold]{harv}[/bold]"
            )

        self.console.print(final_answer_panel(
            answer=answer if answer else (err or "(no answer)"),
            steps_used=state.budget.steps_used,
            wall_s=wall_s,
            tokens_in=state.budget.tokens_in,
            tokens_out=state.budget.tokens_out,
            cost_usd=state.budget.cost_usd,
            budget_reason=budget_reason,
        ))

        self.session.add_turn(Turn(
            question=user_text,
            answer=answer,
            wall_s=wall_s,
            steps_used=state.budget.steps_used,
            tokens_in=state.budget.tokens_in,
            tokens_out=state.budget.tokens_out,
            cost_usd=state.budget.cost_usd,
            provider=self.provider,
            model=self.model or "",
            policy=self.policy_name,
            budget_reason=budget_reason,
            error=err,
        ), state)

        # Auto-save to the resumable session store (best-effort; never
        # let a persistence hiccup interrupt the conversation).
        try:
            from .sessions import autosave
            autosave(self.session)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------

    def _read_line(self) -> str | None:
        """Plain `❯ ` prompt, orange, non-editable.

        The caret is passed as `input()`'s prompt argument — readline
        treats the prompt region as a hard left boundary, so backspace
        / Ctrl-U / Home cannot delete it. ANSI color codes are wrapped
        in `\\001`/`\\002` (RL_PROMPT_*_IGNORE) so readline doesn't
        count them toward the visible prompt width.
        """
        # Orange (#cb4b16), bold; matches scout.you. \001/\002 mark the
        # non-printing escape so readline measures prompt width correctly.
        ORANGE_ON = "\001\x1b[1;38;2;203;75;22m\002"
        RESET = "\001\x1b[0m\002"
        prompt = f"{ORANGE_ON}❯{RESET} "
        try:
            self.console.print()          # 1 blank line above
            line = input(prompt)
            # 3 blank lines of breathing room between turns.
            for _ in range(3):
                self.console.print()
            return line
        except EOFError:
            self.console.print()
            return None
        except KeyboardInterrupt:
            self.console.print()
            return ""

    def run(self) -> int:
        try:
            return self._run_loop()
        finally:
            self.close_mcp()

    def _run_loop(self) -> int:
        self.print_header()
        self.console.print("[dim]Type a question, or /help for commands. Ctrl-D to exit.[/dim]")
        while True:
            line = self._read_line()
            if line is None:
                self.console.print("\n[dim]goodbye.[/dim]")
                return 0
            line = line.strip()
            if not line:
                continue
            if is_command(line):
                try:
                    should_exit = dispatch(self, line)
                except Exception as exc:
                    self.console.print(f"[red]command error:[/red] {exc}")
                    continue
                if should_exit:
                    return 0
                continue
            try:
                self._run_task(line)
            except Exception as exc:
                self.console.print(f"[red]task crashed:[/red] {exc}")


# ---------------------------------------------------------------------------
# Interactive UserIO for `react+`
# ---------------------------------------------------------------------------


class _CliUserIO:
    """Rich-backed UserIO for the interactive REPL.

    Implements the `UserIO` Protocol (see core/user_io.py):
      • `ask(question)`: shows the model's question in a styled box,
        reads one line from stdin, returns it (or empty string on EOF).
      • `confirm(...)`: shows the tool name + truncated args, prompts
        with allow_once / allow_always / deny.

    Lives in app.py because it's CLI-shaped; the protocol is in core/
    so non-CLI surfaces (web, test harness) can supply their own.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._status: Any = None

    def bind_status(self, status: Any) -> None:
        """Attach the active thinking-spinner so blocking prompts can
        pause it (it's a live renderer that otherwise eats stdin)."""
        self._status = status

    def _pause_spinner(self) -> None:
        if self._status is not None:
            try:
                self._status.stop()
            except Exception:
                pass

    def _resume_spinner(self) -> None:
        if self._status is not None:
            try:
                self._status.start()
            except Exception:
                pass

    def ask(self, question: str) -> str:
        self.console.print()
        self.console.print(f"[cyan]? agent asks:[/cyan] {question}")
        self._pause_spinner()
        try:
            return input("[your reply] > ").strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print("[dim](no reply)[/dim]")
            return ""
        finally:
            self._resume_spinner()

    def confirm(self, *, tool_name: str, args: dict, risk: str) -> str:
        import json as _json
        try:
            args_disp = _json.dumps(args, indent=2, default=str)
        except Exception:
            args_disp = repr(args)
        # Truncate long arg blobs so the modal stays scannable.
        if len(args_disp) > 600:
            args_disp = args_disp[:600] + "\n  …(truncated)"
        self.console.print()
        self.console.print(
            f"[yellow]⚠ permission needed[/yellow]  "
            f"[bold]{tool_name}[/bold] [dim](risk: {risk})[/dim]"
        )
        for line in args_disp.splitlines():
            self.console.print(f"  [dim]{line}[/dim]")
        self.console.print(
            "[1] allow once   [2] allow always (this session)   [3] deny"
        )
        self._pause_spinner()
        try:
            raw = input("> [1/2/3, default 3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print("[red]denied[/red]")
            return "deny"
        finally:
            self._resume_spinner()
        if raw == "1":
            return "allow_once"
        if raw == "2":
            return "allow_always"
        return "deny"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="myagent",
        description="Interactive CLI for the agent. Type questions, get answers.",
    )
    ap.add_argument("--provider", default=os.environ.get("MYAGENT_PROVIDER", "openai"),
                    choices=list_providers(),
                    help="LLM provider to start with.")
    ap.add_argument("--model", default=os.environ.get("MYAGENT_MODEL", "gpt-5-nano"),
                    help="Override the provider's default model.")
    ap.add_argument("--policy", default="react+", choices=POLICY_NAMES,
                    help="Policy to run.")
    ap.add_argument("--n-candidates", type=int, default=3,
                    help="N candidate plans (BFS/DFS/best-first).")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="LLM temperature (where applicable).")

    ap.add_argument("--budget-steps", type=int, default=15,
                    help="Max steps per task.")
    ap.add_argument("--budget-wall", type=float, default=300.0,
                    help="Max wall-seconds per task.")
    ap.add_argument("--budget-tokens", type=int, default=None,
                    help="Max total tokens per task (default: unlimited).")
    ap.add_argument("--budget-cost", type=float, default=5.0,
                    help="Max USD cost per task (default: $5).")

    ap.add_argument("--no-shell", action="store_true",
                    help="Drop run_shell from the tool registry.")
    ap.add_argument("--no-plan", action="store_true",
                    help="Drop plan from the tool registry.")
    ap.add_argument("--sandbox", choices=("process", "docker"), default=None,
                    help="Isolation backend for run_python / run_shell. "
                         "'process' (default) runs on the host; 'docker' "
                         "confines each call to a network-less, read-only "
                         "container. Falls back to the BANNA_SANDBOX env var.")
    ap.add_argument("--sandbox-image",
                    default=os.environ.get("BANNA_SANDBOX_IMAGE"),
                    help="Base Docker image for --sandbox=docker "
                         "(default python:3.12-slim). Packages installed on "
                         "demand are layered on top. Falls back to the "
                         "BANNA_SANDBOX_IMAGE env var.")
    ap.add_argument("--skills", action="store_true",
                    help="Enable skill-library injection + harvest. Off by default.")
    ap.add_argument("--resume", nargs="?", const="__pick__", default=None,
                    metavar="ID",
                    help="Resume a previous conversation. `--resume` with no "
                         "argument shows a picker of recent sessions; "
                         "`--resume <id>` resumes that session directly; "
                         "`--resume last` resumes the most recent.")

    return ap.parse_args(argv)


def _load_dotenv() -> tuple[Path | None, int]:
    """Auto-load a .env file from the cwd, `~/.config/banna/`, or
    `~/.config/myagent/` (legacy).

    Lines are parsed as KEY=VALUE; lines starting with # and blank lines
    are skipped. **Existing os.environ vars are not overwritten** —
    anything you `export`'d in your shell beats the .env file. Surrounding
    quotes on the value are stripped for convenience.

    Returns (path_loaded, n_vars) so the CLI can report the source.
    """
    candidates = [
        Path.cwd() / ".env",
        Path.home() / ".config" / "banna" / ".env",
        Path.home() / ".config" / "myagent" / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        n = 0
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # Tolerate `export KEY=VALUE` shell syntax.
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if not k or not k.replace("_", "").isalnum():
                    continue
                # Strip surrounding quotes.
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                    v = v[1:-1]
                if k in os.environ:
                    continue  # shell wins
                os.environ[k] = v
                n += 1
        except OSError:
            continue
        return path, n
    return None, 0


def main(argv: list[str] | None = None) -> int:
    # Subcommand dispatch (banna init / config / providers) before
    # argparse, so legacy `banna --policy X` still works.
    raw = list(sys.argv[1:] if argv is None else argv)
    from . import subcommands as _sub
    if _sub.is_subcommand(raw):
        return _sub.dispatch(raw)

    # First-run wizard: triggered when `~/.config/banna/config.toml`
    # doesn't exist AND no provider key is reachable. Users who already
    # `export OPENAI_API_KEY` in their shell keep working without a
    # nag; users who pip-installed fresh get the walkthrough.
    from .config_store import is_first_run, read_config

    dotenv_path, dotenv_n = _load_dotenv()
    cfg_default: dict = (read_config().get("default") or {}) if not is_first_run() else {}

    if is_first_run() and not any(
        os.environ.get(v) for v in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")
    ):
        from .setup_wizard import run_wizard
        wiz = run_wizard()
        dotenv_path, dotenv_n = _load_dotenv()
        cfg_default = {"provider": wiz.provider, "model": wiz.model}

    args = _parse_args(argv)

    # Saved config defaults take effect *only when the user didn't pass
    # the corresponding flag*. We detect that by scanning `raw` rather
    # than comparing against argparse defaults (which embed env-var
    # fallbacks that can mask "user didn't say anything").
    def _flag_present(flag: str) -> bool:
        return any(a == flag or a.startswith(flag + "=") for a in raw)

    provider = args.provider if _flag_present("--provider") \
        else cfg_default.get("provider", args.provider)
    model = args.model if _flag_present("--model") \
        else cfg_default.get("model", args.model)
    policy_name = args.policy if _flag_present("--policy") \
        else cfg_default.get("policy", args.policy)
    # A saved config may name a policy this build no longer exposes (e.g. an
    # older `policy = "react"` from before react+ became the sole CLI
    # policy). Don't crash on a stale default — fall back to the argparse
    # default and tell the user, so the REPL still starts.
    if policy_name not in POLICY_NAMES:
        fallback = args.policy if args.policy in POLICY_NAMES else POLICY_NAMES[0]
        print(
            f"note: configured policy {policy_name!r} is not available in "
            f"this build; using {fallback!r}. "
            f"Set a new default with `banna config set policy {fallback}`.",
            file=sys.stderr,
        )
        policy_name = fallback

    app = MyAgentApp(
        provider=provider,
        model=model,
        policy_name=policy_name,
        temperature=args.temperature,
        n_candidates=args.n_candidates,
        budget_steps=args.budget_steps,
        budget_wall_s=args.budget_wall,
        budget_tokens=args.budget_tokens,
        budget_cost_usd=args.budget_cost,
        no_shell=args.no_shell,
        no_plan=args.no_plan,
        sandbox=args.sandbox,
        sandbox_image=args.sandbox_image,
        skills_enabled=args.skills,
    )
    if dotenv_path is not None:
        app.console.print(
            f"[dim]loaded {dotenv_n} env var(s) from {dotenv_path}[/dim]"
        )
    if args.resume is not None:
        _apply_resume(app, args.resume)

    try:
        app.rebuild_llm()
    except Exception as exc:
        print(f"failed to build LLM client for provider={args.provider}: {exc}",
              file=sys.stderr)
        return 2
    app.rebuild_tools()
    app.rebuild_policy()
    return app.run()


def _apply_resume(app: "MyAgentApp", which: str) -> None:
    """Restore a prior session into `app` per the --resume argument.

    `__pick__` (bare --resume) lists recent sessions and prompts; `last`
    resumes the most recent; anything else is treated as a session id or
    path. On any miss we warn and start fresh rather than abort.
    """
    from .sessions import latest_session, list_sessions, load_session

    target_id: str | None = None
    if which == "__pick__":
        infos = list_sessions(limit=15)
        if not infos:
            app.console.print("[yellow]no saved sessions to resume[/yellow]")
            return
        app.console.print("[bold]recent sessions:[/bold]")
        for n, info in enumerate(infos, 1):
            app.console.print(
                f"  [bold]{n}[/bold]. {info.id}  "
                f"[dim]({info.n_turns} turn(s))[/dim]  {info.first_question[:60]}")
        try:
            raw = input("resume which? [number, or Enter to skip]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""
        if not raw:
            return
        try:
            target_id = infos[int(raw) - 1].id
        except (ValueError, IndexError):
            app.console.print("[yellow]invalid choice; starting fresh[/yellow]")
            return
    elif which == "last":
        info = latest_session()
        if info is None:
            app.console.print("[yellow]no saved sessions to resume[/yellow]")
            return
        target_id = info.id
    else:
        target_id = which

    try:
        app.session = load_session(target_id)
    except FileNotFoundError:
        app.console.print(f"[yellow]no such session: {target_id}; starting fresh[/yellow]")
        return
    app.console.print(
        f"[green]resumed[/green] {len(app.session.turns)} turn(s) "
        f"from session [bold]{target_id}[/bold]")


if __name__ == "__main__":
    sys.exit(main())
