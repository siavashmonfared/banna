"""Slash-command dispatch.

Each handler has signature `cmd_<name>(app, args: list[str]) -> bool`.
The bool is `should_exit` — return True to break the REPL.

The `COMMANDS` dict maps command name → handler. `dispatch(app, line)`
parses a slash line and calls the right handler.
"""
from __future__ import annotations

import os
import shlex
from typing import Any, Callable

from rich.panel import Panel
from rich.text import Text

from .display import tools_table, turns_table


# Per-policy human-readable knobs we accept for `/policy <name>`.
#
# `react+` is the only policy the public CLI exposes. It subclasses the
# bare ReAct loop (which ships as its parent class, not as a selectable
# policy), adding the interactive `ask_user` affordance, a per-tool
# permission gate, and error-scoping prompt guardrails.
POLICY_NAMES = (
    "react+",
)


# Curated per-provider model lists for `/model` interactive picker.
# Ollama is queried live via /api/tags; everything else is a hand-kept
# shortlist. Users can always type a name not in the list.
KNOWN_MODELS: dict[str, tuple[str, ...]] = {
    "anthropic": (
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
    ),
    "bedrock": (
        # ---- Claude 4.x (current, US cross-region inference profiles) ----
        # Inference-profile IDs route across us-east-1 / us-west-2 / etc.
        # which sidesteps regional capacity issues. Foundation-model IDs
        # without the `us.` prefix are region-locked; if you need to pin
        # a region, drop the prefix.
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        # ---- Cross-continent "global." inference profiles (where AWS
        # has published them) — broader failover, slightly higher
        # latency. Available for newer models.
        "global.anthropic.claude-opus-4-5-20251101-v1:0",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        # ---- EU / APAC inference profiles for users outside the US.
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "apac.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "apac.anthropic.claude-haiku-4-5-20251001-v1:0",
        # ---- Claude 4.0 / 4.1 (still listed on Bedrock for ablation /
        # back-compat). May not be enabled in every account — check
        # `aws bedrock list-foundation-models` to confirm.
        "us.anthropic.claude-opus-4-1-20250805-v1:0",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        # ---- Claude 3.x family — older, cheaper, broadly available.
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "us.anthropic.claude-3-haiku-20240307-v1:0",
        # ---- Region-locked foundation IDs (no `us.`/`eu.`/`apac.`/
        # `global.` prefix). Use these if you've pinned to one region
        # and don't want cross-region routing.
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ),
    "openai": (
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "o4-mini",
    ),
    "gemini": (
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ),
    "ollama": (),  # populated dynamically
}


# Map cloud provider → its API key env var. Used by /provider and
# /model to detect missing keys before submitting a doomed request.
# Ollama is keyless. Bedrock uses AWS_REGION + (keys or profile), which
# is checked at client-construction time and produces its own
# ProviderError; we don't second-guess it here.
PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


# ---------------------------------------------------------------------------
# Interactive helpers — numbered picker, typed prompts
# ---------------------------------------------------------------------------


def _pick(
    app: Any,
    *,
    label: str,
    options: list[str],
    current: str | None,
    allow_custom: bool = False,
) -> str | None:
    """Numbered picker. Returns chosen value, or None to keep current.

    Accepts:
      - a number (1-based)
      - the exact name
      - a unique prefix (e.g. 'react' matches if 'react' is unique)
      - any free-form string (only when allow_custom=True)
    """
    if not options and not allow_custom:
        app.console.print(f"[red]no {label} options available[/red]")
        return None

    app.console.print(f"\n[bold]{label}[/bold]")
    for i, opt in enumerate(options, 1):
        mark = " [green](current)[/green]" if opt == current else ""
        app.console.print(f"  [cyan]{i}.[/cyan] {opt}{mark}")
    if not options:
        app.console.print("  [dim](no preset list — type a name)[/dim]")

    hint_parts = []
    if options:
        hint_parts.append(f"1-{len(options)}")
    if allow_custom:
        hint_parts.append("type a name")
    hint_parts.append("Enter to keep")
    prompt = f"choose [{', '.join(hint_parts)}]: "

    try:
        raw = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        app.console.print("\n[dim](cancelled)[/dim]")
        return None
    if not raw:
        return None

    # Numeric?
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1]
        app.console.print(f"[red]out of range:[/red] {idx}")
        return None

    # Exact match
    if raw in options:
        return raw
    # Unique prefix
    matches = [o for o in options if o.startswith(raw)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        app.console.print(f"[red]ambiguous, matched:[/red] {', '.join(matches)}")
        return None
    if allow_custom:
        return raw
    app.console.print(f"[red]not in list:[/red] {raw}")
    return None


def _ask_value(
    app: Any,
    *,
    label: str,
    current: Any,
    parser,
    placeholder: str = "",
) -> Any:
    """Prompt for a value with the current shown as default.

    `parser` is `int` / `float` / `str` or any callable raising on bad input.
    Empty input keeps the current value. Returns the new (or current) value.
    """
    cur_disp = placeholder if current is None else str(current)
    try:
        raw = input(f"  {label} [{cur_disp}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        app.console.print("\n[dim](cancelled)[/dim]")
        return current
    if not raw:
        return current
    # Allow "none"/"unlimited" for nullable fields.
    if raw.lower() in ("none", "null", "unlimited", "off"):
        return None
    try:
        return parser(raw)
    except (ValueError, TypeError) as exc:
        app.console.print(f"[red]bad value: {exc}[/red]  (kept {cur_disp})")
        return current


def _ollama_models(app: Any) -> list[str]:
    """Live-query the Ollama daemon for installed models."""
    try:
        import requests  # local: only needed when Ollama is the provider

        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        r = requests.get(f"{base}/api/tags", timeout=2.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as exc:
        app.console.print(f"[yellow]could not query Ollama:[/yellow] {exc}")
        return []


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


# One-line synopsis per command for the top-level /help.
_HELP_TEXT = """\
Slash commands

  /help [<cmd>]                  show this list, or detailed help for one command
  /model [<name>]                show or change LLM model
  /provider <name>               switch provider (anthropic, openai, gemini, bedrock, ollama)
  /policy [<name>]               show or change policy ({policies})
  /temperature <float>           change temperature (where applicable)
  /budget [steps=N] [wall=N]     change default per-task budget
  /tools                         list registered tools
  /keys                          show provider API key status (masked)
  /status                        summary: provider, model, policy, budget, toggles, cost
  /cost [detailed|rates]         show estimated $ cost + tokens for the session
  /compact [on|off|key=val]      toggle trace compaction (off by default)
  /skills [on|off|show|clear]    toggle / inspect the skill library
  /memory [show|clear|path]      inspect or manage persistent memory store
  /clear [all]                   clear transcript (and memory if 'all')
  /show <topic>                  trace | last | transcript | evidence | claims | traj | fields
  /save <path>                   write transcript to JSONL
  /load <path>                   load transcript from JSONL (replaces current)
  /exit, /quit                   leave the REPL  (Ctrl-D also works)

Anything else is sent as a question to the agent.

Use [bold]/help <command>[/bold] (e.g. [bold]/help budget[/bold]) for detailed
options + examples for that command.
""".replace("{policies}", " | ".join(POLICY_NAMES))


def cmd_help(app: Any, args: list[str]) -> bool:
    """Show this list, or detailed help for one command.

    Usage:
      /help               list all commands
      /help <command>     show <command>'s docstring + accepted forms

    Examples:
      /help budget        # explains /budget and how key=value parsing works
      /help show          # lists every /show topic
    """
    if not args:
        app.console.print(Panel(Text.from_markup(_HELP_TEXT.rstrip()),
                                title="help", title_align="left",
                                border_style="cyan"))
        return False
    name = args[0].lstrip("/").lower()
    handler = COMMANDS.get(name)
    if handler is None:
        app.console.print(
            f"[red]unknown command:[/red] /{name}\n"
            f"available: {', '.join(sorted(set(COMMANDS)))}"
        )
        return False
    doc = (handler.__doc__ or "(no docstring)").strip()
    # Re-indent: docstrings are usually 4-space-indented after the first line.
    lines = []
    raw_lines = doc.splitlines()
    if raw_lines:
        lines.append(raw_lines[0])  # first line as-is
        for ln in raw_lines[1:]:
            lines.append(ln.lstrip() if ln.strip() else "")
    body = "\n".join(lines)
    app.console.print(Panel(
        body,
        title=f"/{name}",
        title_align="left",
        border_style="cyan",
    ))
    return False


# ---------------------------------------------------------------------------
# Model / provider / policy
# ---------------------------------------------------------------------------


def cmd_model(app: Any, args: list[str]) -> bool:
    """Pick a model. Bare `/model` shows a numbered list; `/model <name>` jumps."""
    if args:
        new = args[0]
    else:
        app.console.print(f"current model: [bold]{app.model or '(provider default)'}[/bold]")
        if app.provider == "ollama":
            options = _ollama_models(app)
            if not options:
                app.console.print(
                    "[yellow]no models registered with Ollama; "
                    "run [bold]ollama pull <name>[/bold] first[/yellow]"
                )
        else:
            options = list(KNOWN_MODELS.get(app.provider, ()))
        new = _pick(app, label=f"models for {app.provider}",
                    options=options, current=app.model, allow_custom=True)
        if new is None:
            return False
    app.model = new
    try:
        app.rebuild_llm()
        app.rebuild_policy()
    except Exception as exc:
        app.console.print(f"[red]failed to switch model: {exc}[/red]")
        return False
    app.console.print(f"model → [bold]{new}[/bold]")
    return False


def _ensure_api_key(app: Any, provider: str) -> bool:
    """If the provider needs an API key and none is set, offer to fix it.

    Returns True when the provider is ready to use (either keyless,
    already has a key, or the user just supplied one). Returns False
    when the user backed out — caller should NOT switch.
    """
    env_var = PROVIDER_API_KEY_ENV.get(provider)
    if not env_var:
        return True  # ollama / bedrock / anything keyless via env-var
    if os.environ.get(env_var):
        return True
    app.console.print(
        f"[yellow]no {env_var} found in your environment.[/yellow] "
        f"{provider} won't work until one is set."
    )
    app.console.print(
        "  1. paste a key now (saved to this session only — not persisted)"
    )
    app.console.print(f"  2. go back (keep current provider: [bold]{app.provider}[/bold])")
    try:
        raw = input("  > [1/2, default 2]: ").strip()
    except (EOFError, KeyboardInterrupt):
        app.console.print("\n[dim](cancelled)[/dim]")
        return False
    if raw != "1":
        return False
    try:
        key = input(f"  paste {env_var} (input echoed): ").strip()
    except (EOFError, KeyboardInterrupt):
        app.console.print("\n[dim](cancelled)[/dim]")
        return False
    if not key:
        app.console.print("[dim]empty key — cancelled[/dim]")
        return False
    os.environ[env_var] = key
    app.console.print(
        f"[dim]✓ {env_var} set for this session. "
        f"To persist, add [bold]export {env_var}={key[:4]}…[/bold] to your shell rc.[/dim]"
    )
    return True


def cmd_provider(app: Any, args: list[str]) -> bool:
    """Pick a provider. Bare `/provider` shows a numbered list."""
    from ..llm.registry import list_providers

    if args:
        new = args[0].lower()
    else:
        app.console.print(f"current provider: [bold]{app.provider}[/bold]")
        new = _pick(app, label="providers", options=list(list_providers()),
                    current=app.provider, allow_custom=False)
        if new is None:
            return False
    # Before switching, check the API key is available. The agent loop
    # now fails fast on missing keys (ProviderError(retryable=False)),
    # so catching it here means the user never sees the "0 steps,
    # stopped: provider_error" branch.
    if not _ensure_api_key(app, new):
        return False
    app.provider = new
    try:
        app.rebuild_llm()
    except Exception as exc:
        app.console.print(f"[red]failed to switch provider: {exc}[/red]")
        return False
    app.console.print(
        f"provider → [bold]{new}[/bold]   "
        f"[dim](model is now {app.model or '(provider default)'} — use /model to change)[/dim]"
    )
    return False


def cmd_policy(app: Any, args: list[str]) -> bool:
    """Pick a policy. Bare `/policy` shows a numbered list."""
    if args:
        new = args[0]
        if new not in POLICY_NAMES:
            app.console.print(
                f"[red]unknown policy:[/red] {new}\n"
                f"available: {', '.join(POLICY_NAMES)}"
            )
            return False
    else:
        app.console.print(f"current policy: [bold]{app.policy_name}[/bold]")
        new = _pick(app, label="policies", options=list(POLICY_NAMES),
                    current=app.policy_name, allow_custom=False)
        if new is None:
            return False
    app.policy_name = new
    app.rebuild_policy()

    app.console.print(f"policy → [bold]{new}[/bold]")
    return False


def cmd_temperature(app: Any, args: list[str]) -> bool:
    """Set temperature. Bare `/temperature` prompts for a value."""
    if args:
        try:
            new = float(args[0])
        except ValueError:
            app.console.print(f"[red]not a number:[/red] {args[0]}")
            return False
    else:
        app.console.print(
            "\n[bold]temperature[/bold]    "
            "[dim]controls LLM randomness; 0.0=deterministic, 1.0=creative[/dim]"
        )
        new = _ask_value(app, label="temperature", current=app.temperature,
                         parser=float)
        if new is None:
            new = app.temperature
        if new == app.temperature:
            return False
    app.temperature = new
    app.rebuild_policy()
    app.console.print(f"temperature → [bold]{new:.2f}[/bold]")
    return False


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def cmd_budget(app: Any, args: list[str]) -> bool:
    """Change the per-task budget.

    Bare `/budget` prompts for each value. Power-user form still works:
        /budget steps=12 wall=300 tokens=20000 cost=0.50
    """
    if args:
        for tok in args:
            if "=" not in tok:
                app.console.print(f"[red]expected key=value, got:[/red] {tok}")
                continue
            k, v = tok.split("=", 1)
            k = k.strip().lower()
            try:
                if k in ("steps", "max_steps"):
                    app.budget_steps = int(v)
                elif k in ("wall", "wall_s", "max_wall_s"):
                    app.budget_wall_s = float(v)
                elif k in ("tokens", "max_tokens", "max_tokens_total"):
                    app.budget_tokens = int(v) if v.lower() not in ("none", "unlimited") else None
                elif k in ("cost", "cost_usd", "max_cost", "max_cost_usd"):
                    app.budget_cost_usd = (
                        float(v) if v.lower() not in ("none", "unlimited") else None
                    )
                else:
                    app.console.print(f"[yellow]unknown budget key:[/yellow] {k}")
            except ValueError:
                app.console.print(f"[red]bad value for {k}:[/red] {v}")
    else:
        app.console.print(
            "\n[bold]budget per task[/bold]    "
            "[dim](Enter to keep, 'unlimited' to remove a cap)[/dim]"
        )
        app.budget_steps = _ask_value(
            app, label="max steps", current=app.budget_steps, parser=int,
        )
        app.budget_wall_s = _ask_value(
            app, label="max wall_s", current=app.budget_wall_s, parser=float,
        )
        app.budget_tokens = _ask_value(
            app, label="max tokens", current=app.budget_tokens, parser=int,
            placeholder="unlimited",
        )
        app.budget_cost_usd = _ask_value(
            app, label="max USD cost", current=app.budget_cost_usd, parser=float,
            placeholder="unlimited",
        )

    tok_disp = "unlimited" if app.budget_tokens is None else str(app.budget_tokens)
    cost_disp = (
        "unlimited" if app.budget_cost_usd is None
        else f"${app.budget_cost_usd:.4f}"
    )
    app.console.print(
        f"budget → [bold]{app.budget_steps}[/bold] steps · "
        f"[bold]{app.budget_wall_s:.0f}s[/bold] wall · "
        f"[bold]{tok_disp}[/bold] tokens · "
        f"[bold]{cost_disp}[/bold] cost"
    )
    return False


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def cmd_tools(app: Any, args: list[str]) -> bool:
    """List tools registered with the current agent.

    Usage:
      /tools

    Shows each tool's name + description as a table. Tools are
    registered at startup based on `--no-shell` / `--no-plan` and
    whether the session has a memory store. To toggle individual
    tools at runtime, edit cli/app.py::rebuild_tools (no slash
    command for it yet).
    """
    specs = app.tools.to_tool_specs() if app.tools else []
    app.console.print(tools_table(specs))
    return False


def cmd_skills(app: Any, args: list[str]) -> bool:
    """Inspect / toggle the skill library.

    /skills                show count + on/off state
    /skills on|off         toggle injection + harvest
    /skills show [N]       list the most recent N skills (default 10)
    /skills clear          drop ALL skills from the library (does not
                            touch other memory entries)
    """
    lib = app.session.skill_library
    if lib is None:
        app.console.print("[red]no skill library on this session[/red]")
        return False

    sub = args[0].lower() if args else "status"

    if sub == "status":
        n = len(lib)
        state = "on" if app.skills_enabled else "off"
        app.console.print(
            f"skills: [bold]{state}[/bold]   library has [bold]{n}[/bold] skill(s)"
        )
        return False

    if sub in ("on", "true", "enable"):
        app.skills_enabled = True
        app.console.print(f"skills → [bold]on[/bold]   library has {len(lib)}")
        return False

    if sub in ("off", "false", "disable"):
        app.skills_enabled = False
        app.console.print("skills → [bold]off[/bold]")
        return False

    if sub == "show":
        n_show = int(args[1]) if len(args) > 1 else 10
        skills = lib.all()
        if not skills:
            app.console.print("[dim]no skills yet[/dim]")
            return False
        for s in skills[-n_show:]:
            app.console.print(
                f"  [bold]{s.name}[/bold]  [dim]{s.signature}[/dim]"
            )
            if s.source_task_id:
                app.console.print(
                    f"    [dim]from task {s.source_task_id}  "
                    f"verifier={s.verifier_name}[/dim]"
                )
        return False

    if sub == "clear":
        # Drop only kind="skill" entries; leave other memory entries.
        store = app.session.memory_store
        skill_ids = [e.id for e in store.all() if e.kind == "skill"]
        for sid in skill_ids:
            try:
                store.delete(sid)
            except Exception:
                pass
        app.console.print(
            f"[dim]cleared {len(skill_ids)} skill(s) from the library[/dim]"
        )
        return False

    app.console.print(f"[red]unknown subcommand:[/red] /skills {sub}")
    return False


def cmd_status(app: Any, args: list[str]) -> bool:
    """Summary of the current session: model, policy, budget, toggles, cost.

    Usage:
      /status              show everything in one panel-free dump
    """
    from ..llm.pricing import estimate_cost

    # ---- totals from session turns ------------------------------------
    turns = app.session.turns
    total_in = sum(t.tokens_in for t in turns)
    total_out = sum(t.tokens_out for t in turns)
    total_cost = 0.0
    unknown_models: set[str] = set()
    for t in turns:
        c, known = estimate_cost(t.provider, t.model, t.tokens_in, t.tokens_out)
        if known:
            total_cost += c
        else:
            unknown_models.add(f"{t.provider}/{t.model}")

    # ---- toggles / settings -------------------------------------------
    def _bool(v: bool) -> str:
        return "[scout.ok]on[/scout.ok]" if v else "[scout.muted]off[/scout.muted]"

    tok_disp = "unlimited" if app.budget_tokens is None else f"{app.budget_tokens:,}"
    tool_names = ([s.name for s in app.tools.to_tool_specs()]
                  if app.tools else [])
    skills_enabled = bool(getattr(app, "skills_enabled", False))
    compact_enabled = bool(getattr(app, "compact_enabled", False))
    memory_path = getattr(app.session, "memory_path", None)
    memory_n = (len(app.session.memory_store.all())
                if app.session.memory_store is not None else 0)

    # ---- print --------------------------------------------------------
    p = app.console.print
    p()
    p("[scout.agent]● banna · status[/scout.agent]")
    p(f"  [scout.muted]provider [/scout.muted][scout.text]{app.provider}[/scout.text]"
      f"   [scout.muted]model [/scout.muted][scout.text]{app.model or '(provider default)'}[/scout.text]"
      f"   [scout.muted]policy [/scout.muted][scout.text]{app.policy_name}[/scout.text]")
    p(f"  [scout.muted]temperature [/scout.muted][scout.text]{app.temperature:.2f}[/scout.text]"
      f"   [scout.muted]n_candidates [/scout.muted][scout.text]{app.n_candidates}[/scout.text]")
    p(f"  [scout.muted]budget [/scout.muted]"
      f"[scout.text]{app.budget_steps}[/scout.text][scout.muted] steps · [/scout.muted]"
      f"[scout.text]{app.budget_wall_s:.0f}s[/scout.text][scout.muted] wall · [/scout.muted]"
      f"[scout.text]{tok_disp}[/scout.text][scout.muted] tokens[/scout.muted]")
    p(f"  [scout.muted]toggles [/scout.muted]"
      f"skills={_bool(skills_enabled)}   "
      f"compact={_bool(compact_enabled)}")
    if compact_enabled:
        p(f"    [scout.muted]compact threshold={app.compact_threshold_tokens} tok   "
          f"keep_last_n={app.compact_keep_last_n}[/scout.muted]")
    p(f"  [scout.muted]tools [/scout.muted]"
      f"[scout.text]{', '.join(tool_names) if tool_names else '(none)'}[/scout.text]")
    p(f"  [scout.muted]memory [/scout.muted]"
      f"[scout.text]{memory_n}[/scout.text][scout.muted] entries · "
      f"{memory_path or '(in-memory only)'}[/scout.muted]")
    p()
    p(f"  [scout.muted]session [/scout.muted]"
      f"[scout.text]{len(turns)}[/scout.text][scout.muted] turn(s) · [/scout.muted]"
      f"[scout.text]{total_in:,}[/scout.text][scout.muted] in · [/scout.muted]"
      f"[scout.text]{total_out:,}[/scout.text][scout.muted] out tokens[/scout.muted]")
    p(f"  [scout.muted]cost (est.) [/scout.muted]"
      f"[scout.ok]${total_cost:.4f}[/scout.ok]")
    if unknown_models:
        p(f"  [scout.warn]unknown pricing for:[/scout.warn] "
          f"[scout.muted]{', '.join(sorted(unknown_models))}[/scout.muted]")
    p()
    return False


def cmd_cost(app: Any, args: list[str]) -> bool:
    """Show estimated $ cost + token usage for this session.

    Usage:
      /cost                session totals (in/out tokens + estimated $)
      /cost detailed       per-turn breakdown table
      /cost rates          dump the pricing table being used

    Cost is computed from each Turn's recorded provider, model, and
    token counts using ``llm/pricing.py``. The numbers are estimates,
    not bill-accurate — provider list prices change, cached-input
    discounts and batch-API rebates aren't accounted for, and Bedrock
    pricing varies by region and provisioned throughput.

    Override rates by setting MYAGENT_PRICES (JSON dict) before launch:
      MYAGENT_PRICES='{"openai/gpt-5-nano":[0.05,0.40]}' myAgent
    """
    from rich.table import Table
    from ..llm.pricing import all_prices, estimate_cost

    sub = (args[0] if args else "").lower()

    # ----- /cost rates ---------------------------------------------------
    if sub == "rates":
        prices = all_prices()
        t = Table(title="pricing (USD per 1M tokens)",
                  title_style="cyan", show_lines=False)
        t.add_column("provider/model", style="bold")
        t.add_column("input $/M", justify="right")
        t.add_column("output $/M", justify="right")
        for k in sorted(prices):
            inp, out = prices[k]
            t.add_row(k, f"{inp:.3f}", f"{out:.3f}")
        app.console.print(t)
        app.console.print(
            "[dim]Override with the MYAGENT_PRICES env var (JSON dict). "
            "Verify against the provider's pricing page before relying on "
            "these numbers.[/dim]"
        )
        return False

    turns = app.session.turns
    if not turns:
        app.console.print(
            "[dim]no turns yet — run a question first, then /cost again.[/dim]"
        )
        return False

    # ----- /cost detailed ------------------------------------------------
    detailed = sub in ("detailed", "verbose", "per-turn")
    total_in = 0
    total_out = 0
    total_cost = 0.0
    unknown_models: set[str] = set()

    if detailed:
        t = Table(title="per-turn cost", title_style="cyan", show_lines=False)
        t.add_column("#", style="dim", justify="right")
        t.add_column("model", style="bold")
        t.add_column("policy", style="dim")
        t.add_column("in tok", justify="right")
        t.add_column("out tok", justify="right")
        t.add_column("$ est.", justify="right")
        for i, turn in enumerate(turns):
            cost, known = estimate_cost(
                turn.provider, turn.model, turn.tokens_in, turn.tokens_out,
            )
            total_in += turn.tokens_in
            total_out += turn.tokens_out
            if known:
                total_cost += cost
                cost_s = f"${cost:.4f}"
            else:
                unknown_models.add(f"{turn.provider}/{turn.model}")
                cost_s = "[yellow]?[/yellow]"
            t.add_row(
                str(i),
                f"{turn.provider}/{turn.model or '(default)'}",
                turn.policy,
                f"{turn.tokens_in:,}",
                f"{turn.tokens_out:,}",
                cost_s,
            )
        app.console.print(t)
    else:
        for turn in turns:
            cost, known = estimate_cost(
                turn.provider, turn.model, turn.tokens_in, turn.tokens_out,
            )
            total_in += turn.tokens_in
            total_out += turn.tokens_out
            if known:
                total_cost += cost
            else:
                unknown_models.add(f"{turn.provider}/{turn.model}")

    # ----- summary line --------------------------------------------------
    app.console.print()
    app.console.print(
        f"[bold]session total:[/bold] "
        f"{total_in:,} in · {total_out:,} out tokens "
        f"({total_in + total_out:,} combined)"
    )
    app.console.print(
        f"  estimated cost: [bold green]${total_cost:.4f}[/bold green] "
        f"[dim]across {len(turns)} turn(s)[/dim]"
    )
    if unknown_models:
        app.console.print(
            f"  [yellow]unknown pricing for:[/yellow] "
            f"{', '.join(sorted(unknown_models))}"
        )
        app.console.print(
            "  [dim]add an entry in llm/pricing.py or set "
            "MYAGENT_PRICES env to fix.[/dim]"
        )
    app.console.print(
        "[dim]rates approximate; not bill-accurate. /cost rates to inspect.[/dim]"
    )
    return False


def cmd_keys(app: Any, args: list[str]) -> bool:
    """Show which provider API keys are set in the environment.

    Usage:
      /keys

    Shows a table: provider | env var | status (set / unset) | masked
    preview (first 4 chars + ****). Never prints the full key value, so
    the output is safe to paste into bug reports / chat.

    To set a missing key:
      • Edit ~/.bashrc with:  export OPENAI_API_KEY=sk-...
      • Or copy .env.example to .env in the project root and edit it;
        myAgent auto-loads .env on startup.
    """
    import os as _os
    from rich.table import Table

    rows = [
        ("anthropic",  "ANTHROPIC_API_KEY",  "Claude direct"),
        ("openai",     "OPENAI_API_KEY",     "GPT models"),
        ("gemini",     "GOOGLE_API_KEY",     "Gemini + Google search"),
        ("bedrock",    "AWS_PROFILE",        "(also reads AWS_REGION)"),
        ("ollama",     "OLLAMA_BASE_URL",    "(no key needed; URL only)"),
        ("tavily",     "TAVILY_API_KEY",     "(only if cascade includes tavily)"),
        ("yacy",       "YACY_BASE_URL",      "(no key; URL only)"),
    ]

    t = Table(title="provider keys", title_style="cyan", show_lines=False)
    t.add_column("provider", style="bold")
    t.add_column("env var")
    t.add_column("status")
    t.add_column("preview")
    t.add_column("notes", style="dim")

    def _mask(v: str) -> str:
        if not v:
            return ""
        if len(v) <= 8:
            return "****"
        return f"{v[:4]}****{v[-2:]}"

    for prov, env_var, note in rows:
        v = _os.environ.get(env_var, "")
        if v:
            status = "[green]set[/green]"
            preview = _mask(v)
        else:
            status = "[red]unset[/red]"
            preview = "—"
        t.add_row(prov, env_var, status, preview, note)
    app.console.print(t)
    app.console.print(
        "[dim]To set a key: edit ~/.bashrc with `export <VAR>=<value>`, or "
        "copy .env.example to .env and edit; myAgent auto-loads .env at startup.[/dim]"
    )
    return False


def cmd_compact(app: Any, args: list[str]) -> bool:
    """Toggle trace compaction.

    /compact            show current state
    /compact on|off     toggle
    /compact threshold=N keep=N   tune the trigger
    """
    if not args:
        state = "on" if app.compact_enabled else "off"
        app.console.print(
            f"compaction: [bold]{state}[/bold]   "
            f"threshold=[bold]{app.compact_threshold_tokens}[/bold] tokens   "
            f"keep_last_n=[bold]{app.compact_keep_last_n}[/bold]"
        )
        return False
    for tok in args:
        t = tok.lower()
        if t in ("on", "true", "enable"):
            app.compact_enabled = True
        elif t in ("off", "false", "disable"):
            app.compact_enabled = False
        elif "=" in t:
            k, v = t.split("=", 1)
            try:
                if k in ("threshold", "threshold_tokens"):
                    app.compact_threshold_tokens = int(v)
                elif k in ("keep", "keep_last", "keep_last_n"):
                    app.compact_keep_last_n = int(v)
                else:
                    app.console.print(f"[yellow]unknown key:[/yellow] {k}")
            except ValueError:
                app.console.print(f"[red]bad value for {k}:[/red] {v}")
        else:
            app.console.print(f"[red]unknown arg:[/red] {tok}")
    state = "on" if app.compact_enabled else "off"
    app.console.print(
        f"compaction → [bold]{state}[/bold]   "
        f"threshold={app.compact_threshold_tokens}   "
        f"keep_last_n={app.compact_keep_last_n}"
    )
    return False


# ---------------------------------------------------------------------------
# Clear / save / load
# ---------------------------------------------------------------------------


def cmd_clear(app: Any, args: list[str]) -> bool:
    """Clear the transcript. By default keeps persistent memory.

    Use `/clear all` to also wipe the persistent memory store
    (truncates ~/.config/myagent/memory.jsonl).
    """
    wipe = bool(args) and args[0].lower() in ("all", "memory", "everything")
    app.session.clear(wipe_memory=wipe)
    app.console.clear()
    app.print_header()
    if wipe:
        app.console.print("[dim]session + persistent memory cleared.[/dim]")
    else:
        app.console.print(
            "[dim]transcript cleared. Persistent memory kept "
            "(use [bold]/clear all[/bold] to wipe it too).[/dim]"
        )
    return False


def cmd_memory(app: Any, args: list[str]) -> bool:
    """Inspect / manage the persistent memory store.

    /memory               show count + path
    /memory show          list recent entries
    /memory clear         truncate the persistent memory file
    /memory path <path>   point at a different JSONL file (for this session only)
    """
    store = app.session.memory_store
    path = app.session.memory_path
    sub = args[0].lower() if args else "status"

    if sub == "status":
        n = len(store.all() if hasattr(store, "all") else [])
        loc = str(path) if path else "(in-memory only)"
        app.console.print(
            f"memory: [bold]{n}[/bold] entries  ·  path: [dim]{loc}[/dim]"
        )
        return False

    if sub == "show":
        n_show = int(args[1]) if len(args) > 1 else 10
        entries = store.all() if hasattr(store, "all") else []
        if not entries:
            app.console.print("[dim]memory is empty[/dim]")
            return False
        for e in entries[-n_show:]:
            content = (e.content or "")[:100]
            app.console.print(
                f"  [dim]{e.id[:8]}[/dim]  [cyan]{e.kind:<8}[/cyan] {content}"
            )
        return False

    if sub == "clear":
        app.session.clear(wipe_memory=True)
        # Don't redraw header; just confirm.
        app.console.print("[dim]persistent memory cleared.[/dim]")
        return False

    if sub == "path":
        if len(args) < 2:
            app.console.print("[red]usage:[/red] /memory path <path>")
            return False
        from pathlib import Path
        from ..memory.embeddings import HashEmbedder
        from ..memory.jsonl_store import JSONLStore
        new_path = Path(args[1]).expanduser().resolve()
        new_path.parent.mkdir(parents=True, exist_ok=True)
        app.session.memory_path = new_path
        app.session.memory_store = JSONLStore(new_path, embedder=HashEmbedder(dim=256))
        # Rebuild tools so the memory tool binds to the new store.
        app.rebuild_tools()
        app.console.print(f"memory path → [bold]{new_path}[/bold]")
        return False

    app.console.print(f"[red]unknown subcommand:[/red] /memory {sub}")
    return False


def cmd_save(app: Any, args: list[str]) -> bool:
    """Write the current session transcript (Q/A pairs) to a JSONL file.

    Usage:
      /save <path>

    Writes each Turn (question, answer, wall_s, steps_used, tokens,
    policy, model, …) as one JSON line, plus a header line with
    started_at and n_turns. Does NOT include the trace itself or the
    persistent memory store — those are separate. Use /load <path>
    to restore a transcript later.
    """
    if not args:
        app.console.print("[red]usage:[/red] /save <path>")
        return False
    p = app.session.save_jsonl(args[0])
    app.console.print(f"[green]saved[/green] {len(app.session.turns)} turns → [bold]{p}[/bold]")
    return False


def cmd_load(app: Any, args: list[str]) -> bool:
    """Replace the current session transcript with one loaded from JSONL.

    Usage:
      /load <path>

    Reads a file previously written by /save. Replaces the in-memory
    Turn list; the persistent memory store is unaffected.
    """
    if not args:
        app.console.print("[red]usage:[/red] /load <path>")
        return False
    try:
        from .session import Session
        sess = Session.load_jsonl(args[0])
    except Exception as exc:
        app.console.print(f"[red]load failed:[/red] {exc}")
        return False
    app.session = sess
    app.console.print(
        f"[green]loaded[/green] {len(app.session.turns)} turns from [bold]{args[0]}[/bold]"
    )
    return False


# ---------------------------------------------------------------------------
# /show
# ---------------------------------------------------------------------------


def cmd_show(app: Any, args: list[str]) -> bool:
    """Inspect session-level data after a task runs.

    Usage:
      /show transcript           table of all Q/A turns this session
      /show last                 one-line summary per step of the last task
      /show trace                FULL per-step dump: action + observation +
                                 wall_s + tokens + truncated data + meta
      /show evidence             URLs + snippets the agent collected
      /show claims               propositions the agent made (with verdicts)

    `trace` is what you usually want for post-mortem debugging — every
    step's action text, observation data, wall time, tokens, and meta
    fields are shown.
    """
    if not args:
        app.console.print(
            "[red]usage:[/red] /show "
            "<transcript|last|trace|evidence|claims>"
        )
        return False
    topic = args[0].lower()
    if topic in ("transcript", "history"):
        if not app.session.turns:
            app.console.print("[dim]transcript empty[/dim]")
            return False
        app.console.print(turns_table(app.session.turns))
        return False

    if topic == "last":
        st = app.session.last_state
        if st is None:
            app.console.print("[dim]no last task yet[/dim]")
            return False
        app.console.print(f"[bold]Question:[/bold] {st.question}")
        for s in st.trace.steps:
            app.console.print(f"  step {s.idx} {s.action.kind.value}: "
                              f"{s.action.tool_name or ''} "
                              f"{(s.action.text or s.action.answer or '')[:100]}")
        if st.trace.final_answer:
            app.console.print(f"[bold green]→ {st.trace.final_answer}[/bold green]")
        return False

    if topic in ("trace", "full"):
        st = app.session.last_state
        if st is None:
            app.console.print("[dim]no last task yet[/dim]")
            return False
        import json as _json
        from ..core.types import ActionKind as _AK
        app.console.print(f"[bold]Question:[/bold] {st.question}")
        app.console.print(f"[dim]run_id: {st.trace.run_id}   started: "
                          f"{st.trace.started_at}   "
                          f"steps: {len(st.trace.steps)}   "
                          f"evidence: {len(st.evidence)}   "
                          f"claims: {len(st.claims)}[/dim]\n")
        for s in st.trace.steps:
            a = s.action
            o = s.observation
            head = f"[bold cyan]step {s.idx}[/bold cyan] [bold]{a.kind.value}[/bold]"
            if a.kind == _AK.TOOL_CALL:
                args_s = _json.dumps(a.tool_args, default=str)
                if len(args_s) > 200:
                    args_s = args_s[:197] + "…"
                app.console.print(f"{head}  [yellow]{a.tool_name}[/yellow]({args_s})")
            elif a.kind == _AK.THINK:
                txt = (a.text or "").strip()
                if len(txt) > 400:
                    txt = txt[:397] + "…"
                app.console.print(f"{head}  [dim italic]{txt}[/dim italic]")
            elif a.kind == _AK.FINAL_ANSWER:
                ans = (a.answer or "").strip()
                if len(ans) > 400:
                    ans = ans[:397] + "…"
                app.console.print(f"{head}  [green]{ans}[/green]")
            # Observation summary
            obs_bits: list[str] = []
            obs_bits.append("ok" if o.ok else f"err:{o.error or '?'}")
            if o.wall_s:
                obs_bits.append(f"{o.wall_s:.2f}s")
            if o.tokens_in or o.tokens_out:
                obs_bits.append(f"{o.tokens_in}→{o.tokens_out} tok")
            if o.data:
                d = _json.dumps(o.data, default=str)
                if len(d) > 240:
                    d = d[:237] + "…"
                obs_bits.append(d)
            elif o.text and a.kind == _AK.THINK:
                # avoid printing the same THINK text twice
                pass
            elif o.text:
                t = o.text.strip().replace("\n", " ")
                if len(t) > 240:
                    t = t[:237] + "…"
                obs_bits.append(t)
            app.console.print(f"  [dim]obs:[/dim] {'  '.join(obs_bits)}")
            # Action.meta highlights
            interesting = {
                k: v for k, v in (a.meta or {}).items()
                if k in ("policy", "plan_step", "phase", "winner_id",
                         "static_argmax_id", "evicted", "verifier_passed",
                         "verifier_retries", "branch", "tokens_in", "tokens_out")
            }
            if interesting:
                meta_s = "  ".join(f"[dim]{k}=[/dim]{v}" for k, v in interesting.items())
                app.console.print(f"  [dim]meta:[/dim] {meta_s}")
            app.console.print()
        if st.trace.final_answer:
            app.console.print(f"[bold green]→ {st.trace.final_answer}[/bold green]")
        return False

    if topic == "evidence":
        st = app.session.last_state
        if st is None or not st.evidence:
            app.console.print("[dim]no evidence collected[/dim]")
            return False
        for ev in st.evidence:
            app.console.print(
                f"  [dim]{ev.evidence_id}[/dim] {ev.source[:80]}\n"
                f"    {ev.content[:160]}"
            )
        return False

    if topic == "claims":
        st = app.session.last_state
        if st is None or not st.claims:
            app.console.print("[dim]no claims[/dim]")
            return False
        for cl in st.claims:
            mark = "[green]●[/green]" if cl.supports else "[yellow]○[/yellow]"
            app.console.print(f"  {mark} {cl.text[:120]}")
        return False


    app.console.print(f"[red]unknown topic:[/red] {topic}")
    return False


# ---------------------------------------------------------------------------
# Exit
# ---------------------------------------------------------------------------


def cmd_exit(app: Any, args: list[str]) -> bool:
    """Leave the REPL.

    Usage:
      /exit
      /quit

    Ctrl-D at the prompt also exits cleanly.
    """
    app.console.print("[dim]goodbye.[/dim]")
    return True


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


COMMANDS: dict[str, Callable[[Any, list[str]], bool]] = {
    "help": cmd_help,
    "?": cmd_help,
    "model": cmd_model,
    "provider": cmd_provider,
    "policy": cmd_policy,
    "temperature": cmd_temperature,
    "temp": cmd_temperature,
    "budget": cmd_budget,
    "tools": cmd_tools,
    "keys": cmd_keys,
    "status": cmd_status,
    "cost": cmd_cost,
    "compact": cmd_compact,
    "skills": cmd_skills,
    "clear": cmd_clear,
    "memory": cmd_memory,
    "save": cmd_save,
    "load": cmd_load,
    "show": cmd_show,
    "exit": cmd_exit,
    "quit": cmd_exit,
}


def is_command(line: str) -> bool:
    return line.strip().startswith("/")


def dispatch(app: Any, line: str) -> bool:
    """Parse a `/command args...` line and run the handler.

    Returns True if the REPL should exit.
    """
    rest = line.strip().lstrip("/")
    if not rest:
        return False
    try:
        tokens = shlex.split(rest)
    except ValueError:
        tokens = rest.split()
    if not tokens:
        return False
    name = tokens[0].lower()
    args = tokens[1:]
    handler = COMMANDS.get(name)
    if handler is None:
        app.console.print(
            f"[red]unknown command:[/red] /{name}   (try /help)"
        )
        return False
    return handler(app, args)
