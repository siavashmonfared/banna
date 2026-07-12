"""Launch-time session-setup TUI (Textual).

Shown at every `banna` start (TTY only; `--no-setup` skips). A single
dashboard pre-filled from config.toml — one keypress (`s`) accepts it,
so returning users lose ~nothing. Rows:

    provider · model · policy · budget · theme · sandbox
    · temperature · skills · n_candidates

Interaction model (confirmed design):
    ↑/↓   move        ←/→   toggle the row's value in place
    ↵     open the row's picker (typed entry for numbers)
    s     start       d     save as default & start        q  quit

Missing API keys are detected live (shell env / ./.env /
~/.config/banna/.env), and the key screen shows the exact file path +
`VAR=value` format, validates with a 1-token call, or hands off to
Ollama for a local/open-weight model instead.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Footer, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from .model_catalog import (
    CURATED,
    KEY_VARS,
    PROVIDER_LABELS,
    PROVIDER_ORDER,
    ProviderStatus,
    key_search_paths,
    refresh_ollama,
    save_api_key,
    scan_providers,
    validate_api_key,
)
from .themes import DEFAULT_THEME, THEME_BLURBS, get_palette, list_themes

# One-line blurbs for every policy any banna build might expose. The
# *actual* offering is whatever this build's commands.POLICY_NAMES says
# — the public repo gates to a subset, and offering a gated name here
# would crash _build_policy at startup.
_ALL_POLICY_BLURBS: dict[str, str] = {
    "react": "plain tool loop",
    "react+": "react + evidence recall & coverage pressure",
    "planner_react": "plan first, then act",
    "bfs_over_plans": "breadth-first search over candidate plans",
    "dfs_over_plans": "depth-first search over candidate plans",
    "best_first_over_plans": "scored best-first search over plans",
    "verifier_retry": "react wrapped in verifier-gated retries",
    "best_of_n": "n independent runs, majority vote",
    "banna_thinking": "field-based deliberation (experimental)",
}

from .commands import POLICY_NAMES  # noqa: E402  (single source of truth)

POLICY_BLURBS: tuple[tuple[str, str], ...] = tuple(
    (name, _ALL_POLICY_BLURBS.get(name, "")) for name in POLICY_NAMES
)

SANDBOXES = ("process", "docker")

# n_candidates only matters to the plan-search / sampling policies. When
# none of them are exposed in this build (the public CLI), the dashboard
# hides the row entirely.
_N_CANDIDATE_POLICIES = frozenset({
    "bfs_over_plans", "dfs_over_plans", "best_first_over_plans",
    "best_of_n", "banna_thinking",
})
SHOW_N_CANDIDATES = bool(_N_CANDIDATE_POLICIES & set(POLICY_NAMES))


# ---------------------------------------------------------------------------
# Values in/out
# ---------------------------------------------------------------------------


@dataclass
class SetupValues:
    """Everything the dashboard edits. Seeded from config.toml + flags."""
    provider: str = "openai"
    model: str = "gpt-5-nano"
    policy: str = "react"
    budget_steps: int = 15
    budget_wall_s: float = 300.0
    budget_tokens: int | None = None      # None = unlimited
    budget_cost_usd: float | None = 5.0   # None = unlimited
    theme: str = DEFAULT_THEME
    sandbox: str = "process"
    temperature: float = 0.7
    skills: bool = False
    n_candidates: int = 3


@dataclass
class SetupResult:
    values: SetupValues
    save_default: bool = False


def _budget_display(v: SetupValues) -> str:
    tokens = "∞ tokens" if v.budget_tokens is None else f"{v.budget_tokens:,} tokens"
    cost = "∞ cost" if v.budget_cost_usd is None else f"${v.budget_cost_usd:.2f}"
    return f"{v.budget_steps} steps · {v.budget_wall_s:g}s · {cost} · {tokens}"


def _provider_models(provider: str, scan: dict[str, ProviderStatus]) -> list[str]:
    """Cycle/pick list for the model row: curated ids, or live Ollama tags."""
    if provider == "ollama":
        st = scan.get("ollama")
        if st is None or not st.ollama_models:
            # The startup scan may predate `ollama serve` — re-probe so a
            # daemon started after banna is still picked up.
            st = refresh_ollama(scan)
        return [m.get("name", "?") for m in st.ollama_models]
    return [m for m, _ in CURATED.get(provider, ())]


def _cycle(options: list[str], current: str, delta: int) -> str:
    if not options:
        return current
    try:
        i = options.index(current)
    except ValueError:
        return options[0]
    return options[(i + delta) % len(options)]


# ---------------------------------------------------------------------------
# Modal screens
# ---------------------------------------------------------------------------


class BannaModal(ModalScreen):
    """Shared chrome: esc dismisses, palette-tinted panel."""

    BINDINGS: ClassVar = [Binding("escape", "dismiss_none", "back", show=True)]

    DEFAULT_CSS = """
    BannaModal {
        align: center middle;
    }
    BannaModal > Vertical {
        width: 76; max-height: 80%; height: auto;
        border: round $secondary;
        padding: 1 2;
    }
    BannaModal .title { text-style: bold; margin-bottom: 1; }
    BannaModal .hint  { color: $text-muted; margin-top: 1; }
    """

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class ProviderScreen(BannaModal):
    """Pick a provider; every row shows live key/server status + source."""

    def __init__(self, scan: dict[str, ProviderStatus], current: str) -> None:
        super().__init__()
        self._scan = scan
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("provider", classes="title")
            ol = OptionList(id="providers")
            yield ol
            yield Label("↑↓ move · ↵ select · esc back", classes="hint")

    def on_mount(self) -> None:
        ol = self.query_one("#providers", OptionList)
        st_o = self._scan.get("ollama")
        if st_o is None or not st_o.ok:
            # Startup scan may predate `ollama serve`; a dead daemon
            # answers instantly (connection refused), so this is cheap.
            refresh_ollama(self._scan)
        for i, name in enumerate(PROVIDER_ORDER):
            st = self._scan[name]
            t = Text()
            t.append(f"{PROVIDER_LABELS[name]:<18}", "bold")
            if st.ok:
                t.append("✓ ", "green")
                t.append(st.detail)
                if st.source:
                    t.append(f"   ({st.source})", "dim")
            else:
                t.append("✗ ", "red")
                t.append(st.detail, "dim")
            ol.add_option(Option(t, id=name))
            if name == self._current:
                ol.highlighted = i

    def on_option_list_option_selected(self, ev: OptionList.OptionSelected) -> None:
        name = str(ev.option_id)
        st = self._scan[name]
        if st.ok:
            self.dismiss(name)
        elif name in KEY_VARS:
            # Cloud provider without a key → key entry sub-screen.
            def _after(outcome: str | None) -> None:
                if outcome == "saved":
                    self.dismiss(name)
                elif outcome == "ollama":
                    self.dismiss("ollama")
            self.app.push_screen(KeyScreen(name, self._scan), _after)
        elif name == "ollama":
            if refresh_ollama(self._scan).ok:
                self.dismiss("ollama")
                return
            self.app.push_screen(InfoScreen(
                "ollama — not running",
                "No server at localhost:11434.\n\n"
                "  1. install: https://ollama.com/\n"
                "  2. run:     ollama serve\n"
                "  3. pull a tools-capable model, e.g.\n"
                "              ollama pull qwen3\n\n"
                "then restart banna (or re-open this picker)."))
        else:  # bedrock
            self.app.push_screen(InfoScreen(
                "bedrock — AWS credentials not found",
                "Bedrock authenticates through the boto3 chain.\n\n"
                "Set in your shell (or ~/.config/banna/.env):\n"
                "  AWS_REGION=us-east-1        (or AWS_DEFAULT_REGION)\n"
                "  AWS_PROFILE=your-profile    (or key pair via\n"
                "  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)\n\n"
                "then restart banna."))


class KeyScreen(BannaModal):
    """Paste-and-validate an API key; shows exactly where it will live.

    Dismisses with "saved", "ollama", or None.
    """

    def __init__(self, provider: str, scan: dict[str, ProviderStatus]) -> None:
        super().__init__()
        self._provider = provider
        self._scan = scan

    def compose(self) -> ComposeResult:
        vars_ = KEY_VARS[self._provider]
        searched = "\n".join(
            f"    {i}. {label:<28}{' / '.join(vars_) if p is None else ''}"
            for i, (label, p) in enumerate(key_search_paths(), start=1)
        )
        cfg_env = key_search_paths()[-1][0]
        with Vertical():
            yield Label(f"{PROVIDER_LABELS[self._provider]} — no API key found",
                        classes="title")
            yield Static(f"  searched, in order:\n{searched}\n")
            yield Static(
                f"  paste a key below → validated with a 1-token call,\n"
                f"  then saved to {cfg_env} (mode 0600)\n"
                f"  format:  {vars_[0]}=sk-...\n")
            yield Input(password=True, placeholder=f"{vars_[0]} value…", id="key")
            yield Static("", id="status")
            with Horizontal():
                yield Button("validate & save", id="save", variant="success")
                yield Button("use ollama instead", id="ollama")
                yield Button("back", id="back")

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "back":
            self.dismiss(None)
        elif ev.button.id == "ollama":
            st = self._scan.get("ollama")
            if st and st.ok:
                self.dismiss("ollama")
            else:
                self.query_one("#status", Static).update(
                    Text("✗ ollama isn't running either — install from "
                         "https://ollama.com/ and `ollama serve`", "red"))
        elif ev.button.id == "save":
            self._submit_key()

    def _submit_key(self) -> None:
        key = self.query_one("#key", Input).value.strip()
        if not key:
            self.query_one("#status", Static).update(Text("✗ empty key", "red"))
            return
        self.query_one("#status", Static).update(
            Text("… validating key with a 1-token test call", "yellow"))
        self._validate(key)

    def on_input_submitted(self, ev: Input.Submitted) -> None:
        self._submit_key()

    @work(thread=True, exclusive=True)
    def _validate(self, key: str) -> None:
        ok, err = validate_api_key(self._provider, key)
        if ok:
            path = save_api_key(self._provider, key)
            self.app.call_from_thread(self._on_saved, str(path))
        else:
            self.app.call_from_thread(self._on_failed, err)

    def _on_saved(self, path: str) -> None:
        # Refresh the shared scan in place so the dashboard sees the key.
        self._scan.update(scan_providers(ollama_timeout_s=0.3))
        self.dismiss("saved")

    def _on_failed(self, err: str) -> None:
        self.query_one("#status", Static).update(Text(f"✗ rejected: {err}", "red"))


class InfoScreen(BannaModal):
    """Read-only notice with a back button."""

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title, self._body = title, body

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title, classes="title")
            yield Static(self._body)
            yield Label("esc back", classes="hint")


class ModelScreen(BannaModal):
    """Curated model shortlist + free-form entry. Ollama lists live tags
    and probes tool-calling support before accepting."""

    def __init__(self, provider: str, scan: dict[str, ProviderStatus],
                 current: str) -> None:
        super().__init__()
        self._provider = provider
        self._scan = scan
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"model — {PROVIDER_LABELS[self._provider]}", classes="title")
            yield OptionList(id="models")
            yield Input(placeholder="custom model id…", id="custom", classes="hidden")
            yield Static("", id="status")
            yield Label("↑↓ move · ↵ select · esc back", classes="hint")

    def on_mount(self) -> None:
        ol = self.query_one("#models", OptionList)
        self.query_one("#custom", Input).display = False
        ids: list[str] = []
        if self._provider == "ollama":
            st = self._scan.get("ollama")
            if st is None or not st.ollama_models:
                st = refresh_ollama(self._scan)
            for m in st.ollama_models:
                name = m.get("name", "?")
                size = m.get("size", 0)
                blurb = f"{size / 1e9:.1f} GB" if size else ""
                ol.add_option(Option(self._row(name, blurb), id=name))
                ids.append(name)
        else:
            for name, blurb in CURATED.get(self._provider, ()):
                ol.add_option(Option(self._row(name, blurb), id=name))
                ids.append(name)
        ol.add_option(Option(self._row("other…", "type any model id"), id="__other__"))
        if self._current in ids:
            ol.highlighted = ids.index(self._current)

    @staticmethod
    def _row(name: str, blurb: str) -> Text:
        t = Text()
        t.append(f"{name:<44}", "bold")
        if blurb:
            t.append(blurb, "dim")
        return t

    def on_option_list_option_selected(self, ev: OptionList.OptionSelected) -> None:
        model = str(ev.option_id)
        if model == "__other__":
            inp = self.query_one("#custom", Input)
            inp.display = True
            inp.focus()
            return
        if self._provider == "ollama":
            self.query_one("#status", Static).update(
                Text(f"… probing {model} for tool-calling support", "yellow"))
            self._probe(model)
        else:
            self.dismiss(model)

    def on_input_submitted(self, ev: Input.Submitted) -> None:
        model = ev.value.strip()
        if model:
            self.dismiss(model)

    @work(thread=True, exclusive=True)
    def _probe(self, model: str) -> None:
        from .setup_wizard import _probe_ollama_tool_support
        ok, reason = _probe_ollama_tool_support(model)
        if ok:
            self.app.call_from_thread(self.dismiss, model)
        else:
            self.app.call_from_thread(
                self.query_one("#status", Static).update,
                Text(f"✗ {model} {reason} — banna's loop needs tool calls; "
                     f"pick e.g. qwen3 / qwen3-coder / gpt-oss", "red"))


class PolicyScreen(BannaModal):
    def __init__(self, current: str) -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("policy", classes="title")
            yield OptionList(id="policies")
            yield Label("↑↓ move · ↵ select · esc back", classes="hint")

    def on_mount(self) -> None:
        ol = self.query_one("#policies", OptionList)
        for i, (name, blurb) in enumerate(POLICY_BLURBS):
            t = Text()
            t.append(f"{name:<24}", "bold")
            t.append(blurb, "dim")
            ol.add_option(Option(t, id=name))
            if name == self._current:
                ol.highlighted = i

    def on_option_list_option_selected(self, ev: OptionList.OptionSelected) -> None:
        self.dismiss(str(ev.option_id))


class ThemeScreen(BannaModal):
    """Theme picker with a live glyph preview of the highlighted palette."""

    def __init__(self, current: str) -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("theme", classes="title")
            yield OptionList(id="themes")
            yield Static("", id="preview")
            yield Label("↑↓ move (preview updates) · ↵ select · esc back",
                        classes="hint")

    def on_mount(self) -> None:
        ol = self.query_one("#themes", OptionList)
        for i, name in enumerate(list_themes()):
            t = Text()
            t.append(f"{name:<18}", "bold")
            if name != "mono":
                p = get_palette(name)
                for slot in ("green", "orange", "blue", "violet", "yellow", "red"):
                    t.append("██", p[slot])
                t.append("  ")
            t.append(THEME_BLURBS.get(name, ""), "dim")
            ol.add_option(Option(t, id=name))
            if name == self._current:
                ol.highlighted = i
        self._preview(self._current)

    def _preview(self, name: str) -> None:
        if name == "mono":
            self.query_one("#preview", Static).update(
                Text("\n  ❯ you   ● banna   ✓ ok   ✗ err   ▾ reasoning\n"))
            return
        p = get_palette(name)
        t = Text("\n  ")
        t.append("❯ you", f"bold {p['orange']}")
        t.append("   ")
        t.append("● banna", f"bold {p['green']}")
        t.append("   ")
        t.append("✓ ok", p["green"])
        t.append("   ")
        t.append("✗ err", p["red"])
        t.append("   ")
        t.append("▾ reasoning", p["violet"])
        t.append("\n")
        self.query_one("#preview", Static).update(t)

    def on_option_list_option_highlighted(self, ev: OptionList.OptionHighlighted) -> None:
        if ev.option_id:
            self._preview(str(ev.option_id))

    def on_option_list_option_selected(self, ev: OptionList.OptionSelected) -> None:
        self.dismiss(str(ev.option_id))


class StepInput(Input):
    """Numeric Input where ←/→ step the value in place (matching the
    dashboard's ←/→-to-toggle convention). ↑/↓ are deliberately left
    unhandled so the enclosing screen can use them to move focus
    between fields.

    Blank is meaningful for the unlimited budget fields: stepping up
    from blank starts at `minimum` (or one step); stepping below the
    minimum on an `allow_blank` field returns to blank (= unlimited).
    """

    def __init__(self, value: str = "", *, step: float = 1.0,
                 minimum: float | None = None, maximum: float | None = None,
                 integer: bool = False, allow_blank: bool = False,
                 **kwargs: Any) -> None:
        kwargs.setdefault("type", "integer" if integer else "number")
        super().__init__(value, **kwargs)
        self._step = step
        self._min = minimum
        self._max = maximum
        self._integer = integer
        self._allow_blank = allow_blank

    def _fmt(self, x: float) -> str:
        return str(round(x)) if self._integer else f"{x:g}"

    def _bump(self, sign: int) -> None:
        raw = self.value.strip()
        try:
            cur: float | None = float(raw) if raw else None
        except ValueError:
            cur = None
        if cur is None:
            if sign < 0:
                return  # blank (= unlimited) stays blank going down
            start = self._min if self._min is not None else self._step
            self.value = self._fmt(start)
        else:
            new = cur + sign * self._step
            floor = self._min if self._min is not None else 0.0
            if new < floor:
                # Below the floor: unlimited-capable fields clear back
                # to blank, the rest clamp.
                self.value = "" if self._allow_blank else self._fmt(floor)
            elif self._max is not None and new > self._max:
                self.value = self._fmt(self._max)
            else:
                self.value = self._fmt(new)
        self.cursor_position = len(self.value)

    def key_right(self, event) -> None:
        event.stop()
        self._bump(+1)

    def key_left(self, event) -> None:
        event.stop()
        self._bump(-1)


class BudgetScreen(BannaModal):
    """Numeric entry for the four budget dimensions — ↑/↓ moves between
    fields, ←/→ steps the focused value, or type a number."""

    def key_up(self, event) -> None:
        event.stop()
        self.focus_previous()

    def key_down(self, event) -> None:
        event.stop()
        self.focus_next()

    DEFAULT_CSS = """
    BudgetScreen .brow { height: 3; }
    BudgetScreen .brow Label { width: 24; padding-top: 1; }
    BudgetScreen .brow Input { width: 1fr; }
    """

    def __init__(self, v: SetupValues) -> None:
        super().__init__()
        self._v = v

    def compose(self) -> ComposeResult:
        v = self._v
        with Vertical():
            yield Label("budget — per task", classes="title")
            with Horizontal(classes="brow"):
                yield Label("max steps")
                yield StepInput(str(v.budget_steps), integer=True,
                                step=1, minimum=1, id="steps")
            with Horizontal(classes="brow"):
                yield Label("max wall time (s)")
                yield StepInput(f"{v.budget_wall_s:g}",
                                step=10, minimum=1, id="wall")
            with Horizontal(classes="brow"):
                yield Label("max tokens (blank = ∞)")
                yield StepInput("" if v.budget_tokens is None else str(v.budget_tokens),
                                integer=True, step=10_000, minimum=10_000,
                                allow_blank=True, id="tokens")
            with Horizontal(classes="brow"):
                yield Label("max cost USD (blank = ∞)")
                yield StepInput("" if v.budget_cost_usd is None else f"{v.budget_cost_usd:g}",
                                step=0.5, minimum=0.5,
                                allow_blank=True, id="cost")
            yield Label("↑/↓ field · ←/→ step · type a number · ↵ apply · esc back",
                        classes="hint")
            with Horizontal():
                yield Button("apply", id="apply", variant="success")
                yield Button("cancel", id="cancel")

    def _collect(self) -> dict[str, Any] | None:
        try:
            steps = int(self.query_one("#steps", Input).value or "15")
            wall = float(self.query_one("#wall", Input).value or "300")
            tokens_raw = self.query_one("#tokens", Input).value.strip()
            cost_raw = self.query_one("#cost", Input).value.strip()
            return {
                "budget_steps": max(1, steps),
                "budget_wall_s": max(1.0, wall),
                "budget_tokens": int(tokens_raw) if tokens_raw else None,
                "budget_cost_usd": float(cost_raw) if cost_raw else None,
            }
        except ValueError:
            return None

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "cancel":
            self.dismiss(None)
        else:
            got = self._collect()
            if got is None:
                self.notify("numbers only", severity="error")
            else:
                self.dismiss(got)

    def on_input_submitted(self, ev: Input.Submitted) -> None:
        got = self._collect()
        if got is not None:
            self.dismiss(got)


class NumberScreen(BannaModal):
    """One typed number (temperature, n_candidates) — ↑/↓ or ←/→ steps
    it (single field, so vertical arrows have nothing else to do)."""

    def key_up(self, event) -> None:
        event.stop()
        self.query_one("#num", StepInput)._bump(+1)

    def key_down(self, event) -> None:
        event.stop()
        self.query_one("#num", StepInput)._bump(-1)

    def __init__(self, label: str, value: str, *, integer: bool,
                 step: float = 1.0, minimum: float | None = None,
                 maximum: float | None = None) -> None:
        super().__init__()
        self._label, self._value, self._integer = label, value, integer
        self._step, self._min, self._max = step, minimum, maximum

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._label, classes="title")
            yield StepInput(self._value, integer=self._integer,
                            step=self._step, minimum=self._min,
                            maximum=self._max, id="num")
            yield Label("↑/↓ or ←/→ step · ↵ apply · esc back", classes="hint")

    def on_input_submitted(self, ev: Input.Submitted) -> None:
        raw = ev.value.strip()
        if not raw:
            self.dismiss(None)
            return
        try:
            self.dismiss(int(raw) if self._integer else float(raw))
        except ValueError:
            self.notify("numbers only", severity="error")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

_ROWS = ("provider", "model", "policy", "budget", "theme", "sandbox",
         "temperature", "skills") + (("n_candidates",) if SHOW_N_CANDIDATES else ())


class DashboardScreen(Screen):
    """The main settings dashboard."""

    BINDINGS: ClassVar = [
        Binding("left", "cycle(-1)", "toggle ←", show=True),
        Binding("right", "cycle(1)", "toggle →", show=True),
        Binding("s", "start", "start", show=True),
        Binding("d", "save_start", "save as default & start", show=True),
        Binding("q", "quit_setup", "quit", show=True),
    ]

    DEFAULT_CSS = """
    DashboardScreen { align: center top; }
    DashboardScreen #hero { margin: 1 0 0 2; }
    DashboardScreen #frame {
        width: 84; height: auto; margin: 0 2;
        border: round $secondary; padding: 1 2;
    }
    DashboardScreen #rows { height: auto; }
    DashboardScreen .hint { color: $text-muted; margin: 0 2; }
    """

    AUTO_FOCUS = "#rows"

    def __init__(self, app_ref: SetupApp) -> None:
        super().__init__()
        self._setup = app_ref

    # -- compose / render ------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(self._hero(), id="hero")
        with Vertical(id="frame"):
            yield OptionList(id="rows")
        yield Label("↑↓ move · ←/→ toggle · ↵ edit · s start · "
                    "d save as default & start · q quit", classes="hint")
        yield Footer()

    def _hero(self) -> Text:
        from .theme import render_hero_mark_rows
        rows = render_hero_mark_rows()
        brand = ("bold" if self._setup.values.theme == "mono"
                 else f"bold {get_palette(self._setup.values.theme)['green']}")
        t = Text()
        for i, row in enumerate(rows):
            t.append_text(Text.from_markup(row))
            if i == 0:
                t.append("   ")
                t.append("banna", brand)
                t.append(" — session setup", "dim")
            t.append("\n")
        return t

    def on_mount(self) -> None:
        self._refresh_rows()

    def _refresh_rows(self, keep_highlight: bool = True) -> None:
        ol = self.query_one("#rows", OptionList)
        old = ol.highlighted if keep_highlight else 0
        ol.clear_options()
        for key in _ROWS:
            ol.add_option(Option(self._row_text(key), id=key))
        ol.add_option(Option(Text("─" * 72, "dim"), disabled=True, id="__sep__"))
        ol.add_option(Option(
            Text("  [ start session ]              (or press s)", "bold"),
            id="__start__"))
        ol.add_option(Option(
            Text("  [ save as default & start ]    (or press d)", "bold"),
            id="__save__"))
        ol.highlighted = old if old is not None else 0

    def _row_text(self, key: str) -> Text:
        v = self._setup.values
        scan = self._setup.scan
        t = Text()
        label = {
            "provider": "Provider", "model": "Model", "policy": "Policy",
            "budget": "Budget", "theme": "Theme", "sandbox": "Sandbox",
            "temperature": "Temperature", "skills": "Skills",
            "n_candidates": "n_candidates",
        }[key]
        t.append(f"  {label:<14}", "bold")
        if key == "provider":
            st = scan.get(v.provider)
            t.append(f"{PROVIDER_LABELS.get(v.provider, v.provider):<28}")
            if st and st.ok:
                t.append("✓ ", "green")
                t.append(st.detail, "dim")
                if st.source:
                    t.append(f" ({st.source})", "dim")
            elif st:
                t.append("✗ ", "red")
                t.append(st.detail, "dim")
        elif key == "model":
            t.append(v.model)
        elif key == "policy":
            t.append(v.policy)
            blurb = dict(POLICY_BLURBS).get(v.policy, "")
            if blurb:
                t.append(f"   {blurb}", "dim")
        elif key == "budget":
            t.append(_budget_display(v))
        elif key == "theme":
            t.append(f"{v.theme:<20}")
            if v.theme != "mono":
                p = get_palette(v.theme)
                for slot in ("green", "orange", "blue", "violet"):
                    t.append("██", p[slot])
        elif key == "sandbox":
            t.append(v.sandbox)
            t.append("   docker = network-less, read-only container" if
                     v.sandbox == "docker" else "   runs on the host", "dim")
        elif key == "temperature":
            t.append(f"{v.temperature:g}")
        elif key == "skills":
            t.append("on" if v.skills else "off")
            t.append("   skill-library injection + harvest", "dim")
        elif key == "n_candidates":
            t.append(str(v.n_candidates))
            t.append("   plans for bfs/dfs/best_first/best_of_n", "dim")
        return t

    # -- value mutation ---------------------------------------------------

    def _set(self, **kw: Any) -> None:
        self._setup.values = replace(self._setup.values, **kw)
        if "theme" in kw:
            self._setup.apply_palette()
            self.query_one("#hero", Static).update(self._hero())
        self._refresh_rows()

    def action_cycle(self, delta: int) -> None:
        ol = self.query_one("#rows", OptionList)
        if ol.highlighted is None:
            return
        opt = ol.get_option_at_index(ol.highlighted)
        key = str(opt.id)
        v = self._setup.values
        if key == "provider":
            new = _cycle(list(PROVIDER_ORDER), v.provider, delta)
            models = _provider_models(new, self._setup.scan)
            self._set(provider=new, model=models[0] if models else v.model)
        elif key == "model":
            models = _provider_models(v.provider, self._setup.scan)
            self._set(model=_cycle(models, v.model, delta))
        elif key == "policy":
            self._set(policy=_cycle(list(POLICY_NAMES), v.policy, delta))
        elif key == "theme":
            self._set(theme=_cycle(list(list_themes()), v.theme, delta))
        elif key == "sandbox":
            self._set(sandbox=_cycle(list(SANDBOXES), v.sandbox, delta))
        elif key == "temperature":
            self._set(temperature=round(
                min(2.0, max(0.0, v.temperature + 0.1 * delta)), 2))
        elif key == "skills":
            self._set(skills=not v.skills)
        elif key == "n_candidates":
            self._set(n_candidates=min(16, max(1, v.n_candidates + delta)))
        elif key == "budget":
            self.app.push_screen(BudgetScreen(v), self._on_budget)

    # -- enter → pickers ---------------------------------------------------

    def on_option_list_option_selected(self, ev: OptionList.OptionSelected) -> None:
        key = str(ev.option_id)
        v = self._setup.values
        if key == "__start__":
            self.action_start()
        elif key == "__save__":
            self.action_save_start()
        elif key == "provider":
            self.app.push_screen(ProviderScreen(self._setup.scan, v.provider),
                                 self._on_provider)
        elif key == "model":
            self.app.push_screen(ModelScreen(v.provider, self._setup.scan, v.model),
                                 self._on_model)
        elif key == "policy":
            self.app.push_screen(PolicyScreen(v.policy), self._on_policy)
        elif key == "budget":
            self.app.push_screen(BudgetScreen(v), self._on_budget)
        elif key == "theme":
            self.app.push_screen(ThemeScreen(v.theme), self._on_theme)
        elif key in ("sandbox", "skills"):
            self.action_cycle(1)
        elif key == "temperature":
            self.app.push_screen(
                NumberScreen("temperature (0–2)", f"{v.temperature:g}",
                             integer=False, step=0.1, minimum=0.0, maximum=2.0),
                self._on_temperature)
        elif key == "n_candidates":
            self.app.push_screen(
                NumberScreen("n_candidates (1–16)", str(v.n_candidates),
                             integer=True, step=1, minimum=1, maximum=16),
                self._on_n_candidates)

    def _on_provider(self, name: str | None) -> None:
        if name:
            models = _provider_models(name, self._setup.scan)
            self._set(provider=name,
                      model=models[0] if models else self._setup.values.model)

    def _on_model(self, model: str | None) -> None:
        if model:
            self._set(model=model)

    def _on_policy(self, policy: str | None) -> None:
        if policy:
            self._set(policy=policy)

    def _on_budget(self, got: dict | None) -> None:
        if got:
            self._set(**got)

    def _on_theme(self, theme: str | None) -> None:
        if theme:
            self._set(theme=theme)

    def _on_temperature(self, val: float | None) -> None:
        if val is not None:
            self._set(temperature=min(2.0, max(0.0, float(val))))

    def _on_n_candidates(self, val: int | None) -> None:
        if val is not None:
            self._set(n_candidates=min(16, max(1, int(val))))

    # -- start / quit -------------------------------------------------------

    def _ready_to_start(self) -> bool:
        v = self._setup.values
        st = self._setup.scan.get(v.provider)
        if st is None or st.ok:
            return True
        if v.provider in KEY_VARS:
            def _after(outcome: str | None) -> None:
                if outcome == "saved":
                    self._refresh_rows()
                    self._setup.finish()
                elif outcome == "ollama":
                    self._on_provider("ollama")
            self.app.push_screen(KeyScreen(v.provider, self._setup.scan), _after)
        else:
            self.notify(f"{v.provider}: {st.detail}", severity="error")
        return False

    def action_start(self) -> None:
        if self._ready_to_start():
            self._setup.finish()

    def action_save_start(self) -> None:
        if self._ready_to_start():
            self._setup.finish(save_default=True)

    def action_quit_setup(self) -> None:
        self._setup.exit(None)


class SetupApp(App):
    """Every-launch session-setup app. `run()` returns SetupResult | None."""

    TITLE = "banna — session setup"

    def __init__(self, values: SetupValues,
                 scan: dict[str, ProviderStatus] | None = None) -> None:
        super().__init__()
        self.values = values
        self.scan: dict[str, ProviderStatus] = scan or {}
        self._pending_save = False

    def on_mount(self) -> None:
        if not self.scan:
            self.scan.update(scan_providers())
        self._register_banna_themes()
        self.apply_palette()
        self.push_screen(DashboardScreen(self))

    def _register_banna_themes(self) -> None:
        """Register each banna palette as a Textual theme so the whole
        app chrome (highlights, buttons, borders) follows the pick."""
        from textual.theme import Theme as TextualTheme
        for name in list_themes():
            if name == "mono":
                continue  # mono maps to the built-in ansi theme
            p = get_palette(name)
            dark = "dark" in name or name == "dracula"
            try:
                self.register_theme(TextualTheme(
                    name=f"banna-{name}",
                    primary=p["blue"],
                    secondary=p["violet"],
                    warning=p["yellow"],
                    error=p["red"],
                    success=p["green"],
                    accent=p["orange"],
                    foreground=p["base01"],
                    background=p["base3"],
                    surface=p["base2"],
                    panel=p["base2"],
                    dark=dark,
                ))
            except Exception:
                # Older Textual without this Theme signature — the TUI
                # still works in its default colors.
                return

    def apply_palette(self) -> None:
        """Switch the Textual theme to the currently selected banna theme."""
        name = self.values.theme
        try:
            if name == "mono":
                self.theme = "textual-ansi"
            elif f"banna-{name}" in self.available_themes:
                self.theme = f"banna-{name}"
        except Exception:
            pass  # theme registration failed; keep defaults

    def finish(self, save_default: bool = False) -> None:
        self.exit(SetupResult(values=self.values, save_default=save_default))


def run_setup(values: SetupValues) -> SetupResult | None:
    """Scan providers, run the TUI, return the outcome (None = quit)."""
    scan = scan_providers()
    return SetupApp(values, scan).run()


__all__ = [
    "POLICY_BLURBS",
    "SANDBOXES",
    "SetupApp",
    "SetupResult",
    "SetupValues",
    "run_setup",
]
