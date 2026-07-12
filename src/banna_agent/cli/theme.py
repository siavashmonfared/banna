"""Banna theme — Solarized-Light TUI ported from scout-theme.html.

Unpacked from the bundler/manifest payload in scout-theme.html (Section
03 semantic tokens + Section 05 input box). This is the canonical
design; do not invent colors elsewhere.

Semantic tokens (from §03):
    fg / body          base00  #657b83
    fg / emphasis      base01  #586e75   — agent prose
    fg / dim           base1   #93a1a1
    fg / faint         base2   #eee8d5   — frame strokes
    bg                 base3   #fdf6e3
    user prompt        orange  #cb4b16   — ❯ you label + caret
    agent / success    green   #859900   — scout (now banna) label, ✓
    tool path / link   blue    #268bd2
    reasoning          violet  #6c71c4   — ▾ reasoning header
    in-flight          yellow  #b58900   — braille spinner
    error / removed    red     #dc322f

Glyphs (from §04):
    ❯  prompt + caret    ▾  open section    ▸  closed section
    ✓  tool success      ✗  tool failure    │  reasoning rule
    ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏  spinner (90 ms)
"""
from __future__ import annotations

from typing import Any

from rich.theme import Theme


# Solarized Light palette ------------------------------------------------------
SOLARIZED = {
    "base03":  "#002b36",
    "base02":  "#073642",
    "base01":  "#586e75",   # body text on dark / muted on light
    "base00":  "#657b83",
    "base0":   "#839496",
    "base1":   "#93a1a1",   # light muted
    "base2":   "#eee8d5",   # panel-ish
    "base3":   "#fdf6e3",   # background
    "yellow":  "#b58900",
    "orange":  "#cb4b16",
    "red":     "#dc322f",
    "magenta": "#d33682",
    "violet":  "#6c71c4",
    "blue":    "#268bd2",
    "cyan":    "#2aa198",
    "green":   "#859900",
}

# Brand: banna (formerly "scout").
BRAND = "banna"

# 8×8 pixel-art mascot. Grid + palette are imported live from
# scout-mascot.html (project root) so the design source of truth lives
# in the .html and any edits there flow through here. Falls back to a
# pinned copy if the file is missing.
_MASCOT_FALLBACK_GRID = (
    "..cccc..",
    ".cccccc.",
    ".ssssss.",
    ".semmes.",
    ".sssoss.",
    ".yyyyyy.",
    ".yoooy..",
    ".kk..kk.",
)
_MASCOT_FALLBACK_PALETTE = {
    "c": "#859900", "s": "#fce8c4", "e": "#073642", "m": "#073642",
    "o": "#cb4b16", "y": "#b58900", "k": "#073642",
}


def _load_mascot_from_html() -> tuple[tuple[str, ...], dict[str, str]]:
    """Parse scout-mascot.html to extract GRID and PALETTE.

    The file embeds a Python reference implementation in its `<pre
    class="block">` markup; we extract GRID/PALETTE from there. If the
    file isn't reachable, return the pinned fallback.
    """
    import re
    from pathlib import Path

    here = Path(__file__).resolve()
    # Walk up looking for scout-mascot.html in any parent (project root).
    for parent in [here.parent, *here.parents]:
        candidate = parent / "scout-mascot.html"
        if candidate.is_file():
            try:
                src = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                break
            # Strip HTML tags so the inline Python becomes readable text.
            text = re.sub(r"<[^>]+>", "", src)
            # Palette: "c": (133, 153, 0),    # …
            pal: dict[str, str] = {}
            for m in re.finditer(
                r'"([a-z])"\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
                text,
            ):
                key, r, g, b = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
                pal[key] = f"#{r:02x}{g:02x}{b:02x}"
            # Grid: list of 8 quoted 8-char strings.
            grid_rows = re.findall(r'"([.a-z]{8})"', text)
            grid_rows = [g for g in grid_rows if any(c != "." for c in g)]
            # First 8 distinct 8-char rows that form the sprite.
            if len(grid_rows) >= 8 and len(pal) >= 4:
                return tuple(grid_rows[:8]), pal
            break
    return _MASCOT_FALLBACK_GRID, _MASCOT_FALLBACK_PALETTE


HERO_MARK_GRID, HERO_PALETTE = _load_mascot_from_html()


def _shift_legs(grid: tuple[str, ...], dx_left: int, dx_right: int) -> tuple[str, ...]:
    """Build a running-pose variant by shifting the foot pixels in the
    bottom row of the sprite.

    Only the last row (`.kk..kk.`) is touched. Left foot at cols 1-2
    moves by `dx_left`, right foot at cols 5-6 moves by `dx_right`.
    """
    rows = list(grid)
    if len(rows) < 8:
        return grid
    base = list("." * len(rows[-1]))
    # Default rest position from grid[-1] if it matches expectations.
    feet_char = "k"
    # Left foot: place 2 'k's starting at col 1 + dx_left.
    lx = max(0, min(len(base) - 2, 1 + dx_left))
    base[lx] = feet_char
    base[lx + 1] = feet_char
    rx = max(0, min(len(base) - 2, 5 + dx_right))
    base[rx] = feet_char
    base[rx + 1] = feet_char
    rows[-1] = "".join(base)
    return tuple(rows)


# Locally synthesized running cycle (the bundle references HERO_RUN_A/B
# but doesn't define them — we make our own by shifting the feet).
HERO_FRAMES = (
    HERO_MARK_GRID,                                    # idle
    _shift_legs(HERO_MARK_GRID, dx_left=-1, dx_right=+1),  # legs spread
    HERO_MARK_GRID,                                    # back to idle
    _shift_legs(HERO_MARK_GRID, dx_left=+1, dx_right=-1),  # legs crossed
)

# macOS-style traffic-light window controls. These sit in the .titlebar
# in scout-theme.html, just under the h1. The green one is the
# "luigi-like" dot the user pointed at — Luigi's signature green.
TRAFFIC_RED = "#ff5f57"
TRAFFIC_YELLOW = "#febc2e"
TRAFFIC_GREEN = "#28c840"


def render_brand_title(path: str | None = None) -> str:
    """Rich-markup title string for the header panel.

    Layout:  ● banna — ~/path
    The green bullet matches the spec's agent-label glyph; the full 4-row
    mascot is rendered separately above the panel.
    """
    dot = "[scout.accent]●[/scout.accent]"   # themed: green slot of the active palette
    name = f"[scout.text]{BRAND}[/scout.text]"
    if path:
        return f"{dot} {name} [scout.muted]— {path}[/scout.muted]"
    return f"{dot} {name}"


def _render_grid_rows(grid: tuple[str, ...]) -> list[str]:
    """Half-block render for any 8-wide pixel grid (§04 transparency rules)."""
    h = len(grid)
    w = max(len(r) for r in grid)
    rows: list[str] = []
    for y in range(0, h, 2):
        top = grid[y]
        bot = grid[y + 1] if y + 1 < h else "." * w
        parts: list[str] = []
        for x in range(w):
            t = top[x] if x < len(top) else "."
            b = bot[x] if x < len(bot) else "."
            tc = HERO_PALETTE.get(t); bc = HERO_PALETTE.get(b)
            if tc is None and bc is None:
                parts.append(" ")
            elif tc is None:
                parts.append(f"[{bc}]▄[/]")
            elif bc is None:
                parts.append(f"[{tc}]▀[/]")
            else:
                parts.append(f"[{tc} on {bc}]▀[/]")
        rows.append("".join(parts))
    return rows


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def render_hero_frame_ansi(frame_idx: int = 0, scale: int = 1) -> list[str]:
    """Render one HERO_FRAMES frame as a list of raw-ANSI lines.

    Used by the input-line code which writes directly to stdout
    (Rich markup wouldn't work mid-input). Each cell ends with ESC[0m
    so the terminal state resets before the next character.

    `scale` widens each half-block cell horizontally: scale=1 → 8 cols,
    scale=2 → 16 cols. Vertical scale is fixed at 1 (each row remains
    one terminal line, carrying 2 pixel rows via the ▀ trick).
    """
    RESET = "\x1b[0m"
    grid = HERO_FRAMES[frame_idx % len(HERO_FRAMES)]
    rows: list[str] = []
    scale = max(1, int(scale))
    for y in range(0, len(grid), 2):
        top = grid[y]
        bot = grid[y + 1] if y + 1 < len(grid) else "." * len(top)
        cells: list[str] = []
        for x in range(len(top)):
            t = top[x]; b = bot[x]
            tc = HERO_PALETTE.get(t)
            bc = HERO_PALETTE.get(b)
            if tc is None and bc is None:
                cells.append(" " * scale)
            elif tc is None:
                r, g, bb = _hex_to_rgb(bc)
                cells.append(f"\x1b[38;2;{r};{g};{bb}m" + ("▄" * scale) + RESET)
            elif bc is None:
                r, g, bb = _hex_to_rgb(tc)
                cells.append(f"\x1b[38;2;{r};{g};{bb}m" + ("▀" * scale) + RESET)
            else:
                tr, tg, tbb = _hex_to_rgb(tc)
                br, bg, bbb = _hex_to_rgb(bc)
                cells.append(
                    f"\x1b[38;2;{tr};{tg};{tbb};48;2;{br};{bg};{bbb}m"
                    + ("▀" * scale) + RESET
                )
        rows.append("".join(cells))
    return rows


def render_hero_frame(frame_idx: int = 0) -> str:
    """Rich-markup string for one running-cycle frame (4 lines, 8 cells)."""
    grid = HERO_FRAMES[frame_idx % len(HERO_FRAMES)]
    return "\n".join(_render_grid_rows(grid))


def animate_hero(
    console: Any,
    *,
    frames: int = 12,
    fps: int = 8,
    leading_text: str = "",
) -> None:
    """Briefly animate the mascot in-place using rich.live, then leave the
    idle frame in scrollback.

    `leading_text` (Rich markup) is printed to the right of the sprite —
    e.g. a "starting…" label.
    """
    import time
    from rich.live import Live
    from rich.text import Text

    def compose(i: int) -> Text:
        grid = HERO_FRAMES[i % len(HERO_FRAMES)]
        lines = _render_grid_rows(grid)
        # Right-align leading_text on the second row.
        out = Text()
        for k, line in enumerate(lines):
            out.append_text(Text.from_markup(line))
            if k == 1 and leading_text:
                out.append("  ")
                out.append_text(Text.from_markup(leading_text))
            out.append("\n")
        return out

    interval = 1.0 / max(1, fps)
    with Live(compose(0), console=console, refresh_per_second=fps, transient=False) as live:
        for i in range(1, frames):
            time.sleep(interval)
            live.update(compose(i))
        # Final idle pose.
        live.update(compose(0))


def render_hero_mark_rows() -> list[str]:
    """Render the mascot as Rich-markup lines per scout-mascot.html §04.

    Half-block trick: each terminal cell carries two stacked pixels.
        '▀' (fg=top, bg=bot)   when both pixels are filled
        '▀' (fg=top, no bg)    when only top is filled
        '▄' (fg=bot, no bg)    when only bottom is filled
        ' '                    when both are transparent
    Transparent cells use the terminal's own background — no fill.
    """
    g = HERO_MARK_GRID
    h = len(g)
    w = max(len(r) for r in g)
    rows: list[str] = []
    for y in range(0, h, 2):
        top = g[y]
        bot = g[y + 1] if y + 1 < h else "." * w
        parts: list[str] = []
        for x in range(w):
            t = top[x] if x < len(top) else "."
            b = bot[x] if x < len(bot) else "."
            tc = HERO_PALETTE.get(t)
            bc = HERO_PALETTE.get(b)
            if tc is None and bc is None:
                parts.append(" ")
            elif tc is None:
                parts.append(f"[{bc}]▄[/]")
            elif bc is None:
                parts.append(f"[{tc}]▀[/]")
            else:
                parts.append(f"[{tc} on {bc}]▀[/]")
        rows.append("".join(parts))
    return rows


def render_hero_mark_inline(row_pair: tuple[int, int] = (2, 3)) -> str:
    """Single-line variant for inline placement (e.g. in a panel title).

    Samples one pair of adjacent pixel rows and renders them as a single
    line of half-block cells. Default is rows 2+3 — the face — which is
    the most recognizable bit of the sprite. Pass (0, 1) for the green
    hat only, (4, 5) for the shirt, etc.
    """
    g = HERO_MARK_GRID
    top = g[row_pair[0]]
    bot = g[row_pair[1]]
    parts: list[str] = []
    for x in range(len(top)):
        t = top[x]; b = bot[x]
        tc = HERO_PALETTE.get(t); bc = HERO_PALETTE.get(b)
        if tc is None and bc is None:
            parts.append(" ")
        elif tc is None:
            parts.append(f"[{bc}]▄[/]")
        elif bc is None:
            parts.append(f"[{tc}]▀[/]")
        else:
            parts.append(f"[{tc} on {bc}]▀[/]")
    return "".join(parts)


def render_hero_mark_compact() -> str:
    """Two-row, 8-col mascot for inline header use — top half + bottom half.

    Returns a Rich-markup string with one embedded newline; renders as
    a 2-line, 8-cell sprite. Use this when the title bar can grow by
    one line but not four. (Each line is 2 pixel rows via half-block.)
    """
    rows = render_hero_mark_rows()
    # Take top 2 lines = pixel rows 0–3 (head only). Skip body+feet.
    return "\n".join(rows[:2])


# Back-compat alias for earlier code that used the mushroom emoji.
BRAND_ICON = "●"
BRAND_TITLE = render_brand_title()

# Prompt glyph from §05 of the scout spec. The `❯` is the user caret
# and is rendered in `orange` — not green. (Earlier draft was wrong.)
PROMPT = "❯ "
PROMPT_ACCENT_STYLE = "bold #cb4b16"   # solarized orange

# Box-drawing characters (§04, §05).
BOX = {
    "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
    "h":  "─", "v":  "│",
}

# Right-edge hints printed inside the top/bottom rules of the input box.
INPUT_TOP_HINT_RIGHT = "↵ send · ⌃c stop"
INPUT_BOTTOM_HINT_RIGHT = "/ cmds · @ files"
INPUT_LABEL = "you"

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_INTERVAL_MS = 90


def scout_theme() -> Theme:
    """Return the Rich Theme that paints the CLI in the *active* theme.

    The style map lives in `themes._theme_from_palette`; which palette
    fills it is chosen via `themes.set_active` (from `[ui] theme` in
    config.toml or the launch TUI). Defaults to Solarized Light, so
    every caller that predates theming is unchanged.
    """
    from .themes import build_theme
    return build_theme()


def render_input_box_top(width: int) -> str:
    """Top rule of the TUI input box (Rich-markup string).

        ╭─ you ────────────────────────╮
    """
    label = f" [scout.you]{INPUT_LABEL}[/scout.you] "
    label_visible = len(INPUT_LABEL) + 2
    # Account for: corners (2), 1 leading dash, label visible width,
    # remaining dashes.
    fill = width - 2 - 1 - label_visible
    if fill < 4:
        fill = 4
    left = "[scout.border]" + BOX["tl"] + BOX["h"] + "[/scout.border]"
    mid = "[scout.border]" + (BOX["h"] * fill) + BOX["tr"] + "[/scout.border]"
    return left + label + mid


def render_input_box_bottom(width: int) -> str:
    """Bottom rule of the TUI input box (Rich-markup string).

        ╰──────────────────────────────╯
    """
    fill = width - 2
    if fill < 4:
        fill = 4
    left = "[scout.border]" + BOX["bl"] + "[/scout.border]"
    mid = "[scout.border]" + (BOX["h"] * fill) + BOX["br"] + "[/scout.border]"
    return left + mid


__all__ = [
    "scout_theme", "SOLARIZED",
    "PROMPT", "PROMPT_ACCENT_STYLE",
    "BRAND", "BRAND_ICON", "BRAND_TITLE",
    "TRAFFIC_RED", "TRAFFIC_YELLOW", "TRAFFIC_GREEN",
    "HERO_MARK_GRID", "HERO_PALETTE",
    "render_brand_title", "render_hero_mark_rows", "render_hero_mark_inline",
    "render_hero_frame", "render_hero_frame_ansi",
    "animate_hero", "HERO_FRAMES",
    "BOX", "SPINNER_FRAMES", "SPINNER_INTERVAL_MS",
    "INPUT_LABEL", "INPUT_TOP_HINT_RIGHT", "INPUT_BOTTOM_HINT_RIGHT",
    "render_input_box_top", "render_input_box_bottom",
]
