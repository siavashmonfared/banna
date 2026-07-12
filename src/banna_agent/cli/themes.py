"""Theme registry for the banna CLI.

Every theme is a palette dict using the *same slot names* as the
canonical Solarized palette in `theme.py` (base3 = background …
green = agent accent), so one mapping function turns any palette into
the Rich Theme the whole CLI paints with.

The active theme is a module-level global: `set_active(name)` before
any Console is constructed (done by `main()` from `[ui] theme` in
config.toml / the launch TUI), and `theme.scout_theme()` keeps working
unchanged for every existing caller.

Note: the pixel-art mascot and the raw-ANSI input-box accents keep
their Solarized colors — they're brand, not theme.
"""
from __future__ import annotations

from rich.theme import Theme

# ---------------------------------------------------------------------------
# Palettes — same slot names as theme.SOLARIZED
# ---------------------------------------------------------------------------

_SOLARIZED_LIGHT = {
    "base03": "#002b36", "base02": "#073642",
    "base01": "#586e75",   # emphasis / agent prose
    "base00": "#657b83",   # body
    "base0":  "#839496",
    "base1":  "#93a1a1",   # muted
    "base2":  "#eee8d5",   # panel
    "base3":  "#fdf6e3",   # background
    "yellow": "#b58900", "orange": "#cb4b16", "red": "#dc322f",
    "magenta": "#d33682", "violet": "#6c71c4", "blue": "#268bd2",
    "cyan": "#2aa198", "green": "#859900",
}

# Same hues, light/dark roles swapped (canonical Solarized Dark).
_SOLARIZED_DARK = {
    **_SOLARIZED_LIGHT,
    "base3":  "#002b36",   # background
    "base2":  "#073642",   # panel
    "base1":  "#586e75",   # muted
    "base01": "#93a1a1",   # emphasis
    "base00": "#839496",   # body
    "base0":  "#657b83",
    "base03": "#fdf6e3",
    "base02": "#eee8d5",
}

_GRUVBOX_DARK = {
    "base03": "#fbf1c7", "base02": "#ebdbb2",
    "base01": "#ebdbb2",   # emphasis
    "base00": "#d5c4a1",   # body
    "base0":  "#bdae93",
    "base1":  "#928374",   # muted
    "base2":  "#3c3836",   # panel
    "base3":  "#282828",   # background
    "yellow": "#fabd2f", "orange": "#fe8019", "red": "#fb4934",
    "magenta": "#d3869b", "violet": "#d3869b", "blue": "#83a598",
    "cyan": "#8ec07c", "green": "#b8bb26",
}

_DRACULA = {
    "base03": "#f8f8f2", "base02": "#f8f8f2",
    "base01": "#f8f8f2",   # emphasis
    "base00": "#e6e6dc",   # body
    "base0":  "#bfbfb2",
    "base1":  "#6272a4",   # muted
    "base2":  "#44475a",   # panel
    "base3":  "#282a36",   # background
    "yellow": "#f1fa8c", "orange": "#ffb86c", "red": "#ff5555",
    "magenta": "#ff79c6", "violet": "#bd93f9", "blue": "#8be9fd",
    "cyan": "#8be9fd", "green": "#50fa7b",
}

PALETTES: dict[str, dict[str, str]] = {
    "solarized-light": _SOLARIZED_LIGHT,
    "solarized-dark": _SOLARIZED_DARK,
    "gruvbox-dark": _GRUVBOX_DARK,
    "dracula": _DRACULA,
    "mono": {},   # sentinel — build_theme returns an uncolored theme
}

THEME_BLURBS: dict[str, str] = {
    "solarized-light": "the banna default — warm paper background",
    "solarized-dark": "same hues on the dark base",
    "gruvbox-dark": "retro warm dark",
    "dracula": "high-contrast dark purple",
    "mono": "no color — pipes, screen readers, minimal terminals",
}

DEFAULT_THEME = "solarized-light"

_active = DEFAULT_THEME


def list_themes() -> tuple[str, ...]:
    return tuple(PALETTES)


def get_active() -> str:
    return _active


def set_active(name: str) -> str:
    """Set the process-wide theme. Unknown names fall back to default."""
    global _active
    _active = name if name in PALETTES else DEFAULT_THEME
    return _active


def get_palette(name: str | None = None) -> dict[str, str]:
    """Palette dict for `name` (default: active theme). Mono returns the
    Solarized slots so glyph-drawing code always has hex values."""
    p = PALETTES.get(name or _active) or _SOLARIZED_LIGHT
    return p or _SOLARIZED_LIGHT


def build_theme(name: str | None = None) -> Theme:
    """Rich Theme for `name` (default: the active theme)."""
    key = name or _active
    if key == "mono":
        return _mono_theme()
    return _theme_from_palette(get_palette(key))


def _theme_from_palette(s: dict[str, str]) -> Theme:
    """The canonical style map (moved from theme.scout_theme), applied
    to any palette that fills the Solarized slot names."""
    return Theme({
        # ---- remap bare color names so existing markup just works ------
        "cyan":     s["cyan"],
        "green":    s["green"],
        "yellow":   s["yellow"],
        "red":      s["red"],
        "magenta":  s["magenta"],
        "blue":     s["blue"],
        "white":    s["base01"],          # readable on the theme's bg
        "dim":      s["base1"],           # softer than default dim

        # ---- bold variants (preserve `bold cyan` etc.) -----------------
        "bold cyan":    f"bold {s['cyan']}",
        "bold green":   f"bold {s['green']}",
        "bold yellow":  f"bold {s['yellow']}",
        "bold red":     f"bold {s['red']}",
        "bold magenta": f"bold {s['magenta']}",
        "bold blue":    f"bold {s['blue']}",

        # ---- scout.* / banna.* semantic aliases (from §03 spec) -------
        "scout.bg":         s["base3"],
        "scout.panel":      s["base2"],
        "scout.border":     s["base1"],
        "scout.muted":      s["base1"],
        "scout.text":       s["base01"],
        "scout.body":       s["base00"],
        "scout.you":        f"bold {s['orange']}",
        "scout.you.caret":  s["orange"],
        "scout.agent":      f"bold {s['green']}",
        "scout.reasoning":  s["violet"],
        "scout.link":       s["blue"],
        "scout.inflight":   s["yellow"],
        "scout.ok":         s["green"],
        "scout.err":        s["red"],
        "scout.warn":       s["yellow"],
        "scout.title":      f"bold {s['green']}",
        "scout.accent":     s["green"],
        "scout.info":       s["cyan"],
        "scout.prompt":     f"bold {s['orange']}",
        "scout.border.dim": s["base1"],

        # Banna aliases — same things, friendlier names.
        "banna.you":       f"bold {s['orange']}",
        "banna.agent":     f"bold {s['green']}",
        "banna.frame":     s["base1"],
        "banna.link":      s["blue"],
        "banna.reasoning": s["violet"],
        "banna.spinner":   s["yellow"],
    })


def apply_terminal_colors(stream=None) -> bool:
    """Paint the *terminal itself* in the active theme via OSC 10/11.

    Rich styles only color the text they print; the terminal's default
    background/foreground stay whatever the emulator was configured
    with — which is why a picked theme otherwise "disappears" once the
    Textual setup screen exits into the REPL. OSC 11 (background) and
    OSC 10 (foreground) re-program those defaults for the whole session.
    Supported by every mainstream emulator (xterm, gnome-terminal,
    kitty, alacritty, wezterm, iTerm2); unsupported ones ignore it.

    Returns True when the escape was written (caller should later call
    `reset_terminal_colors`); False for mono / non-TTY streams.
    """
    import sys
    if _active == "mono":
        return False
    s = stream or sys.stdout
    if not (hasattr(s, "isatty") and s.isatty()):
        return False
    p = get_palette()
    try:
        s.write(f"\x1b]10;{p['base01']}\x07\x1b]11;{p['base3']}\x07")
        s.flush()
    except Exception:
        return False
    return True


def reset_terminal_colors(stream=None) -> None:
    """Undo `apply_terminal_colors` (OSC 110/111 = reset to defaults)."""
    import sys
    s = stream or sys.stdout
    try:
        if hasattr(s, "isatty") and s.isatty():
            s.write("\x1b]110\x07\x1b]111\x07")
            s.flush()
    except Exception:
        pass


def _mono_theme() -> Theme:
    """Colorless theme: bare color names go to terminal default, bold
    stays bold, semantic slots keep only weight/emphasis."""
    plain = "default"
    styles: dict[str, str] = {}
    for name in ("cyan", "green", "yellow", "red", "magenta", "blue", "white"):
        styles[name] = plain
        styles[f"bold {name}"] = "bold"
    styles["dim"] = "dim"
    for name in (
        "scout.bg", "scout.panel", "scout.border", "scout.muted", "scout.text",
        "scout.body", "scout.you.caret", "scout.reasoning", "scout.link",
        "scout.inflight", "scout.ok", "scout.err", "scout.warn", "scout.accent",
        "scout.info", "scout.border.dim",
        "banna.frame", "banna.link", "banna.reasoning", "banna.spinner",
    ):
        styles[name] = plain
    for name in ("scout.you", "scout.agent", "scout.title", "scout.prompt",
                 "banna.you", "banna.agent"):
        styles[name] = "bold"
    return Theme(styles)


__all__ = [
    "DEFAULT_THEME",
    "PALETTES",
    "THEME_BLURBS",
    "apply_terminal_colors",
    "build_theme",
    "get_active",
    "get_palette",
    "list_themes",
    "reset_terminal_colors",
    "set_active",
]
