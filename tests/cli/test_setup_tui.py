"""Tests for the launch TUI stack: model_catalog, themes, setup_tui.

The Textual app is exercised headless via Pilot (no TTY needed); the
catalog/key tests isolate config paths with XDG_CONFIG_HOME + tmp cwd.
"""
from __future__ import annotations

import asyncio
import os
import stat

import pytest

from banna_agent.cli import model_catalog as mc
from banna_agent.cli import themes
from banna_agent.cli.model_catalog import ProviderStatus
from banna_agent.cli.setup_tui import (
    POLICY_NAMES,
    SetupApp,
    SetupValues,
    _budget_display,
)

# ---------------------------------------------------------------------------
# model_catalog
# ---------------------------------------------------------------------------


def test_curated_lists_nonempty_for_cloud_providers():
    for p in ("anthropic", "openai", "gemini", "bedrock"):
        assert mc.CURATED[p], p
        for model, blurb in mc.CURATED[p]:
            assert model and blurb


def test_known_models_includes_bedrock_extras():
    models = mc.known_models("bedrock")
    assert "us.anthropic.claude-3-haiku-20240307-v1:0" in models
    assert models[0] == mc.CURATED["bedrock"][0][0]


def test_gemini_accepts_both_key_vars():
    assert set(mc.KEY_VARS["gemini"]) == {"GEMINI_API_KEY", "GOOGLE_API_KEY"}


@pytest.fixture()
def isolated_config(tmp_path, monkeypatch):
    """Point config + cwd at tmp dirs and clear provider env vars."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path / "cwd" if (tmp_path / "cwd").mkdir() or True else tmp_path)
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                "GOOGLE_API_KEY", "GOOGLE_SEARCH_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_find_key_missing_everywhere(isolated_config):
    assert mc.find_key("anthropic") is None


def test_find_key_reports_shell_env(isolated_config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    var, source = mc.find_key("anthropic")
    assert var == "ANTHROPIC_API_KEY"
    assert source == "shell env"


def test_find_key_reports_config_env_file(isolated_config, monkeypatch):
    from banna_agent.cli.config_store import env_path
    p = env_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("GEMINI_API_KEY=abc123\n")
    monkeypatch.setenv("GEMINI_API_KEY", "abc123")  # as _load_dotenv would
    var, source = mc.find_key("gemini")
    assert var == "GEMINI_API_KEY"
    assert source == str(p)


def test_save_api_key_writes_all_vars_mode_0600(isolated_config):
    path = mc.save_api_key("gemini", "gkey-42")
    text = path.read_text()
    assert "GEMINI_API_KEY=gkey-42" in text
    assert "GOOGLE_API_KEY=gkey-42" in text
    assert os.environ["GOOGLE_API_KEY"] == "gkey-42"
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600


def test_scan_providers_reports_missing(isolated_config, monkeypatch):
    monkeypatch.setattr(mc, "detect_ollama", lambda timeout_s=1.0: None)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    scan = mc.scan_providers()
    assert set(scan) == set(mc.PROVIDER_ORDER)
    assert not scan["anthropic"].ok
    assert "ANTHROPIC_API_KEY" in scan["anthropic"].detail
    assert not scan["ollama"].ok
    assert not scan["bedrock"].ok


# ---------------------------------------------------------------------------
# themes
# ---------------------------------------------------------------------------


def test_every_theme_builds_a_rich_theme():
    for name in themes.list_themes():
        t = themes.build_theme(name)
        assert "scout.agent" in t.styles
        assert "banna.link" in t.styles


def test_set_active_falls_back_on_unknown():
    assert themes.set_active("no-such-theme") == themes.DEFAULT_THEME
    assert themes.set_active("dracula") == "dracula"
    themes.set_active(themes.DEFAULT_THEME)


def test_scout_theme_follows_active_theme():
    from banna_agent.cli.theme import scout_theme
    themes.set_active("gruvbox-dark")
    try:
        t = scout_theme()
        assert str(t.styles["scout.link"].color.name) == "#83a598"
    finally:
        themes.set_active(themes.DEFAULT_THEME)


# ---------------------------------------------------------------------------
# setup_tui — headless Pilot
# ---------------------------------------------------------------------------


def _fake_scan() -> dict[str, ProviderStatus]:
    """All providers ready — no network, no key screens."""
    scan = {
        name: ProviderStatus(name, True, "FAKE_KEY", source="shell env")
        for name in mc.PROVIDER_ORDER
    }
    scan["ollama"] = ProviderStatus(
        "ollama", True, "localhost:11434 · 2 models",
        ollama_models=[{"name": "qwen3:8b", "size": 5_000_000_000},
                       {"name": "llama3.1:8b", "size": 4_700_000_000}])
    return scan


def _run_app(coro_fn):
    """Run an async pilot scenario against a fresh SetupApp; return the app."""
    app = SetupApp(SetupValues(), _fake_scan())

    async def main():
        async with app.run_test(size=(100, 40)) as pilot:
            await coro_fn(pilot)

    asyncio.run(main())
    return app


def test_pilot_start_returns_seed_values():
    async def scenario(pilot):
        await pilot.press("s")

    app = _run_app(scenario)
    result = app.return_value
    assert result is not None and not result.save_default
    assert result.values == SetupValues()


def test_pilot_right_cycles_provider_and_resets_model():
    async def scenario(pilot):
        await pilot.press("right")     # provider row: openai -> gemini
        await pilot.press("s")

    app = _run_app(scenario)
    v = app.return_value.values
    assert v.provider == "gemini"
    assert v.model == mc.CURATED["gemini"][0][0]


def test_pilot_cycle_policy_theme_and_save_flag():
    async def scenario(pilot):
        await pilot.press("down", "down")      # -> policy row
        await pilot.press("right")             # react -> react+
        await pilot.press("down", "down")      # -> theme row
        await pilot.press("right")             # solarized-light -> solarized-dark
        await pilot.press("d")                 # save as default & start

    app = _run_app(scenario)
    result = app.return_value
    assert result.save_default
    assert result.values.policy == POLICY_NAMES[1]
    assert result.values.theme == "solarized-dark"


def test_pilot_quit_returns_none():
    async def scenario(pilot):
        await pilot.press("q")

    app = _run_app(scenario)
    assert app.return_value is None


def test_budget_display_formats_unlimited():
    v = SetupValues(budget_tokens=None, budget_cost_usd=None)
    s = _budget_display(v)
    assert "∞ tokens" in s and "∞ cost" in s


def test_pilot_budget_modal_arrow_keys():
    """↑/↓ moves between budget fields; ←/→ steps the focused value."""
    async def scenario(pilot):
        await pilot.press("down", "down", "down")   # -> budget row
        await pilot.press("enter")                  # open BudgetScreen
        await pilot.pause()
        await pilot.press("right", "right")         # steps: 15 -> 17
        await pilot.press("down")                   # ↓ -> wall field
        await pilot.press("left")                   # wall: 300 -> 290
        await pilot.press("down")                   # ↓ -> tokens field (blank = ∞)
        await pilot.press("right")                  # blank -> 10000
        await pilot.press("right")                  # -> 20000
        await pilot.press("left", "left", "left")   # -> blank again (∞)
        await pilot.press("enter")                  # apply
        await pilot.pause()
        await pilot.press("s")

    app = _run_app(scenario)
    v = app.return_value.values
    assert v.budget_steps == 17
    assert v.budget_wall_s == 290.0
    assert v.budget_tokens is None


def test_n_candidates_row_gated_by_exposed_policies():
    """The dashboard shows n_candidates only when this build exposes a
    policy that uses it (the public CLI exposes none)."""
    from banna_agent.cli.commands import POLICY_NAMES as exposed
    from banna_agent.cli.setup_tui import _ROWS
    uses_n = {"bfs_over_plans", "dfs_over_plans", "best_first_over_plans",
              "best_of_n", "banna_thinking"}
    assert ("n_candidates" in _ROWS) == bool(uses_n & set(exposed))


def test_pilot_number_modal_arrow_stepping_temperature():
    async def scenario(pilot):
        await pilot.press("down", "down", "down", "down", "down", "down")  # -> temperature
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("up", "up", "up")   # 0.7 -> 1.0
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("s")

    app = _run_app(scenario)
    assert app.return_value.values.temperature == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# config persistence (app._persist_setup)
# ---------------------------------------------------------------------------


def test_persist_setup_writes_sections_and_preserves_packages(isolated_config):
    from banna_agent.cli.app import _persist_setup
    from banna_agent.cli.config_store import read_config, write_config

    write_config({"packages": {"numpy": "numpy==2.0.0"}})
    v = SetupValues(provider="anthropic", model="claude-sonnet-5",
                    policy="react+", theme="dracula", skills=True,
                    budget_tokens=100_000, budget_cost_usd=None)
    _persist_setup(v)

    cfg = read_config()
    assert cfg["default"]["provider"] == "anthropic"
    assert cfg["default"]["model"] == "claude-sonnet-5"
    assert cfg["default"]["policy"] == "react+"
    assert cfg["default"]["skills"] is True
    assert cfg["budget"]["tokens"] == 100_000
    assert "cost_usd" not in cfg["budget"]      # None = unlimited = omitted
    assert cfg["ui"]["theme"] == "dracula"
    assert cfg["packages"]["numpy"] == "numpy==2.0.0"
