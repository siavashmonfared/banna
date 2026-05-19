"""Tests for the first-run setup wizard and supporting modules."""
from __future__ import annotations

import io
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from banna_agent.cli import config_store, setup_wizard, subcommands


# ---------------------------------------------------------------------------
# config_store
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_config(monkeypatch, tmp_path: Path) -> Path:
    """Redirect ~/.config/banna/ to a tmp path."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path / "banna"


def test_config_dir_honors_xdg(isolated_config: Path) -> None:
    assert config_store.config_dir() == isolated_config


def test_is_first_run_true_when_no_config(isolated_config: Path) -> None:
    assert config_store.is_first_run() is True


def test_write_read_config_roundtrip(isolated_config: Path) -> None:
    config_store.write_config({"default": {
        "provider": "openai",
        "model": "gpt-5-nano",
        "temperature": 0.7,
    }})
    assert config_store.is_first_run() is False
    data = config_store.read_config()
    assert data["default"]["provider"] == "openai"
    assert data["default"]["model"] == "gpt-5-nano"
    assert data["default"]["temperature"] == 0.7


def test_write_env_creates_mode_0600(isolated_config: Path) -> None:
    path = config_store.write_env({"OPENAI_API_KEY": "sk-test"})
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


def test_write_env_merges_existing(isolated_config: Path) -> None:
    config_store.write_env({"OPENAI_API_KEY": "k1"})
    config_store.write_env({"ANTHROPIC_API_KEY": "k2"})
    env = config_store.read_env()
    assert env["OPENAI_API_KEY"] == "k1"
    assert env["ANTHROPIC_API_KEY"] == "k2"


def test_write_env_drops_empty_values(isolated_config: Path) -> None:
    config_store.write_env({"OPENAI_API_KEY": "k1", "JUNK": ""})
    env = config_store.read_env()
    assert "OPENAI_API_KEY" in env
    assert "JUNK" not in env


# ---------------------------------------------------------------------------
# Ollama detection
# ---------------------------------------------------------------------------


def test_detect_ollama_returns_none_when_unreachable() -> None:
    with patch("banna_agent.cli.setup_wizard.requests.get",
               side_effect=Exception("connection refused")):
        assert setup_wizard._detect_ollama() is None


def test_detect_ollama_returns_models_list() -> None:
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "models": [
            {"name": "llama3.1:8b", "size": 4_700_000_000},
            {"name": "qwen2.5:14b", "size": 8_200_000_000},
        ]
    }
    with patch("banna_agent.cli.setup_wizard.requests.get",
               return_value=fake_response):
        models = setup_wizard._detect_ollama()
    assert models is not None
    assert len(models) == 2
    assert models[0]["name"] == "llama3.1:8b"


# ---------------------------------------------------------------------------
# Wizard happy paths
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_stdin(monkeypatch):
    """Replace sys.stdin with a StringIO seeded from a list of lines."""
    def _setup(answers: list[str]) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(answers) + "\n"))
    return _setup


def test_wizard_ollama_path_picks_model_and_saves_config(
    isolated_config: Path, fake_stdin, monkeypatch, capsys,
) -> None:
    fake_models = [
        {"name": "llama3.1:8b", "size": 4_700_000_000},
        {"name": "qwen2.5:14b", "size": 8_200_000_000},
    ]
    monkeypatch.setattr(setup_wizard, "_detect_ollama", lambda timeout_s=1.0: fake_models)
    # Provider=1 (Ollama, listed first when detected), model=2 (qwen2.5)
    fake_stdin(["1", "2"])
    result = setup_wizard.run_wizard()
    assert result.provider == "ollama"
    assert result.model == "qwen2.5:14b"
    assert result.api_key is None
    # config.toml written
    data = config_store.read_config()
    assert data["default"]["provider"] == "ollama"
    assert data["default"]["model"] == "qwen2.5:14b"


def test_wizard_openai_path_validates_key_and_saves_env(
    isolated_config: Path, fake_stdin, monkeypatch,
) -> None:
    # No Ollama detected → menu is OpenAI/Anthropic/Gemini/Ollama
    monkeypatch.setattr(setup_wizard, "_detect_ollama", lambda timeout_s=1.0: None)
    monkeypatch.setitem(setup_wizard._VALIDATORS, "openai",
                        lambda key: (True, ""))
    # Provider=1 (OpenAI when no Ollama), then paste key, then pick model=1
    fake_stdin(["1", "sk-test-good", "1"])
    result = setup_wizard.run_wizard()
    assert result.provider == "openai"
    assert result.api_key == "sk-test-good"
    env = config_store.read_env()
    assert env["OPENAI_API_KEY"] == "sk-test-good"
    # Also injected into current process so the REPL can start immediately.
    assert os.environ.get("OPENAI_API_KEY") == "sk-test-good"


def test_wizard_invalid_then_valid_key_retries(
    isolated_config: Path, fake_stdin, monkeypatch,
) -> None:
    monkeypatch.setattr(setup_wizard, "_detect_ollama", lambda timeout_s=1.0: None)
    calls = []
    def validator(key):
        calls.append(key)
        return (key == "sk-good", "AuthenticationError: bad key")
    monkeypatch.setitem(setup_wizard._VALIDATORS, "openai", validator)
    # Provider=1 OpenAI → bad key, retry y, good key → model=1
    fake_stdin(["1", "sk-bad", "y", "sk-good", "1"])
    result = setup_wizard.run_wizard()
    assert calls == ["sk-bad", "sk-good"]
    assert result.api_key == "sk-good"


def test_wizard_anthropic_path_uses_anthropic_validator(
    isolated_config: Path, fake_stdin, monkeypatch,
) -> None:
    monkeypatch.setattr(setup_wizard, "_detect_ollama", lambda timeout_s=1.0: None)
    monkeypatch.setitem(setup_wizard._VALIDATORS, "anthropic",
                        lambda k: (True, ""))
    # Provider=2 Anthropic (when no Ollama), key, model=1
    fake_stdin(["2", "ant-test", "1"])
    result = setup_wizard.run_wizard()
    assert result.provider == "anthropic"
    env = config_store.read_env()
    assert env["ANTHROPIC_API_KEY"] == "ant-test"


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------


def test_is_subcommand_recognizes_init_config_providers() -> None:
    assert subcommands.is_subcommand(["init"])
    assert subcommands.is_subcommand(["config", "get"])
    assert subcommands.is_subcommand(["providers"])
    assert not subcommands.is_subcommand(["--policy", "react"])
    assert not subcommands.is_subcommand([])


def test_config_set_then_get_roundtrips(
    isolated_config: Path, capsys,
) -> None:
    subcommands.dispatch(["config", "set", "provider", "openai"])
    subcommands.dispatch(["config", "set", "model", "gpt-5-nano"])
    rc = subcommands.dispatch(["config", "get", "provider"])
    out = capsys.readouterr().out
    assert "openai" in out.splitlines()[-1]
    assert rc == 0


def test_config_get_missing_key_returns_error(
    isolated_config: Path, capsys,
) -> None:
    config_store.write_config({"default": {"provider": "openai"}})
    rc = subcommands.dispatch(["config", "get", "model"])
    assert rc == 1


def test_config_path_prints_the_toml_path(
    isolated_config: Path, capsys,
) -> None:
    rc = subcommands.dispatch(["config", "path"])
    out = capsys.readouterr().out.strip()
    assert out.endswith("banna/config.toml")
    assert rc == 0


def test_providers_lists_each_provider(
    isolated_config: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config_store.write_env({"OPENAI_API_KEY": "sk-test"})
    with patch("banna_agent.cli.subcommands.requests.get",
               side_effect=Exception("nope")):
        subcommands.dispatch(["providers"])
    out = capsys.readouterr().out
    assert "openai" in out
    assert "anthropic" in out
    assert "gemini" in out
    assert "ollama" in out
    assert "(no key)" in out  # anthropic+gemini should be unset
