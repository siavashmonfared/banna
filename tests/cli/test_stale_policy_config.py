"""A saved config naming a policy this build doesn't expose must not crash
the CLI — it should warn and fall back to a valid default."""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # No provider keys → but we never reach LLM build; we stop at policy
    # resolution. Reload config_store so it reads the patched config dir.
    from banna_agent.cli import config_store
    importlib.reload(config_store)
    return config_store


def _write_config(config_store, policy: str):
    config_store.write_config({"default": {
        "provider": "openai", "model": "gpt-5-nano", "policy": policy}})


def test_stale_policy_falls_back(isolated, monkeypatch, capsys):
    from banna_agent.cli import app as app_mod
    importlib.reload(app_mod)
    _write_config(isolated, "definitely_not_a_policy")

    captured = {}

    class _Console:
        def print(self, *a, **k): pass

    class _StubApp:
        def __init__(self, **kw):
            captured.update(kw)
            self.console = _Console()
            self.session = None
        def rebuild_llm(self): pass
        def rebuild_tools(self): pass
        def rebuild_policy(self): pass
        def run(self): return 0

    monkeypatch.setattr(app_mod, "MyAgentApp", _StubApp)
    # No --policy flag → config's stale value is read, then must be rejected.
    rc = app_mod.main([])
    assert rc == 0
    # Fell back to a name the build actually exposes.
    from banna_agent.cli.commands import POLICY_NAMES
    assert captured["policy_name"] in POLICY_NAMES
    err = capsys.readouterr().err
    assert "not available in this build" in err


def test_valid_policy_passes_through(isolated, monkeypatch):
    from banna_agent.cli import app as app_mod
    importlib.reload(app_mod)
    from banna_agent.cli.commands import POLICY_NAMES
    _write_config(isolated, POLICY_NAMES[0])

    captured = {}

    class _Console:
        def print(self, *a, **k): pass

    class _StubApp:
        def __init__(self, **kw):
            captured.update(kw)
            self.console = _Console()
            self.session = None
        def rebuild_llm(self): pass
        def rebuild_tools(self): pass
        def rebuild_policy(self): pass
        def run(self): return 0

    monkeypatch.setattr(app_mod, "MyAgentApp", _StubApp)
    rc = app_mod.main([])
    assert rc == 0
    assert captured["policy_name"] == POLICY_NAMES[0]
