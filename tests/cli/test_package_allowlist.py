"""Persistence for the docker-sandbox package allowlist (`[packages]`)."""
from __future__ import annotations

import pytest


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from banna_agent.cli import config_store
    return config_store


def test_empty_when_unset(cfg) -> None:
    assert cfg.read_package_allowlist() == {}


def test_add_merges_and_roundtrips(cfg) -> None:
    cfg.write_package_allowlist({"cv2": "opencv-python==4.10.0.84"})
    cfg.write_package_allowlist({"numpy": "numpy==2.1.1"})
    assert cfg.read_package_allowlist() == {
        "cv2": "opencv-python==4.10.0.84",
        "numpy": "numpy==2.1.1",
    }


def test_replace_removes(cfg) -> None:
    cfg.write_package_allowlist({"cv2": "opencv-python==4.10.0.84",
                                 "numpy": "numpy==2.1.1"})
    pk = cfg.read_package_allowlist()
    del pk["cv2"]
    cfg.write_package_allowlist(pk, replace=True)
    assert cfg.read_package_allowlist() == {"numpy": "numpy==2.1.1"}


def test_preserves_other_sections(cfg) -> None:
    cfg.write_config({"default": {"provider": "openai", "model": "gpt-5-nano"}})
    cfg.write_package_allowlist({"cv2": "opencv-python==4.10.0.84"})
    data = cfg.read_config()
    assert data["default"]["provider"] == "openai"
    assert data["packages"]["cv2"] == "opencv-python==4.10.0.84"


# --- built-in default allowlist ---------------------------------------------

def test_default_allowlist_is_pinned_and_copied() -> None:
    from banna_agent.tools.package_policy import DEFAULT_ALLOWLIST, default_allowlist
    a = default_allowlist()
    assert a == DEFAULT_ALLOWLIST and a is not DEFAULT_ALLOWLIST  # fresh copy
    a["numpy"] = "tampered"
    assert DEFAULT_ALLOWLIST["numpy"] != "tampered"  # mutation doesn't leak
    # Common stack is present and every entry is version-pinned.
    for name in ("numpy", "pandas", "scipy", "matplotlib", "sklearn"):
        assert "==" in DEFAULT_ALLOWLIST[name]


def test_default_allowlist_maps_import_to_distribution() -> None:
    from banna_agent.tools.package_policy import DEFAULT_ALLOWLIST
    # Import name != pip distribution name for these.
    assert DEFAULT_ALLOWLIST["sklearn"].startswith("scikit-learn==")
    assert DEFAULT_ALLOWLIST["cv2"].startswith("opencv-python-headless==")
    assert DEFAULT_ALLOWLIST["PIL"].startswith("pillow==")
    assert DEFAULT_ALLOWLIST["bs4"].startswith("beautifulsoup4==")
    assert DEFAULT_ALLOWLIST["yaml"].startswith("pyyaml==")


def test_user_config_overrides_default(cfg) -> None:
    # The merge the CLI performs: defaults first, user config on top.
    from banna_agent.tools.package_policy import PackagePolicy, default_allowlist
    cfg.write_package_allowlist({"numpy": "numpy==1.26.4",        # override a default
                                 "cowsay": "cowsay==6.1"})        # extend with new
    merged = {**default_allowlist(), **cfg.read_package_allowlist()}
    p = PackagePolicy(allowlist=merged)
    assert p.resolve("numpy") == "numpy==1.26.4"      # user pin wins
    assert p.resolve("cowsay") == "cowsay==6.1"        # user addition
    assert p.resolve("pandas").startswith("pandas==")  # untouched default
