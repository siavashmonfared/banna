"""Persistent config for the `banna` CLI.

Two files under `~/.config/banna/` (XDG_CONFIG_HOME honored if set):

  - `config.toml`   non-secret defaults (provider, model, policy)
  - `.env`          API keys, mode 0600

A config is considered to "exist" iff `config.toml` is present. That
single check drives the first-run wizard trigger, regardless of whether
the user happens to have keys exported in their shell env.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib


def config_dir() -> Path:
    """Return `$XDG_CONFIG_HOME/banna` (or `~/.config/banna`)."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "banna"


def config_toml_path() -> Path:
    return config_dir() / "config.toml"


def env_path() -> Path:
    return config_dir() / ".env"


def is_first_run() -> bool:
    """True iff `~/.config/banna/config.toml` does not exist."""
    return not config_toml_path().is_file()


def read_config() -> dict:
    """Return the contents of `config.toml`, or `{}` if missing/broken."""
    p = config_toml_path()
    if not p.is_file():
        return {}
    try:
        return tomllib.loads(p.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def write_config(data: dict) -> Path:
    """Write `data` to `config.toml`, replacing prior content.

    The file is small + human-readable. We hand-render rather than
    pulling in a third-party TOML writer.
    """
    p = config_toml_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for section, body in data.items():
        if isinstance(body, dict):
            lines.append(f"[{section}]")
            for k, v in body.items():
                lines.append(f"{k} = {_render(v)}")
            lines.append("")
        else:
            lines.append(f"{section} = {_render(body)}")
    p.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return p


def _render(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return '"' + str(v).replace('\\', '\\\\').replace('"', '\\"') + '"'


def read_package_allowlist() -> dict[str, str]:
    """Return the sandbox package allowlist (`import_name -> "dist==ver"`).

    Backed by the `[packages]` section of `config.toml`; `{}` if unset.
    """
    pkgs = read_config().get("packages") or {}
    return {k: v for k, v in pkgs.items() if isinstance(v, str)}


def write_package_allowlist(entries: dict[str, str], *, replace: bool = False) -> Path:
    """Persist allowlist `entries` into the `[packages]` section.

    By default `entries` are merged into the existing allowlist; pass
    `replace=True` to overwrite it wholesale (used by `remove`). Other config
    sections (e.g. `[default]`) are preserved.
    """
    data = read_config()
    pkgs = {} if replace else dict(data.get("packages") or {})
    pkgs.update(entries)
    data["packages"] = pkgs
    return write_config(data)


def read_env() -> dict[str, str]:
    """Parse `~/.config/banna/.env` into a `{KEY: VALUE}` dict."""
    p = env_path()
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        out[k] = v
    return out


def write_env(values: dict[str, str]) -> Path:
    """Write `values` to `~/.config/banna/.env`, mode 0600.

    Existing entries are merged: keys in `values` overwrite; keys not
    in `values` are preserved.
    """
    p = env_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    current = read_env()
    current.update({k: v for k, v in values.items() if v})
    lines = [f"{k}={v}" for k, v in sorted(current.items())]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass  # Windows / unusual fs; not fatal
    return p
