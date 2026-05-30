"""Top-level subcommands for the `banna` CLI: init / config / providers.

These are dispatched manually before argparse runs, so the legacy
`banna --policy X` invocation (no subcommand) keeps working untouched.
"""
from __future__ import annotations

import sys
from typing import Sequence

import requests

from .config_store import (
    config_toml_path,
    env_path,
    read_config,
    read_env,
    read_package_allowlist,
    write_config,
    write_package_allowlist,
)
from .setup_wizard import _PROVIDER_KEY_VAR, _VALIDATORS, run_wizard


SUBCOMMANDS = ("init", "config", "providers")


def is_subcommand(argv: Sequence[str]) -> bool:
    return bool(argv) and argv[0] in SUBCOMMANDS


def dispatch(argv: Sequence[str]) -> int:
    name = argv[0]
    rest = list(argv[1:])
    if name == "init":
        return _cmd_init(rest)
    if name == "config":
        return _cmd_config(rest)
    if name == "providers":
        return _cmd_providers(rest)
    print(f"unknown subcommand: {name}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# init — re-run the wizard
# ---------------------------------------------------------------------------


def _cmd_init(args: list[str]) -> int:
    """`banna init` — (re-)run the interactive setup wizard."""
    run_wizard()
    return 0


# ---------------------------------------------------------------------------
# config — get / set fields in config.toml
# ---------------------------------------------------------------------------


_CONFIG_USAGE = (
    "usage:\n"
    "  banna config get [key]         show the whole config, or a single key\n"
    "  banna config set <key> <value> set a key (e.g. `banna config set provider openai`)\n"
    "  banna config path              print the config file path\n"
    "  banna config packages ...      manage the docker-sandbox install allowlist"
)

_PACKAGES_USAGE = (
    "usage:\n"
    "  banna config packages list                       show the allowlist\n"
    "  banna config packages add <import> <dist==ver>   allow a package\n"
    "                                                   (e.g. `add cv2 opencv-python==4.10.0.84`)\n"
    "  banna config packages remove <import>            remove a package"
)


def _cmd_config(args: list[str]) -> int:
    if not args:
        print(_CONFIG_USAGE)
        return 0
    sub = args[0]

    if sub == "packages":
        return _cmd_config_packages(args[1:])

    if sub == "path":
        print(config_toml_path())
        return 0

    if sub == "get":
        data = read_config()
        if len(args) >= 2:
            key = args[1]
            val = data.get("default", {}).get(key)
            if val is None:
                print(f"(unset: {key})", file=sys.stderr)
                return 1
            print(val)
            return 0
        # Dump the whole thing.
        if not data:
            print("(no config yet — run `banna init`)", file=sys.stderr)
            return 1
        for section, body in data.items():
            print(f"[{section}]")
            if isinstance(body, dict):
                for k, v in body.items():
                    print(f"  {k} = {v}")
        return 0

    if sub == "set":
        if len(args) < 3:
            print("usage: banna config set <key> <value>", file=sys.stderr)
            return 2
        key, value = args[1], args[2]
        data = read_config()
        data.setdefault("default", {})[key] = value
        path = write_config(data)
        print(f"set default.{key} = {value!r}  →  {path}")
        return 0

    print(_CONFIG_USAGE, file=sys.stderr)
    return 2


def _cmd_config_packages(args: list[str]) -> int:
    """`banna config packages {list,add,remove}` — docker-sandbox allowlist."""
    verb = args[0] if args else "list"

    if verb == "list":
        from ..tools.package_policy import default_allowlist
        defaults = default_allowlist()
        user = read_package_allowlist()
        print("built-in defaults (trusted, installed with no prompt):")
        for name, spec in sorted(defaults.items()):
            override = f"  → overridden: {user[name]}" if name in user else ""
            print(f"  {name} = {spec}{override}")
        extra = {k: v for k, v in user.items() if k not in defaults}
        print("\nyour additions:")
        if extra:
            for name, spec in sorted(extra.items()):
                print(f"  {name} = {spec}")
        else:
            print("  (none — `banna config packages add <import> <dist==ver>`)")
        return 0

    if verb == "add":
        if len(args) < 3:
            print("usage: banna config packages add <import_name> <dist==version>",
                  file=sys.stderr)
            return 2
        import_name, spec = args[1], args[2]
        path = write_package_allowlist({import_name: spec})
        print(f"allowlisted {import_name} = {spec!r}  →  {path}")
        return 0

    if verb == "remove":
        if len(args) < 2:
            print("usage: banna config packages remove <import_name>",
                  file=sys.stderr)
            return 2
        import_name = args[1]
        pkgs = read_package_allowlist()
        if import_name not in pkgs:
            print(f"(not in allowlist: {import_name})", file=sys.stderr)
            return 1
        del pkgs[import_name]
        write_package_allowlist(pkgs, replace=True)
        print(f"removed {import_name}")
        return 0

    print(_PACKAGES_USAGE, file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# providers — list configured providers + key validation status
# ---------------------------------------------------------------------------


def _cmd_providers(args: list[str]) -> int:
    """`banna providers` — show which providers have a key and whether it works."""
    import os
    env_file = read_env()

    rows = []
    for provider, var in _PROVIDER_KEY_VAR.items():
        # Key may live in shell env OR our .env file.
        src = None
        if os.environ.get(var):
            src = "shell"
        elif env_file.get(var):
            src = str(env_path())
        rows.append((provider, var, src))

    # Ollama is special — no key, just a server check.
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.0)
        ollama_status = "up" if r.status_code == 200 else "responding but not 200"
    except Exception:
        ollama_status = "not detected"

    width = max(len(p) for p, _, _ in rows) + 2
    print()
    for provider, var, src in rows:
        if src is None:
            status = "(no key)"
        else:
            status = f"key from {src}"
            if "--validate" in args:
                ok, err = _VALIDATORS[provider]((os.environ.get(var) or env_file.get(var)) or "")
                status += "  ✓ live" if ok else f"  ✗ {err[:60]}"
        print(f"  {provider:<{width}} {status}")
    print(f"  {'ollama':<{width}} {ollama_status}")
    print()
    print("hint: pass --validate to make a 1-token test call against each cloud provider")
    return 0
