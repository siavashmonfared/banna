"""Interactive first-run setup wizard for `banna`.

Walks the user through:

  1. Picking a provider — Ollama (auto-detected at localhost:11434),
     OpenAI, Anthropic, or Gemini.
  2. For cloud providers: paste API key, validate with a 1-token test
     call, save to `~/.config/banna/.env` mode 0600.
  3. For Ollama: list installed models, let the user pick one.
  4. Pick a default model (curated short list per provider).
  5. Save `~/.config/banna/config.toml` and report the path.

All I/O is via stdin/stdout/stderr so the wizard works without a TTY
in test harnesses. The Rich console is used only for color when
available (falls back to plain prints if `console` is None).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import requests

from .config_store import write_config, write_env


# Curated short lists per provider — small enough to read at a glance,
# wide enough to cover "cheap & fast" vs "smart". Users can always
# override with `--model` later.
_PROVIDER_MODELS: dict[str, list[tuple[str, str]]] = {
    "openai": [
        ("gpt-5-nano", "fastest, cheapest"),
        ("gpt-4o-mini", "well-balanced"),
        ("gpt-4o", "most capable"),
    ],
    "anthropic": [
        ("claude-haiku-4-5-20251001", "fastest, cheapest"),
        ("claude-sonnet-4-6", "well-balanced"),
        ("claude-opus-4-7", "most capable"),
    ],
    "gemini": [
        ("gemini-2.0-flash", "fastest, cheapest"),
        ("gemini-2.0-pro", "well-balanced"),
        ("gemini-2.5-pro", "most capable"),
    ],
}

_PROVIDER_KEY_VAR: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


@dataclass
class WizardResult:
    """What the wizard captured. Returned to `main()`."""
    provider: str
    model: str
    api_key: str | None         # None for Ollama (no key needed)
    config_path: str
    env_path: str | None        # None when no key was saved


# ---------------------------------------------------------------------------
# Ollama auto-detect
# ---------------------------------------------------------------------------


def _detect_ollama(timeout_s: float = 1.0) -> list[dict[str, Any]] | None:
    """Return the list of Ollama models if a server responds, else None.

    Models are dicts with at least `name` (e.g. "llama3.1:8b") and
    usually `size` in bytes.
    """
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=timeout_s)
        if r.status_code != 200:
            return None
        data = r.json()
        models = data.get("models")
        if not isinstance(models, list):
            return None
        return models
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------


def _validate_openai_key(api_key: str) -> tuple[bool, str]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "."}],
            max_completion_tokens=1,
        )
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _validate_anthropic_key(api_key: str) -> tuple[bool, str]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "."}],
        )
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _validate_gemini_key(api_key: str) -> tuple[bool, str]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel("gemini-2.0-flash")
        m.generate_content(".", generation_config={"max_output_tokens": 1})
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


_VALIDATORS = {
    "openai": _validate_openai_key,
    "anthropic": _validate_anthropic_key,
    "gemini": _validate_gemini_key,
}


# ---------------------------------------------------------------------------
# Pure-IO helpers (so we can mock in tests)
# ---------------------------------------------------------------------------


def _ask(prompt: str, *, default: str | None = None) -> str:
    """Read one line from stdin with an optional default."""
    suffix = f" [{default}]" if default else ""
    sys.stdout.write(f"{prompt}{suffix} ")
    sys.stdout.flush()
    line = sys.stdin.readline()
    if not line:
        # EOF — treat as empty
        return default or ""
    line = line.rstrip("\n").strip()
    return line or (default or "")


def _ask_choice(prompt: str, n_options: int, *, default: int = 1) -> int:
    """Read a 1-indexed choice in [1, n_options]. Retries on bad input."""
    while True:
        raw = _ask(prompt, default=str(default))
        try:
            i = int(raw)
        except ValueError:
            print("  please enter a number")
            continue
        if 1 <= i <= n_options:
            return i
        print(f"  out of range; pick 1..{n_options}")


def _say(msg: str) -> None:
    print(msg)


# ---------------------------------------------------------------------------
# Provider sub-flows
# ---------------------------------------------------------------------------


def _ollama_flow(models: list[dict[str, Any]]) -> tuple[str, str]:
    """Returns (provider, model). Lets the user pick from installed Ollama models."""
    _say("\n  Ollama detected at localhost:11434. Installed models:")
    for i, m in enumerate(models, start=1):
        name = m.get("name", "?")
        size_b = m.get("size", 0)
        size_gb = f"{size_b / 1e9:.1f} GB" if size_b else ""
        _say(f"    {i}. {name}    {size_gb}".rstrip())
    idx = _ask_choice("\n  Pick a model:", n_options=len(models), default=1)
    chosen = models[idx - 1].get("name", "")
    return "ollama", chosen


def _cloud_flow(provider: str) -> tuple[str, str, str]:
    """Returns (provider, model, api_key). Loops on validation failure."""
    var = _PROVIDER_KEY_VAR[provider]
    _say(f"\n  Paste your {var}. (input is echoed; it'll be saved to")
    _say("  ~/.config/banna/.env with mode 0600)")
    while True:
        key = _ask("  >").strip()
        if not key:
            _say("  empty input; aborting")
            sys.exit(2)
        _say("  validating key with a 1-token test call…")
        ok, err = _VALIDATORS[provider](key)
        if ok:
            _say("  ✓ key works")
            break
        _say(f"  ✗ rejected: {err}")
        retry = _ask("  try again? (y/n)", default="y").lower()
        if not retry.startswith("y"):
            sys.exit(2)

    # Pick a model.
    options = _PROVIDER_MODELS[provider]
    _say(f"\n  Pick a default model for {provider}:")
    for i, (name, label) in enumerate(options, start=1):
        _say(f"    {i}. {name}    ({label})")
    idx = _ask_choice("\n  Model:", n_options=len(options), default=1)
    model = options[idx - 1][0]
    return provider, model, key


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_wizard() -> WizardResult:
    """Run the interactive first-run wizard. Saves config + .env, returns
    a `WizardResult`."""
    _say("\n● banna — first-run setup")
    _say("  No LLM provider configured. Let's pick one.\n")

    # Build the menu. Ollama appears first if detected so the obvious
    # zero-cost path is also the most prominent.
    ollama_models = _detect_ollama()
    options: list[tuple[str, str]] = []
    if ollama_models:
        options.append(("ollama", f"Ollama (local, {len(ollama_models)} model"
                                  f"{'s' if len(ollama_models) != 1 else ''} installed)"))
    options.extend([
        ("openai", "OpenAI       (cloud, paid)"),
        ("anthropic", "Anthropic    (cloud, paid)"),
        ("gemini", "Gemini       (cloud, free tier)"),
    ])
    if not ollama_models:
        options.append(("ollama", "Ollama       (local, not detected at :11434)"))

    for i, (_, label) in enumerate(options, start=1):
        _say(f"  {i}. {label}")
    idx = _ask_choice("\nProvider:", n_options=len(options), default=1)
    provider = options[idx - 1][0]

    # Dispatch.
    if provider == "ollama":
        if not ollama_models:
            _say("\n  Ollama isn't running at localhost:11434.")
            _say("  Install: https://ollama.com/  →  then `ollama serve` + `ollama pull llama3.1:8b`")
            _say("  Re-run `banna init` once that's set up.")
            sys.exit(2)
        provider, model = _ollama_flow(ollama_models)
        api_key = None
    else:
        provider, model, api_key = _cloud_flow(provider)

    # Persist.
    config_path = write_config({"default": {
        "provider": provider,
        "model": model,
        "policy": "verifier_retry",
    }})
    env_path = None
    if api_key:
        env_path = write_env({_PROVIDER_KEY_VAR[provider]: api_key})
        # Also load into the current process so the REPL starts working
        # immediately without a restart.
        os.environ[_PROVIDER_KEY_VAR[provider]] = api_key

    _say("")
    _say(f"  ✓ saved {config_path}")
    if env_path:
        _say(f"  ✓ saved {env_path} (mode 0600)")
    _say(f"  ✓ ready. running banna with provider={provider} model={model}\n")

    return WizardResult(
        provider=provider,
        model=model,
        api_key=api_key,
        config_path=str(config_path),
        env_path=str(env_path) if env_path else None,
    )
