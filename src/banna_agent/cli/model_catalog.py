"""Curated model catalog + provider credential scanning.

Single source of truth shared by the launch TUI (`setup_tui`), the
`/model` and `/provider` slash commands (`commands.py`), and the
non-TTY first-run wizard (`setup_wizard.py`):

  - `CURATED`      per-provider (model, blurb) shortlists — hand-kept,
                   correct API model ids, cheap→capable order
  - `KEY_VARS`     which env vars satisfy each provider (first is
                   canonical; any of them counts as "key found")
  - `scan_providers()`  live status per provider: key found + where
                   (shell env / ./.env / ~/.config/banna/.env), Ollama
                   server reachability + installed models, Bedrock AWS
                   credential presence
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config_store import env_path

# ---------------------------------------------------------------------------
# Curated model shortlists
# ---------------------------------------------------------------------------

# (model_id, blurb) per provider, ordered cheap/fast → most capable.
# Ollama is queried live via /api/tags; bedrock ids mirror the anthropic
# list through cross-region inference profiles.
CURATED: dict[str, tuple[tuple[str, str], ...]] = {
    "anthropic": (
        ("claude-haiku-4-5-20251001", "fastest, cheapest"),
        ("claude-sonnet-5", "well-balanced"),
        ("claude-opus-4-8", "most capable"),
        ("claude-fable-5", "frontier (Claude 5 family)"),
    ),
    "openai": (
        ("gpt-5-nano", "fastest, cheapest"),
        ("gpt-5-mini", "well-balanced"),
        ("gpt-5", "most capable"),
        ("o4-mini", "reasoning, budget"),
    ),
    "gemini": (
        ("gemini-2.0-flash", "fastest, cheapest"),
        ("gemini-2.5-flash", "well-balanced"),
        ("gemini-2.5-pro", "most capable"),
    ),
    "bedrock": (
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "fastest, cheapest (US profile)"),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "well-balanced (US profile)"),
        ("us.anthropic.claude-opus-4-5-20251101-v1:0", "most capable (US profile)"),
        ("global.anthropic.claude-sonnet-4-5-20250929-v1:0", "global failover profile"),
    ),
    "ollama": (),  # populated live from /api/tags
    # OpenAI-compatible third-party providers (reuse the OpenAI adapter).
    "kimi": (
        ("kimi-k2.5", "fast, cheap"),
        ("kimi-k2.6", "well-balanced"),
        ("kimi-k3", "most capable"),
        ("moonshot-v1-128k", "long-context"),
    ),
    "glm": (
        ("glm-4.5-air", "fast, cheap"),
        ("glm-4.5", "well-balanced"),
        ("glm-4.6", "most capable"),
    ),
    # Hugging Face router hosts many open-weights models. Use a bare repo
    # id (append `:provider` to pin a specific inference provider). Paste
    # any HF model id via the `/model` "type a name" row; see the live
    # list at https://router.huggingface.co/v1/models.
    "huggingface": (
        ("thinkingmachines/Inkling", "Thinking Machines Inkling"),
        ("moonshotai/Kimi-K3", "Kimi K3 (open weights)"),
        ("Qwen/Qwen3.6-27B", "Qwen 3.6 27B"),
    ),
}

# Extended bedrock id list kept for the `/model` picker — regional and
# legacy ids beyond the curated four above.
BEDROCK_EXTRA_MODELS: tuple[str, ...] = (
    "global.anthropic.claude-opus-4-5-20251101-v1:0",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    "apac.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "apac.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
)


def known_models(provider: str) -> tuple[str, ...]:
    """Flat model-id tuple for a provider (curated + bedrock extras)."""
    base = tuple(m for m, _ in CURATED.get(provider, ()))
    if provider == "bedrock":
        return base + BEDROCK_EXTRA_MODELS
    return base


# ---------------------------------------------------------------------------
# Credential env vars
# ---------------------------------------------------------------------------

# Env vars that satisfy each cloud provider. First entry is canonical
# (what we write when saving a pasted key). Gemini historically split
# between GEMINI_API_KEY (wizard) and GOOGLE_API_KEY (llm/config.py
# reads it) — both are accepted, and saving a key writes both so either
# consumer works.
KEY_VARS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "kimi": ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
    "glm": ("ZAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"),
    "huggingface": ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN"),
}

# Human labels for the launch TUI. Internal provider names stay as the
# registry knows them ("gemini"), display adds the vendor.
PROVIDER_LABELS: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "google (gemini)",
    "ollama": "ollama (local)",
    "bedrock": "bedrock (AWS)",
    "kimi": "kimi (moonshot)",
    "glm": "glm (z.ai)",
    "huggingface": "hugging face (router)",
}

# Order the launch TUI lists providers in.
PROVIDER_ORDER: tuple[str, ...] = (
    "anthropic", "openai", "gemini", "ollama", "bedrock",
    "kimi", "glm", "huggingface",
)


# ---------------------------------------------------------------------------
# Live provider scan
# ---------------------------------------------------------------------------


@dataclass
class ProviderStatus:
    """One provider's readiness for the launch TUI."""
    name: str                       # registry name ("gemini", not "google")
    ok: bool                        # usable right now
    detail: str                     # "ANTHROPIC_API_KEY", "4 models", reason if not ok
    source: str = ""                # where the key was found ("shell env", a path)
    ollama_models: list[dict[str, Any]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return PROVIDER_LABELS.get(self.name, self.name)


def _parse_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE parser matching config_store.read_env, any path."""
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        out[k] = v
    return out


def key_search_paths() -> list[tuple[str, Path | None]]:
    """Where keys are looked for, in load order. (label, path|None=shell)."""
    return [
        ("shell env", None),
        ("./.env", Path.cwd() / ".env"),
        (str(env_path()), env_path()),
    ]


def find_key(provider: str) -> tuple[str, str] | None:
    """Return (env_var, source_label) for the first key found, else None.

    Files are checked for *which* source supplied the value; the actual
    value the client uses always comes from os.environ (loaded by
    `_load_dotenv` at startup, shell env winning).
    """
    vars_ = KEY_VARS.get(provider, ())
    file_maps = [
        (label, _parse_env_file(p)) for label, p in key_search_paths() if p is not None
    ]
    for var in vars_:
        if not os.environ.get(var):
            # Not loaded into the process — still report a file hit so the
            # TUI can say "found in X (restart load)" rather than "missing".
            for label, kv in file_maps:
                if kv.get(var):
                    return var, label
            continue
        for label, kv in file_maps:
            if kv.get(var) == os.environ[var]:
                return var, label
        return var, "shell env"
    return None


def detect_ollama(timeout_s: float = 1.0) -> list[dict[str, Any]] | None:
    """List of installed Ollama models if the server responds, else None."""
    from .setup_wizard import _detect_ollama
    return _detect_ollama(timeout_s=timeout_s)


def _ollama_status(*, timeout_s: float = 1.0) -> ProviderStatus:
    """Probe the Ollama daemon and build its ProviderStatus."""
    host = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    host_disp = host.split("://", 1)[-1].rstrip("/")
    models = detect_ollama(timeout_s=timeout_s)
    if models:
        return ProviderStatus(
            "ollama", True,
            f"{host_disp} · {len(models)} model{'s' if len(models) != 1 else ''}",
            ollama_models=models)
    return ProviderStatus("ollama", False, f"not running at {host_disp}")


def refresh_ollama(scan: dict[str, ProviderStatus], *,
                   timeout_s: float = 1.0) -> ProviderStatus:
    """Re-probe the Ollama daemon and update `scan` in place.

    The launch TUI scans providers once at startup; a daemon started
    (or a model pulled) after that would otherwise stay invisible for
    the whole session. Callers re-probe just before they need the
    model list.
    """
    st = _ollama_status(timeout_s=timeout_s)
    scan["ollama"] = st
    return st


def scan_providers(*, ollama_timeout_s: float = 1.0) -> dict[str, ProviderStatus]:
    """Live readiness scan across all providers, in PROVIDER_ORDER."""
    out: dict[str, ProviderStatus] = {}
    for name in PROVIDER_ORDER:
        if name == "ollama":
            out[name] = _ollama_status(timeout_s=ollama_timeout_s)
        elif name == "bedrock":
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            profile = os.environ.get("AWS_PROFILE")
            if region or profile:
                bits = [b for b in (
                    f"AWS_REGION={region}" if region else "",
                    f"AWS_PROFILE={profile}" if profile else "") if b]
                out[name] = ProviderStatus(name, True, " · ".join(bits), source="shell env")
            else:
                out[name] = ProviderStatus(name, False, "no AWS_REGION / AWS_PROFILE")
        else:
            hit = find_key(name)
            if hit:
                var, source = hit
                out[name] = ProviderStatus(name, True, var, source=source)
            else:
                out[name] = ProviderStatus(
                    name, False, "missing — " + " / ".join(KEY_VARS[name]))
    return out


def save_api_key(provider: str, key: str) -> Path:
    """Persist `key` to ~/.config/banna/.env (mode 0600) and os.environ.

    Writes *every* accepted var for the provider (gemini gets both
    GEMINI_API_KEY and GOOGLE_API_KEY) so all consumers agree.
    """
    from .config_store import write_env
    values = {var: key for var in KEY_VARS.get(provider, ())}
    path = write_env(values)
    for var in values:
        os.environ[var] = key
    return path


def validate_api_key(provider: str, key: str) -> tuple[bool, str]:
    """1-token live validation. Returns (ok, error_message)."""
    from .setup_wizard import _VALIDATORS
    fn = _VALIDATORS.get(provider)
    if fn is None:
        return True, ""
    return fn(key)
