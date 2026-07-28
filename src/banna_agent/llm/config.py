"""Provider configuration & credential resolution.

Resolves API keys, regions, and base URLs from a small, documented set of
environment variables. One object per provider — passed to the adapter at
construction time. Separates "where does the key come from?" from "how is
the request shaped?" so the adapters stay thin.

Env var summary (see `.env.example`):
    ANTHROPIC_API_KEY          direct Anthropic
    AWS_REGION / AWS_PROFILE   Bedrock (via boto3 chain); optional explicit
                                ANTHROPIC_BEDROCK_BASE_URL for custom endpoints
    OPENAI_API_KEY             OpenAI; optional OPENAI_BASE_URL for proxies
    GOOGLE_API_KEY             Gemini (also used by tools/search.py)
    OLLAMA_BASE_URL            Ollama local server (default http://localhost:11434)
    MOONSHOT_API_KEY           Kimi / Moonshot (OpenAI-compatible)
    ZAI_API_KEY / ZHIPU_API_KEY GLM / Z.ai / Zhipu (OpenAI-compatible)
    HF_TOKEN                   Hugging Face router (OpenAI-compatible; hosts
                                Inkling and many open-weights models)

The last three providers are OpenAI wire-compatible, so they reuse the
OpenAI adapter with a preset base_url and their own key. Each base_url is
overridable via a `<PROVIDER>_BASE_URL` env var for proxies / regional
endpoints (e.g. Moonshot's `.cn` domain).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class AnthropicConfig:
    """Direct Anthropic API configuration."""
    api_key: str | None = None
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "AnthropicConfig":
        return cls(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        )


@dataclass
class BedrockConfig:
    """AWS Bedrock-hosted Anthropic models configuration."""
    aws_region: str | None = None
    aws_profile: str | None = None
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "BedrockConfig":
        return cls(
            aws_region=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
            aws_profile=os.environ.get("AWS_PROFILE"),
            base_url=os.environ.get("ANTHROPIC_BEDROCK_BASE_URL"),
        )


@dataclass
class OpenAIConfig:
    """OpenAI / OpenAI-compat configuration."""
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            organization=os.environ.get("OPENAI_ORG") or os.environ.get("OPENAI_ORGANIZATION"),
        )


@dataclass
class GeminiConfig:
    """Gemini via generativelanguage.googleapis.com."""
    api_key: str | None = None
    base_url: str = "https://generativelanguage.googleapis.com"

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        return cls(
            api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_SEARCH_API_KEY"),
            base_url=os.environ.get("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com",
        )


@dataclass
class OllamaConfig:
    """Ollama local-server configuration."""
    base_url: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        return cls(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))


# OpenAI-compatible third-party providers: (canonical key vars, base_url
# env var, default base_url). All three speak the OpenAI Chat Completions
# wire format, so they reuse OpenAIConfig + OpenAIClient with a preset
# base_url. The base is overridable via `<PROVIDER>_BASE_URL`.
_OPENAI_COMPAT: dict[str, tuple[tuple[str, ...], str, str]] = {
    "kimi": (("MOONSHOT_API_KEY", "KIMI_API_KEY"),
             "MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1"),
    "glm": (("ZAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"),
            "GLM_BASE_URL", "https://api.z.ai/api/paas/v4"),
    # Hugging Face inference router: one token reaches many hosted
    # open-weights models (incl. Thinking Machines' Inkling). The `:auto`
    # model suffix lets HF pick an available inference provider.
    "huggingface": (("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN"),
                    "HF_BASE_URL", "https://router.huggingface.co/v1"),
}


def _openai_compat_config(provider: str) -> OpenAIConfig:
    key_vars, base_env, default_base = _OPENAI_COMPAT[provider]
    api_key = next((os.environ[v] for v in key_vars if os.environ.get(v)), None)
    return OpenAIConfig(
        api_key=api_key,
        base_url=os.environ.get(base_env) or default_base,
    )


def resolve_provider_config(provider: str) -> Any:
    """Build the right config object for a provider name."""
    p = provider.lower()
    if p == "anthropic":
        return AnthropicConfig.from_env()
    if p == "bedrock":
        return BedrockConfig.from_env()
    if p == "openai":
        return OpenAIConfig.from_env()
    if p == "gemini":
        return GeminiConfig.from_env()
    if p == "ollama":
        return OllamaConfig.from_env()
    if p in _OPENAI_COMPAT:
        return _openai_compat_config(p)
    raise ValueError(f"unknown provider: {provider!r}")
