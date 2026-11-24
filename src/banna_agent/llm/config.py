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
    OPENROUTER_API_KEY         OpenRouter (OpenAI-compatible; optional
                                OPENROUTER_BASE_URL)
    OLLAMA_BASE_URL            Ollama local server (default http://localhost:11434)

OpenRouter speaks the OpenAI wire format, so it reuses the OpenAI adapter
with a preset base_url and its own key; OPENROUTER_BASE_URL overrides the
base for proxies.
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
    # `None` lets the adapter size the context window from each request.
    # Set an int (or OLLAMA_NUM_CTX) to pin it — worth doing when VRAM is
    # tight and you'd rather cap the KV cache than fit the whole prompt.
    num_ctx: int | None = None

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        raw = os.environ.get("OLLAMA_NUM_CTX") or os.environ.get(
            "OLLAMA_CONTEXT_LENGTH")
        try:
            num_ctx = int(raw) if raw else None
        except ValueError:
            num_ctx = None
        return cls(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            num_ctx=num_ctx,
        )


# OpenAI-compatible providers: (key vars, base_url env var, default
# base_url). They speak the OpenAI Chat Completions wire format, so they
# reuse OpenAIConfig + OpenAIClient with a preset base_url.
_OPENAI_COMPAT: dict[str, tuple[tuple[str, ...], str, str]] = {
    # OpenRouter: one key reaches models from many vendors, ids are
    # `vendor/model` (e.g. anthropic/claude-haiku-4.5).
    "openrouter": (("OPENROUTER_API_KEY",),
                   "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
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
