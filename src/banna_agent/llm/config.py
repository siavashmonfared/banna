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
    raise ValueError(f"unknown provider: {provider!r}")
