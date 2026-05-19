"""Provider registry and factory.

Usage:

    from banna_agent.llm.registry import make_client, list_providers

    client = make_client("anthropic", model="claude-opus-4-7")
    client = make_client("bedrock", model="us.anthropic.claude-opus-4-7-v1:0")
    client = make_client("openai",  model="gpt-5.5")
    client = make_client("gemini",  model="gemini-3.1-pro-preview")
    client = make_client("ollama",  model="qwen3:8b")

The factory reads env vars via `llm/config.py`. Pass `env_override=` to
inject a specific config (useful for tests and multi-account setups).

The registry also lets new adapters register themselves at import time:

    from banna_agent.llm.registry import register_client
    register_client("my_provider", MyClient)
"""
from __future__ import annotations

from typing import Any, Callable

from .anthropic import AnthropicClient
from .base import LLMClient
from .config import (
    AnthropicConfig,
    BedrockConfig,
    GeminiConfig,
    OllamaConfig,
    OpenAIConfig,
    resolve_provider_config,
)
from .gemini import GeminiClient
from .ollama import OllamaClient
from .openai import OpenAIClient

# Factory signature: (config, **kwargs) -> LLMClient
ClientFactory = Callable[[Any], LLMClient]


def _anthropic_factory(cfg: AnthropicConfig, **kw: Any) -> LLMClient:
    return AnthropicClient(
        model=kw.get("model", "claude-opus-4-5-20251101"),
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        system_default=kw.get("system_default"),
        transport="direct",
    )


def _bedrock_factory(cfg: BedrockConfig, **kw: Any) -> LLMClient:
    # Sensible default: latest Anthropic inference-profile id
    default_model = "us.anthropic.claude-opus-4-5-20251101-v1:0"
    return AnthropicClient(
        model=kw.get("model", default_model),
        transport="bedrock",
        aws_region=cfg.aws_region,
        aws_profile=cfg.aws_profile,
        base_url=cfg.base_url,
        system_default=kw.get("system_default"),
    )


def _openai_factory(cfg: OpenAIConfig, **kw: Any) -> LLMClient:
    return OpenAIClient(
        model=kw.get("model", "gpt-5-mini"),
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        organization=cfg.organization,
        system_default=kw.get("system_default"),
    )


def _gemini_factory(cfg: GeminiConfig, **kw: Any) -> LLMClient:
    return GeminiClient(
        model=kw.get("model", "gemini-2.5-pro"),
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        system_default=kw.get("system_default"),
    )


def _ollama_factory(cfg: OllamaConfig, **kw: Any) -> LLMClient:
    return OllamaClient(
        model=kw.get("model", "qwen2.5:7b"),
        base_url=cfg.base_url,
        system_default=kw.get("system_default"),
    )


_REGISTRY: dict[str, ClientFactory] = {
    "anthropic": _anthropic_factory,
    "bedrock": _bedrock_factory,
    "openai": _openai_factory,
    "gemini": _gemini_factory,
    "ollama": _ollama_factory,
}


def register_client(name: str, factory: ClientFactory) -> None:
    """Register a custom factory. Overwrites an existing entry with the same name."""
    _REGISTRY[name] = factory


def list_providers() -> list[str]:
    return sorted(_REGISTRY)


def make_client(
    provider: str,
    *,
    model: str | None = None,
    env_override: Any = None,
    **kwargs: Any,
) -> LLMClient:
    """Build a client by provider name.

    `provider` must be registered. `model` is optional; each factory has
    a sensible default. `env_override` bypasses env-var resolution —
    useful for tests and multi-account orchestration.
    """
    try:
        factory = _REGISTRY[provider]
    except KeyError as exc:
        raise ValueError(
            f"unknown provider {provider!r}; registered: {sorted(_REGISTRY)}"
        ) from exc
    cfg = env_override if env_override is not None else resolve_provider_config(provider)
    return factory(cfg, model=model, **kwargs)
