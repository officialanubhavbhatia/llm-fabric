"""Construction of provider instances from configuration.

Providers are built lazily and then reused, because each one holds an HTTP client
with its own connection pool and building them per request would discard those
connections. Tests inject providers by name through `overrides`, which is how the
router is exercised without network access.

Ollama and vLLM are reached through the OpenAI-compatible chat-completions
contract. They are not embedded engines: the fabric does not scrape their
native `/metrics` endpoints and does not load model weights itself.
"""

from __future__ import annotations

import threading

from llm_fabric.config import Settings
from llm_fabric.errors import ConfigurationError, ProviderUnavailableError
from llm_fabric.serving.adapters import (
    AnthropicProvider,
    LiteLLMProvider,
    MockProvider,
    OpenAIProvider,
)
from llm_fabric.serving.base import Provider

_KNOWN_PROVIDERS = (
    "mock",
    "openai",
    "anthropic",
    "ollama",
    "vllm",
    "openai-compatible",
    "litellm",
)


def _is_openai_compatible(name: str) -> bool:
    return (
        name in {"ollama", "vllm", "openai-compatible"}
        or name.startswith("ollama-")
        or name.startswith("vllm-")
    )


def _is_litellm(name: str) -> bool:
    return name == "litellm" or name.startswith("litellm-")


class ProviderFactory:
    def __init__(
        self,
        settings: Settings,
        overrides: dict[str, Provider] | None = None,
    ) -> None:
        self._settings = settings
        self._instances: dict[str, Provider] = dict(overrides or {})
        self._owned: set[str] = set()
        self._lock = threading.Lock()

    def get(self, name: str, *, base_url: str | None = None) -> Provider:
        key = f"{name}@{base_url.rstrip('/')}" if base_url else name
        if existing := self._instances.get(key):
            return existing

        with self._lock:
            if existing := self._instances.get(key):
                return existing
            try:
                provider = self._build(name, base_url=base_url)
            except ConfigurationError as exc:
                # Missing credentials or an unknown provider name must be
                # retryable so a later candidate in the chain can still serve.
                raise ProviderUnavailableError(str(exc)) from exc
            self._instances[key] = provider
            self._owned.add(key)
            return provider

    def constructible(self, name: str) -> bool:
        """True when this factory can return a provider for `name`."""
        try:
            self.get(name)
        except ProviderUnavailableError:
            return False
        return True

    def _compatible_base_url(self, name: str, *, base_url: str | None = None) -> str:
        if base_url:
            return base_url
        settings = self._settings
        if name in settings.provider_base_urls:
            return settings.provider_base_urls[name]
        if _is_litellm(name):
            return settings.litellm_base_url
        if name == "ollama" or name.startswith("ollama-"):
            return settings.ollama_base_url
        if name == "vllm" or name.startswith("vllm-"):
            return settings.vllm_base_url
        return settings.openai_base_url

    def _compatible_api_key(self, name: str) -> str:
        settings = self._settings
        if _is_litellm(name):
            return settings.litellm_api_key or "litellm"
        if name == "ollama" or name.startswith("ollama-"):
            return settings.ollama_api_key or "ollama"
        if name == "vllm" or name.startswith("vllm-"):
            return settings.vllm_api_key or "vllm"
        return settings.openai_api_key or name

    def _build(self, name: str, *, base_url: str | None = None) -> Provider:
        settings = self._settings
        if name == "mock":
            return MockProvider(delay_s=settings.mock_delay_s)
        if name == "openai":
            return OpenAIProvider(
                api_key=settings.openai_api_key,
                base_url=base_url or settings.openai_base_url,
                timeout_s=settings.request_timeout_s,
            )
        if name == "anthropic":
            return AnthropicProvider(
                api_key=settings.anthropic_api_key,
                base_url=base_url or settings.anthropic_base_url,
                timeout_s=settings.request_timeout_s,
            )
        if _is_litellm(name):
            return LiteLLMProvider(
                api_key=self._compatible_api_key(name),
                base_url=self._compatible_base_url(name, base_url=base_url),
                timeout_s=settings.request_timeout_s,
                name=name,
                require_api_key=False,
            )
        if _is_openai_compatible(name):
            return OpenAIProvider(
                api_key=self._compatible_api_key(name),
                base_url=self._compatible_base_url(name, base_url=base_url),
                timeout_s=settings.request_timeout_s,
                name=name,
                require_api_key=False,
            )
        raise ConfigurationError(
            f"unknown provider '{name}' (available: {list(_KNOWN_PROVIDERS)} "
            "plus ollama-* / vllm-* / litellm-* pool names)"
        )

    async def aclose(self) -> None:
        """Close only the providers this factory created, never injected ones."""
        with self._lock:
            owned = list(self._owned)
        for name in owned:
            await self._instances[name].aclose()
        with self._lock:
            self._instances = {
                name: provider
                for name, provider in self._instances.items()
                if name not in self._owned
            }
            self._owned.clear()
