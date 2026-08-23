"""Construction of provider instances from configuration.

Providers are built lazily and then reused, because each one holds an HTTP client
with its own connection pool and building them per request would discard those
connections. Tests inject providers by name through `overrides`, which is how the
router is exercised without network access.
"""

from __future__ import annotations

from llm_fabric.config import Settings
from llm_fabric.errors import ConfigurationError
from llm_fabric.serving.adapters import AnthropicProvider, MockProvider, OpenAIProvider
from llm_fabric.serving.base import Provider

_KNOWN_PROVIDERS = ("mock", "openai", "anthropic")


class ProviderFactory:
    def __init__(
        self,
        settings: Settings,
        overrides: dict[str, Provider] | None = None,
    ) -> None:
        self._settings = settings
        self._instances: dict[str, Provider] = dict(overrides or {})
        self._owned: set[str] = set()

    def get(self, name: str) -> Provider:
        if existing := self._instances.get(name):
            return existing

        provider = self._build(name)
        self._instances[name] = provider
        self._owned.add(name)
        return provider

    def _build(self, name: str) -> Provider:
        settings = self._settings
        if name == "mock":
            return MockProvider()
        if name == "openai":
            return OpenAIProvider(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout_s=settings.request_timeout_s,
            )
        if name == "anthropic":
            return AnthropicProvider(
                api_key=settings.anthropic_api_key,
                base_url=settings.anthropic_base_url,
                timeout_s=settings.request_timeout_s,
            )
        raise ConfigurationError(f"unknown provider '{name}' (available: {list(_KNOWN_PROVIDERS)})")

    async def aclose(self) -> None:
        """Close only the providers this factory created, never injected ones."""
        for name in self._owned:
            await self._instances[name].aclose()
        self._instances = {
            name: provider for name, provider in self._instances.items() if name not in self._owned
        }
        self._owned.clear()
