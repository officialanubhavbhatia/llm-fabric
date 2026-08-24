"""Construction of provider instances from configuration.

Providers are built lazily and then reused, because each one holds an HTTP client
with its own connection pool and building them per request would discard those
connections. Tests inject providers by name through `overrides`, which is how the
router is exercised without network access.
"""

from __future__ import annotations

import threading

from llm_fabric.config import Settings
from llm_fabric.errors import ConfigurationError, ProviderUnavailableError
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
        self._lock = threading.Lock()

    def get(self, name: str) -> Provider:
        if existing := self._instances.get(name):
            return existing

        with self._lock:
            if existing := self._instances.get(name):
                return existing
            try:
                provider = self._build(name)
            except ConfigurationError as exc:
                # Missing credentials or an unknown provider name must be
                # retryable so a later candidate in the chain can still serve.
                raise ProviderUnavailableError(str(exc)) from exc
            self._instances[name] = provider
            self._owned.add(name)
            return provider

    def constructible(self, name: str) -> bool:
        """True when this factory can return a provider for `name`."""
        try:
            self.get(name)
        except ProviderUnavailableError:
            return False
        return True

    def _build(self, name: str) -> Provider:
        settings = self._settings
        if name == "mock":
            return MockProvider(delay_s=settings.mock_delay_s)
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
