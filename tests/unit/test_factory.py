"""Provider factory: Ollama and vLLM do not require an OpenAI key."""

from __future__ import annotations

import pytest

from llm_fabric.config import Settings
from llm_fabric.errors import ProviderUnavailableError
from llm_fabric.serving.adapters.openai import OpenAIProvider
from llm_fabric.serving.factory import ProviderFactory


def test_ollama_builds_without_an_openai_key() -> None:
    factory = ProviderFactory(Settings(_env_file=None, openai_api_key=None))
    provider = factory.get("ollama")
    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "ollama"


def test_vllm_pool_names_share_the_openai_compatible_adapter() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key=None,
        provider_base_urls={"vllm-coding": "http://vllm-coding.example:8000/v1"},
    )
    factory = ProviderFactory(settings)
    provider = factory.get("vllm-coding")
    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "vllm-coding"


def test_openai_still_requires_a_key() -> None:
    factory = ProviderFactory(Settings(_env_file=None, openai_api_key=None))
    with pytest.raises(ProviderUnavailableError, match="API key"):
        factory.get("openai")
