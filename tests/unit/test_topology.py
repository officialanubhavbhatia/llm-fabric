"""Inference topology: LiteLLM transport, declared runtime, bounded retries."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from llm_fabric.config import Settings, validate_startup
from llm_fabric.errors import (
    ConfigurationError,
    ContextTooLargeError,
    LiteLLMUnavailableError,
    ModelUnavailableError,
    OllamaUnavailableError,
    RateLimitedError,
    VllmUnavailableError,
)
from llm_fabric.router.registry import ModelRegistry, ModelSpec
from llm_fabric.serving.adapters._http import raise_for_status, translate_transport_error
from llm_fabric.serving.adapters.litellm import LiteLLMProvider
from llm_fabric.serving.factory import ProviderFactory
from llm_fabric.serving.topology import (
    RuntimeKind,
    TransportKind,
    defaults_for_provider,
    refuse_retry_amplification,
)


def test_litellm_builds_without_an_openai_key() -> None:
    factory = ProviderFactory(Settings(_env_file=None, openai_api_key=None))
    provider = factory.get("litellm")
    assert isinstance(provider, LiteLLMProvider)
    assert provider.name == "litellm"


def test_litellm_pool_names_share_the_transport_adapter() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key=None,
        provider_base_urls={"litellm-gpu": "http://litellm-gpu.example:4000/v1"},
    )
    factory = ProviderFactory(settings)
    provider = factory.get("litellm-gpu")
    assert isinstance(provider, LiteLLMProvider)
    assert provider.name == "litellm-gpu"


def test_runtime_is_not_inferred_from_the_model_id() -> None:
    spec = ModelSpec(id="g00-smollm2-135m", provider="litellm", runtime=RuntimeKind.OLLAMA)
    assert spec.runtime is RuntimeKind.OLLAMA
    assert spec.transport is TransportKind.LITELLM
    assert spec.provider_adapter == "litellm"
    adapter, transport, runtime = defaults_for_provider("litellm")
    assert adapter == "litellm"
    assert transport is TransportKind.LITELLM
    assert runtime is RuntimeKind.EXTERNAL


def test_ollama_provider_cannot_set_litellm_transport() -> None:
    with pytest.raises(ConfigurationError, match="transport=litellm"):
        ModelRegistry.from_mapping(
            {
                "models": [
                    {
                        "id": "mixed",
                        "provider": "ollama",
                        "transport": "litellm",
                    }
                ]
            }
        )


def test_litellm_ollama_registry_declares_transport_and_runtime() -> None:
    registry = ModelRegistry.from_yaml(Path("config/models.litellm-ollama.yaml"))
    spec = registry.get("litellm-ollama-smollm2-135m")
    assert spec.provider == "litellm"
    assert spec.provider_adapter == "litellm"
    assert spec.transport is TransportKind.LITELLM
    assert spec.runtime is RuntimeKind.OLLAMA
    assert spec.provider_model == "smollm2-135m"


def test_direct_ollama_defaults_are_not_litellm() -> None:
    spec = ModelSpec(id="local-small", provider="ollama", provider_model="llama3.2")
    assert spec.provider_adapter == "ollama"
    assert spec.transport is TransportKind.DIRECT
    assert spec.runtime is RuntimeKind.OLLAMA


def test_transport_retries_above_one_are_refused() -> None:
    with pytest.raises(ConfigurationError, match="transport retries"):
        refuse_retry_amplification(fabric_attempts=3, transport_retries=2)


def test_retry_product_above_cap_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="retry product"):
        refuse_retry_amplification(fabric_attempts=10, transport_retries=0)


def test_startup_refuses_litellm_num_retries_above_one() -> None:
    with pytest.raises(ConfigurationError, match="transport retries"):
        validate_startup(
            Settings(
                _env_file=None,
                environment="development",
                allow_anonymous=True,
                litellm_num_retries=2,
            )
        )


def test_http_429_is_rate_limited() -> None:
    response = httpx.Response(429, json={"error": {"message": "slow down"}})
    with pytest.raises(RateLimitedError) as exc:
        raise_for_status("openai", response)
    assert exc.value.error_type == "rate_limited"


def test_connect_errors_name_the_runtime() -> None:
    ollama = translate_transport_error("ollama", httpx.ConnectError("down"))
    assert isinstance(ollama, OllamaUnavailableError)
    assert ollama.error_type == "ollama_unavailable"
    litellm = translate_transport_error("litellm", httpx.ConnectError("down"))
    assert isinstance(litellm, LiteLLMUnavailableError)
    assert litellm.error_type == "litellm_unavailable"
    vllm = translate_transport_error("vllm-coding", httpx.ConnectError("down"))
    assert isinstance(vllm, VllmUnavailableError)
    assert vllm.error_type == "vllm_unavailable"


def test_model_not_found_and_context_overflow_stay_distinct() -> None:
    missing = httpx.Response(404, json={"error": {"message": "model not found"}})
    with pytest.raises(ModelUnavailableError) as missing_exc:
        raise_for_status("litellm", missing)
    assert missing_exc.value.error_type == "model_unavailable"
    overflow = httpx.Response(400, json={"error": {"message": "maximum context length exceeded"}})
    with pytest.raises(ContextTooLargeError) as overflow_exc:
        raise_for_status("vllm", overflow)
    assert overflow_exc.value.error_type == "context_too_large"
