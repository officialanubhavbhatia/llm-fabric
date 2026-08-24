"""Inference topology: adapter, transport, and runtime.

These are deployment attributes. They are never inferred from a model id string.
LiteLLM is a transport, not a second route planner.
"""

from __future__ import annotations

from enum import StrEnum

from llm_fabric.errors import ConfigurationError

# Worst-case provider calls = fabric attempts × (1 + transport retries).
MAX_RETRY_PRODUCT = 9
MAX_TRANSPORT_RETRIES = 1


class TransportKind(StrEnum):
    DIRECT = "direct"
    LITELLM = "litellm"


class RuntimeKind(StrEnum):
    OLLAMA = "ollama"
    VLLM = "vllm"
    EXTERNAL = "external"
    MOCK = "mock"


def retry_product(*, fabric_attempts: int, transport_retries: int) -> int:
    return max(1, fabric_attempts) * (1 + max(0, transport_retries))


def refuse_retry_amplification(*, fabric_attempts: int, transport_retries: int) -> None:
    if transport_retries < 0:
        raise ConfigurationError("transport retries cannot be negative")
    if transport_retries > MAX_TRANSPORT_RETRIES:
        raise ConfigurationError(
            f"transport retries may be 0 or {MAX_TRANSPORT_RETRIES}; MyVista owns semantic fallback"
        )
    product = retry_product(fabric_attempts=fabric_attempts, transport_retries=transport_retries)
    if product > MAX_RETRY_PRODUCT:
        raise ConfigurationError(
            f"retry product {product} exceeds cap {MAX_RETRY_PRODUCT} "
            f"(attempts={fabric_attempts} × (1+transport_retries={transport_retries}))"
        )


def defaults_for_provider(provider: str) -> tuple[str, TransportKind, RuntimeKind]:
    """adapter name, transport, runtime from the provider field — not the model id."""
    if provider == "mock":
        return "mock", TransportKind.DIRECT, RuntimeKind.MOCK
    if provider == "litellm" or provider.startswith("litellm-"):
        return "litellm", TransportKind.LITELLM, RuntimeKind.EXTERNAL
    if provider == "ollama" or provider.startswith("ollama-"):
        return "ollama", TransportKind.DIRECT, RuntimeKind.OLLAMA
    if provider == "vllm" or provider.startswith("vllm-"):
        return "vllm", TransportKind.DIRECT, RuntimeKind.VLLM
    if provider in {"openai", "anthropic", "openai-compatible"}:
        return provider, TransportKind.DIRECT, RuntimeKind.EXTERNAL
    return provider, TransportKind.DIRECT, RuntimeKind.EXTERNAL
