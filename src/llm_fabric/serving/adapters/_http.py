"""Shared HTTP plumbing for provider adapters.

The important part here is the error mapping: transport failures, timeouts, and
5xx/429 responses become `RetryableError`, which is the signal the router uses to
try the next candidate. A 4xx that is the caller's fault does not become
retryable, because retrying it would just fail again more slowly.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from llm_fabric.errors import (
    AuthenticationError,
    ContextTooLargeError,
    InvalidRequestError,
    LiteLLMUnavailableError,
    ModelUnavailableError,
    OllamaUnavailableError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitedError,
    RetryableError,
    VllmUnavailableError,
)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
_CONTEXT_MARKERS = ("context length", "context window", "maximum context", "too many tokens")


def unavailable_error(provider: str, detail: str) -> ProviderUnavailableError:
    if provider == "litellm" or provider.startswith("litellm-"):
        return LiteLLMUnavailableError(detail)
    if provider == "ollama" or provider.startswith("ollama-"):
        return OllamaUnavailableError(detail)
    if provider == "vllm" or provider.startswith("vllm-"):
        return VllmUnavailableError(detail)
    return ProviderUnavailableError(detail)


def build_client(base_url: str, headers: dict[str, str], timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers=headers,
        timeout=httpx.Timeout(timeout_s),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        follow_redirects=False,
    )


def _extract_message(payload: object, fallback: str) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            nested = error.get("message")
            if isinstance(nested, str):
                return nested
        if isinstance(error, str):
            return error
        message = payload.get("message")
        if isinstance(message, str):
            return message
    return fallback


def raise_for_status(provider: str, response: httpx.Response, body: object = None) -> None:
    if response.status_code < 400:
        return

    if body is None:
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            body = None

    message = _extract_message(body, response.text[:500] or response.reason_phrase)
    detail = f"{provider}: {message}"
    lowered = message.lower()

    if response.status_code in {401, 403}:
        raise AuthenticationError(detail)
    if any(marker in lowered for marker in _CONTEXT_MARKERS):
        raise ContextTooLargeError(detail)
    if response.status_code == 429:
        raise RateLimitedError(detail)
    if response.status_code == 404 or "model" in lowered and "not found" in lowered:
        raise ModelUnavailableError(detail)
    if response.status_code in _RETRYABLE_STATUS:
        raise unavailable_error(provider, detail)
    if 400 <= response.status_code < 500:
        raise InvalidRequestError(detail)
    raise RetryableError(detail)


def translate_transport_error(provider: str, exc: httpx.HTTPError) -> RetryableError:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeoutError(f"{provider}: request timed out")
    return unavailable_error(provider, f"{provider}: {exc.__class__.__name__}: {exc}")


async def iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield the `data:` payload of each SSE event, skipping comments and `[DONE]`."""
    async for raw_line in response.aiter_lines():
        line = raw_line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        yield data
