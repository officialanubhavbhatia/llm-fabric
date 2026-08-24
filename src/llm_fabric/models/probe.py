"""Live capability probe. Measures what a deployment actually answers.

Unsupported capabilities are recorded as unsupported, not as probe failures.
When the provider cannot be reached the result is `status: unavailable` and no
latency numbers are invented.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from llm_fabric import __version__ as fabric_version
from llm_fabric.config import Settings, get_settings
from llm_fabric.contract.openai import ChatMessage
from llm_fabric.eval.provenance import current_commit
from llm_fabric.router.registry import ModelRegistry, ModelSpec
from llm_fabric.serving.adapters.mock import MockProvider
from llm_fabric.serving.base import InferenceRequest, Provider
from llm_fabric.serving.factory import ProviderFactory

PROBE_VERSION = "model-probe-v1"

_CHAT_PROMPT = "Reply with the single word pong."
_JSON_PROMPT = 'Return only JSON: {"ok": true}'
_STREAM_PROMPT = "Count from 1 to 3."


def _capability(supported: bool | None, reason: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"supported": supported}
    if reason:
        payload["reason"] = reason
    return payload


def _unavailable(
    *, deployment: str | None, provider: str, model: str, detail: str
) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "deployment": deployment,
        "provider": provider,
        "model": model,
        "probe_version": PROBE_VERSION,
        "detail": detail,
    }


def _env_metadata(settings: Settings) -> dict[str, Any]:
    return {
        "environment": settings.environment,
        "fabric_version": fabric_version,
        "commit": current_commit(),
        "python": "3.12",
    }


async def _chat(
    provider: Provider, model: str, prompt: str, *, max_tokens: int = 64
) -> tuple[str, float]:
    started = time.perf_counter()
    result = await provider.generate(
        InferenceRequest(
            model=model,
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return result.text, elapsed_ms


async def _stream_ttft(
    provider: Provider, model: str
) -> tuple[bool, float | None, float | None, int]:
    started = time.perf_counter()
    ttft_ms: float | None = None
    chunks = 0
    try:
        async for event in provider.stream(
            InferenceRequest(
                model=model,
                messages=[ChatMessage(role="user", content=_STREAM_PROMPT)],
                max_tokens=32,
                temperature=0.0,
            )
        ):
            text = getattr(event, "text", None)
            if text and ttft_ms is None:
                ttft_ms = (time.perf_counter() - started) * 1000
            if text:
                chunks += 1
    except Exception:  # noqa: BLE001 - probe records absence rather than failing
        return False, None, None, chunks
    total_ms = (time.perf_counter() - started) * 1000
    return True, ttft_ms, total_ms, chunks


def _endpoint_class(provider: str) -> str:
    if provider == "mock":
        return "in-process"
    if provider == "openai" or provider == "anthropic":
        return "native-http"
    return "openai-compatible"


async def probe_with_provider(
    provider: Provider,
    *,
    model: str,
    deployment: str | None,
    provider_name: str,
    spec: ModelSpec | None,
    settings: Settings,
) -> dict[str, Any]:
    capabilities: dict[str, Any] = {}
    try:
        text, chat_ms = await _chat(provider, model, _CHAT_PROMPT)
    except Exception as exc:  # noqa: BLE001
        return _unavailable(
            deployment=deployment,
            provider=provider_name,
            model=model,
            detail=str(exc),
        )

    capabilities["chat"] = _capability(True)
    capabilities["connectivity"] = _capability(True)

    streamed, ttft_ms, stream_ms, chunks = await _stream_ttft(provider, model)
    capabilities["streaming"] = _capability(
        streamed and chunks > 0,
        None if streamed and chunks > 0 else "probe_failed",
    )

    json_ok = False
    json_reason: str | None = "probe_failed"
    try:
        json_text, _ = await _chat(provider, model, _JSON_PROMPT)
        stripped = json_text.strip()
        json_ok = stripped.startswith("{") and "ok" in stripped.lower()
        if json_ok:
            json_reason = None
        elif spec is not None and not spec.capabilities.supports_json_schema:
            json_reason = "model_capability"
    except Exception:  # noqa: BLE001
        json_reason = "probe_failed"
    capabilities["json"] = _capability(json_ok, json_reason)

    # Tool calling is not executed by this fabric. Do not mark absence as a failure.
    capabilities["tools"] = _capability(
        False,
        "model_capability"
        if spec is None or not spec.capabilities.supports_tools
        else "not_executed",
    )

    for name in ("reasoning", "coding", "summarization"):
        capabilities[name] = {
            "supported": None,
            "reason": "not_objectively_probed",
        }

    output_tokens = max(1, len(text.split()))
    total_latency = chat_ms
    tokens_per_sec = (output_tokens / (total_latency / 1000.0)) if total_latency > 0 else None

    return {
        "status": "ok",
        "deployment": deployment,
        "provider": provider_name,
        "provider_endpoint_class": _endpoint_class(provider_name),
        "model": model,
        "model_id": spec.id if spec else model,
        "model_revision": spec.revision if spec else None,
        "model_digest": spec.digest if spec else None,
        "huggingface_id": spec.huggingface_id if spec else None,
        "probe_version": PROBE_VERSION,
        "timestamp": time.time(),
        "reachable": True,
        "capabilities": capabilities,
        "performance": {
            "ttft_ms": ttft_ms,
            "total_latency_ms": round(total_latency, 3),
            "stream_total_ms": round(stream_ms, 3) if stream_ms is not None else None,
            "output_tokens_per_second": round(tokens_per_sec, 3) if tokens_per_sec else None,
        },
        "environment": _env_metadata(settings),
    }


async def probe_deployment(
    deployment_id: str, *, registry: ModelRegistry, settings: Settings | None = None
) -> dict[str, Any]:
    resolved = settings or get_settings()
    spec = registry.get(deployment_id)
    factory = ProviderFactory(resolved)
    try:
        provider = factory.get(spec.provider, base_url=spec.api_base)
    except Exception as exc:  # noqa: BLE001
        return _unavailable(
            deployment=spec.id,
            provider=spec.provider,
            model=spec.provider_model,
            detail=str(exc),
        )
    try:
        return await probe_with_provider(
            provider,
            model=spec.provider_model,
            deployment=spec.id,
            provider_name=spec.provider,
            spec=spec,
            settings=resolved,
        )
    finally:
        await factory.aclose()


async def probe_provider_model(
    *,
    provider: str,
    model: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    if provider == "mock":
        backend: Provider = MockProvider()
        return await probe_with_provider(
            backend,
            model=model,
            deployment=None,
            provider_name="mock",
            spec=None,
            settings=resolved,
        )

    factory = ProviderFactory(resolved)
    try:
        try:
            backend = factory.get(provider)
        except Exception as exc:  # noqa: BLE001
            return _unavailable(deployment=None, provider=provider, model=model, detail=str(exc))

        if provider != "anthropic":
            base = resolved.provider_base_urls.get(provider)
            if base is None and (provider == "ollama" or provider.startswith("ollama-")):
                base = resolved.ollama_base_url
            elif base is None and (provider == "vllm" or provider.startswith("vllm-")):
                base = resolved.vllm_base_url
            elif base is None and provider in {"openai", "openai-compatible"}:
                base = resolved.openai_base_url
            if base:
                try:
                    async with httpx.AsyncClient(base_url=base, timeout=5.0) as client:
                        response = await client.get("/models")
                        if response.status_code >= 500:
                            return _unavailable(
                                deployment=None,
                                provider=provider,
                                model=model,
                                detail=f"HTTP {response.status_code} from {base}/models",
                            )
                except httpx.HTTPError as exc:
                    return _unavailable(
                        deployment=None, provider=provider, model=model, detail=str(exc)
                    )

        return await probe_with_provider(
            backend,
            model=model,
            deployment=None,
            provider_name=provider,
            spec=None,
            settings=resolved,
        )
    finally:
        await factory.aclose()
