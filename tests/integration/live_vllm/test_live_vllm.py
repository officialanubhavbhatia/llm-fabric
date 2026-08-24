"""Optional live vLLM path. SKIPPED when the server is not running."""

from __future__ import annotations

import os

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.router.grades import Grade
from llm_fabric.router.health import HealthTracker
from llm_fabric.router.plan import (
    ExclusionRule,
    RoutePlanner,
    RouteRequest,
    TenantRoutingPolicies,
    TenantRoutingPolicy,
)
from llm_fabric.router.registry import ModelRegistry

pytestmark = pytest.mark.skipif(
    os.environ.get("LLM_FABRIC_LIVE_VLLM") != "1",
    reason="vLLM live inference not measured (set LLM_FABRIC_LIVE_VLLM=1)",
)


def _vllm_up() -> bool:
    base = os.environ.get("LLM_FABRIC_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    try:
        response = httpx.get(f"{base.rstrip('/')}/models", timeout=1.0)
    except httpx.HTTPError:
        return False
    if response.status_code >= 500:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    ids = {str(item.get("id")) for item in payload.get("data") or []}
    owners = {str(item.get("owned_by")) for item in payload.get("data") or []}
    # Port 8000 is often a local Fabric mock gateway. That is not vLLM.
    return "mock-small" not in ids and "llm-fabric" not in owners


def _base() -> str:
    return os.environ.get("LLM_FABRIC_VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")


def _model() -> str:
    configured = os.environ.get("LLM_FABRIC_LIVE_VLLM_MODEL")
    if configured:
        return configured
    payload = httpx.get(f"{_base()}/models", timeout=5.0).json()
    return str(payload["data"][0]["id"])


def _registry(*, enabled: bool = True, context_window: int = 8192) -> ModelRegistry:
    return ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "vllm-live",
                    "provider": "vllm",
                    "provider_model": _model(),
                    "grade": "L10",
                    "context_window": context_window,
                    "capabilities": ["chat", "streaming"],
                    "lifecycle": "approved",
                    "enabled": enabled,
                    "fallbacks": ["mock-small"],
                },
                {
                    "id": "mock-small",
                    "provider": "mock",
                    "grade": "L2",
                    "capabilities": ["chat", "streaming"],
                    "lifecycle": "approved",
                },
            ],
            "aliases": [
                {
                    "id": "auto",
                    "policy": "declared",
                    "candidates": ["vllm-live", "mock-small"],
                }
            ],
        }
    )


@pytest.mark.skipif(not _vllm_up(), reason="vLLM OpenAI-compatible endpoint not reachable")
def test_vllm_openai_compatible_lists_models() -> None:
    response = httpx.get(f"{_base()}/models", timeout=5.0)
    assert response.status_code == 200


@pytest.mark.skipif(not _vllm_up(), reason="vLLM OpenAI-compatible endpoint not reachable")
def test_vllm_chat_and_response_normalization() -> None:
    response = httpx.post(
        f"{_base()}/chat/completions",
        json={
            "model": _model(),
            "messages": [{"role": "user", "content": "Reply pong"}],
            "max_tokens": 8,
            "temperature": 0,
        },
        timeout=60.0,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"]


@pytest.mark.skipif(not _vllm_up(), reason="vLLM OpenAI-compatible endpoint not reachable")
def test_vllm_streaming_contract() -> None:
    with httpx.stream(
        "POST",
        f"{_base()}/chat/completions",
        json={
            "model": _model(),
            "messages": [{"role": "user", "content": "Reply pong"}],
            "max_tokens": 8,
            "stream": True,
        },
        timeout=60.0,
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "data:" in body
    assert "[DONE]" in body


@pytest.mark.skipif(not _vllm_up(), reason="vLLM OpenAI-compatible endpoint not reachable")
def test_fabric_routes_to_vllm_with_headers() -> None:
    app = create_app(
        settings=Settings(environment="test", vllm_base_url=_base()),
        registry=_registry(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "vllm-live",
                "messages": [{"role": "user", "content": "Reply pong"}],
                "max_tokens": 8,
            },
        )
    assert response.status_code == 200
    assert response.headers["x-fabric-served-model"] == "vllm-live"
    assert response.headers["x-fabric-provider"] == "vllm"
    assert response.headers["x-fabric-selected-tier"] == "L10"


@pytest.mark.skipif(not _vllm_up(), reason="vLLM OpenAI-compatible endpoint not reachable")
def test_vllm_timeout_uses_typed_fallback() -> None:
    app = create_app(
        settings=Settings(
            environment="test",
            vllm_base_url=_base(),
            request_timeout_s=0.000001,
        ),
        registry=_registry(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "vllm-live",
                "messages": [{"role": "user", "content": "Reply pong"}],
            },
        )
    assert response.status_code == 200
    assert response.headers["x-fabric-served-model"] == "mock-small"
    assert int(response.headers["x-fabric-failovers"]) >= 1


@pytest.mark.skipif(not _vllm_up(), reason="vLLM OpenAI-compatible endpoint not reachable")
def test_vllm_context_disabled_ceiling_and_breaker_remain_distinct() -> None:
    registry = _registry(context_window=32)
    context = RoutePlanner(registry).plan(RouteRequest("vllm-live", prompt_tokens=10_000))
    assert any(item.rule is ExclusionRule.CONTEXT_TOO_SMALL for item in context.excluded)

    disabled = RoutePlanner(_registry(enabled=False)).plan(RouteRequest("vllm-live"))
    assert any(item.rule is ExclusionRule.DISABLED for item in disabled.excluded)

    tenant = RoutePlanner(
        registry,
        tenant_policies=TenantRoutingPolicies(
            [
                TenantRoutingPolicy(
                    tenant_id="limited",
                    maximum_grade=Grade.GRADE04,
                )
            ]
        ),
    ).plan(RouteRequest("vllm-live", tenant_id="limited"))
    assert any(item.rule is ExclusionRule.GRADE_ABOVE_MAXIMUM for item in tenant.excluded)

    health = HealthTracker()
    health.force_open("vllm-live", reason="live-test")
    breaker = RoutePlanner(registry, health=health).plan(RouteRequest("vllm-live"))
    assert any(item.rule is ExclusionRule.CIRCUIT_OPEN for item in breaker.excluded)
