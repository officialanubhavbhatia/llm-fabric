"""Chaos: expected degraded behaviour, not merely 'no exception'."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_fabric.chaos import EXPECTED, DegradedMode
from llm_fabric.config import Settings
from llm_fabric.errors import DependencyUnavailableError, QuotaExceededError
from llm_fabric.gateway.app import create_app
from llm_fabric.guardrails import ExecutionGuardrail, GuardrailAction, InputGuardrail
from llm_fabric.observability.analytics import AnalyticsEvent, BufferedAnalyticsSink
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider
from llm_fabric.storage.redis import RedisCache, RedisRevocationStore
from llm_fabric.tenancy.quota import QuotaLedger, QuotaPolicy
from llm_fabric.tenancy.scope import TenantScope

pytestmark = pytest.mark.chaos


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "cheap",
                    "provider": "mock",
                    "provider_model": "cheap-v1",
                    "context_window": 128,
                    "capabilities": ["chat"],
                    "fallbacks": ["premium"],
                },
                {
                    "id": "premium",
                    "provider": "mock",
                    "provider_model": "premium-v1",
                    "context_window": 8192,
                    "capabilities": ["chat"],
                },
                {
                    "id": "broken",
                    "provider": "failing",
                    "provider_model": "broken-v1",
                    "capabilities": ["chat"],
                    "fallbacks": ["cheap"],
                },
            ]
        }
    )


def test_expected_modes_are_named() -> None:
    assert EXPECTED["otel_collector_unavailable"] is DegradedMode.CONTINUE_WITHOUT_TELEMETRY
    assert EXPECTED["provider_500"] is DegradedMode.FALLBACK
    assert EXPECTED["context_overflow"] is DegradedMode.REJECT


def test_provider_failure_falls_back(registry: ModelRegistry) -> None:
    assert EXPECTED["provider_500"] is DegradedMode.FALLBACK
    app = create_app(
        settings=Settings(allow_anonymous=True),
        registry=registry,
        provider_overrides={"mock": MockProvider(), "failing": MockProvider(fail=True)},
        meter=InMemoryMeter(),
    )
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "broken", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.headers["x-fabric-failovers"] == "1"
    assert response.headers["x-fabric-served-model"] == "cheap"


def test_context_overflow_is_rejected(registry: ModelRegistry) -> None:
    assert EXPECTED["context_overflow"] is DegradedMode.REJECT
    app = create_app(
        settings=Settings(allow_anonymous=True),
        registry=registry,
        provider_overrides={"mock": MockProvider()},
        meter=InMemoryMeter(),
    )
    app.state.controls.set_context_ceiling(50)
    client = TestClient(app)
    huge = "word " * 5_000
    response = client.post(
        "/v1/chat/completions",
        json={"model": "cheap", "messages": [{"role": "user", "content": huge}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] in {"invalid_request_error", "context_too_large"}


def test_extreme_output_tokens_are_rejected() -> None:
    assert EXPECTED["extreme_output_tokens"] is DegradedMode.REJECT
    decision = InputGuardrail().evaluate({"text": "hi", "max_tokens": 99_999}, tenant_id="acme")
    assert decision.action is GuardrailAction.BLOCK


def test_tool_loop_is_rejected() -> None:
    assert EXPECTED["tool_loop"] is DegradedMode.REJECT
    decision = ExecutionGuardrail(allowed_tools=("search",), max_iterations=2).evaluate(
        {"tool": "search", "iterations": 9}, tenant_id="acme"
    )
    assert decision.action is GuardrailAction.BLOCK


def test_telemetry_outage_does_not_grow_unbounded() -> None:
    assert EXPECTED["otel_collector_unavailable"] is DegradedMode.CONTINUE_WITHOUT_TELEMETRY
    sink = BufferedAnalyticsSink(max_events=8)
    for index in range(50):
        sink.emit(AnalyticsEvent(kind="trace", tenant_id="acme", payload={"i": index}))
    assert sink.dropped >= 1
    assert len(sink.drain(100)) <= 8


def test_quota_refusal_is_fail_closed() -> None:
    assert EXPECTED["queue_saturation"] is DegradedMode.REJECT
    ledger = QuotaLedger(default_tenant_policy=QuotaPolicy(requests_per_minute=1))
    scope = TenantScope(tenant_id="acme", user_id="alice")
    ledger.admit(scope)
    with pytest.raises(QuotaExceededError):
        ledger.admit(scope)


def test_revocation_fail_closed_when_redis_raises() -> None:
    assert EXPECTED["redis_unavailable_production_revocation"] is DegradedMode.FAIL_CLOSED

    class Dead:
        def exists(self, *_args, **_kwargs):
            raise __import__("redis").RedisError("down")

    store = RedisRevocationStore(Dead(), fail_closed=True)  # type: ignore[arg-type]
    with pytest.raises(DependencyUnavailableError):
        store.is_revoked(token_id="x")


def test_cache_miss_when_redis_is_down() -> None:
    assert EXPECTED["cache_unavailable"] is DegradedMode.DEGRADE

    class Dead:
        def get(self, *_args, **_kwargs):
            raise __import__("redis").RedisError("down")

    cache = RedisCache(Dead(), fail_soft=True)  # type: ignore[arg-type]
    assert cache.get("acme", "exact_response", "abc") is None
