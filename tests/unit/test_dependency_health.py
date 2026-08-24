"""P0-FIX-4: dependency health, readiness aggregation, and admission."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from llm_fabric.config import Settings
from llm_fabric.deps.health import (
    DependencyClass,
    DependencyHealth,
    HealthStatus,
    detection_bound_s,
)
from llm_fabric.deps.monitor import DependencyMonitor
from llm_fabric.errors import DependencyUnavailableError
from llm_fabric.gateway.app import create_app
from llm_fabric.observability.metering import DurableMeter, InMemoryMeter
from llm_fabric.observability.usage_event import TokenSource, UsageEvent, UsageStatus
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider
from llm_fabric.storage.redis import RedisRevocationStore

CHAT = "/v1/chat/completions"
REGISTRY = {
    "models": [
        {
            "id": "cheap",
            "provider": "mock",
            "provider_model": "cheap-v1",
            "capabilities": ["chat"],
        }
    ]
}


def _app(
    *,
    health: DependencyHealth,
    provider: MockProvider | None = None,
    meter: InMemoryMeter | None = None,
):
    provider = provider or MockProvider()
    return create_app(
        settings=Settings(api_keys=[], allow_anonymous=True),
        registry=ModelRegistry.from_mapping(REGISTRY),
        provider_overrides={"mock": provider},
        meter=meter or InMemoryMeter(),
        dependency_health=health,
    ), provider


def test_mandatory_vs_optional_classification() -> None:
    health = DependencyHealth(postgres=True, redis=True, telemetry=True)
    snaps = health.snapshots()
    assert snaps["postgres"].classification is DependencyClass.MANDATORY_SERVING
    assert snaps["redis"].classification is DependencyClass.MANDATORY_SERVING
    assert snaps["telemetry"].classification is DependencyClass.OPTIONAL_FAIL_SOFT
    assert snaps["postgres"].required is True
    assert snaps["telemetry"].required is False


def test_initial_mandatory_is_not_ready_until_first_success() -> None:
    health = DependencyHealth(postgres=True, redis=True)
    assert health.serving_ready() is False
    health.observe_probe_success("postgres")
    assert health.serving_ready() is False
    health.observe_probe_success("redis")
    assert health.serving_ready() is True
    assert health.snapshot("postgres") is not None
    assert health.snapshot("postgres").status is HealthStatus.HEALTHY


def test_hysteresis_probe_failure_goes_suspect_then_unhealthy() -> None:
    health = DependencyHealth(postgres=True, fail_threshold=2, recovery_threshold=2)
    health.observe_probe_success("postgres")
    assert health.serving_ready() is True
    health.observe_probe_failure("postgres")
    assert health.snapshot("postgres").status is HealthStatus.SUSPECT
    assert health.serving_ready() is True
    health.observe_probe_failure("postgres")
    assert health.snapshot("postgres").status is HealthStatus.UNHEALTHY
    assert health.serving_ready() is False


def test_hysteresis_recovery_requires_threshold() -> None:
    health = DependencyHealth(postgres=True, fail_threshold=1, recovery_threshold=2)
    health.observe_probe_success("postgres")
    health.observe_probe_failure("postgres")
    assert health.snapshot("postgres").status is HealthStatus.UNHEALTHY
    health.observe_probe_success("postgres")
    assert health.snapshot("postgres").status is HealthStatus.RECOVERING
    assert health.serving_ready() is False
    health.observe_probe_success("postgres")
    assert health.snapshot("postgres").status is HealthStatus.HEALTHY
    assert health.serving_ready() is True


def test_serving_failure_is_immediately_unhealthy() -> None:
    health = DependencyHealth(postgres=True, fail_threshold=5)
    health.observe_probe_success("postgres")
    health.observe_serving_failure("postgres")
    assert health.snapshot("postgres").status is HealthStatus.UNHEALTHY
    assert health.serving_ready() is False


def test_optional_otel_failure_does_not_remove_readiness() -> None:
    health = DependencyHealth(postgres=True, telemetry=True)
    health.observe_probe_success("postgres")
    assert health.serving_ready() is True
    health.mark_optional_unhealthy("telemetry")
    assert health.snapshot("telemetry").status is HealthStatus.UNHEALTHY
    assert health.serving_ready() is True
    public = health.public_dependencies()
    assert public["telemetry"]["required"] is False
    assert public["telemetry"]["status"] == "unhealthy"


def test_public_dependencies_do_not_include_secrets() -> None:
    health = DependencyHealth(postgres=True)
    health.observe_probe_failure("postgres", reason="postgresql://fabric:secret@host/db")
    dumped = str(health.public_dependencies())
    assert "secret" not in dumped
    assert "postgresql://" not in dumped
    assert health.snapshot("postgres").reason == "probe_failed"


def test_admission_allow_and_reject() -> None:
    health = DependencyHealth(postgres=True, redis=True)
    health.observe_probe_success("postgres")
    health.observe_probe_success("redis")
    assert health.refusal() is None
    health.observe_serving_failure("postgres")
    refusal = health.refusal()
    assert isinstance(refusal, DependencyUnavailableError)
    assert refusal.status_code == 503
    assert refusal.retryable is True


def test_readiness_aggregation_ignores_optional() -> None:
    health = DependencyHealth(postgres=True, redis=True, telemetry=True)
    health.observe_probe_success("postgres")
    health.observe_probe_success("redis")
    health.mark_optional_unhealthy("telemetry")
    assert health.serving_ready() is True
    health.observe_serving_failure("redis")
    assert health.serving_ready() is False


def test_chat_is_rejected_before_provider_when_postgres_unhealthy() -> None:
    health = DependencyHealth(postgres=True, redis=True)
    health.observe_probe_success("postgres")
    health.observe_probe_success("redis")
    health.observe_serving_failure("postgres")
    app, provider = _app(health=health)
    with TestClient(app) as client:
        live = client.get("/healthz")
        ready = client.get("/readyz")
        chat = client.post(
            CHAT, json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]}
        )
    assert live.status_code == 200
    assert ready.status_code == 503
    assert ready.json()["ready"] is False
    assert ready.json()["status"] == "dependency_unhealthy"
    assert chat.status_code == 503
    assert chat.json()["error"]["type"] == "dependency_unavailable"
    assert chat.json()["error"]["retryable"] is True
    assert "request_id" in chat.json()["error"]
    assert "trace_id" in chat.json()["error"]
    assert provider.generate_calls == 0


def test_chat_is_rejected_before_provider_when_redis_unhealthy() -> None:
    health = DependencyHealth(postgres=True, redis=True)
    health.observe_probe_success("postgres")
    health.observe_probe_success("redis")
    health.observe_serving_failure("redis")
    app, provider = _app(health=health)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        chat = client.post(
            CHAT, json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]}
        )
    assert chat.status_code == 503
    assert provider.generate_calls == 0


def test_admission_allows_when_healthy() -> None:
    health = DependencyHealth(postgres=True, redis=True)
    health.observe_probe_success("postgres")
    health.observe_probe_success("redis")
    app, provider = _app(health=health)
    with TestClient(app) as client:
        chat = client.post(
            CHAT, json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]}
        )
    assert chat.status_code == 200
    assert provider.generate_calls == 1


def test_otel_optional_failure_keeps_chat_and_readyz() -> None:
    health = DependencyHealth(postgres=True, redis=True, telemetry=True)
    health.observe_probe_success("postgres")
    health.observe_probe_success("redis")
    health.mark_optional_unhealthy("telemetry")
    app, provider = _app(health=health)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        ready = client.get("/readyz")
        chat = client.post(
            CHAT, json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]}
        )
    assert ready.status_code == 200
    assert ready.json()["dependencies"]["telemetry"]["required"] is False
    assert ready.json()["dependencies"]["telemetry"]["status"] == "unhealthy"
    assert chat.status_code == 200
    assert provider.generate_calls == 1


def test_passive_postgres_failure_trips_admission() -> None:
    health = DependencyHealth(postgres=True)
    health.observe_probe_success("postgres")

    class BoomLedger:
        def insert(self, event: UsageEvent) -> None:
            del event
            raise OperationalError("statement", {}, Exception("connection refused"))

    meter = DurableMeter(MagicMock())
    meter._ledger = BoomLedger()  # type: ignore[assignment]
    meter._retry = deque()
    meter._lock = Lock()
    meter.bind_dependency_health(health)
    event = UsageEvent(
        event_id="e1",
        request_id="r1",
        invocation_id="e1",
        tenant_id="t",
        provider="mock",
        model="cheap",
        status=UsageStatus.SUCCESS.value,
        token_source=TokenSource.UNAVAILABLE.value,
    )
    meter.record_events([event])
    assert health.snapshot("postgres").status is HealthStatus.UNHEALTHY
    assert meter.retry_depth == 1

    app, provider = _app(health=health)
    with TestClient(app) as client:
        chat = client.post(
            CHAT, json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]}
        )
    assert chat.status_code == 503
    assert provider.generate_calls == 0
    assert meter.retry_depth == 1


def test_passive_redis_failure_trips_health() -> None:
    health = DependencyHealth(redis=True)
    health.observe_probe_success("redis")

    class Dead:
        def exists(self, *_args: object, **_kwargs: object) -> bool:
            raise __import__("redis").RedisError("down")

    store = RedisRevocationStore(Dead(), fail_closed=True)  # type: ignore[arg-type]
    store.bind_dependency_health(health)
    with pytest.raises(DependencyUnavailableError):
        store.is_revoked(token_id="x")
    assert health.snapshot("redis").status is HealthStatus.UNHEALTHY
    assert health.serving_ready() is False


def test_recovery_returns_admission() -> None:
    health = DependencyHealth(postgres=True, fail_threshold=1, recovery_threshold=1)
    health.observe_probe_success("postgres")
    health.observe_serving_failure("postgres")
    app, provider = _app(health=health)
    with TestClient(app) as client:
        denied = client.post(
            CHAT, json={"model": "cheap", "messages": [{"role": "user", "content": "no"}]}
        )
        assert denied.status_code == 503
        health.observe_probe_success("postgres")
        ok = client.post(
            CHAT, json={"model": "cheap", "messages": [{"role": "user", "content": "yes"}]}
        )
    assert ok.status_code == 200
    assert provider.generate_calls == 1


def test_post_outage_requests_do_not_fill_retry_buffer() -> None:
    health = DependencyHealth(postgres=True)
    health.observe_probe_success("postgres")
    health.observe_serving_failure("postgres")
    meter = InMemoryMeter()
    app, provider = _app(health=health, meter=meter)
    with TestClient(app) as client:
        for _ in range(100):
            response = client.post(
                CHAT,
                json={"model": "cheap", "messages": [{"role": "user", "content": "x"}]},
            )
            assert response.status_code == 503
    assert provider.generate_calls == 0
    assert meter.recent_events(limit=200) == []


@pytest.mark.asyncio
async def test_probe_timeout_is_bounded() -> None:
    health = DependencyHealth(postgres=True, fail_threshold=1)
    monitor = DependencyMonitor(
        health,
        database_url="postgresql://fabric:fabric@127.0.0.1:1/fabric",
        interval_s=0.05,
        timeout_s=0.2,
        jitter=False,
    )

    def hang() -> None:
        time.sleep(5)

    monitor._check_postgres = hang  # type: ignore[method-assign]
    started = time.perf_counter()
    await monitor.probe_once()
    elapsed = time.perf_counter() - started
    assert elapsed < 1.5
    assert health.snapshot("postgres").status is HealthStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_monitor_stop_cancels_task() -> None:
    health = DependencyHealth()
    monitor = DependencyMonitor(health, interval_s=30, timeout_s=1, jitter=False)
    monitor._database_url = "postgresql://example/db"
    monitor._check_postgres = lambda: None  # type: ignore[method-assign]
    await monitor.start()
    assert monitor.task is not None
    await monitor.stop()
    assert monitor.task is None


def test_detection_bound_formula() -> None:
    settings = Settings(_env_file=None)
    assert settings.health_detection_bound_s == detection_bound_s(
        interval_s=settings.health_probe_interval_s,
        timeout_s=settings.health_probe_timeout_s,
        fail_threshold=settings.health_fail_threshold,
    )
    assert settings.health_detection_bound_s == 6.0


def test_lifespan_monitor_does_not_leak() -> None:
    health = DependencyHealth()
    app, _provider = _app(health=health)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        monitor = client.app.state.dependency_monitor
        assert monitor is not None
    assert monitor.task is None or monitor.task.done()
