"""Serving-path ContextRecord coverage."""

from __future__ import annotations

from fastapi.testclient import TestClient

from llm_fabric.gateway.app import create_app
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.observability.usage_event import (
    UsageOperation,
    provider_invocations_without_context_record,
)
from llm_fabric.serving.adapters.mock import MockProvider


def test_provider_invocations_without_context_record_is_zero(registry, settings, providers) -> None:
    meter = InMemoryMeter()
    app = create_app(
        settings=settings,
        registry=registry,
        provider_overrides=providers,
        meter=meter,
    )
    with TestClient(app) as client:
        buffered = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert buffered.status_code == 200
        assert buffered.headers["x-fabric-context-record-id"]
        streamed = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert streamed.status_code == 200
        list(streamed.iter_lines())
    events = meter.recent_events(limit=200)
    user_events = [
        event for event in events if event.operation == UsageOperation.USER_RESPONSE.value
    ]
    assert user_events
    assert provider_invocations_without_context_record(events) == 0
    assert all(event.context_record_id for event in user_events)


def test_command_center_context_view_is_available(registry, settings) -> None:
    meter = InMemoryMeter()
    app = create_app(
        settings=settings,
        registry=registry,
        provider_overrides={"mock": MockProvider()},
        meter=meter,
    )
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
            ).status_code
            == 200
        )
        context = client.get("/v1/observability/dashboards/context").json()
        kv = client.get("/v1/observability/dashboards/kv_cache").json()
    assert context["available"] is True
    assert context["data"]["records"]
    assert context["data"]["supported_metrics_without_provenance"] == 0
    assert context["data"]["overflow_rejection"]["provenance"] == "UNAVAILABLE"
    assert "before_optimization" in context["data"]
    assert kv["available"] is True
    assert kv["data"]["scope"] == "DEPLOYMENT"
    assert "filters" in kv["data"]
    assert kv["data"]["filters"]["pod"]["available"] is False
    assert "this request" not in (kv.get("note") or "").lower() or "not this request" in kv["note"]
    matrix = kv["data"]["capability_matrix"]
    ollama = next(row for row in matrix["runtimes"] if row["runtime"] == "ollama")
    assert ollama["categories"]["kv_utilization"]["supported"] is False
    assert "DOES NOT EXPOSE" in ollama["categories"]["kv_utilization"]["note"]
