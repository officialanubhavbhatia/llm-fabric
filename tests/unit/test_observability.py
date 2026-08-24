"""Observability: labels stay bounded, missing backends stay unavailable."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from llm_fabric.observability.dashboards import VIEWS, DashboardAssembler
from llm_fabric.observability.engine import (
    OLLAMA_MEASUREMENTS,
    EngineMetricsHub,
    EngineSnapshot,
    ollama_unavailable,
    vllm_unavailable,
)
from llm_fabric.observability.langfuse import (
    HttpLangfuseAdapter,
    LangfuseSettings,
    NullLangfuse,
    build_langfuse,
)
from llm_fabric.observability.metering import InMemoryMeter, UsageRecord
from llm_fabric.observability.otel import (
    BUILT_STAGES,
    FabricTracer,
    RecordedSpan,
    SpanJournal,
    normalize_otlp_http_endpoint,
)
from llm_fabric.observability.prom import PATH_LABELS, FabricMetrics
from llm_fabric.observability.telemetry import (
    Telemetry,
    bind_telemetry,
    optional_span,
    reset_telemetry,
)
from llm_fabric.router.health import HealthTracker
from llm_fabric.router.registry import ModelRegistry


def _usage(**overrides: object) -> UsageRecord:
    base: dict[str, object] = {
        "request_id": "r1",
        "requested_model": "auto",
        "served_model": "cheap",
        "provider": "mock",
        "policy": "cost_first",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost_usd": 0.001,
        "cost_is_estimated": False,
        "latency_ms": 12.5,
        "streamed": False,
        "failover_count": 0,
        "tenant_id": "acme",
        "user_id": "alice",
    }
    base.update(overrides)
    return UsageRecord(**base)  # type: ignore[arg-type]


def test_prometheus_collapses_unknown_paths() -> None:
    metrics = FabricMetrics(CollectorRegistry())
    assert metrics.path_label("/v1/chat/completions") == "/v1/chat/completions"
    assert metrics.path_label("/v1/models/cheap") == "/v1/models/{id}"
    assert metrics.path_label("/v1/observability/dashboards/overview") == (
        "/v1/observability/dashboards/{view}"
    )
    assert metrics.path_label("/injected?x=" + "a" * 200) == "other"
    assert "other" in PATH_LABELS


def test_prometheus_never_labels_request_or_tenant_identity() -> None:
    metrics = FabricMetrics(CollectorRegistry())
    metrics.observe_http(method="POST", path="/v1/chat/completions", status=200, duration_s=0.01)
    metrics.observe_usage(
        prompt_tokens=3,
        completion_tokens=2,
        cost_usd=0.01,
        cost_is_estimated=True,
        provider="mock",
        policy="cost_first",
        failover_count=1,
        latency_s=0.02,
        error=False,
    )
    rendered = metrics.render().decode()
    assert "request_id" not in rendered
    assert "tenant_id" not in rendered
    assert "user_id" not in rendered
    assert "fabric_requests_total" in rendered
    assert 'path="/v1/chat/completions"' in rendered
    assert 'estimated="true"' in rendered


def test_engine_snapshots_are_unavailable_until_an_adapter_exists() -> None:
    ollama = ollama_unavailable().snapshot()
    vllm = vllm_unavailable().snapshot()
    assert ollama.available is False
    assert vllm.available is False
    assert ollama.measurements["loaded_models"] is None
    assert "kv_cache_utilization" not in ollama.measurements
    assert all(
        name in vllm.measurements
        for name in (
            "kv_cache_usage",
            "prefix_cache_hits",
            "running_requests",
        )
    )
    assert all(value is None for value in vllm.measurements.values())
    assert "not configured" in ollama.note or "stay" in ollama.note
    for name in OLLAMA_MEASUREMENTS:
        assert name in ollama.measurements


def test_langfuse_is_a_noop_without_credentials() -> None:
    sink = build_langfuse(host=None, public_key=None, secret_key=None)
    assert isinstance(sink, NullLangfuse)
    assert sink.enabled is False


@pytest.mark.asyncio
async def test_langfuse_adapter_posts_ingestion_without_prompt_text() -> None:
    captured: list[dict] = []

    class _Client:
        async def post(self, url, json, headers):  # noqa: ANN001, ANN201
            captured.append({"url": url, "json": json, "headers": headers})

            class _Resp:
                def raise_for_status(self) -> None:
                    return None

            return _Resp()

        async def aclose(self) -> None:
            return None

    adapter = HttpLangfuseAdapter(
        LangfuseSettings(host="https://langfuse.example", public_key="pk", secret_key="sk"),
        client=_Client(),  # type: ignore[arg-type]
    )
    await adapter.export_trace(
        [
            RecordedSpan(
                name="request",
                trace_id="abc",
                span_id="def",
                parent_span_id=None,
                duration_ms=1.0,
                status="ok",
                attributes={"http_method": "POST"},
                tenant_id="acme",
                user_id="alice",
                started_at=1.0,
            )
        ]
    )
    assert captured
    body = captured[0]["json"]["batch"]
    dumped = str(body)
    assert "user said" not in dumped
    assert any(event["type"] == "trace-create" for event in body)


def test_optional_span_is_a_noop_without_bound_telemetry() -> None:
    with optional_span("llm"):
        pass


def test_optional_span_journals_when_telemetry_is_bound() -> None:
    telemetry = Telemetry(tracer=FabricTracer())
    token = bind_telemetry(telemetry)
    try:
        with optional_span("llm", gen_ai_system="mock"):
            pass
    finally:
        reset_telemetry(token)
    names = [span.name for span in telemetry.tracer.journal.recent()]
    assert "llm" in names


def test_every_named_dashboard_exists_and_unbuilt_ones_stay_empty() -> None:
    registry = ModelRegistry.from_mapping(
        {"models": [{"id": "cheap", "provider": "mock", "input_price_per_mtok": 0.1}]}
    )
    meter = InMemoryMeter()
    meter.record(_usage())
    assembler = DashboardAssembler(
        meter=meter,
        journal=SpanJournal(),
        health=HealthTracker(),
        registry=registry,
        engines=EngineMetricsHub(),
    )
    for view in VIEWS:
        payload = assembler.render(view, tenant_id="acme", fleet=False, scope_note="test")
        assert payload["view"] == view
        if view in {"batching", "threads"}:
            assert payload["available"] is False
            assert payload["data"] is None
            assert payload["note"]
        else:
            assert payload["available"] is True
            assert payload["data"] is not None


def test_overview_does_not_invent_quality_or_safety() -> None:
    registry = ModelRegistry.from_mapping(
        {"models": [{"id": "cheap", "provider": "mock", "input_price_per_mtok": 0.1}]}
    )
    meter = InMemoryMeter()
    meter.record(_usage())
    payload = DashboardAssembler(
        meter=meter,
        journal=SpanJournal(),
        health=HealthTracker(),
        registry=registry,
        engines=EngineMetricsHub(),
    ).render("overview", tenant_id="acme", fleet=False, scope_note="test")
    assert "quality" not in payload["data"]
    assert "safety" in payload["unavailable_fields"]
    assert payload["data"]["requests"] == 1
    coverage = payload["data"]["coverage"]
    assert coverage["intent_serving"] is None
    assert coverage["context_record"] is None
    assert coverage["supported_telemetry_provenance"] is None
    assert "tps" in payload["unavailable_fields"]


def test_unbuilt_lifecycle_stages_are_listed_not_timed() -> None:
    assert "context" in BUILT_STAGES
    assert "eval" not in BUILT_STAGES
    assert "request" in BUILT_STAGES


class _LiveVllm:
    def snapshot(self) -> EngineSnapshot:
        return EngineSnapshot(
            provider="vllm",
            available=True,
            measurements={"kv_cache_usage": 0.42, "running_requests": 3.0},
            unsupported=(),
            note="scraped from the engine",
        )


def test_vllm_measurements_appear_only_when_an_adapter_reports_them() -> None:
    hub = EngineMetricsHub({"vllm": _LiveVllm()})
    snap = hub.for_provider("vllm")
    assert snap.available is True
    assert snap.measurements["kv_cache_usage"] == pytest.approx(0.42)
    assert EngineMetricsHub().for_provider("vllm").available is False


def test_otlp_http_endpoint_appends_traces_path() -> None:
    assert (
        normalize_otlp_http_endpoint("http://otel-collector:4318")
        == "http://otel-collector:4318/v1/traces"
    )
    assert (
        normalize_otlp_http_endpoint("http://otel-collector:4318/v1/traces")
        == "http://otel-collector:4318/v1/traces"
    )
