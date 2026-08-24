"""Assemble Command Center views from sources that actually exist.

Every named view is reachable. A view whose backend is not built returns
`available: false` and an empty `data`, never a zero that would read as a
measurement. Fields inside an otherwise-available view that cannot be produced
are listed in `unavailable_fields`.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_fabric.context.record import ContextRecord
from llm_fabric.eval.schema import EvalRun
from llm_fabric.heal.drift import analyze
from llm_fabric.heal.schema import Incident, RemediationRecord
from llm_fabric.intent.metrics import percentile
from llm_fabric.observability.capability import capability_matrix
from llm_fabric.observability.engine import (
    UNAVAILABLE_INFERENCE_METRICS,
    EngineMetricsHub,
    EngineSnapshot,
)
from llm_fabric.observability.metering import UsageMeter, UsageRecord
from llm_fabric.observability.metric import (
    CountProvenance,
    MetricScope,
    Observed,
    provenance_missing,
)
from llm_fabric.observability.ollama_metrics import OLLAMA_DOES_NOT_EXPOSE
from llm_fabric.observability.otel import BUILT_STAGES, LIFECYCLE_STAGES, SpanJournal
from llm_fabric.observability.usage_event import (
    UsageEvent,
    UsageOperation,
    provider_invocations_without_context_record,
    provider_invocations_without_intent,
)
from llm_fabric.router.health import HealthTracker
from llm_fabric.router.registry import ModelRegistry, ModelSpec

VIEWS: tuple[str, ...] = (
    "overview",
    "users",
    "tenants",
    "requests",
    "traces",
    "threads",
    "intents",
    "models",
    "promotion",
    "tiers",
    "kv_cache",
    "batching",
    "routing",
    "fallbacks",
    "tokens",
    "context",
    "economics",
    "evals",
    "drift",
    "reliability",
)

_NOT_BUILT = {
    "batching": (
        "vLLM does not expose a stable batch-utilization series on /metrics "
        "in the versions this catalog parses. The field is UNAVAILABLE, not zero."
    ),
    "threads": (
        "Chat completions are not persisted as conversations, so traces "
        "cannot be grouped into threads. The conversation store exists; "
        "nothing writes to it from the serving path."
    ),
}


def _envelope(
    view: str,
    *,
    available: bool,
    data: Any = None,
    unavailable_fields: Sequence[str] = (),
    note: str | None = None,
    scope: str,
    source: str = "meter",
    estimated_fields: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "view": view,
        "available": available,
        "supported": available,
        "source": source if available else "none",
        "estimated": bool(estimated_fields),
        "estimated_fields": list(estimated_fields),
        "collected_at": time.time() if available else None,
        "data": data,
        "unavailable_fields": list(unavailable_fields),
        "note": note,
        "scope": scope,
    }


def _measured_by_deployment() -> dict[str, Any]:
    path = Path("datasets/eval/models/leaderboard.json")
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows = payload.get("models") or payload.get("leaderboard") or []
    return {str(row.get("deployment")): row for row in rows if row.get("deployment")}


def _latencies(records: Sequence[UsageRecord]) -> dict[str, float | None]:
    values = [record.latency_ms for record in records]
    return {
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "count": len(values) or None,
    }


def _unavailable(
    name: str, *, scope: MetricScope, note: str, unit: str | None = None
) -> dict[str, Any]:
    return Observed.unavailable(name, scope=scope, note=note, unit=unit).as_dict()


def _measured(
    name: str,
    value: float | int | None,
    *,
    provenance: CountProvenance,
    scope: MetricScope,
    note: str | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    if value is None:
        return _unavailable(
            name,
            scope=scope,
            note=note or f"{name} was not measured on this sample",
            unit=unit,
        )
    return Observed(
        name=name,
        value=value,
        provenance=provenance,
        scope=scope,
        unit=unit,
        note=note,
    ).as_dict()


def _engine_observation(engine: EngineSnapshot, name: str) -> dict[str, Any] | None:
    for item in engine.observations:
        if item.name == name:
            return item.as_dict()
    if name in engine.measurements:
        value = engine.measurements[name]
        if value is None:
            return _unavailable(
                name,
                scope=MetricScope.DEPLOYMENT,
                note=engine.note or "engine did not report this series",
            )
        return Observed(
            name=name,
            value=value,
            provenance=CountProvenance.PROVIDER_MEASURED,
            scope=MetricScope.DEPLOYMENT,
            note=engine.note or None,
        ).as_dict()
    return None


def _runtime_unavailable(name: str, spec: ModelSpec, engine: EngineSnapshot) -> dict[str, Any]:
    runtime = spec.runtime.value
    if runtime == "ollama" or spec.provider == "ollama":
        note = OLLAMA_DOES_NOT_EXPOSE
    elif runtime == "vllm" or spec.provider == "vllm":
        note = engine.note or "vLLM /metrics was not scraped on this process"
    elif spec.provider_adapter == "litellm" or spec.provider == "litellm":
        note = "UNAVAILABLE — LiteLLM is transport, not an inference engine"
    else:
        note = engine.note or f"{spec.provider} does not expose {name}"
    return _unavailable(name, scope=MetricScope.DEPLOYMENT, note=note)


def _prefix_hit_ratio(engine: EngineSnapshot) -> dict[str, Any]:
    hits = _engine_observation(engine, "prefix_cache_hit_tokens")
    queries = _engine_observation(engine, "prefix_cache_query_tokens")
    gauge = _engine_observation(engine, "prefix_cache_hit_ratio") or _engine_observation(
        engine, "prefix_cache_hit_ratio_legacy_gauge"
    )
    if gauge is not None and gauge.get("value") is not None:
        return gauge
    hit_value = None if hits is None else hits.get("value")
    query_value = None if queries is None else queries.get("value")
    if hit_value is not None and query_value not in (None, 0):
        return Observed(
            name="prefix_cache_hit_ratio",
            value=round(float(hit_value) / float(query_value), 6),
            provenance=CountProvenance.DERIVED,
            scope=MetricScope.DEPLOYMENT,
            note="hits / queries from the vLLM scrape; DEPLOYMENT scope, not request KV use",
        ).as_dict()
    return _unavailable(
        "prefix_cache_hit_ratio",
        scope=MetricScope.DEPLOYMENT,
        note="prefix-cache hit ratio requires a vLLM scrape with hits and queries",
    )


class DashboardAssembler:
    def __init__(
        self,
        *,
        meter: UsageMeter,
        journal: SpanJournal,
        health: HealthTracker,
        registry: ModelRegistry,
        engines: EngineMetricsHub,
        intent_snapshot: Mapping[str, Any] | None = None,
        eval_runs: Sequence[EvalRun] = (),
        incidents: Sequence[Incident] = (),
        remediations: Sequence[RemediationRecord] = (),
        promotion_state_path: Path | None = None,
        context_records: Sequence[ContextRecord] = (),
    ) -> None:
        self._meter = meter
        self._journal = journal
        self._health = health
        self._registry = registry
        self._engines = engines
        self._intent = intent_snapshot
        self._eval_runs = eval_runs
        self._incidents = incidents
        self._remediations = remediations
        self._promotion_state_path = promotion_state_path
        self._context_records = context_records

    def _events(
        self, *, tenant_id: str | None, fleet: bool, request_id: str | None = None
    ) -> list[UsageEvent]:
        return self._meter.recent_events(
            limit=10_000,
            tenant_id=None if fleet else tenant_id,
            observe=fleet,
            request_id=request_id,
        )

    def _coverage(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
    ) -> dict[str, Any]:
        events = self._events(tenant_id=tenant_id, fleet=fleet)
        user_events = [
            event for event in events if event.operation == UsageOperation.USER_RESPONSE.value
        ]
        without_intent = provider_invocations_without_intent(events)
        without_context = provider_invocations_without_context_record(events)
        user_n = len(user_events)
        observations = [
            obs
            for record in self._context_records
            if fleet or tenant_id is None or record.tenant_id == tenant_id
            for obs in record.observations()
        ]
        missing_provenance = provenance_missing(observations)
        supported_n = len(observations)
        return {
            "user_response_invocations": user_n or None,
            "buffered_requests": len(records) or None,
            "provider_invocations_without_intent": without_intent if user_n else None,
            "provider_invocations_without_context_record": without_context if user_n else None,
            "supported_metrics_without_provenance": (missing_provenance if supported_n else None),
            "intent_serving": round((user_n - without_intent) / user_n, 4) if user_n else None,
            "context_record": (round((user_n - without_context) / user_n, 4) if user_n else None),
            "supported_telemetry_provenance": (
                round((supported_n - missing_provenance) / supported_n, 4) if supported_n else None
            ),
            "note": (
                "Coverage is attachment of IntentResult and ContextRecord on "
                "USER_RESPONSE invocations, and provenance on compiled context "
                "observations. Empty buffer is not 100%. This is not "
                "classification accuracy."
            ),
        }

    def render(
        self,
        view: str,
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        if view not in VIEWS:
            raise ValueError(f"unknown view '{view}'")
        records = self._meter.recent(
            limit=10_000, tenant_id=None if fleet else tenant_id, observe=fleet
        )
        if view in _NOT_BUILT:
            return _envelope(
                view,
                available=False,
                note=_NOT_BUILT[view],
                scope=scope_note,
            )
        builder = {
            "overview": self._overview,
            "users": self._users,
            "tenants": self._tenants,
            "requests": self._requests,
            "traces": self._traces,
            "intents": self._intents,
            "models": self._models,
            "promotion": self._promotion,
            "tiers": self._tiers,
            "routing": self._routing,
            "fallbacks": self._fallbacks,
            "tokens": self._tokens,
            "context": self._context,
            "kv_cache": self._kv_cache,
            "economics": self._economics,
            "evals": self._evals,
            "drift": self._drift,
            "reliability": self._reliability,
        }[view]
        return builder(records, tenant_id=tenant_id, fleet=fleet, scope_note=scope_note)

    def _overview(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        successes = [r for r in records if r.error is None]
        errors = [r for r in records if r.error is not None]
        tokens = sum(
            (record.ledger_prompt_tokens + record.ledger_completion_tokens)
            if record.ledger_prompt_tokens is not None
            and record.ledger_completion_tokens is not None
            else record.total_tokens
            for record in records
        )
        cost = sum(r.cost_usd for r in records)
        estimated = sum(1 for r in records if r.cost_is_estimated)
        rps = None
        if len(records) >= 2:
            span = max(r.created_at for r in records) - min(r.created_at for r in records)
            if span >= 1.0:
                rps = round(len(records) / span, 3)
        by_provider: dict[str, int] = defaultdict(int)
        by_model: dict[str, int] = defaultdict(int)
        by_tier: dict[str, int] = defaultdict(int)
        for record in records:
            by_provider[record.provider] += 1
            by_model[record.served_model] += 1
            if record.selected_tier:
                by_tier[record.selected_tier] += 1
        fallbacks = sum(r.failover_count for r in records)
        reliability = (len(successes) / len(records)) if records else None
        return _envelope(
            "overview",
            available=True,
            data={
                "coverage": self._coverage(records, tenant_id=tenant_id, fleet=fleet),
                "requests": len(records),
                "reliability": reliability,
                "rps": rps,
                "successes": len(successes),
                "errors": len(errors),
                "error_rate": (len(errors) / len(records)) if records else None,
                "latency": _latencies(records),
                "tokens": tokens,
                "cost_usd": round(cost, 6),
                "requests_with_estimated_cost": estimated,
                "failovers": fallbacks,
                "fallback_rate": (fallbacks / len(records)) if records else None,
                "by_provider": dict(by_provider),
                "by_served_model": dict(by_model),
                "by_tier": dict(by_tier) or None,
            },
            unavailable_fields=["quality", "safety", "tps", "queue_depth"],
            estimated_fields=["cost_usd"] if estimated else (),
            note=(
                "Quality, safety and tokens-per-second have no source in this "
                "build and are not shown as numbers. Process-buffer rps is "
                "gateway request rate, not inference tokens/sec. Queue depth "
                "is not a fleet queue; in-flight counts live on reliability."
            ),
            scope=scope_note,
        )

    def _users(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del tenant_id, fleet
        by_user: dict[tuple[str, str | None], list[UsageRecord]] = defaultdict(list)
        for record in records:
            by_user[(record.tenant_id, record.user_id)].append(record)
        rows = []
        for (tenant, user), group in sorted(by_user.items(), key=lambda item: -len(item[1])):
            rows.append(
                {
                    "tenant_id": tenant,
                    "user_id": user,
                    "requests": len(group),
                    "tokens": sum(r.total_tokens for r in group),
                    "cost_usd": round(sum(r.cost_usd for r in group), 6),
                    "errors": sum(1 for r in group if r.error),
                    "latency": _latencies(group),
                }
            )
        return _envelope(
            "users",
            available=True,
            data={"users": rows},
            unavailable_fields=["quality"],
            note="Quality per user is not measured.",
            scope=scope_note,
        )

    def _tenants(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        if not fleet:
            return _envelope(
                "tenants",
                available=True,
                data={"tenants": [self._tenant_row(tenant_id or "", records)]},
                unavailable_fields=["quality"],
                note="Fleet-wide tenant comparison requires the fabric:observe scope.",
                scope=scope_note,
            )
        by_tenant: dict[str, list[UsageRecord]] = defaultdict(list)
        for record in records:
            by_tenant[record.tenant_id].append(record)
        rows = [self._tenant_row(tid, group) for tid, group in sorted(by_tenant.items())]
        return _envelope(
            "tenants",
            available=True,
            data={"tenants": rows},
            unavailable_fields=["quality"],
            scope=scope_note,
        )

    def _tenant_row(self, tenant_id: str, records: Sequence[UsageRecord]) -> dict[str, Any]:
        return {
            "tenant_id": tenant_id,
            "requests": len(records),
            "tokens": sum(r.total_tokens for r in records),
            "cost_usd": round(sum(r.cost_usd for r in records), 6),
            "errors": sum(1 for r in records if r.error),
            "failovers": sum(r.failover_count for r in records),
            "latency": _latencies(records),
        }

    def _requests(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        events = self._events(tenant_id=tenant_id, fleet=fleet)
        by_request: dict[str, list[UsageEvent]] = defaultdict(list)
        for event in events:
            by_request[event.request_id].append(event)
        ctx_by_id = {record.context_record_id: record for record in self._context_records}
        shown = records[:100]
        return _envelope(
            "requests",
            available=True,
            data={
                "recent": [
                    self._request_row(
                        record,
                        events=by_request.get(record.request_id, ()),
                        context=(
                            ctx_by_id.get(record.context_record_id)
                            if record.context_record_id
                            else None
                        ),
                    )
                    for record in shown
                ],
                "shown": min(100, len(records)),
                "buffered": len(records),
            },
            note=(
                "Durable postgres ledger."
                if self._meter.durable
                else "In-memory buffer, this process only, lost on restart. "
                "Engine snapshot is DEPLOYMENT-scoped, not this request's KV use."
            ),
            scope=scope_note,
        )

    def _request_row(
        self,
        record: UsageRecord,
        *,
        events: Sequence[UsageEvent] = (),
        context: ContextRecord | None = None,
    ) -> dict[str, Any]:
        user_events = [
            event for event in events if event.operation == UsageOperation.USER_RESPONSE.value
        ]
        last = user_events[0] if user_events else (events[0] if events else None)
        engine = self._engines.for_provider(record.provider)
        accounting = (
            context.as_dict(include_content=False)["accounting"] if context is not None else None
        )
        context_section: dict[str, Any]
        if accounting is not None:
            context_section = {
                "context_record_id": context.context_record_id if context is not None else None,
                "before_tokens": accounting.get("context_tokens_before_optimization"),
                "after_tokens": accounting.get("context_tokens_after_optimization"),
                "stable_prefix": accounting.get("stable_prefix_tokens"),
                "volatile_suffix": accounting.get("volatile_prompt_tokens"),
                "deduplicated": accounting.get("tokens_deduplicated"),
                "compressed": accounting.get("tokens_compressed"),
                "dropped": accounting.get("tokens_dropped"),
                "context_utilization": accounting.get("context_utilization_percent"),
            }
        elif record.context_record_id:
            context_section = {
                "context_record_id": record.context_record_id,
                "note": (
                    "ContextRecord id is on the usage row; "
                    "the compiler buffer no longer holds the body."
                ),
            }
        else:
            context_section = {
                "context_record_id": None,
                "note": "no ContextRecord on this usage row",
            }
        request_unavail = "not measured on this request (buffered mock often has no stream timings)"
        return {
            "request_id": record.request_id,
            "tenant_id": record.tenant_id,
            "served_model": record.served_model,
            "latency_ms": record.latency_ms,
            "total_tokens": record.total_tokens,
            "intent_id": record.intent_id,
            "error": record.error,
            "identity": {
                "request_id": record.request_id,
                "trace_id": record.trace_id,
                "tenant_id": record.tenant_id,
                "user_id": record.user_id,
                "project_id": last.project_id if last is not None else None,
            },
            "intent": {
                "domain_task": record.intent_id,
                "confidence": record.intent_confidence,
                "classifier_layer": record.intent_layer,
                "taxonomy_version": last.taxonomy_version if last is not None else None,
                "classifier_version": last.classifier_version if last is not None else None,
                "cache_hit": record.intent_cache_hit,
                "state": _unavailable(
                    "serving_state",
                    scope=MetricScope.REQUEST,
                    note=(
                        "serving_state is counted on the IntentOS process view; "
                        "it is not persisted on UsageRecord"
                    ),
                ),
            },
            "context": context_section,
            "route": {
                "grade": last.grade if last is not None else None,
                "deployment": last.deployment_id if last is not None else record.served_model,
                "provider_adapter": last.provider_adapter if last is not None else None,
                "transport": last.transport if last is not None else None,
                "runtime": last.runtime if last is not None else None,
                "provider": record.provider,
                "policy": record.policy,
                "selected_tier": record.selected_tier,
                "actual_served_model": (
                    (last.actual_served_model if last is not None else None) or record.served_model
                ),
                "fallback_depth": (
                    last.fallback_depth if last is not None else record.failover_count
                ),
            },
            "tokens": {
                "prompt": record.prompt_tokens,
                "completion": record.completion_tokens,
                "total": record.total_tokens,
                "cached": _measured(
                    "cached_tokens",
                    last.cached_tokens if last is not None else None,
                    provenance=CountProvenance.PROVIDER_MEASURED,
                    scope=MetricScope.REQUEST,
                    unit="tokens",
                    note="only when the backend reported cached tokens",
                ),
                "provenance": last.token_source if last is not None else None,
            },
            "performance": {
                "total_ms": record.latency_ms,
                "ttft_ms": _measured(
                    "ttft_ms",
                    record.ttft_ms,
                    provenance=CountProvenance.DERIVED,
                    scope=MetricScope.REQUEST,
                    unit="ms",
                    note="gateway first-byte on streaming requests; " + request_unavail,
                ),
                "tpot_ms": _measured(
                    "tpot_ms",
                    record.tpot_ms,
                    provenance=CountProvenance.DERIVED,
                    scope=MetricScope.REQUEST,
                    unit="ms",
                    note="gateway inter-token on streaming requests; " + request_unavail,
                ),
                "queue": _unavailable(
                    "queue_ms",
                    scope=MetricScope.REQUEST,
                    note="request queue wait is not measured on this gateway path",
                    unit="ms",
                ),
                "prefill": _unavailable(
                    "prefill_ms",
                    scope=MetricScope.REQUEST,
                    note="prefill duration requires a runtime that reports it",
                    unit="ms",
                ),
                "decode": _unavailable(
                    "decode_ms",
                    scope=MetricScope.REQUEST,
                    note=(
                        "decode duration requires a runtime that reports it; "
                        "gateway e2e is not decode"
                    ),
                    unit="ms",
                ),
                "prefill_tps": _unavailable(
                    "prefill_tps",
                    scope=MetricScope.REQUEST,
                    note="prefill TPS requires prompt tokens and prefill duration",
                ),
                "decode_tps": _unavailable(
                    "decode_tps",
                    scope=MetricScope.REQUEST,
                    note="decode TPS requires completion tokens and decode duration",
                ),
            },
            "engine_snapshot": {
                "scope": "DEPLOYMENT",
                "note": "Engine KV/prefix/running/waiting gauges are not this request's KV use.",
                **engine.as_dict(),
            },
        }

    def _traces(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del records
        trees = self._journal.traces(limit=50, tenant_id=None if fleet else tenant_id)
        return _envelope(
            "traces",
            available=True,
            data={
                "expected_tree": list(LIFECYCLE_STAGES),
                "built_stages": sorted(BUILT_STAGES),
                "unbuilt_stages": [s for s in LIFECYCLE_STAGES if s not in BUILT_STAGES],
                "traces": trees,
            },
            note=(
                "local-pod diagnostic only — not authoritative fleet trace "
                "history. This journal is per process and is lost on restart. "
                "Fleet traces belong in the OTLP backend when "
                "LLM_FABRIC_OTEL_EXPORTER_OTLP_ENDPOINT is set."
            ),
            scope=scope_note,
        )

    def _intents(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del records, tenant_id, fleet
        snapshot = dict(self._intent or {})
        routing_on = bool(snapshot.get("routing_enabled"))
        cascade_present = bool(snapshot.get("cascade_present"))
        serving_requests = int(snapshot.get("serving_requests") or 0)
        known = int(snapshot.get("known") or 0)
        unknown = int(snapshot.get("unknown") or 0)
        abstain = int(snapshot.get("abstentions") or 0)
        safe_fallback = int(snapshot.get("safe_fallback") or 0)
        missing = int(snapshot.get("missing") or 0)
        coverage_denom = serving_requests or (known + unknown + abstain + safe_fallback)
        coverage = None
        if coverage_denom:
            coverage = round((coverage_denom - missing) / coverage_denom, 4)
        cache_hits = snapshot.get("cache_hits") or {}
        serving = {
            "coverage": coverage,
            "requests": serving_requests if coverage_denom else None,
            "known": known if coverage_denom else None,
            "unknown": unknown if coverage_denom else None,
            "abstain": abstain if coverage_denom else None,
            "safe_fallback": safe_fallback if coverage_denom else None,
            "missing": missing,
            "known_pct": round(known / coverage_denom, 4) if coverage_denom else None,
            "unknown_pct": round(unknown / coverage_denom, 4) if coverage_denom else None,
            "abstain_pct": round(abstain / coverage_denom, 4) if coverage_denom else None,
            "classifier_layers": snapshot.get("by_layer"),
            "layer_distribution": snapshot.get("by_layer"),
            "confidence": snapshot.get("confidence"),
            "latency_ms": snapshot.get("latency_ms"),
            "l0_hits": (cache_hits or {}).get("exact"),
            "l1_hits": (cache_hits or {}).get("semantic"),
            "exact_cache_hits": (cache_hits or {}).get("exact"),
            "semantic_cache_hits": (cache_hits or {}).get("semantic"),
            "classifier_version": snapshot.get("classifier_version"),
            "taxonomy_version": snapshot.get("taxonomy_version"),
        }
        return _envelope(
            "intents",
            available=True,
            data={
                "safety_gates": {
                    "hard_negative_accuracy": 0.50,
                    "required": 0.58,
                    "routing": "OFF",
                    "source": "frozen_eval_v1.1",
                    "serving_path_classification": cascade_present and routing_on,
                },
                "serving": serving,
                "classifications": snapshot.get("classifications") or None,
                "abstention_rate": snapshot.get("abstention_rate"),
                "unknown": snapshot.get("unknown") or None,
                "escalations": snapshot.get("escalations") or None,
                "disagreements": snapshot.get("disagreements") or None,
                "cache_hits": cache_hits or None,
                "by_layer": snapshot.get("by_layer"),
                "confidence": snapshot.get("confidence"),
                "latency_ms": snapshot.get("latency_ms"),
                "full": snapshot or None,
            },
            unavailable_fields=["misclassifications", "newly_discovered_clusters", "drift"],
            note=(
                "Safety gates first: hard-negative accuracy on the frozen 98 "
                "is 0.50 (required >= 0.58). Serving coverage is whether every "
                "provider invocation carried an IntentResult (known / unknown / "
                "abstain / safe_fallback), not classification accuracy. "
                "Do not read coverage as classification accuracy. Drift is "
                "unavailable without labelled traffic."
            ),
            source="frozen_eval",
            scope=scope_note,
        )

    def _models(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del tenant_id, fleet
        by_model: dict[str, list[UsageRecord]] = defaultdict(list)
        for record in records:
            by_model[record.served_model].append(record)
        health = self._health.all_snapshots()
        measured = _measured_by_deployment()
        rows = []
        for spec in self._registry.enabled_models():
            group = by_model.get(spec.id, [])
            snap = health.get(spec.deployment_id)
            engine = self._engines.for_provider(spec.provider)
            errors = sum(1 for r in group if r.error)
            measured_row = measured.get(spec.id) or {}
            ttft_vals = [r.ttft_ms for r in group if r.ttft_ms is not None]
            tpot_vals = [r.tpot_ms for r in group if r.tpot_ms is not None]
            rps = None
            if len(group) >= 2:
                span = max(r.created_at for r in group) - min(r.created_at for r in group)
                if span >= 1.0:
                    rps = round(len(group) / span, 3)
            prompt_tps = _engine_observation(engine, "prefill_tps") or _runtime_unavailable(
                "prompt_tps", spec, engine
            )
            decode_tps = _engine_observation(engine, "decode_tps") or _runtime_unavailable(
                "decode_tps", spec, engine
            )
            kv_util = _engine_observation(engine, "kv_cache_utilization") or _runtime_unavailable(
                "kv_utilization", spec, engine
            )
            prefix_ratio = _prefix_hit_ratio(engine)
            if prefix_ratio.get("value") is None and (
                spec.runtime.value == "ollama" or spec.provider == "ollama"
            ):
                prefix_ratio = _runtime_unavailable("prefix_cache_hit_rate", spec, engine)
            cached_tokens = _engine_observation(
                engine, "cached_prompt_tokens"
            ) or _runtime_unavailable("cached_tokens", spec, engine)
            running = _engine_observation(engine, "running_requests") or _runtime_unavailable(
                "running", spec, engine
            )
            waiting = _engine_observation(engine, "waiting_requests") or _runtime_unavailable(
                "waiting", spec, engine
            )
            preemptions = _engine_observation(engine, "preemptions") or _runtime_unavailable(
                "preemptions", spec, engine
            )
            rows.append(
                {
                    "deployment": spec.id,
                    "provider": spec.provider,
                    "grade": spec.grade.value if spec.grade else None,
                    "runtime": spec.runtime.value,
                    "transport": spec.transport.value,
                    "provider_adapter": spec.provider_adapter or None,
                    "hardware": spec.placement.hardware,
                    "tier_range": [tier.value for tier in spec.tiers] or None,
                    "lifecycle": spec.lifecycle.value,
                    "production_eligible": spec.lifecycle.value == "approved"
                    and spec.promotion_identity_match
                    and spec.enabled,
                    "declared_tiers": [tier.value for tier in spec.tiers] or None,
                    "approved_tiers": [tier.value for tier in spec.approved_tiers] or None,
                    "revision": spec.revision,
                    "pool": spec.pool,
                    "health": snap.state.value if snap else "unknown",
                    "requests": len(group) or None,
                    "errors": errors if group else None,
                    "rps": _measured(
                        "rps",
                        rps,
                        provenance=CountProvenance.DERIVED,
                        scope=MetricScope.MODEL,
                        note=(
                            "process-buffer request rate for this deployment, "
                            "not inference tokens/sec"
                        ),
                    ),
                    "p95_ms": _latencies(group)["p95_ms"],
                    "ttft": {
                        "p50_ms": percentile(ttft_vals, 0.50) if ttft_vals else None,
                        "p95_ms": percentile(ttft_vals, 0.95) if ttft_vals else None,
                        "p99_ms": percentile(ttft_vals, 0.99) if ttft_vals else None,
                        "count": len(ttft_vals) or None,
                    },
                    "tpot_ms": percentile(tpot_vals, 0.50) if tpot_vals else None,
                    "prompt_tps": prompt_tps,
                    "decode_tps": decode_tps,
                    "kv_utilization": kv_util,
                    "prefix_cache_hit_rate": prefix_ratio,
                    "cached_tokens": cached_tokens,
                    "running": running,
                    "waiting": waiting,
                    "preemptions": preemptions,
                    "error_rate": (errors / len(group)) if group else None,
                    "quality_status": {
                        "declared": spec.quality.as_dict(),
                        "measured": measured_row or None,
                        "label": (
                            "measured"
                            if measured_row
                            else ("declared" if spec.quality.as_dict() else "unknown")
                        ),
                    },
                    "probe_status": measured_row.get("probe_status") or "unknown",
                    "capabilities": {
                        "declared": sorted(spec.capabilities.declared),
                        "measured": measured_row.get("capabilities"),
                    },
                    "capabilities_declared": ",".join(sorted(spec.capabilities.declared)) or None,
                    "capabilities_measured": (
                        "measured" if measured_row.get("capabilities") else "unknown"
                    ),
                    "quality_declared": "declared" if spec.quality.as_dict() else "unknown",
                    "quality_measured": "measured" if measured_row else "not evaluated",
                    "cost": {
                        "declared": spec.api_cost_knowledge.value,
                        "measured": "unknown",
                    },
                    "engine_metrics": engine.as_dict(),
                }
            )
        return _envelope(
            "models",
            available=True,
            data={"models": rows},
            unavailable_fields=["quality_live", "batching"],
            note=(
                "Declared fields come from the registry YAML. Measured quality "
                "comes from datasets/eval/models/leaderboard.json when present. "
                "KV, prefix-cache, running/waiting, and engine TPS are "
                "DEPLOYMENT scrapes shown per runtime; unsupported cells say "
                "unavailable. Process-buffer rps is not inference tokens/sec. "
                "Registry metadata is never shown as a benchmark fact."
            ),
            scope=scope_note,
        )

    def _promotion(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del tenant_id, fleet
        from llm_fabric.models.promotion import PromotionStore, status_payload

        store = PromotionStore.load(
            self._promotion_state_path
            if self._promotion_state_path is not None
            else PromotionStore().path
        )
        by_model: dict[str, list[UsageRecord]] = defaultdict(list)
        for record in records:
            by_model[record.served_model].append(record)
        health = self._health.all_snapshots()
        rows = []
        for spec in self._registry.enabled_models():
            status = status_payload(spec, store)
            group = by_model.get(spec.id, [])
            snap = health.get(spec.deployment_id)
            errors = sum(1 for r in group if r.error)
            probe = status.get("probe") or {}
            evaluation = status.get("evaluation") or {}
            shadow = status.get("shadow") or {}
            approval = status.get("approval") or {}
            rows.append(
                {
                    **status,
                    "model": spec.provider_model,
                    "revision": spec.revision,
                    "digest": spec.digest,
                    "probe_status": "passed" if probe.get("passed") else "unknown",
                    "evaluation_status": ("passed" if evaluation.get("passed") else "unknown"),
                    "shadow_status": "recorded" if shadow.get("path") else "unknown",
                    "approval_status": ("approved" if approval.get("approved") else "not approved"),
                    "promotion_history_count": len(status.get("history") or []),
                    "health": snap.state.value if snap else "unknown",
                    "requests": len(group),
                    "p95_ms": _latencies(group)["p95_ms"],
                    "error_rate": (errors / len(group)) if group else None,
                    "capabilities": {
                        "declared": sorted(spec.capabilities.declared),
                        "measured": "unknown",
                    },
                }
            )
        return _envelope(
            "promotion",
            available=True,
            data={"models": rows},
            note=(
                "Lifecycle is operator promotion state. Evaluated is not approved. "
                "Registered high-tier YAML is not production eligible."
            ),
            scope=scope_note,
        )

    def _routing(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del tenant_id, fleet
        by_policy: dict[str, int] = defaultdict(int)
        by_served: dict[str, int] = defaultdict(int)
        by_tier: dict[str, int] = defaultdict(int)
        edges: dict[tuple[str, str], int] = defaultdict(int)
        for record in records:
            by_policy[record.policy] += 1
            by_served[record.served_model] += 1
            if record.selected_tier:
                by_tier[record.selected_tier] += 1
            if record.attempts:
                previous = None
                for attempt in record.attempts:
                    if previous is not None:
                        edges[(previous, attempt.model_id)] += 1
                    previous = attempt.model_id
        fallback_edges = [
            {"from": source, "to": target, "count": count}
            for (source, target), count in sorted(edges.items())
        ]
        tier_to_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        model_to_provider: dict[str, str] = {}
        for spec in self._registry.enabled_models():
            model_to_provider[spec.id] = spec.provider
            for tier in spec.tiers:
                tier_to_model[tier.value][spec.id] += 0
        for record in records:
            if record.selected_tier:
                tier_to_model[record.selected_tier][record.served_model] += 1
        return _envelope(
            "routing",
            available=True,
            data={
                "by_policy": dict(by_policy),
                "by_served_model": dict(by_served),
                "by_tier": dict(by_tier),
                "tier_to_model": {tier: dict(models) for tier, models in tier_to_model.items()},
                "model_to_provider": model_to_provider,
                "fallback_edges": fallback_edges,
                "failovers": sum(r.failover_count for r in records),
            },
            note="Counts come from completed requests in this process, not from a routing eval.",
            scope=scope_note,
        )

    def _tiers(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del tenant_id, fleet
        from llm_fabric.router.tiers import ALL_TIERS

        buckets: dict[str, list[UsageRecord]] = {tier.value: [] for tier in ALL_TIERS}
        unlabelled: list[UsageRecord] = []
        for record in records:
            if record.selected_tier and record.selected_tier in buckets:
                buckets[record.selected_tier].append(record)
            else:
                unlabelled.append(record)
        histogram = []
        for tier in ALL_TIERS:
            group = buckets[tier.value]
            errors = sum(1 for r in group if r.error)
            histogram.append(
                {
                    "tier": tier.value,
                    "requests": len(group),
                    "success_rate": ((len(group) - errors) / len(group)) if group else None,
                    "latency_p95_ms": _latencies(group)["p95_ms"],
                    "fallbacks": sum(r.failover_count for r in group),
                }
            )
        return _envelope(
            "tiers",
            available=True,
            data={
                "histogram": histogram,
                "unlabelled_requests": len(unlabelled),
            },
            note=(
                "Compact L0–L30 request histogram. Null success/latency means "
                "no requests landed on that tier in this process buffer."
            ),
            scope=scope_note,
        )

    def _fallbacks(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del tenant_id, fleet
        hops: list[dict[str, Any]] = []
        for record in records:
            if not record.failover_count:
                continue
            hops.append(
                {
                    "request_id": record.request_id,
                    "tenant_id": record.tenant_id,
                    "requested_model": record.requested_model,
                    "served_model": record.served_model,
                    "failovers": record.failover_count,
                    "attempts": [
                        {
                            "model_id": attempt.model_id,
                            "provider": attempt.provider,
                            "duration_ms": attempt.duration_ms,
                            "error": attempt.error,
                        }
                        for attempt in record.attempts
                    ],
                }
            )
        return _envelope(
            "fallbacks",
            available=True,
            data={"events": hops[:100], "total_failovers": sum(r.failover_count for r in records)},
            scope=scope_note,
        )

    def _tokens(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del records
        invocations = self._meter.invocation_totals(
            tenant_id=None if fleet else tenant_id, observe=fleet
        )
        estimated = invocations.estimated_invocations
        return _envelope(
            "tokens",
            available=True,
            data={
                "prompt_tokens": invocations.prompt_tokens,
                "completion_tokens": invocations.completion_tokens,
                "total_tokens": invocations.total_tokens,
                "invocations": invocations.invocations,
                "requests_with_estimated_counts": estimated,
                "by_token_source": invocations.by_token_source,
                "unavailable_invocations": invocations.unavailable_invocations,
            },
            unavailable_fields=["cached_tokens", "reasoning_tokens"],
            estimated_fields=["prompt_tokens", "completion_tokens"] if estimated else (),
            note=(
                "Invocation ledger, including fallbacks. Cached and reasoning "
                "tokens are recorded only when a backend reports them. Values "
                "are marked PROVIDER_MEASURED, LOCAL_TOKENIZER_ESTIMATE, "
                "DERIVED, or UNAVAILABLE."
            ),
            source="usage_ledger" if self._meter.durable else "meter",
            scope=scope_note,
        )

    def _context(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        items = [
            record
            for record in self._context_records
            if fleet or tenant_id is None or record.tenant_id == tenant_id
        ]
        payloads = [record.as_dict(include_content=False) for record in items[-50:]]
        observations = [obs for record in items for obs in record.observations()]
        missing = provenance_missing(observations)
        covered = sum(1 for record in records if record.context_record_id)

        def _named(record: ContextRecord, name: str) -> Observed | None:
            for obs in record.observations():
                if obs.name == name:
                    return obs
            return None

        def _sum_named(name: str) -> dict[str, Any]:
            values = []
            provenance = CountProvenance.DERIVED
            for record in items:
                obs = _named(record, name)
                if obs is None or obs.value is None:
                    continue
                values.append(obs.value)
            if not values:
                return _unavailable(
                    name,
                    scope=MetricScope.REQUEST,
                    note="no compiled ContextRecords in this process buffer",
                )
            return Observed(
                name=name,
                value=sum(values),
                provenance=provenance,
                scope=MetricScope.REQUEST,
                unit="tokens" if "percent" not in name else "percent",
                note=f"sum over {len(values)} compiled records in this process buffer",
            ).as_dict()

        distribution: dict[str, int] = defaultdict(int)
        for record in items:
            for block in record.blocks:
                distribution[block.type.value] += 1
        utilization_values = [
            obs.value
            for record in items
            if (obs := _named(record, "context_utilization_percent")) is not None
            and obs.value is not None
        ]
        return _envelope(
            "context",
            available=True,
            data={
                "distribution": dict(distribution) or None,
                "before_optimization": _sum_named("context_tokens_before_optimization"),
                "after_optimization": _sum_named("context_tokens_after_optimization"),
                "context_utilization": {
                    "mean_percent": (
                        round(sum(utilization_values) / len(utilization_values), 4)
                        if utilization_values
                        else None
                    ),
                    "count": len(utilization_values) or None,
                    "note": (
                        "requires a known model context limit; otherwise UNAVAILABLE per record"
                    ),
                },
                "stable_prefix": _sum_named("stable_prefix_tokens"),
                "compression": _sum_named("tokens_compressed"),
                "deduplication": _sum_named("tokens_deduplicated"),
                "dropped_tokens": _sum_named("tokens_dropped"),
                "overflow_rejection": _unavailable(
                    "overflow_rejection",
                    scope=MetricScope.FLEET,
                    note="no overflow-rejection counter exists in this build",
                ),
                "records": payloads,
                "shown": len(payloads),
                "buffered": len(items),
                "requests_with_context_record": covered,
                "requests_without_context_record": max(0, len(records) - covered),
                "supported_metrics_without_provenance": missing,
                "stable_prefix_note": (
                    "stable_prefix_tokens labels prompt shape. It is not a vLLM prefix-cache hit."
                ),
            },
            unavailable_fields=["overflow_rejection"],
            note=(
                "Context compiler ran on the serving path. Absent block types "
                "are 0 (counted). Unknown windows are UNAVAILABLE. Raw prompt "
                "text is withheld. Compression is 0 when no compressor is configured."
            ),
            source="context_compiler",
            scope=scope_note,
        )

    def _kv_cache(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del records, tenant_id
        snapshots = list(self._engines.all_snapshots())
        engines = []
        for engine in snapshots:
            payload = engine.as_dict()
            payload["kv_current"] = _engine_observation(engine, "kv_cache_utilization")
            payload["prefix_hit_ratio"] = _prefix_hit_ratio(engine)
            payload["cached_prompt_tokens"] = _engine_observation(engine, "cached_prompt_tokens")
            payload["running"] = _engine_observation(engine, "running_requests")
            payload["waiting"] = _engine_observation(engine, "waiting_requests")
            payload["preemptions"] = _engine_observation(engine, "preemptions")
            payload["ttft"] = _engine_observation(engine, "ttft_seconds_sum")
            payload["tpot"] = _engine_observation(engine, "inter_token_latency_seconds_sum")
            payload["prefill_tps"] = _engine_observation(engine, "prefill_tps")
            payload["decode_tps"] = _engine_observation(engine, "decode_tps")
            if engine.provider == "ollama":
                for name in (
                    "kv_current",
                    "prefix_hit_ratio",
                    "cached_prompt_tokens",
                    "running",
                    "waiting",
                    "preemptions",
                    "prefill_tps",
                    "decode_tps",
                ):
                    if payload.get(name) is None:
                        payload[name] = _unavailable(
                            name,
                            scope=MetricScope.DEPLOYMENT,
                            note=OLLAMA_DOES_NOT_EXPOSE,
                        )
            engines.append(payload)
        gpu = [
            Observed.unavailable(
                name,
                scope=MetricScope.POD,
                note=(
                    "UNAVAILABLE — DCGM exporter is not scraped by this process. "
                    "Configure Prometheus to scrape nvidia-dcgm-exporter; do not "
                    "treat GPU series as request metrics."
                ),
            ).as_dict()
            for name in ("gpu_utilization", "gpu_memory", "gpu_temperature", "gpu_power")
        ]
        return _envelope(
            "kv_cache",
            available=True,
            data={
                "scope": "DEPLOYMENT",
                "filters": {
                    "fleet": {
                        "available": True,
                        "selected": fleet,
                        "note": "engine scrapes are this process; not a multi-cluster fleet",
                    },
                    "deployment": sorted({engine.provider for engine in snapshots}),
                    "pod": {
                        "available": False,
                        "note": "pod identity is not on engine scrapes",
                    },
                    "model": [spec.id for spec in self._registry.enabled_models()],
                },
                "engines": engines,
                "capability_matrix": capability_matrix(),
                "gpu": {
                    "scope": "POD",
                    "source": "dcgm-exporter",
                    "observations": gpu,
                },
            },
            unavailable_fields=["batch_utilization", "request_kv_percent", "pod"],
            note=(
                "Engine KV/prefix/queue gauges are DEPLOYMENT-scoped. They are "
                "not this request's KV use. Ollama cells that the runtime does "
                "not expose stay UNAVAILABLE. GPU series come from DCGM when "
                "Prometheus scrapes it, not from the gateway. Filters hide "
                "existing rows; they do not invent a vLLM scrape."
            ),
            source="engine_scrape",
            scope=scope_note,
        )

    def _economics(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del tenant_id, fleet
        successes = [r for r in records if r.error is None]
        cost = sum(r.cost_usd for r in records)
        prompt = sum(r.prompt_tokens for r in records)
        completion = sum(r.completion_tokens for r in records)
        by_model: dict[str, float] = defaultdict(float)
        by_provider: dict[str, float] = defaultdict(float)
        by_intent: dict[str, float] = defaultdict(float)
        for record in records:
            by_model[record.served_model] += record.cost_usd
            by_provider[record.provider] += record.cost_usd
            if record.intent_id:
                by_intent[record.intent_id] += record.cost_usd
        return _envelope(
            "economics",
            available=True,
            data={
                "cost_usd": round(cost, 6),
                "cost_per_request": round(cost / len(records), 6) if records else None,
                "cost_per_successful_request": (
                    round(sum(r.cost_usd for r in successes) / len(successes), 6)
                    if successes
                    else None
                ),
                "cost_per_1k_input_tokens": (round(cost / prompt * 1000, 6) if prompt else None),
                "cost_per_1k_output_tokens": (
                    round(cost / completion * 1000, 6) if completion else None
                ),
                "by_model": {k: round(v, 6) for k, v in by_model.items()},
                "by_provider": {k: round(v, 6) for k, v in by_provider.items()},
                "by_intent": {k: round(v, 6) for k, v in by_intent.items()} or None,
                "requests_with_estimated_cost": sum(1 for r in records if r.cost_is_estimated),
            },
            unavailable_fields=[
                "gpu_hours",
                "electricity_cost",
                "cache_savings",
                "routing_savings",
                "context_compression_savings",
                "cost_per_evaluation_point",
            ],
            note=(
                "Spend is registry prices times tokens. Self-hosted inference "
                "is not treated as free — an unpriced deployment contributes "
                "nothing rather than $0. Cache, routing and compression "
                "savings have no baseline to subtract from."
            ),
            scope=scope_note,
        )

    def _drift(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del fleet
        report = analyze(
            records,
            tenant_id=tenant_id or "public",
            health=self._health,
            registry=self._registry,
            eval_runs=self._eval_runs,
        )
        unavailable = [
            signal.metric for signal in report.signals if signal.severity.value == "unavailable"
        ]
        return _envelope(
            "drift",
            available=True,
            data={
                "report": report.as_dict(),
                "incidents": [item.as_dict() for item in self._incidents[:50]],
                "remediations": [item.as_dict() for item in self._remediations[:50]],
            },
            unavailable_fields=unavailable,
            note=report.note,
            scope=scope_note,
        )

    def _evals(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del records, fleet
        runs = [
            run.as_dict()
            for run in self._eval_runs
            if tenant_id is None or run.tenant_id == tenant_id
        ]
        return _envelope(
            "evals",
            available=True,
            data={"runs": runs[:50], "count": len(runs)},
            unavailable_fields=["historical_clickhouse", "agent_evals", "safety_evals"],
            note=(
                "Runs stored in this process. Agent and safety suites are not "
                "built. DeepEval and lm-evaluation-harness stay unavailable "
                "until those packages are installed and mapped."
            ),
            scope=scope_note,
        )

    def _reliability(
        self,
        records: Sequence[UsageRecord],
        *,
        tenant_id: str | None,
        fleet: bool,
        scope_note: str,
    ) -> dict[str, Any]:
        del tenant_id, fleet
        snapshots = [snap.as_dict() for snap in self._health.all_snapshots().values()]
        errors = sum(1 for r in records if r.error)
        return _envelope(
            "reliability",
            available=True,
            data={
                "requests": len(records),
                "errors": errors,
                "error_rate": (errors / len(records)) if records else None,
                "failovers": sum(r.failover_count for r in records),
                "deployments": snapshots,
                "engine": [snap.as_dict() for snap in self._engines.all_snapshots()],
            },
            unavailable_fields=list(UNAVAILABLE_INFERENCE_METRICS),
            note=(
                "Circuit state and EWMA latency/error are observed by this "
                "process. GPU, KV-cache and batch metrics stay unavailable "
                "until an engine adapter reports them."
            ),
            scope=scope_note,
        )
