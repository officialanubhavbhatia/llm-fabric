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

from llm_fabric.eval.schema import EvalRun
from llm_fabric.heal.drift import analyze
from llm_fabric.heal.schema import Incident, RemediationRecord
from llm_fabric.intent.metrics import percentile
from llm_fabric.observability.engine import (
    UNAVAILABLE_INFERENCE_METRICS,
    EngineMetricsHub,
)
from llm_fabric.observability.metering import UsageMeter, UsageRecord
from llm_fabric.observability.otel import BUILT_STAGES, LIFECYCLE_STAGES, SpanJournal
from llm_fabric.router.health import HealthTracker
from llm_fabric.router.registry import ModelRegistry

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
    "kv_cache": (
        "Chat can use Ollama or vLLM through the OpenAI-compatible adapter. "
        "KV-cache utilisation is not scraped from those engines and is not "
        "synthesized. Ollama does not expose a KV-cache series the fabric can "
        "honestly show."
    ),
    "batching": (
        "Continuous batching is a vLLM engine property. Fabric talks to vLLM "
        "through the OpenAI-compatible HTTP API and does not scrape vLLM "
        "/metrics, so batch size and batch utilisation are unavailable."
    ),
    "context": (
        "The context compiler is not on the serving path. Context tokens "
        "before/after optimisation, tokens compressed and tokens dropped "
        "have never been measured."
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
        del tenant_id, fleet
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
            if span > 0:
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
        return _envelope(
            "overview",
            available=True,
            data={
                "requests": len(records),
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
                "build. Queue depth is not a fleet queue; in-flight counts live "
                "on the reliability view."
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
        del tenant_id, fleet
        return _envelope(
            "requests",
            available=True,
            data={
                "recent": [self._request_row(record) for record in records[:100]],
                "shown": min(100, len(records)),
                "buffered": len(records),
            },
            note=(
                "Durable postgres ledger."
                if self._meter.durable
                else "In-memory buffer, this process only, lost on restart."
            ),
            scope=scope_note,
        )

    def _request_row(self, record: UsageRecord) -> dict[str, Any]:
        row = record.as_dict()
        # attempts already serialise; keep the payload lean for the table.
        row.pop("attempts", None)
        return row

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
        if self._intent is None:
            return _envelope(
                "intents",
                available=True,
                data={
                    "safety_gates": {
                        "hard_negative_accuracy": 0.50,
                        "required": 0.58,
                        "routing": "OFF",
                        "source": "frozen_eval_v1.1",
                        "serving_path_classification": False,
                    },
                    "live_classifications": None,
                },
                unavailable_fields=["classifications", "misclassifications"],
                note=(
                    "Safety gates first: hard-negative accuracy on the frozen 98 "
                    "is 0.50 (required >= 0.58). Serving-path IntentOS routing is "
                    "OFF. These figures are the locked evaluation, not live traffic."
                ),
                source="frozen_eval",
                scope=scope_note,
            )
        snapshot = dict(self._intent)
        return _envelope(
            "intents",
            available=True,
            data={
                "safety_gates": {
                    "hard_negative_accuracy": 0.50,
                    "required": 0.58,
                    "routing": "OFF",
                    "source": "frozen_eval_v1.1",
                    "serving_path_classification": False,
                },
                "classifications": snapshot.get("classifications"),
                "abstention_rate": snapshot.get("abstention_rate"),
                "unknown": snapshot.get("unknown"),
                "escalations": snapshot.get("escalations"),
                "disagreements": snapshot.get("disagreements"),
                "cache_hits": snapshot.get("cache_hits"),
                "by_layer": snapshot.get("by_layer"),
                "confidence": snapshot.get("confidence"),
                "latency_ms": snapshot.get("latency_ms"),
                "full": snapshot,
            },
            unavailable_fields=["misclassifications", "newly_discovered_clusters"],
            note=(
                "Safety gates first: serving-path IntentOS routing is OFF because "
                "hard-negative accuracy on the frozen 98 is 0.50 (required >= 0.58). "
                "Those figures live in docs, not in live traffic. Counters below are "
                "cascade activity, not accuracy. Misclassifications require labelled "
                "traffic, which is not collected on the serving path."
            ),
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
            rows.append(
                {
                    "deployment": spec.id,
                    "provider": spec.provider,
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
                    "requests": len(group),
                    "p95_ms": _latencies(group)["p95_ms"],
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
            unavailable_fields=["quality_live", "kv_cache", "batching", "tps"],
            note=(
                "Declared fields come from the registry YAML. Measured fields "
                "come from datasets/eval/models/leaderboard.json when present. "
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
