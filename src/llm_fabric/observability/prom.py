"""Prometheus-compatible metrics with bounded label cardinality.

The constitution forbids putting unbounded-cardinality values into Prometheus
labels. Request ids, user ids, tenant ids, prompts and trace ids never appear
as labels. Paths are collapsed onto a closed set of route templates. Model ids
are admitted only while fewer than `MAX_MODEL_LABELS` distinct values have been
seen; anything past that is labelled `other`.
"""

from __future__ import annotations

from typing import Final

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

#: Closed set of HTTP path labels. Anything else becomes `other`.
PATH_LABELS: Final[frozenset[str]] = frozenset(
    {
        "/healthz",
        "/readyz",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/command-center",
        "/v1/chat/completions",
        "/v1/models",
        "/v1/models/{id}",
        "/v1/usage",
        "/v1/intents/classify",
        "/v1/routes/health",
        "/v1/observability/dashboards/{view}",
        "/v1/observability/traces",
        "/v1/dev/token",
        "other",
    }
)

MAX_MODEL_LABELS = 64
MAX_PROVIDER_LABELS = 16
MAX_POLICY_LABELS = 16

# Latency buckets covering a cache hit through a slow generation.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class FabricMetrics:
    """One registry per process. Tests pass their own so they do not leak."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self._known_models: set[str] = set()
        self._known_providers: set[str] = set()
        self._known_policies: set[str] = set()

        self.requests_total = Counter(
            "fabric_requests_total",
            "HTTP requests completed",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.active_requests = Gauge(
            "fabric_active_requests",
            "Requests currently being served by this process",
            registry=self.registry,
        )
        self.request_duration_seconds = Histogram(
            "fabric_request_duration_seconds",
            "End-to-end HTTP request duration",
            ["method", "path"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.tokens_total = Counter(
            "fabric_tokens_total",
            "Tokens accounted by this process",
            ["token_kind"],
            registry=self.registry,
        )
        self.cost_usd_total = Counter(
            "fabric_cost_usd_total",
            "Spend at registry prices. Estimated costs are counted and labelled.",
            ["estimated"],
            registry=self.registry,
        )
        self.fallbacks_total = Counter(
            "fabric_fallbacks_total",
            "Failover hops that actually ran",
            ["reason"],
            registry=self.registry,
        )
        self.route_decisions_total = Counter(
            "fabric_route_decisions_total",
            "Completed routing decisions",
            ["policy", "provider"],
            registry=self.registry,
        )
        self.inference_duration_seconds = Histogram(
            "fabric_inference_duration_seconds",
            "Provider-call duration, one observation per attempt",
            ["provider", "outcome"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.intent_classifications_total = Counter(
            "fabric_intent_classifications_total",
            "Intent cascade outcomes",
            ["layer", "cache_hit"],
            registry=self.registry,
        )
        self.intent_abstentions_total = Counter(
            "fabric_intent_abstentions_total",
            "Intent cascade abstentions",
            registry=self.registry,
        )
        self.intent_unknown_total = Counter(
            "fabric_intent_unknown_total",
            "Intent cascade unknown/OOD outcomes",
            registry=self.registry,
        )
        self.intent_cache_hits_total = Counter(
            "fabric_intent_cache_hits_total",
            "Intent cache hits",
            ["cache"],
            registry=self.registry,
        )
        self.intent_classifier_latency_seconds = Histogram(
            "fabric_intent_classifier_latency_seconds",
            "End-to-end intent classification latency",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.intent_escalations_total = Counter(
            "fabric_intent_escalations_total",
            "Times the L5 escalation classifier ran",
            registry=self.registry,
        )
        self.intent_disagreements_total = Counter(
            "fabric_intent_disagreements_total",
            "Times cheap classifier layers named different intents",
            registry=self.registry,
        )
        self.intent_serving_requests_total = Counter(
            "fabric_intent_serving_requests_total",
            "Chat requests that produced an IntentResult before routing",
            registry=self.registry,
        )
        self.intent_known_total = Counter(
            "fabric_intent_known_total",
            "Serving-path IntentResults in the known state",
            registry=self.registry,
        )
        self.intent_safe_fallback_total = Counter(
            "fabric_intent_safe_fallback_total",
            "Serving-path IntentResults in the safe_fallback state",
            registry=self.registry,
        )
        self.intent_error_total = Counter(
            "fabric_intent_error_total",
            "Intent cascade failures that degraded instead of skipping",
            registry=self.registry,
        )
        self.intent_missing_total = Counter(
            "fabric_intent_missing_total",
            "Provider executions that had to synthesize a SAFE_FALLBACK IntentResult",
            registry=self.registry,
        )
        self.ttft_seconds = Histogram(
            "fabric_ttft_seconds",
            "Time to first streamed byte. Unobserved on buffered responses.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.tpot_seconds = Histogram(
            "fabric_tpot_seconds",
            "Time per output token after the first streamed byte. Unobserved without streaming.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.context_compile_seconds = Histogram(
            "fabric_context_compile_seconds",
            "Context compiler latency on the serving path",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.context_tokens = Counter(
            "fabric_context_tokens_total",
            "Compiled context tokens",
            ["stage"],
            registry=self.registry,
        )
        self.litellm_transport_seconds = Histogram(
            "fabric_litellm_transport_seconds",
            "LiteLLM HTTP transport duration. Not a vLLM engine metric.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.litellm_transport_errors_total = Counter(
            "fabric_litellm_transport_errors_total",
            "LiteLLM transport failures seen by this process",
            registry=self.registry,
        )
        self.litellm_rate_limit_events_total = Counter(
            "fabric_litellm_rate_limit_events_total",
            "HTTP 429 responses from the LiteLLM proxy",
            registry=self.registry,
        )
        self.litellm_retries_total = Counter(
            "fabric_litellm_retries_total",
            "Adapter-level LiteLLM retries. Router failovers are fabric_fallbacks_total.",
            registry=self.registry,
        )
        # Gauges for quantities the constitution names that this process can
        # actually see. Queue depth is in-flight on this process, not a fleet
        # queue — the name says so.
        self.in_flight_by_provider = Gauge(
            "fabric_in_flight_requests",
            "In-flight inference calls on this process",
            ["provider"],
            registry=self.registry,
        )
        self.dependency_health = Gauge(
            "fabric_dependency_health",
            "1 when this process currently treats the dependency as healthy",
            ["dependency"],
            registry=self.registry,
        )
        self.dependency_failures_total = Counter(
            "fabric_dependency_failures_total",
            "Observed dependency failures on this process",
            ["dependency", "source"],
            registry=self.registry,
        )
        self.dependency_recoveries_total = Counter(
            "fabric_dependency_recoveries_total",
            "Times a dependency returned to healthy on this process",
            ["dependency"],
            registry=self.registry,
        )
        self.readiness_transitions_total = Counter(
            "fabric_readiness_transitions_total",
            "Times this process changed serving-ready state",
            ["from_state", "to_state"],
            registry=self.registry,
        )
        self.admission_rejections_total = Counter(
            "fabric_admission_rejections_total",
            "New inference refused because a mandatory dependency is unhealthy",
            ["dependency"],
            registry=self.registry,
        )

    def path_label(self, path: str) -> str:
        if path in PATH_LABELS:
            return path
        if path.startswith("/v1/models/") and path.count("/") == 3:
            return "/v1/models/{id}"
        if path.startswith("/v1/observability/dashboards/"):
            return "/v1/observability/dashboards/{view}"
        return "other"

    def _cap(self, value: str, known: set[str], ceiling: int) -> str:
        if value in known:
            return value
        if len(known) >= ceiling:
            return "other"
        known.add(value)
        return value

    def model_label(self, model_id: str) -> str:
        return self._cap(model_id or "unknown", self._known_models, MAX_MODEL_LABELS)

    def provider_label(self, provider: str) -> str:
        return self._cap(provider or "unknown", self._known_providers, MAX_PROVIDER_LABELS)

    def policy_label(self, policy: str) -> str:
        return self._cap(policy or "unknown", self._known_policies, MAX_POLICY_LABELS)

    def observe_http(self, *, method: str, path: str, status: int, duration_s: float) -> None:
        labelled_path = self.path_label(path)
        method_upper = method.upper()
        labelled_method = (
            method_upper
            if method_upper in {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"}
            else "OTHER"
        )
        status_class = f"{status // 100}xx" if 100 <= status <= 599 else "unknown"
        self.requests_total.labels(labelled_method, labelled_path, status_class).inc()
        self.request_duration_seconds.labels(labelled_method, labelled_path).observe(duration_s)

    def observe_usage(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        cost_is_estimated: bool,
        provider: str,
        policy: str,
        failover_count: int,
        latency_s: float,
        error: bool,
        ttft_s: float | None = None,
        tpot_s: float | None = None,
    ) -> None:
        self.tokens_total.labels("prompt").inc(prompt_tokens)
        self.tokens_total.labels("completion").inc(completion_tokens)
        self.cost_usd_total.labels("true" if cost_is_estimated else "false").inc(cost_usd)
        self.route_decisions_total.labels(
            self.policy_label(policy), self.provider_label(provider)
        ).inc()
        self.inference_duration_seconds.labels(
            self.provider_label(provider), "error" if error else "ok"
        ).observe(latency_s)
        if failover_count:
            self.fallbacks_total.labels("failover").inc(failover_count)
        if ttft_s is not None:
            self.ttft_seconds.observe(ttft_s)
        if tpot_s is not None:
            self.tpot_seconds.observe(tpot_s)

    def observe_intent(
        self,
        *,
        layer: str,
        cache_hit: bool,
        abstained: bool = False,
        latency_s: float | None = None,
        escalated: bool = False,
        disagreed: bool = False,
        cache_source: str | None = None,
    ) -> None:
        self.intent_classifications_total.labels(layer, "true" if cache_hit else "false").inc()
        if abstained:
            self.intent_abstentions_total.inc()
        if cache_hit:
            kind = cache_source if cache_source in {"l0_exact", "l1_semantic"} else "other"
            self.intent_cache_hits_total.labels(kind).inc()
        if latency_s is not None:
            self.intent_classifier_latency_seconds.observe(latency_s)
        if escalated:
            self.intent_escalations_total.inc()
        if disagreed:
            self.intent_disagreements_total.inc()

    def observe_intent_serving(self, state: str) -> None:
        self.intent_serving_requests_total.inc()
        if state == "known":
            self.intent_known_total.inc()
        elif state == "unknown":
            self.intent_unknown_total.inc()
        elif state == "abstain":
            self.intent_abstentions_total.inc()
        else:
            self.intent_safe_fallback_total.inc()

    def observe_intent_missing(self) -> None:
        self.intent_missing_total.inc()
        self.intent_safe_fallback_total.inc()
        self.intent_serving_requests_total.inc()

    def observe_intent_error(self) -> None:
        self.intent_error_total.inc()

    def observe_context(self, *, compile_s: float, tokens_before: int, tokens_after: int) -> None:
        self.context_compile_seconds.observe(compile_s)
        self.context_tokens.labels("before").inc(tokens_before)
        self.context_tokens.labels("after").inc(tokens_after)

    def observe_litellm_transport(
        self,
        *,
        latency_s: float,
        error: bool = False,
        rate_limited: bool = False,
        retries: int = 0,
    ) -> None:
        self.litellm_transport_seconds.observe(latency_s)
        if error:
            self.litellm_transport_errors_total.inc()
        if rate_limited:
            self.litellm_rate_limit_events_total.inc()
        if retries:
            self.litellm_retries_total.inc(retries)

    def _dependency_label(self, name: str) -> str:
        if name in {"postgres", "redis", "telemetry"}:
            return name
        return "other"

    def set_dependency_health(self, name: str, healthy: bool) -> None:
        self.dependency_health.labels(self._dependency_label(name)).set(1 if healthy else 0)

    def note_dependency_failure(self, name: str, source: str) -> None:
        labelled_source = source if source in {"probe", "serving"} else "other"
        self.dependency_failures_total.labels(self._dependency_label(name), labelled_source).inc()

    def note_dependency_recovery(self, name: str) -> None:
        self.dependency_recoveries_total.labels(self._dependency_label(name)).inc()

    def note_readiness_transition(self, *, from_ready: bool, to_ready: bool) -> None:
        self.readiness_transitions_total.labels(
            "ready" if from_ready else "not_ready",
            "ready" if to_ready else "not_ready",
        ).inc()

    def note_admission_rejection(self, name: str) -> None:
        self.admission_rejections_total.labels(self._dependency_label(name)).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)
