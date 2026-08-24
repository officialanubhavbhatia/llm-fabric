"""Version-aware parser for vLLM Prometheus `/metrics`.

Names are taken from vLLM's documented exposition (V1 and legacy V0 aliases).
If a logical metric's candidate names are all absent, the observation is
UNAVAILABLE — never zero. A configured scrape that returns HTTP 200 but is
missing a series the catalog marked as expected for that version is
METRIC_PIPELINE_BROKEN.

This module does not scrape. Callers fetch `/metrics` off the request path.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from llm_fabric.observability.metric import CountProvenance, MetricHealth, MetricScope, Observed

#: Logical name -> candidate Prometheus names, newest first.
#: Documented V1 (vLLM 0.8+/0.19): kv_cache_usage_perc, prefix_cache_queries/hits,
#: num_requests_running/waiting, prompt_tokens_total, generation_tokens_total,
#: time_to_first_token_seconds, inter_token_latency_seconds,
#: request_queue_time_seconds, request_prefill_time_seconds,
#: request_decode_time_seconds, num_preemptions_total.
#: Legacy: gpu_cache_usage_perc, gpu_prefix_cache_*, time_per_output_token_seconds.
VLLM_METRIC_CATALOG: dict[str, tuple[str, ...]] = {
    "kv_cache_utilization": (
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
    ),
    "prefix_cache_query_tokens": (
        "vllm:prefix_cache_queries",
        "vllm:prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_queries",
    ),
    "prefix_cache_hit_tokens": (
        "vllm:prefix_cache_hits",
        "vllm:prefix_cache_hits_total",
        "vllm:gpu_prefix_cache_hits",
    ),
    "prefix_cache_hit_ratio_legacy_gauge": (
        "vllm:gpu_prefix_cache_hit_rate",
        "vllm:prefix_cache_hit_rate",
    ),
    "cached_prompt_tokens": ("vllm:prefix_cache_hits", "vllm:cached_tokens_total"),
    "prompt_tokens": ("vllm:prompt_tokens_total", "vllm:prompt_tokens"),
    "generated_tokens": ("vllm:generation_tokens_total", "vllm:generation_tokens"),
    "running_requests": ("vllm:num_requests_running",),
    "waiting_requests": ("vllm:num_requests_waiting",),
    "preemptions": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
    "queue_time_seconds_sum": ("vllm:request_queue_time_seconds_sum",),
    "queue_time_seconds_count": ("vllm:request_queue_time_seconds_count",),
    "ttft_seconds_sum": ("vllm:time_to_first_token_seconds_sum",),
    "ttft_seconds_count": ("vllm:time_to_first_token_seconds_count",),
    "inter_token_latency_seconds_sum": (
        "vllm:inter_token_latency_seconds_sum",
        "vllm:time_per_output_token_seconds_sum",
        "vllm:request_time_per_output_token_seconds_sum",
    ),
    "inter_token_latency_seconds_count": (
        "vllm:inter_token_latency_seconds_count",
        "vllm:time_per_output_token_seconds_count",
        "vllm:request_time_per_output_token_seconds_count",
    ),
    "prefill_time_seconds_sum": ("vllm:request_prefill_time_seconds_sum",),
    "prefill_time_seconds_count": ("vllm:request_prefill_time_seconds_count",),
    "decode_time_seconds_sum": ("vllm:request_decode_time_seconds_sum",),
    "decode_time_seconds_count": ("vllm:request_decode_time_seconds_count",),
    "e2e_request_latency_seconds_sum": ("vllm:e2e_request_latency_seconds_sum",),
    "e2e_request_latency_seconds_count": ("vllm:e2e_request_latency_seconds_count",),
    "kv_block_lifetime": (),  # not in the documented exposition
    "kv_eviction": (),
    "idle_before_eviction": (),
    "reuse_gap": (),
}

#: Expected on a healthy V1 scrape. Missing after a successful fetch is broken.
V1_EXPECTED: frozenset[str] = frozenset(
    {
        "kv_cache_utilization",
        "running_requests",
        "waiting_requests",
        "prompt_tokens",
        "generated_tokens",
    }
)

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:[0-9]*\.?[0-9]+|\.[0-9]+)(?:[eE][-+]?\d+)?|NaN|Inf|\+Inf|-Inf)\s*$"
)


@dataclass(frozen=True, slots=True)
class ParsedSample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float


def parse_prometheus_text(text: str) -> list[ParsedSample]:
    samples: list[ParsedSample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        labels = _parse_labels(match.group("labels") or "")
        samples.append(
            ParsedSample(
                name=match.group("name"),
                labels=labels,
                value=_float(match.group("value")),
            )
        )
    return samples


def _parse_labels(raw: str) -> tuple[tuple[str, str], ...]:
    if not raw.strip():
        return ()
    parts: list[tuple[str, str]] = []
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts.append((key.strip(), value.strip().strip('"')))
    return tuple(parts)


def _float(raw: str) -> float:
    if raw in {"NaN", "+Inf", "Inf", "-Inf"}:
        return float(raw.replace("+", ""))
    return float(raw)


def _sum_named(samples: list[ParsedSample], name: str) -> float | None:
    matched = [item.value for item in samples if item.name == name]
    if not matched:
        return None
    return float(sum(matched))


def detect_runtime_version(samples: list[ParsedSample]) -> str:
    names = {item.name for item in samples}
    if "vllm:kv_cache_usage_perc" in names:
        return "vllm-v1"
    if "vllm:gpu_cache_usage_perc" in names:
        return "vllm-v0-legacy"
    if any(name.startswith("vllm:") for name in names):
        return "vllm-unknown"
    return "unknown"


def parse_vllm_metrics(
    text: str,
    *,
    configured: bool = True,
    scrape_ok: bool = True,
) -> list[Observed]:
    """Normalize a `/metrics` body into scoped observations.

    Scope is DEPLOYMENT: these gauges/counters describe the engine process, not
    the calling request.
    """
    if not configured:
        return [
            Observed.unavailable(
                name,
                scope=MetricScope.DEPLOYMENT,
                note="vLLM metrics_endpoint is not configured",
            )
            for name in (
                "kv_cache_utilization",
                "prefix_cache_query_tokens",
                "prefix_cache_hit_tokens",
                "prefix_cache_hit_ratio",
                "cached_prompt_tokens",
                "prompt_tokens",
                "generated_tokens",
                "running_requests",
                "waiting_requests",
                "preemptions",
                "queue_time_seconds",
                "ttft_seconds",
                "inter_token_latency_seconds",
                "prefill_time_seconds",
                "decode_time_seconds",
            )
        ]
    if not scrape_ok:
        return [
            Observed.broken(
                "vllm_metrics_scrape",
                scope=MetricScope.DEPLOYMENT,
                note="configured vLLM /metrics scrape failed",
            )
        ]

    samples = parse_prometheus_text(text)
    version = detect_runtime_version(samples)
    found: dict[str, tuple[str, float]] = {}
    for logical, candidates in VLLM_METRIC_CATALOG.items():
        for candidate in candidates:
            value = _sum_named(samples, candidate)
            if value is None or math.isnan(value):
                continue
            found[logical] = (candidate, value)
            break

    observations: list[Observed] = []
    scope = MetricScope.DEPLOYMENT

    def emit(logical: str, *, unit: str | None = None, derived: bool = False) -> Observed | None:
        if logical not in found:
            if not VLLM_METRIC_CATALOG.get(logical):
                observations.append(
                    Observed.unavailable(
                        logical,
                        scope=scope,
                        note="vLLM does not document this series on /metrics",
                        runtime_version=version,
                    )
                )
                return None
            if logical in V1_EXPECTED and version == "vllm-v1":
                observations.append(
                    Observed.broken(
                        logical,
                        scope=scope,
                        note=f"expected {VLLM_METRIC_CATALOG[logical][0]} missing from V1 scrape",
                        runtime_version=version,
                    )
                )
                return None
            observations.append(
                Observed.unavailable(
                    logical,
                    scope=scope,
                    note="series not present on this vLLM version",
                    runtime_version=version,
                )
            )
            return None
        source, value = found[logical]
        item = Observed(
            name=logical,
            value=value,
            provenance=CountProvenance.DERIVED if derived else CountProvenance.PROVIDER_MEASURED,
            scope=scope,
            source_metric_name=source,
            runtime_version=version,
            health=MetricHealth.OK,
            unit=unit,
        )
        observations.append(item)
        return item

    emit("kv_cache_utilization", unit="fraction")
    queries = emit("prefix_cache_query_tokens")
    hits = emit("prefix_cache_hit_tokens")
    if queries is not None and hits is not None and float(queries.value or 0) > 0:
        observations.append(
            Observed(
                name="prefix_cache_hit_ratio",
                value=round(float(hits.value or 0) / float(queries.value), 6),
                provenance=CountProvenance.DERIVED,
                scope=scope,
                source_metric_name=f"{hits.source_metric_name}/{queries.source_metric_name}",
                runtime_version=version,
                unit="ratio",
                note="counter ratio at scrape time, not a request-level hit",
            )
        )
    elif "prefix_cache_hit_ratio_legacy_gauge" in found:
        source, value = found["prefix_cache_hit_ratio_legacy_gauge"]
        observations.append(
            Observed(
                name="prefix_cache_hit_ratio",
                value=value,
                provenance=CountProvenance.PROVIDER_MEASURED,
                scope=scope,
                source_metric_name=source,
                runtime_version=version,
                unit="ratio",
                note="legacy gauge; V1 replaced this with query/hit counters",
            )
        )
    else:
        observations.append(
            Observed.unavailable(
                "prefix_cache_hit_ratio",
                scope=scope,
                note="need prefix_cache_queries and prefix_cache_hits, or a legacy hit-rate gauge",
                runtime_version=version,
            )
        )
    emit("cached_prompt_tokens")
    emit("prompt_tokens")
    emit("generated_tokens")
    emit("running_requests")
    emit("waiting_requests")
    emit("preemptions")
    _mean_from_histogram(found, observations, version, "queue_time_seconds")
    _mean_from_histogram(found, observations, version, "ttft_seconds")
    _mean_from_histogram(found, observations, version, "inter_token_latency_seconds")
    _mean_from_histogram(found, observations, version, "prefill_time_seconds")
    _mean_from_histogram(found, observations, version, "decode_time_seconds")
    emit("kv_block_lifetime")
    emit("kv_eviction")
    emit("idle_before_eviction")
    emit("reuse_gap")
    kv = next((item for item in observations if item.name == "kv_cache_utilization"), None)
    if kv is not None and kv.value is not None:
        observations.append(
            Observed(
                name="kv_cache_pressure",
                value=kv.value,
                provenance=CountProvenance.DERIVED,
                scope=scope,
                source_metric_name=kv.source_metric_name,
                runtime_version=version,
                unit="fraction",
                note="same series as kv_cache_utilization; not request-scoped",
            )
        )
    else:
        observations.append(
            Observed.unavailable(
                "kv_cache_pressure",
                scope=scope,
                note="kv_cache_utilization not present",
                runtime_version=version,
            )
        )
    return observations


def _mean_from_histogram(
    found: dict[str, tuple[str, float]],
    observations: list[Observed],
    version: str,
    base: str,
) -> None:
    sum_key, count_key = f"{base}_sum", f"{base}_count"
    if sum_key not in found or count_key not in found:
        observations.append(
            Observed.unavailable(
                base,
                scope=MetricScope.DEPLOYMENT,
                note="histogram _sum/_count not present on this scrape",
                runtime_version=version,
            )
        )
        return
    total, count = found[sum_key][1], found[count_key][1]
    if count <= 0:
        observations.append(
            Observed.unavailable(
                base,
                scope=MetricScope.DEPLOYMENT,
                note="histogram count is zero; mean is undefined",
                runtime_version=version,
                source_metric_name=found[count_key][0],
            )
        )
        return
    observations.append(
        Observed(
            name=base,
            value=round(total / count, 6),
            provenance=CountProvenance.DERIVED,
            scope=MetricScope.DEPLOYMENT,
            source_metric_name=f"{found[sum_key][0]}/{found[count_key][0]}",
            runtime_version=version,
            unit="seconds",
            note="mean of engine histogram; not this request's TTFT/TPOT",
        )
    )


def observations_as_dict(items: list[Observed]) -> list[dict[str, Any]]:
    return [item.as_dict() for item in items]


def fetch_vllm_metrics(url: str, *, timeout_s: float = 2.0) -> list[Observed]:
    """HTTP GET `/metrics`. Never call this on the user request path."""
    import httpx

    try:
        response = httpx.get(url, timeout=timeout_s)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - scrape failure is pipeline-broken
        return [
            Observed.broken(
                "vllm_metrics_scrape",
                scope=MetricScope.DEPLOYMENT,
                note=f"{type(exc).__name__}: {exc}",
                source_metric_name=url,
            )
        ]
    return parse_vllm_metrics(response.text, configured=True, scrape_ok=True)
