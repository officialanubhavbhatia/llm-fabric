"""Engine-level measurements from inference backends.

Ollama: scrape `/api/ps` when `api_base` is set. Do not fake KV-cache series.
vLLM: scrape `/metrics` when `metrics_endpoint` is set. Scope is DEPLOYMENT.

Until a scraper is configured every measurement is `unavailable`, never zero.
Scrapes run from Command Center / hub snapshots, never on the chat path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from llm_fabric.observability.metric import MetricScope, Observed
from llm_fabric.observability.ollama_metrics import OLLAMA_DOES_NOT_EXPOSE, OLLAMA_UNSUPPORTED

#: Measurements vLLM's Prometheus endpoint typically exports. Shown only when
#: a vLLM adapter is constructed and has scraped successfully.
VLLM_MEASUREMENTS: tuple[str, ...] = (
    "kv_cache_usage",
    "prefix_cache_queries",
    "prefix_cache_hits",
    "cached_prompt_tokens",
    "running_requests",
    "waiting_requests",
    "prefill_tokens",
    "generated_tokens",
    "preemptions",
    "gpu_cache_usage",
    "batch_size",
)

#: Measurements Ollama's API actually exposes today (`/api/ps`, `/api/tags`).
#: KV-cache utilisation is not among them.
OLLAMA_MEASUREMENTS: tuple[str, ...] = (
    "loaded_models",
    "size_vram_bytes",
    "size_bytes",
    "expires_at",
)

#: Constitution inference metrics that no current adapter can produce.
UNAVAILABLE_INFERENCE_METRICS: tuple[str, ...] = (
    "gpu_memory",
    "gpu_utilization",
    "kv_cache_utilization",
    "prefix_cache_hit_rate",
    "batch_size",
    "batch_utilization",
    "active_sequences",
    "waiting_sequences",
    "preemption",
    "prefill_tps",
    "decode_tps",
    "speculative_decoding_acceptance",
)


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """What one inference engine reported, or why it reported nothing."""

    provider: str
    available: bool
    measurements: dict[str, float | None] = field(default_factory=dict)
    observations: tuple[Observed, ...] = ()
    unsupported: tuple[str, ...] = ()
    note: str = ""
    scope: str = "DEPLOYMENT"

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "available": self.available,
            "scope": self.scope,
            "measurements": self.measurements,
            "observations": [item.as_dict() for item in self.observations],
            "unsupported": list(self.unsupported),
            "note": self.note,
        }


class EngineMetrics(Protocol):
    def snapshot(self) -> EngineSnapshot: ...


class UnavailableEngine:
    """The honest default: the adapter is not present."""

    def __init__(
        self,
        provider: str,
        *,
        reason: str,
        supported_when_present: tuple[str, ...] = (),
    ) -> None:
        self._provider = provider
        self._reason = reason
        self._supported = supported_when_present

    def snapshot(self) -> EngineSnapshot:
        return EngineSnapshot(
            provider=self._provider,
            available=False,
            measurements={name: None for name in self._supported},
            unsupported=UNAVAILABLE_INFERENCE_METRICS
            if self._provider not in {"ollama", "vllm"}
            else tuple(
                name for name in UNAVAILABLE_INFERENCE_METRICS if name not in self._supported
            ),
            note=self._reason,
        )


def ollama_unavailable() -> UnavailableEngine:
    return UnavailableEngine(
        "ollama",
        reason=(
            "Ollama /api/ps is not configured on this process. KV occupancy, "
            "prefix-cache hits, batch utilisation and queue depth stay "
            f"{OLLAMA_DOES_NOT_EXPOSE} even when /api/ps is scraped."
        ),
        supported_when_present=OLLAMA_MEASUREMENTS,
    )


def vllm_unavailable() -> UnavailableEngine:
    return UnavailableEngine(
        "vllm",
        reason=(
            "Chat completions can use the OpenAI-compatible vLLM adapter. "
            "vLLM /metrics (KV cache, running requests, prefix cache) is not "
            "scraped and is not synthesized."
        ),
        supported_when_present=VLLM_MEASUREMENTS,
    )


class EngineMetricsHub:
    """Looks up engine metrics for every known inference backend name."""

    def __init__(self, engines: dict[str, EngineMetrics] | None = None) -> None:
        self._engines: dict[str, EngineMetrics] = {
            "ollama": ollama_unavailable(),
            "vllm": vllm_unavailable(),
        }
        if engines:
            self._engines.update(engines)

    def register(self, provider: str, engine: EngineMetrics) -> None:
        self._engines[provider] = engine

    def for_provider(self, provider: str) -> EngineSnapshot:
        engine = self._engines.get(provider)
        if engine is None:
            return EngineSnapshot(
                provider=provider,
                available=False,
                unsupported=UNAVAILABLE_INFERENCE_METRICS,
                note=f"provider '{provider}' does not expose engine metrics",
            )
        return engine.snapshot()

    def all_snapshots(self) -> list[EngineSnapshot]:
        return [engine.snapshot() for engine in self._engines.values()]


class CachedVllmEngine:
    """Periodic `/metrics` reader. Snapshots are DEPLOYMENT-scoped."""

    def __init__(self, url: str, *, ttl_s: float = 15.0) -> None:
        self._url = url
        self._ttl_s = ttl_s
        self._cached: EngineSnapshot | None = None
        self._cached_at = 0.0

    def snapshot(self) -> EngineSnapshot:
        import time

        from llm_fabric.observability.vllm_metrics import fetch_vllm_metrics

        now = time.monotonic()
        if self._cached is not None and now - self._cached_at < self._ttl_s:
            return self._cached
        observations = fetch_vllm_metrics(self._url)
        healthy = any(item.health.value == "OK" and item.value is not None for item in observations)
        measurements = {
            item.name: (float(item.value) if item.value is not None else None)
            for item in observations
            if item.name
            in {
                "kv_cache_utilization",
                "prefix_cache_query_tokens",
                "prefix_cache_hit_tokens",
                "cached_prompt_tokens",
                "prompt_tokens",
                "generated_tokens",
                "running_requests",
                "waiting_requests",
                "preemptions",
            }
        }
        self._cached = EngineSnapshot(
            provider="vllm",
            available=healthy,
            measurements=measurements,
            observations=tuple(observations),
            note="vLLM /metrics scrape; DEPLOYMENT scope, not request KV use",
            scope="DEPLOYMENT",
        )
        self._cached_at = now
        return self._cached


class CachedOllamaPsEngine:
    """`/api/ps` reader. Loaded-model facts only; KV stays UNAVAILABLE."""

    def __init__(self, api_base: str, *, ttl_s: float = 15.0) -> None:
        self._api_base = api_base.rstrip("/")
        self._ttl_s = ttl_s
        self._cached: EngineSnapshot | None = None
        self._cached_at = 0.0

    def snapshot(self) -> EngineSnapshot:
        import time

        import httpx

        now = time.monotonic()
        if self._cached is not None and now - self._cached_at < self._ttl_s:
            return self._cached
        root = self._api_base[:-3] if self._api_base.endswith("/v1") else self._api_base
        observations: list[Observed] = [
            Observed.unavailable(name, scope=MetricScope.DEPLOYMENT, note=OLLAMA_DOES_NOT_EXPOSE)
            for name in OLLAMA_UNSUPPORTED
        ]
        measurements: dict[str, float | None] = {name: None for name in OLLAMA_MEASUREMENTS}
        available = False
        note = "Ollama /api/ps scrape; KV/prefix/queue remain unavailable"
        try:
            response = httpx.get(f"{root}/api/ps", timeout=2.0)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models") or []
            measurements["loaded_models"] = float(len(models))
            vram = sum(float(item.get("size_vram") or 0) for item in models)
            size = sum(float(item.get("size") or 0) for item in models)
            measurements["size_vram_bytes"] = vram
            measurements["size_bytes"] = size
            available = True
        except Exception as exc:  # noqa: BLE001 - scrape failure is not a KV zero
            observations.append(
                Observed.broken(
                    "ollama_ps_scrape",
                    scope=MetricScope.DEPLOYMENT,
                    note=f"{type(exc).__name__}: {exc}",
                    source_metric_name=f"{root}/api/ps",
                )
            )
            note = "configured Ollama /api/ps scrape failed"
        self._cached = EngineSnapshot(
            provider="ollama",
            available=available,
            measurements=measurements,
            observations=tuple(observations),
            unsupported=OLLAMA_UNSUPPORTED,
            note=note,
            scope="DEPLOYMENT",
        )
        self._cached_at = now
        return self._cached


def build_engine_hub(registry: Any | None = None) -> EngineMetricsHub:
    """Attach scrapers for registry metrics_endpoint values. Off the request path."""
    hub = EngineMetricsHub()
    if registry is None:
        return hub
    vllm_seen: set[str] = set()
    ollama_seen: set[str] = set()
    for spec in registry.enabled_models():
        runtime = getattr(getattr(spec, "runtime", None), "value", None)
        endpoint = getattr(spec, "metrics_endpoint", None)
        if runtime == "vllm" and endpoint and endpoint not in vllm_seen:
            vllm_seen.add(endpoint)
            hub.register("vllm", CachedVllmEngine(endpoint))
        api_base = getattr(spec, "api_base", None)
        if runtime == "ollama" and api_base and api_base not in ollama_seen:
            ollama_seen.add(api_base)
            hub.register("ollama", CachedOllamaPsEngine(api_base))
    return hub
