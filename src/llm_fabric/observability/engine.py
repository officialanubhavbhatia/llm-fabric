"""Engine-level measurements from inference backends.

The constitution names two engines and two honesty rules:

* Ollama: capture what it actually exposes. Do not fake KV-cache statistics.
* vLLM: consume exposed engine metrics when that adapter is enabled.

Neither adapter is built. This module is the seam they will implement, and the
source the Command Center reads. Until an adapter is present every measurement
is `unavailable`, never zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

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
    unsupported: tuple[str, ...] = ()
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "available": self.available,
            "measurements": self.measurements,
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
            "The Ollama adapter is not built. Only measurements Ollama actually "
            "exposes would be shown; KV-cache statistics are not among them "
            "and will never be synthesized."
        ),
        supported_when_present=OLLAMA_MEASUREMENTS,
    )


def vllm_unavailable() -> UnavailableEngine:
    return UnavailableEngine(
        "vllm",
        reason=(
            "The vLLM adapter is not built. When it is enabled the Command "
            "Center will consume the engine's /metrics endpoint rather than "
            "invent values."
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
