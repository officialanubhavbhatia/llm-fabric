"""Counters for the intent subsystem.

Every dimension here is bounded on purpose. Layers, intents and buckets are
finite and known ahead of time; prompts, tenants and trace ids are not, and the
constitution forbids putting unbounded-cardinality values into metric labels.
Tenant-level accounting belongs in metering, where it is already scoped and
already bounded.

The recorded values are facts about what the cascade did — which layer answered,
how often it abstained, what it cost. They are not evidence that the answers
were right. Accuracy comes from the benchmark, against labelled data.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_fabric.intent.schema import ClassifierLayer

#: Fixed confidence buckets. Ten is enough to see a distribution shift and few
#: enough to stay cheap.
CONFIDENCE_BUCKETS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

#: Latency buckets in milliseconds, spanning a cache hit to a slow model call.
LATENCY_BUCKETS_MS: tuple[float, ...] = (0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000)

#: Ceiling on distinct intents tracked. The taxonomy is bounded, but a
#: misconfigured one must not be able to grow this map without limit.
MAX_TRACKED_INTENTS = 512


@dataclass(slots=True)
class _Histogram:
    bounds: tuple[float, ...]
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    observations: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            # One extra slot for the overflow bucket.
            self.counts = [0] * (len(self.bounds) + 1)

    def observe(self, value: float) -> None:
        self.total += value
        self.observations += 1
        for index, bound in enumerate(self.bounds):
            if value <= bound:
                self.counts[index] += 1
                return
        self.counts[-1] += 1

    @property
    def mean(self) -> float | None:
        return self.total / self.observations if self.observations else None

    def as_dict(self) -> dict[str, Any]:
        # Not strict: `counts` carries one extra slot for the overflow bucket,
        # which is labelled separately on the next line.
        labelled = {
            str(bound): count for bound, count in zip(self.bounds, self.counts, strict=False)
        }
        labelled["+Inf"] = self.counts[-1]
        return {"buckets": labelled, "observations": self.observations, "mean": self.mean}


class IntentMetrics:
    """Thread-safe, bounded counters for classification activity."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_layer: dict[str, int] = {layer.value: 0 for layer in ClassifierLayer}
        self._by_intent: dict[str, int] = {}
        self._intents_dropped = 0
        self._classifications = 0
        self._abstentions = 0
        self._unknown = 0
        self._ambiguous = 0
        self._exact_hits = 0
        self._semantic_hits = 0
        self._escalations = 0
        self._disagreements = 0
        self._cost_usd = 0.0
        self._confidence = _Histogram(CONFIDENCE_BUCKETS)
        self._latency = _Histogram(LATENCY_BUCKETS_MS)
        self._layer_latency: dict[str, _Histogram] = {
            layer.value: _Histogram(LATENCY_BUCKETS_MS) for layer in ClassifierLayer
        }

    def record(
        self,
        *,
        layer: ClassifierLayer,
        intent_id: str,
        confidence: float,
        abstained: bool,
        ambiguous: bool,
        latency_ms: float,
        cost_usd: float,
        cache_hit: bool,
    ) -> None:
        with self._lock:
            self._classifications += 1
            self._by_layer[layer.value] = self._by_layer.get(layer.value, 0) + 1
            self._confidence.observe(confidence)
            self._latency.observe(latency_ms)
            self._cost_usd += cost_usd

            if abstained:
                self._abstentions += 1
                self._unknown += 1
            if ambiguous:
                self._ambiguous += 1
            if cache_hit:
                if layer is ClassifierLayer.L0_EXACT_CACHE:
                    self._exact_hits += 1
                elif layer is ClassifierLayer.L1_SEMANTIC_CACHE:
                    self._semantic_hits += 1

            if intent_id in self._by_intent:
                self._by_intent[intent_id] += 1
            elif len(self._by_intent) < MAX_TRACKED_INTENTS:
                self._by_intent[intent_id] = 1
            else:
                self._intents_dropped += 1

    def record_escalation(self) -> None:
        with self._lock:
            self._escalations += 1

    def record_disagreement(self) -> None:
        with self._lock:
            self._disagreements += 1

    def record_layer_latency(self, layer: ClassifierLayer, latency_ms: float) -> None:
        """Time spent in one layer, whether or not it produced the answer."""
        with self._lock:
            self._layer_latency[layer.value].observe(latency_ms)

    @property
    def abstention_rate(self) -> float | None:
        with self._lock:
            if not self._classifications:
                return None
            return self._abstentions / self._classifications

    @property
    def cache_hit_rate(self) -> float | None:
        with self._lock:
            if not self._classifications:
                return None
            return (self._exact_hits + self._semantic_hits) / self._classifications

    def snapshot(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "classifications": self._classifications,
                "abstentions": self._abstentions,
                "unknown": self._unknown,
                "escalations": self._escalations,
                "disagreements": self._disagreements,
                "abstention_rate": (
                    self._abstentions / self._classifications if self._classifications else None
                ),
                "ambiguous": self._ambiguous,
                "cache_hits": {
                    "exact": self._exact_hits,
                    "semantic": self._semantic_hits,
                },
                "by_layer": dict(self._by_layer),
                "by_intent": dict(sorted(self._by_intent.items())),
                "intents_dropped": self._intents_dropped,
                "cost_usd": round(self._cost_usd, 6),
                "confidence": self._confidence.as_dict(),
                "latency_ms": self._latency.as_dict(),
                "layer_latency_ms": {
                    layer: histogram.as_dict()
                    for layer, histogram in self._layer_latency.items()
                    if histogram.observations
                },
            }

    def reset(self) -> None:
        """Only for tests. Production counters are monotonic."""
        with self._lock:
            self._by_layer = {layer.value: 0 for layer in ClassifierLayer}
            self._by_intent.clear()
            self._intents_dropped = 0
            self._classifications = 0
            self._abstentions = 0
            self._unknown = 0
            self._ambiguous = 0
            self._exact_hits = 0
            self._semantic_hits = 0
            self._escalations = 0
            self._disagreements = 0
            self._cost_usd = 0.0
            self._confidence = _Histogram(CONFIDENCE_BUCKETS)
            self._latency = _Histogram(LATENCY_BUCKETS_MS)
            self._layer_latency = {
                layer.value: _Histogram(LATENCY_BUCKETS_MS) for layer in ClassifierLayer
            }


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile. `None` for empty input rather than zero.

    Zero would read as "instant" on a latency chart; absence should read as
    absence.
    """
    if not values:
        return None
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]
