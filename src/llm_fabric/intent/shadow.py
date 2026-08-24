"""Shadow classification: a candidate cascade watches sampled traffic.

The candidate's result is recorded and never returned to the caller. Expensive
layers must not be double-called unless sampling is configured to allow it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from llm_fabric.intent.cascade import IntentCascade, IntentDecision
from llm_fabric.intent.schema import ClassificationRequest
from llm_fabric.tenancy.scope import TenantScope


@dataclass(frozen=True, slots=True)
class ShadowObservation:
    production_intent: str
    candidate_intent: str
    production_confidence: float
    candidate_confidence: float
    production_layer: str
    candidate_layer: str
    differed: bool
    candidate_latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "production_intent": self.production_intent,
            "candidate_intent": self.candidate_intent,
            "production_confidence": round(self.production_confidence, 4),
            "candidate_confidence": round(self.candidate_confidence, 4),
            "production_layer": self.production_layer,
            "candidate_layer": self.candidate_layer,
            "differed": self.differed,
            "candidate_latency_ms": round(self.candidate_latency_ms, 3),
        }


class ShadowClassifier:
    """Optionally run a second cascade on a sampled fraction of requests."""

    def __init__(
        self,
        candidate: IntentCascade,
        *,
        sample_rate: float = 0.0,
        allow_paid_layers: bool = False,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must lie in [0, 1]")
        self._candidate = candidate
        self._sample_rate = sample_rate
        self._allow_paid = allow_paid_layers
        self.observations: list[ShadowObservation] = []

    def should_sample(self, request: ClassificationRequest) -> bool:
        if self._sample_rate <= 0.0:
            return False
        if self._sample_rate >= 1.0:
            return True
        digest = hashlib.sha256(request.text.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        return bucket < self._sample_rate

    async def observe(
        self,
        scope: TenantScope,
        request: ClassificationRequest,
        production: IntentDecision,
    ) -> ShadowObservation | None:
        if not self.should_sample(request):
            return None
        if not self._allow_paid and self._candidate.uses_paid_layers:
            return None
            # Paid layers stay dark unless an operator opted in. The candidate
            # can still be an offline cascade.
            return None

        decision = await self._candidate.classify(scope, request)
        observation = ShadowObservation(
            production_intent=production.classification.intent_id,
            candidate_intent=decision.classification.intent_id,
            production_confidence=production.classification.confidence,
            candidate_confidence=decision.classification.confidence,
            production_layer=production.classification.layer.value,
            candidate_layer=decision.classification.layer.value,
            differed=production.classification.intent_id != decision.classification.intent_id,
            candidate_latency_ms=decision.total_latency_ms,
        )
        self.observations.append(observation)
        return observation
