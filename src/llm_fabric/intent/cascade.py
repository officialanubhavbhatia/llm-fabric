"""The cascading decision engine.

Layers run cheapest-first and stop at the first one whose confidence clears its
threshold:

    L0 exact cache -> L1 semantic cache -> L2 rules -> L3 embedding
    -> L4 structured model -> L5 escalation -> abstain

**Thresholds fall as the cascade deepens**, which looks backwards until you see
what a threshold is for. It is the price of skipping everything below. A regex
match must be near-certain to earn the right to stop the cascade, because the
layers it would have skipped are better than it is. The escalation layer has
nothing better behind it, so demanding the same certainty of it would only
convert answers into abstentions.

Abstention is the honest floor. When no layer clears its bar the engine returns
`unknown` rather than the best of several bad guesses, and the router is
expected to treat that as "route conservatively", not as an error.

The engine never lets a classifier failure become a request failure. Classifying
a prompt is an optimisation; the caller asked for a completion.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from llm_fabric.intent.cache import (
    ExactIntentCache,
    IntentCacheDiscriminators,
    SemanticIntentCache,
)
from llm_fabric.intent.classifiers.base import (
    ClassifierVerdict,
    IntentClassifier,
    materialise,
    rescore,
)
from llm_fabric.intent.classifiers.embedding import EmbeddingClassifier
from llm_fabric.intent.embeddings import Vector
from llm_fabric.intent.features import bound_text, infer_profile_overrides
from llm_fabric.intent.metrics import IntentMetrics
from llm_fabric.intent.schema import (
    ClassificationRequest,
    ClassifierLayer,
    IntentAlternative,
    IntentClassification,
    minimum_capability_grade,
)
from llm_fabric.intent.taxonomy import IntentTaxonomy
from llm_fabric.tenancy.scope import TenantScope

#: Exact-cache lifetime. Longer than the semantic cache because an exact hit is
#: a fact about an identical prompt, not a guess about a similar one.
DEFAULT_EXACT_TTL_SECONDS = 900.0


@dataclass(frozen=True, slots=True)
class CascadeThresholds:
    """Confidence each layer must reach to end the cascade."""

    semantic_cache: float = 0.80
    rules: float = 0.70
    embedding: float = 0.62
    structured: float = 0.55
    escalation: float = 0.40
    #: When L2 and L3 name the same intent, accept at this lower floor.
    agreement_floor: float = 0.48

    def __post_init__(self) -> None:
        for name in (
            "semantic_cache",
            "rules",
            "embedding",
            "structured",
            "escalation",
            "agreement_floor",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} threshold must lie in [0, 1]")

    def for_layer(self, layer: ClassifierLayer) -> float:
        return {
            ClassifierLayer.L0_EXACT_CACHE: 0.0,
            ClassifierLayer.L1_SEMANTIC_CACHE: self.semantic_cache,
            ClassifierLayer.L2_RULES: self.rules,
            ClassifierLayer.L3_EMBEDDING: self.embedding,
            ClassifierLayer.L4_STRUCTURED_LLM: self.structured,
            ClassifierLayer.L5_ESCALATION: self.escalation,
            ClassifierLayer.ABSTAIN: 0.0,
        }[layer]


@dataclass(frozen=True, slots=True)
class LayerAttempt:
    """What one layer did. The unit of intent tracing."""

    layer: ClassifierLayer
    intent_id: str | None
    confidence: float
    threshold: float
    accepted: bool
    latency_ms: float
    cost_usd: float = 0.0
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer.value,
            "intent_id": self.intent_id,
            "confidence": round(self.confidence, 4),
            "threshold": self.threshold,
            "accepted": self.accepted,
            "latency_ms": round(self.latency_ms, 3),
            "cost_usd": round(self.cost_usd, 6),
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class IntentDecision:
    """The classification plus the full record of how it was reached."""

    classification: IntentClassification
    attempts: tuple[LayerAttempt, ...] = ()
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0

    @property
    def layer(self) -> ClassifierLayer:
        return self.classification.layer

    @property
    def abstained(self) -> bool:
        return self.classification.abstain

    def trace_attributes(self) -> dict[str, Any]:
        attributes = dict(self.classification.trace_attributes())
        attributes["intent.layers_tried"] = len(self.attempts)
        attributes["intent.total_latency_ms"] = round(self.total_latency_ms, 3)
        attributes["intent.total_cost_usd"] = round(self.total_cost_usd, 6)
        attributes["intent.ambiguous"] = self.classification.is_ambiguous
        return attributes

    def as_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.as_dict(),
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "total_latency_ms": round(self.total_latency_ms, 3),
            "total_cost_usd": round(self.total_cost_usd, 6),
        }


@dataclass(frozen=True, slots=True)
class CandidateExample:
    """A prompt worth a human's attention.

    Fed to `candidate_sink` for the learning loop. Nothing in this process trains
    on it or promotes anything: the constitution requires sanitisation,
    deduplication, offline evaluation, shadow traffic, a canary and a statistical
    gate before any classifier changes, and none of that happens on the request
    path.
    """

    tenant_id: str
    text: str
    reason: str
    decision: IntentDecision


class IntentCascade:
    """Runs the layers in order and decides where to stop."""

    def __init__(
        self,
        *,
        taxonomy: IntentTaxonomy,
        exact_cache: ExactIntentCache,
        rules: IntentClassifier | None = None,
        embedding: EmbeddingClassifier | None = None,
        structured: IntentClassifier | None = None,
        escalation: IntentClassifier | None = None,
        semantic_cache: SemanticIntentCache | None = None,
        thresholds: CascadeThresholds | None = None,
        metrics: IntentMetrics | None = None,
        exact_ttl_seconds: float = DEFAULT_EXACT_TTL_SECONDS,
        candidate_sink: Callable[[CandidateExample], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        shadow: Any | None = None,
        calibrator: Any | None = None,
        recorder: Callable[..., Any] | None = None,
    ) -> None:
        self._taxonomy = taxonomy
        self._exact = exact_cache
        self._semantic = semantic_cache
        self._rules = rules
        self._embedding = embedding
        self._structured = structured
        self._escalation = escalation
        self._thresholds = thresholds or CascadeThresholds()
        self._metrics = metrics or IntentMetrics()
        self._exact_ttl = exact_ttl_seconds
        self._candidate_sink = candidate_sink
        self._clock = clock
        self._shadow = shadow
        self._calibrator = calibrator
        self._recorder = recorder

        if self._semantic is not None and self._embedding is None:
            raise ValueError(
                "the semantic cache needs an embedding classifier to vectorise prompts"
            )

    @property
    def taxonomy(self) -> IntentTaxonomy:
        return self._taxonomy

    @property
    def metrics(self) -> IntentMetrics:
        return self._metrics

    @property
    def exact_cache(self) -> ExactIntentCache:
        return self._exact

    @property
    def semantic_cache(self) -> SemanticIntentCache | None:
        return self._semantic

    @property
    def uses_paid_layers(self) -> bool:
        return self._structured is not None or self._escalation is not None

    @property
    def thresholds(self) -> CascadeThresholds:
        return self._thresholds

    @property
    def version(self) -> str:
        """A single version covering every layer.

        Cache keys carry one `classifier_version`, but a cached classification
        was produced by the whole cascade. Digesting every layer's version means
        changing any one of them invalidates the cache, which is the only safe
        default.
        """
        parts = [
            f"{name}={classifier.version if classifier else 'none'}"
            for name, classifier in (
                ("rules", self._rules),
                ("embedding", self._embedding),
                ("structured", self._structured),
                ("escalation", self._escalation),
            )
        ]
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]
        return f"cascade-1.1:{digest}"

    def discriminators(self, request: ClassificationRequest) -> IntentCacheDiscriminators:
        return IntentCacheDiscriminators.build(
            request,
            taxonomy_version=self._taxonomy.version,
            classifier_version=self.version,
        )

    async def classify(self, scope: TenantScope, request: ClassificationRequest) -> IntentDecision:
        started = self._clock()
        discriminators = self.discriminators(request)
        attempts: list[LayerAttempt] = []

        # -- L0: exact cache -------------------------------------------------
        layer_started = self._clock()
        cached = self._exact.get(scope, request.text, discriminators)
        elapsed = self._elapsed_ms(layer_started)
        self._metrics.record_layer_latency(ClassifierLayer.L0_EXACT_CACHE, elapsed)
        if cached is not None:
            attempts.append(
                LayerAttempt(
                    layer=ClassifierLayer.L0_EXACT_CACHE,
                    intent_id=cached.intent_id,
                    confidence=cached.confidence,
                    threshold=0.0,
                    accepted=True,
                    latency_ms=elapsed,
                    rationale="exact prompt match",
                )
            )
            served = rescore(
                cached,
                layer=ClassifierLayer.L0_EXACT_CACHE,
                latency_ms=elapsed,
                cache_hit=True,
                cache_source="l0_exact",
                cost_usd=0.0,
            )
            return await self._complete(scope, request, served, attempts, started)
        attempts.append(
            LayerAttempt(
                layer=ClassifierLayer.L0_EXACT_CACHE,
                intent_id=None,
                confidence=0.0,
                threshold=0.0,
                accepted=False,
                latency_ms=elapsed,
                rationale="miss",
            )
        )

        # The prompt is embedded at most once and reused by L1 and L3.
        # A failed embedder must not take down classification or serving.
        vector: Vector | None = None
        bounded = ClassificationRequest(
            text=bound_text(request.text),
            language=request.language,
            policy_version=request.policy_version,
            conversation_state_signature=request.conversation_state_signature,
            metadata=request.metadata,
        )
        if self._embedding is not None:
            try:
                vector = await self._embedding.embed_prompt(bounded.text)
            except Exception:  # noqa: BLE001 - degrade; serving continues
                vector = None

        # -- L1: semantic cache ----------------------------------------------
        if self._semantic is not None and vector:
            layer_started = self._clock()
            match = self._semantic.lookup(scope, vector, discriminators)
            elapsed = self._elapsed_ms(layer_started)
            self._metrics.record_layer_latency(ClassifierLayer.L1_SEMANTIC_CACHE, elapsed)
            threshold = self._thresholds.semantic_cache
            accepted = match is not None and match.entry.classification.confidence >= threshold
            attempts.append(
                LayerAttempt(
                    layer=ClassifierLayer.L1_SEMANTIC_CACHE,
                    intent_id=match.entry.classification.intent_id if match else None,
                    confidence=match.entry.classification.confidence if match else 0.0,
                    threshold=threshold,
                    accepted=accepted,
                    latency_ms=elapsed,
                    rationale=f"similarity {match.similarity:.3f}" if match else "miss",
                )
            )
            if accepted and match is not None:
                served = rescore(
                    match.entry.classification,
                    layer=ClassifierLayer.L1_SEMANTIC_CACHE,
                    latency_ms=elapsed,
                    cache_hit=True,
                    cache_source="l1_semantic",
                    cost_usd=0.0,
                )
                return await self._complete(scope, request, served, attempts, started)

        # -- L2..L5: the classifiers -----------------------------------------
        best: ClassifierVerdict | None = None
        best_layer = ClassifierLayer.ABSTAIN
        shortlist: list[str] = []
        opinions: dict[ClassifierLayer, ClassifierVerdict] = {}

        for classifier, layer in (
            (self._rules, ClassifierLayer.L2_RULES),
            (self._embedding, ClassifierLayer.L3_EMBEDDING),
            (self._structured, ClassifierLayer.L4_STRUCTURED_LLM),
            (self._escalation, ClassifierLayer.L5_ESCALATION),
        ):
            if classifier is None:
                continue

            if (
                layer
                in (
                    ClassifierLayer.L4_STRUCTURED_LLM,
                    ClassifierLayer.L5_ESCALATION,
                )
                and shortlist
            ):
                bounded.metadata.setdefault("intent_shortlist", shortlist)

            layer_started = self._clock()
            if layer is ClassifierLayer.L5_ESCALATION:
                self._metrics.record_escalation()
            try:
                if layer is ClassifierLayer.L3_EMBEDDING and vector and self._embedding is not None:
                    await self._embedding.prepare(self._taxonomy)
                    verdict = self._embedding.score(vector)
                else:
                    verdict = await classifier.classify(bounded, self._taxonomy)
            except Exception as exc:  # noqa: BLE001 - layer outage is not a request outage
                verdict = ClassifierVerdict.no_opinion(f"{layer.value} failed: {exc}")
            elapsed = self._elapsed_ms(layer_started)
            self._metrics.record_layer_latency(layer, elapsed)

            if self._calibrator is not None and verdict.has_opinion:
                verdict = ClassifierVerdict(
                    intent_id=verdict.intent_id,
                    confidence=self._calibrator.calibrate(verdict.confidence),
                    alternatives=verdict.alternatives,
                    cost_usd=verdict.cost_usd,
                    profile_overrides=verdict.profile_overrides,
                    rationale=verdict.rationale,
                )

            threshold = self._thresholds.for_layer(layer)
            usable = verdict.has_opinion and verdict.intent_id in self._taxonomy
            accepted = usable and verdict.confidence >= threshold

            attempts.append(
                LayerAttempt(
                    layer=layer,
                    intent_id=verdict.intent_id,
                    confidence=verdict.confidence,
                    threshold=threshold,
                    accepted=accepted,
                    latency_ms=elapsed,
                    cost_usd=verdict.cost_usd,
                    rationale=verdict.rationale,
                )
            )

            if usable:
                shortlist = _shortlist(verdict, shortlist)
                opinions[layer] = verdict
                if best is None or verdict.confidence > best.confidence:
                    best, best_layer = verdict, layer

            if accepted:
                if layer is ClassifierLayer.L5_ESCALATION:
                    self._metrics.record_escalation()
                classification = self._materialise(
                    verdict, layer=layer, latency_ms=elapsed, request=request
                )
                self._admit(scope, request, discriminators, classification, vector)
                return await self._complete(scope, request, classification, attempts, started)

        agreed = _agreed_verdict(
            opinions.get(ClassifierLayer.L2_RULES),
            opinions.get(ClassifierLayer.L3_EMBEDDING),
            floor=self._thresholds.agreement_floor,
        )
        if agreed is not None:
            classification = self._materialise(
                agreed,
                layer=ClassifierLayer.L3_EMBEDDING,
                latency_ms=self._elapsed_ms(started),
                request=request,
            )
            attempts.append(
                LayerAttempt(
                    layer=ClassifierLayer.L3_EMBEDDING,
                    intent_id=agreed.intent_id,
                    confidence=agreed.confidence,
                    threshold=self._thresholds.agreement_floor,
                    accepted=True,
                    latency_ms=0.0,
                    rationale=agreed.rationale,
                )
            )
            self._admit(scope, request, discriminators, classification, vector)
            return await self._complete(scope, request, classification, attempts, started)

        # -- Abstain ----------------------------------------------------------
        # The best rejected guess is retained: "nearly certain but under the bar"
        # and "no idea" both abstain, and the learning loop needs to tell them
        # apart.
        rejected = (
            (IntentAlternative(intent_id=best.intent_id, confidence=best.confidence),)
            if best is not None and best.intent_id is not None
            else ()
        )
        abstention = IntentClassification.unknown(
            classifier_version=self.version,
            taxonomy_version=self._taxonomy.version,
            confidence=best.confidence if best else 0.0,
            alternatives=rejected,
            latency_ms=self._elapsed_ms(started),
            cost_usd=sum(attempt.cost_usd for attempt in attempts),
        )
        abstention = rescore(
            abstention,
            language=request.language,
            conversation_aware=request.conversation_aware,
            policy_version=request.policy_version,
            embedding_model_version=(
                self._embedding.version if self._embedding is not None else None
            ),
        )
        attempts.append(
            LayerAttempt(
                layer=ClassifierLayer.ABSTAIN,
                intent_id=None,
                confidence=best.confidence if best else 0.0,
                threshold=self._thresholds.for_layer(best_layer),
                accepted=True,
                latency_ms=0.0,
                rationale="no layer reached its threshold",
            )
        )
        return await self._complete(scope, request, abstention, attempts, started)

    # -- internals -----------------------------------------------------------

    def _materialise(
        self,
        verdict: ClassifierVerdict,
        *,
        layer: ClassifierLayer,
        latency_ms: float,
        request: ClassificationRequest,
    ) -> IntentClassification:
        classification = materialise(
            verdict,
            taxonomy=self._taxonomy,
            layer=layer,
            classifier_version=self.version,
            latency_ms=latency_ms,
        )
        node = self._taxonomy.require(verdict.intent_id) if verdict.intent_id else None
        profile = verdict.profile_overrides
        if profile is None and node is not None:
            profile = infer_profile_overrides(request, node.profile)
        secondary = tuple(
            alt.intent_id
            for alt in verdict.alternatives
            if alt.intent_id
            and alt.intent_id in self._taxonomy
            and self._taxonomy.require(alt.intent_id).domain != classification.domain
            and alt.confidence >= 0.25
        )
        embedding_version = None
        if self._embedding is not None:
            embedding_version = self._embedding.version
        prompt_version = None
        if layer in (ClassifierLayer.L4_STRUCTURED_LLM, ClassifierLayer.L5_ESCALATION):
            prompt_version = "structured-system-1"
        if node is not None and profile is not None:
            classification = rescore(
                classification,
                complexity=profile.complexity,
                reasoning_level=profile.reasoning_level,
                context_class=profile.context_class,
                structured_output=profile.structured_output,
            )
        return rescore(
            classification,
            language=request.language,
            conversation_aware=request.conversation_aware,
            policy_version=request.policy_version,
            secondary_intents=secondary,
            embedding_model_version=embedding_version,
            prompt_version=prompt_version,
            minimum_capability_grade=minimum_capability_grade(
                complexity=classification.complexity,
                reasoning_level=classification.reasoning_level,
                agent_required=classification.agent_required,
                abstain=False,
            ),
        )

    async def _complete(
        self,
        scope: TenantScope,
        request: ClassificationRequest,
        classification: IntentClassification,
        attempts: list[LayerAttempt],
        started: float,
    ) -> IntentDecision:
        if _layers_disagreed(attempts):
            self._metrics.record_disagreement()
        decision = self._finish(scope, request, classification, attempts, started)
        if self._recorder is not None:
            self._recorder(scope, decision)
        if self._shadow is not None:
            with suppress(Exception):
                await self._shadow.observe(scope, request, decision)
        return decision

    def _admit(
        self,
        scope: TenantScope,
        request: ClassificationRequest,
        discriminators: IntentCacheDiscriminators,
        classification: IntentClassification,
        vector: Vector | None,
    ) -> None:
        """Populate the caches from a freshly computed classification.

        Abstentions are deliberately not cached. An abstention is the outcome
        most likely to be fixed by a taxonomy or threshold change, and caching it
        would keep serving the old answer after the fix landed.
        """
        if classification.abstain:
            return

        self._exact.put(
            scope,
            request.text,
            discriminators,
            classification,
            ttl_seconds=self._exact_ttl,
        )
        if self._semantic is not None and vector:
            self._semantic.admit(scope, request.text, vector, discriminators, classification)

    def _finish(
        self,
        scope: TenantScope,
        request: ClassificationRequest,
        classification: IntentClassification,
        attempts: list[LayerAttempt],
        started: float,
    ) -> IntentDecision:
        total_latency = self._elapsed_ms(started)
        total_cost = sum(attempt.cost_usd for attempt in attempts)

        decision = IntentDecision(
            classification=classification,
            attempts=tuple(attempts),
            total_latency_ms=total_latency,
            total_cost_usd=total_cost,
        )

        self._metrics.record(
            layer=classification.layer,
            intent_id=classification.intent_id,
            confidence=classification.confidence,
            abstained=classification.abstain,
            ambiguous=classification.is_ambiguous,
            latency_ms=total_latency,
            cost_usd=total_cost,
            cache_hit=classification.cache_hit,
        )

        self._collect_candidate(scope, request, decision)
        self._observe_prom(decision)
        return decision

    def _observe_prom(self, decision: IntentDecision) -> None:
        from llm_fabric.observability.telemetry import current_telemetry

        telemetry = current_telemetry()
        if telemetry is None:
            return
        classification = decision.classification
        telemetry.metrics.observe_intent(
            layer=classification.layer.value,
            cache_hit=classification.cache_hit,
            abstained=classification.abstain,
            latency_s=decision.total_latency_ms / 1000.0,
            escalated=any(attempt.layer.value == "l5_escalation" for attempt in decision.attempts),
            disagreed=_layers_disagreed(decision.attempts),
            cache_source=classification.cache_source,
        )

    def _collect_candidate(
        self, scope: TenantScope, request: ClassificationRequest, decision: IntentDecision
    ) -> None:
        if self._candidate_sink is None or decision.classification.cache_hit:
            return

        reason = ""
        if decision.abstained:
            reason = "abstained"
        elif decision.classification.is_ambiguous:
            reason = "ambiguous"
        elif decision.classification.confidence < self._thresholds.escalation:
            reason = "low_confidence"
        elif _layers_disagreed(decision.attempts):
            reason = "layer_disagreement"

        if reason:
            self._candidate_sink(
                CandidateExample(
                    tenant_id=scope.tenant_id,
                    text=request.text,
                    reason=reason,
                    decision=decision,
                )
            )

    def _elapsed_ms(self, since: float) -> float:
        return (self._clock() - since) * 1000.0


def _agreed_verdict(
    rules: ClassifierVerdict | None,
    embedding: ClassifierVerdict | None,
    *,
    floor: float,
) -> ClassifierVerdict | None:
    """Accept when two independent cheap layers name the same intent.

    Neither layer cleared its own stop threshold, but agreement is stronger
    evidence than either score alone. Combined confidence is not a probability;
    it is a monotone combination used only for gating.
    """
    if (
        rules is None
        or embedding is None
        or not rules.has_opinion
        or not embedding.has_opinion
        or rules.intent_id != embedding.intent_id
    ):
        return None
    if max(rules.confidence, embedding.confidence) < floor:
        return None
    # A close runner-up is a multi-intent or hard-negative signal. Agreement
    # must not manufacture a high-confidence single label from a split.
    if rules.alternatives and rules.alternatives[0].confidence >= 0.18:
        return None
    if embedding.alternatives and embedding.alternatives[0].confidence >= 0.18:
        return None
    combined = 1.0 - (1.0 - rules.confidence) * (1.0 - embedding.confidence)
    return ClassifierVerdict(
        intent_id=rules.intent_id,
        confidence=min(0.99, combined),
        alternatives=embedding.alternatives or rules.alternatives,
        rationale=(
            f"L2/L3 agreed on {rules.intent_id} "
            f"({rules.confidence:.3f}, {embedding.confidence:.3f})"
        ),
    )


def _shortlist(verdict: ClassifierVerdict, existing: list[str]) -> list[str]:
    """Accumulate candidates for the model layers, preserving order, bounded."""
    candidates = list(existing)
    for intent_id in (verdict.intent_id, *(alt.intent_id for alt in verdict.alternatives)):
        if intent_id and intent_id not in candidates:
            candidates.append(intent_id)
    return candidates[:8]


def _layers_disagreed(attempts: tuple[LayerAttempt, ...] | list[LayerAttempt]) -> bool:
    """True when two classifier layers favoured different intents.

    Disagreement is a strong signal that a prompt sits on a boundary the
    taxonomy does not draw well, which is exactly what the learning loop wants
    to see.
    """
    opinions = {
        attempt.intent_id
        for attempt in attempts
        if attempt.intent_id is not None and not attempt.layer.is_cache
    }
    return len(opinions) > 1


@dataclass(slots=True)
class CandidateBuffer:
    """A bounded, in-memory sink for candidate hard examples.

    Deliberately small and lossy. Losing candidates is acceptable; growing
    without limit on the request path is not.
    """

    capacity: int = 256
    items: list[CandidateExample] = field(default_factory=list)
    dropped: int = 0

    def __call__(self, candidate: CandidateExample) -> None:
        if len(self.items) >= self.capacity:
            self.dropped += 1
            return
        self.items.append(candidate)

    def drain(self) -> list[CandidateExample]:
        drained = list(self.items)
        self.items.clear()
        return drained
