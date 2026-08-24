"""What every classifier layer agrees to produce.

A classifier returns an *opinion*, not a decision. It says which intent it
favours and how strongly; the cascade decides whether that is good enough to
return, using the layer's threshold. Keeping those two jobs apart is what lets
the same classifier sit behind a strict threshold in one deployment and a
permissive one in another.

A classifier is also allowed to have no opinion. `ClassifierVerdict.no_opinion()`
is different from a low-confidence guess: the first says "not my department",
the second says "probably this, barely". The cascade treats them differently.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Self, runtime_checkable

from llm_fabric.intent.schema import (
    ClassificationRequest,
    ClassifierLayer,
    IntentAlternative,
    IntentClassification,
    IntentProfile,
    minimum_capability_grade,
    privacy_from_risk,
    safety_from_risk,
)
from llm_fabric.intent.taxonomy import IntentTaxonomy

#: How many runners-up a classifier reports. Bounded: alternatives are for
#: auditing an ambiguous call, not for dumping the whole taxonomy into a trace.
MAX_ALTERNATIVES = 3


@dataclass(frozen=True, slots=True)
class ClassifierVerdict:
    """One classifier's opinion about one prompt."""

    intent_id: str | None
    confidence: float
    alternatives: tuple[IntentAlternative, ...] = ()
    cost_usd: float = 0.0
    #: Set when the classifier learned something about *this* prompt that the
    #: taxonomy's static profile does not capture, such as an unusually long
    #: input or an explicit request for JSON.
    profile_overrides: IntentProfile | None = None
    #: Free-text reason, carried into traces for explainability.
    rationale: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must lie in [0, 1], got {self.confidence!r}")
        if self.intent_id is None and self.confidence != 0.0:
            raise ValueError("a verdict with no intent must carry zero confidence")

    @classmethod
    def no_opinion(cls, rationale: str = "") -> Self:
        return cls(intent_id=None, confidence=0.0, rationale=rationale)

    @property
    def has_opinion(self) -> bool:
        return self.intent_id is not None

    @property
    def margin(self) -> float:
        """Gap between the favoured intent and the nearest runner-up."""
        if not self.alternatives:
            return self.confidence
        return self.confidence - self.alternatives[0].confidence


@runtime_checkable
class IntentClassifier(Protocol):
    """A single layer of the cascade."""

    @property
    def layer(self) -> ClassifierLayer: ...

    @property
    def version(self) -> str:
        """Identifies this classifier's behaviour.

        It is part of every cache key, so changing the classifier's behaviour
        without changing its version silently serves stale classifications.
        """
        ...

    async def classify(
        self, request: ClassificationRequest, taxonomy: IntentTaxonomy
    ) -> ClassifierVerdict: ...


def materialise(
    verdict: ClassifierVerdict,
    *,
    taxonomy: IntentTaxonomy,
    layer: ClassifierLayer,
    classifier_version: str,
    latency_ms: float,
    cache_hit: bool = False,
) -> IntentClassification:
    """Turn an opinion into a full classification using the taxonomy's profile.

    The cheap layers identify *which* intent without knowing its routing shape;
    the taxonomy supplies that. A classifier that did learn something specific
    about this prompt overrides it through `profile_overrides`.
    """
    if verdict.intent_id is None:
        raise ValueError("cannot materialise a verdict that holds no opinion")

    node = taxonomy.require(verdict.intent_id)
    profile = verdict.profile_overrides or node.profile

    return IntentClassification(
        intent_id=node.intent_id,
        domain=node.domain,
        task=node.task,
        subtask=node.subtask,
        complexity=profile.complexity,
        reasoning_level=profile.reasoning_level,
        modality=profile.modality,
        context_class=profile.context_class,
        risk_class=profile.risk_class,
        latency_class=profile.latency_class,
        quality_class=profile.quality_class,
        cost_class=profile.cost_class,
        required_capabilities=node.required_capabilities | profile.required_capabilities,
        agent_required=profile.agent_required,
        tools_required=profile.tools_required,
        structured_output=profile.structured_output,
        retrieval_required="retrieval"
        in (node.required_capabilities | profile.required_capabilities),
        confidence=verdict.confidence,
        alternatives=verdict.alternatives[:MAX_ALTERNATIVES],
        abstain=False,
        classifier_version=classifier_version,
        taxonomy_version=taxonomy.version,
        layer=layer,
        latency_ms=latency_ms,
        cost_usd=verdict.cost_usd,
        cache_hit=cache_hit,
        cache_source=_cache_source(layer, cache_hit),
        privacy_class=privacy_from_risk(profile.risk_class),
        safety_class=safety_from_risk(profile.risk_class),
        minimum_capability_grade=minimum_capability_grade(
            complexity=profile.complexity,
            reasoning_level=profile.reasoning_level,
            agent_required=profile.agent_required,
            abstain=False,
        ),
        recommended_quality_grade=profile.quality_class.value,
    )


def rescore(classification: IntentClassification, **changes: object) -> IntentClassification:
    """Copy a classification with provenance fields changed.

    Used when a cached result is served: the intent is the cached one, but the
    layer, latency and cache flag describe *this* request, not the request that
    populated the cache.
    """
    return replace(classification, **changes)  # type: ignore[arg-type]


def _cache_source(layer: ClassifierLayer, cache_hit: bool) -> str | None:
    del cache_hit
    if layer is ClassifierLayer.L0_EXACT_CACHE:
        return "l0_exact"
    if layer is ClassifierLayer.L1_SEMANTIC_CACHE:
        return "l1_semantic"
    return None
