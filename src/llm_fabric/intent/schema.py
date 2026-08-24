"""The typed result of classifying a prompt.

Every field the constitution names is present and typed. The classes that infer
these values differ wildly in cost and reliability, so the result also records
*which layer produced it*, what it cost, and how confident it is. A consumer
that cannot see the provenance of a classification cannot decide how much to
trust it.

Nothing here performs classification. This module is the vocabulary the whole
subsystem agrees on.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self

#: The intent assigned when no layer reached its confidence threshold.
#: Unknown is a first-class outcome, not a failure: routing a misunderstood
#: prompt confidently is worse than admitting the prompt was not understood.
UNKNOWN_INTENT_ID = "unknown"

#: Optimized route inference is allowed only at or above this confidence, and
#: only for a KNOWN serving state. Medium, low, unknown, abstain, and
#: safe-fallback use a balanced capability floor.
HIGH_CONFIDENCE = 0.90


class ServingClassificationState(StrEnum):
    """How the serving path treats this IntentResult.

    Coverage, not accuracy: every eligible invocation carries one of these.
    UNKNOWN is constitutionally valid. Never force a known label to advertise
    100% coverage.
    """

    KNOWN = "known"
    UNKNOWN = "unknown"
    ABSTAIN = "abstain"
    SAFE_FALLBACK = "safe_fallback"


class ClassifierLayer(StrEnum):
    """Which layer of the cascade produced a result."""

    L0_EXACT_CACHE = "l0_exact_cache"
    L1_SEMANTIC_CACHE = "l1_semantic_cache"
    L2_RULES = "l2_rules"
    L3_EMBEDDING = "l3_embedding"
    L4_STRUCTURED_LLM = "l4_structured_llm"
    L5_ESCALATION = "l5_escalation"
    ABSTAIN = "abstain"
    SAFE_FALLBACK = "safe_fallback"

    @property
    def is_cache(self) -> bool:
        return self in (self.L0_EXACT_CACHE, self.L1_SEMANTIC_CACHE)


class Complexity(StrEnum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ReasoningLevel(StrEnum):
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    DEEP = "deep"
    EXTENDED = "extended"


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class ContextClass(StrEnum):
    """Expected working-set size, not the size of this particular prompt."""

    TINY = "tiny"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    VERY_LONG = "very_long"


class RiskClass(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class PrivacyClass(StrEnum):
    """How tightly the prompt's content should be held.

    Distinct from `RiskClass`: a medical note can be low-risk to generate and
    still be exclusive to a private deployment. Classification proposes this;
    tenant policy decides whether it is permitted.
    """

    STANDARD = "standard"
    STRICT = "strict"
    EXCLUSIVE = "exclusive"


class SafetyClass(StrEnum):
    """Safety attention the route should assume. Not an authorization grant."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class LatencyClass(StrEnum):
    REALTIME = "realtime"
    INTERACTIVE = "interactive"
    STANDARD = "standard"
    BATCH = "batch"


class QualityClass(StrEnum):
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"


class CostClass(StrEnum):
    MINIMAL = "minimal"
    LOW = "low"
    STANDARD = "standard"
    PREMIUM = "premium"


@dataclass(frozen=True, slots=True)
class IntentAlternative:
    """A runner-up the classifier considered.

    Alternatives are what make a classification auditable. A top-1 answer at
    0.51 with a 0.49 runner-up is a materially different fact from a 0.51 with
    nothing behind it, and only the former should be treated as ambiguous.
    """

    intent_id: str
    confidence: float

    def __post_init__(self) -> None:
        _check_confidence(self.confidence, "alternative confidence")


@dataclass(frozen=True, slots=True)
class IntentProfile:
    """The routing-relevant shape of an intent.

    Carried by the taxonomy rather than inferred per request, so the cheap
    layers can emit a complete classification. A classifier that learns better
    values for a specific prompt may override them.
    """

    complexity: Complexity = Complexity.MODERATE
    reasoning_level: ReasoningLevel = ReasoningLevel.LIGHT
    modality: Modality = Modality.TEXT
    context_class: ContextClass = ContextClass.SHORT
    risk_class: RiskClass = RiskClass.LOW
    latency_class: LatencyClass = LatencyClass.INTERACTIVE
    quality_class: QualityClass = QualityClass.STANDARD
    cost_class: CostClass = CostClass.LOW
    agent_required: bool = False
    tools_required: tuple[str, ...] = ()
    structured_output: bool = False
    required_capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class IntentClassification:
    """What the fabric decided a prompt is asking for."""

    intent_id: str
    domain: str
    complexity: Complexity
    reasoning_level: ReasoningLevel
    modality: Modality
    context_class: ContextClass
    risk_class: RiskClass
    latency_class: LatencyClass
    quality_class: QualityClass
    cost_class: CostClass
    confidence: float
    classifier_version: str
    taxonomy_version: str
    task: str | None = None
    subtask: str | None = None
    required_capabilities: frozenset[str] = frozenset()
    agent_required: bool = False
    tools_required: tuple[str, ...] = ()
    structured_output: bool = False
    alternatives: tuple[IntentAlternative, ...] = ()
    abstain: bool = False
    serving_state: ServingClassificationState = ServingClassificationState.KNOWN
    intent_result_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # Provenance. Not in the constitution's field list, but the observability
    # section requires classifier layer, cache hit and confidence as metrics,
    # and they are only knowable here.
    layer: ClassifierLayer = ClassifierLayer.L2_RULES
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cache_hit: bool = False
    cache_source: str | None = None
    language: str | None = None
    retrieval_required: bool = False
    privacy_class: PrivacyClass = PrivacyClass.STANDARD
    safety_class: SafetyClass = SafetyClass.LOW
    conversation_aware: bool = False
    secondary_intents: tuple[str, ...] = ()
    embedding_model_version: str | None = None
    prompt_version: str | None = None
    policy_version: str = "v1"
    #: Abstract capability floors for a future planner. Never a model name.
    minimum_capability_grade: str | None = None
    recommended_quality_grade: str | None = None

    def __post_init__(self) -> None:
        _check_confidence(self.confidence, "confidence")
        if self.serving_state is ServingClassificationState.KNOWN:
            if self.intent_id == UNKNOWN_INTENT_ID:
                raise ValueError("a known serving state cannot carry the unknown intent id")
            if self.abstain:
                raise ValueError(
                    "an abstaining classification must carry the unknown intent id; "
                    "abstention and a concrete answer are mutually exclusive"
                )
            return
        if self.intent_id != UNKNOWN_INTENT_ID:
            raise ValueError(
                f"{self.serving_state.value} serving state must carry the unknown intent id"
            )
        if self.serving_state is ServingClassificationState.ABSTAIN:
            if not self.abstain:
                raise ValueError("an abstaining serving state must set abstain=True")
            return
        if self.abstain:
            raise ValueError(
                f"{self.serving_state.value} is not abstention; use serving_state=abstain"
            )

    @classmethod
    def unknown(
        cls,
        *,
        classifier_version: str,
        taxonomy_version: str,
        confidence: float = 0.0,
        alternatives: tuple[IntentAlternative, ...] = (),
        layer: ClassifierLayer = ClassifierLayer.ABSTAIN,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ) -> Self:
        """Build the abstention result.

        `confidence` is the confidence in the *rejected* best guess, retained
        because "we were nearly sure" and "we had no idea" are different signals
        for the learning loop even though both abstain.
        """
        return cls._unclassified(
            serving_state=ServingClassificationState.ABSTAIN,
            abstain=True,
            layer=layer,
            classifier_version=classifier_version,
            taxonomy_version=taxonomy_version,
            confidence=confidence,
            alternatives=alternatives,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )

    @classmethod
    def unknown_result(
        cls,
        *,
        classifier_version: str,
        taxonomy_version: str,
        confidence: float = 0.0,
        alternatives: tuple[IntentAlternative, ...] = (),
        layer: ClassifierLayer = ClassifierLayer.ABSTAIN,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ) -> Self:
        """Build an UNKNOWN serving result. Not abstention; not a known label."""
        return cls._unclassified(
            serving_state=ServingClassificationState.UNKNOWN,
            abstain=False,
            layer=layer,
            classifier_version=classifier_version,
            taxonomy_version=taxonomy_version,
            confidence=confidence,
            alternatives=alternatives,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )

    @classmethod
    def safe_fallback(
        cls,
        *,
        classifier_version: str = "serving-unclassified",
        taxonomy_version: str = "none",
        confidence: float = 0.0,
        alternatives: tuple[IntentAlternative, ...] = (),
        layer: ClassifierLayer = ClassifierLayer.SAFE_FALLBACK,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ) -> Self:
        """Build a SAFE_FALLBACK result when the cascade cannot run.

        Used when classification is explicitly disabled in development/test, or
        when the cascade itself fails. The serving path still has an IntentResult.
        """
        return cls._unclassified(
            serving_state=ServingClassificationState.SAFE_FALLBACK,
            abstain=False,
            layer=layer,
            classifier_version=classifier_version,
            taxonomy_version=taxonomy_version,
            confidence=confidence,
            alternatives=alternatives,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )

    @classmethod
    def _unclassified(
        cls,
        *,
        serving_state: ServingClassificationState,
        abstain: bool,
        layer: ClassifierLayer,
        classifier_version: str,
        taxonomy_version: str,
        confidence: float,
        alternatives: tuple[IntentAlternative, ...],
        latency_ms: float,
        cost_usd: float,
    ) -> Self:
        return cls(
            intent_id=UNKNOWN_INTENT_ID,
            domain=UNKNOWN_INTENT_ID,
            complexity=Complexity.MODERATE,
            reasoning_level=ReasoningLevel.LIGHT,
            modality=Modality.TEXT,
            context_class=ContextClass.SHORT,
            risk_class=RiskClass.MODERATE,
            latency_class=LatencyClass.STANDARD,
            quality_class=QualityClass.STANDARD,
            cost_class=CostClass.STANDARD,
            confidence=confidence,
            classifier_version=classifier_version,
            taxonomy_version=taxonomy_version,
            alternatives=alternatives,
            abstain=abstain,
            serving_state=serving_state,
            layer=layer,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            privacy_class=PrivacyClass.STRICT,
            safety_class=SafetyClass.MODERATE,
            minimum_capability_grade="standard",
            recommended_quality_grade=QualityClass.STANDARD.value,
        )

    @property
    def allows_optimized_route(self) -> bool:
        """True only for a high-confidence known classification."""
        return (
            self.serving_state is ServingClassificationState.KNOWN
            and not self.abstain
            and self.confidence >= HIGH_CONFIDENCE
        )

    @property
    def is_ambiguous(self) -> bool:
        """True when the runner-up is close enough to make top-1 a coin toss."""
        if not self.alternatives:
            return False
        return (self.confidence - self.alternatives[0].confidence) < AMBIGUITY_MARGIN

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "domain": self.domain,
            "task": self.task,
            "subtask": self.subtask,
            "complexity": self.complexity.value,
            "reasoning_level": self.reasoning_level.value,
            "required_capabilities": sorted(self.required_capabilities),
            "modality": self.modality.value,
            "agent_required": self.agent_required,
            "tools_required": list(self.tools_required),
            "structured_output": self.structured_output,
            "retrieval_required": self.retrieval_required,
            "context_class": self.context_class.value,
            "risk_class": self.risk_class.value,
            "privacy_class": self.privacy_class.value,
            "safety_class": self.safety_class.value,
            "latency_class": self.latency_class.value,
            "quality_class": self.quality_class.value,
            "cost_class": self.cost_class.value,
            "language": self.language,
            "confidence": round(self.confidence, 4),
            "alternatives": [
                {"intent_id": alt.intent_id, "confidence": round(alt.confidence, 4)}
                for alt in self.alternatives
            ],
            "secondary_intents": list(self.secondary_intents),
            "abstain": self.abstain,
            "serving_state": self.serving_state.value,
            "intent_result_id": self.intent_result_id,
            "classifier_version": self.classifier_version,
            "taxonomy_version": self.taxonomy_version,
            "embedding_model_version": self.embedding_model_version,
            "prompt_version": self.prompt_version,
            "policy_version": self.policy_version,
            "layer": self.layer.value,
            "latency_ms": round(self.latency_ms, 3),
            "cost_usd": round(self.cost_usd, 6),
            "cache_hit": self.cache_hit,
            "cache_source": self.cache_source,
            "conversation_aware": self.conversation_aware,
            "minimum_capability_grade": self.minimum_capability_grade,
            "recommended_quality_grade": self.recommended_quality_grade,
        }

    def trace_attributes(self) -> dict[str, Any]:
        """Span attributes for one classification.

        Deliberately excludes the prompt. A trace attribute is not a safe place
        for caller content, and the constitution keeps unbounded-cardinality
        values out of metric labels.
        """
        return {
            "intent.id": self.intent_id,
            "intent.domain": self.domain,
            "intent.confidence": round(self.confidence, 4),
            "intent.abstain": self.abstain,
            "intent.serving_state": self.serving_state.value,
            "intent.result_id": self.intent_result_id,
            "intent.layer": self.layer.value,
            "intent.cache_hit": self.cache_hit,
            "intent.cache_source": self.cache_source or "none",
            "intent.taxonomy_version": self.taxonomy_version,
            "intent.classifier_version": self.classifier_version,
            "intent.embedding_model_version": self.embedding_model_version or "",
            "intent.latency_ms": round(self.latency_ms, 3),
            "intent.conversation_aware": self.conversation_aware,
        }


#: Confidence gap below which top-1 and the runner-up are treated as a tie.
AMBIGUITY_MARGIN = 0.10


@dataclass(slots=True)
class ClassificationRequest:
    """Everything a classifier is allowed to see.

    `conversation_state_signature` and `policy_version` exist because they are
    mandated cache-key components: a classification computed under one
    conversation state or one policy version must not be reused under another.
    """

    text: str
    language: str = "en"
    policy_version: str = "v1"
    conversation_state_signature: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("classification input must be text")

    @property
    def conversation_aware(self) -> bool:
        return bool(self.conversation_state_signature)


def _check_confidence(value: float, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1], got {value!r}")


def privacy_from_risk(risk: RiskClass) -> PrivacyClass:
    if risk is RiskClass.CRITICAL:
        return PrivacyClass.EXCLUSIVE
    if risk is RiskClass.HIGH:
        return PrivacyClass.STRICT
    return PrivacyClass.STANDARD


def safety_from_risk(risk: RiskClass) -> SafetyClass:
    return SafetyClass(risk.value)


def minimum_capability_grade(
    *,
    complexity: Complexity,
    reasoning_level: ReasoningLevel,
    agent_required: bool,
    abstain: bool,
) -> str:
    """Abstract capability floor. Never a model name.

    `economy` / `standard` / `high` / `maximum` are quality bands the future
    route planner can map onto whatever deployments it currently has.
    """
    if abstain:
        return "standard"
    if agent_required or complexity is Complexity.VERY_COMPLEX:
        return "maximum"
    if complexity is Complexity.COMPLEX or reasoning_level in (
        ReasoningLevel.DEEP,
        ReasoningLevel.EXTENDED,
    ):
        return "high"
    if complexity is Complexity.TRIVIAL:
        return "economy"
    return "standard"
