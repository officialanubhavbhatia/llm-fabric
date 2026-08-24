"""The classification result: field completeness and its invariants."""

from __future__ import annotations

import pytest

from llm_fabric.intent.schema import (
    UNKNOWN_INTENT_ID,
    ClassificationRequest,
    ClassifierLayer,
    Complexity,
    ContextClass,
    CostClass,
    IntentAlternative,
    IntentClassification,
    LatencyClass,
    Modality,
    QualityClass,
    ReasoningLevel,
    RiskClass,
)

#: Every field the constitution requires a classification to carry.
MANDATED_FIELDS = (
    "domain",
    "task",
    "subtask",
    "complexity",
    "reasoning_level",
    "required_capabilities",
    "modality",
    "agent_required",
    "tools_required",
    "structured_output",
    "context_class",
    "risk_class",
    "latency_class",
    "quality_class",
    "privacy_class",
    "safety_class",
    "language",
    "cache_source",
    "confidence",
    "alternatives",
    "abstain",
    "serving_state",
    "classifier_version",
    "taxonomy_version",
)


def classification(**overrides: object) -> IntentClassification:
    defaults: dict[str, object] = {
        "intent_id": "coding",
        "domain": "coding",
        "complexity": Complexity.MODERATE,
        "reasoning_level": ReasoningLevel.LIGHT,
        "modality": Modality.TEXT,
        "context_class": ContextClass.SHORT,
        "risk_class": RiskClass.LOW,
        "latency_class": LatencyClass.INTERACTIVE,
        "quality_class": QualityClass.STANDARD,
        "cost_class": CostClass.LOW,
        "confidence": 0.9,
        "classifier_version": "test-1",
        "taxonomy_version": "v1",
    }
    defaults.update(overrides)
    return IntentClassification(**defaults)  # type: ignore[arg-type]


def test_every_mandated_field_is_present() -> None:
    result = classification()

    for field in MANDATED_FIELDS:
        assert hasattr(result, field), f"classification is missing '{field}'"


def test_the_serialised_form_carries_every_mandated_field() -> None:
    payload = classification().as_dict()

    for field in MANDATED_FIELDS:
        assert field in payload, f"serialised classification is missing '{field}'"


def test_confidence_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        classification(confidence=1.4)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        classification(confidence=-0.1)


def test_a_boolean_is_not_accepted_as_a_confidence() -> None:
    with pytest.raises(TypeError):
        classification(confidence=True)


def test_abstention_and_a_concrete_intent_cannot_coexist() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        classification(abstain=True)

    with pytest.raises(ValueError, match="cannot carry the unknown intent id"):
        classification(intent_id=UNKNOWN_INTENT_ID, domain=UNKNOWN_INTENT_ID, abstain=False)


def test_the_unknown_factory_produces_a_valid_abstention() -> None:
    result = IntentClassification.unknown(classifier_version="test-1", taxonomy_version="v1")

    assert result.abstain is True
    assert result.intent_id == UNKNOWN_INTENT_ID
    assert result.serving_state.value == "abstain"
    assert result.layer is ClassifierLayer.ABSTAIN
    assert result.intent_result_id


def test_unknown_and_safe_fallback_are_valid_without_abstaining() -> None:
    unknown = IntentClassification.unknown_result(
        classifier_version="test-1", taxonomy_version="v1"
    )
    fallback = IntentClassification.safe_fallback()

    assert unknown.serving_state.value == "unknown"
    assert unknown.abstain is False
    assert fallback.serving_state.value == "safe_fallback"
    assert fallback.abstain is False
    assert unknown.intent_id == UNKNOWN_INTENT_ID
    assert fallback.intent_id == UNKNOWN_INTENT_ID


def test_an_abstention_retains_the_confidence_of_the_guess_it_rejected() -> None:
    """A near-miss and a blank are both abstentions, and must stay distinguishable."""
    near_miss = IntentClassification.unknown(
        classifier_version="test-1",
        taxonomy_version="v1",
        confidence=0.68,
        alternatives=(IntentAlternative("coding", 0.68),),
    )
    blank = IntentClassification.unknown(classifier_version="test-1", taxonomy_version="v1")

    assert near_miss.confidence == 0.68
    assert near_miss.alternatives[0].intent_id == "coding"
    assert blank.confidence == 0.0
    assert blank.alternatives == ()


def test_a_close_runner_up_makes_a_result_ambiguous() -> None:
    close = classification(confidence=0.55, alternatives=(IntentAlternative("writing", 0.52),))
    clear = classification(confidence=0.55, alternatives=(IntentAlternative("writing", 0.10),))
    alone = classification(confidence=0.55)

    assert close.is_ambiguous is True
    assert clear.is_ambiguous is False
    assert alone.is_ambiguous is False


def test_trace_attributes_never_carry_the_prompt() -> None:
    attributes = classification().trace_attributes()

    assert "intent.id" in attributes
    assert not any("prompt" in key or "text" in key for key in attributes)


def test_a_request_refuses_non_text_input() -> None:
    with pytest.raises(TypeError, match="must be text"):
        ClassificationRequest(text=object())  # type: ignore[arg-type]


def test_cache_relevant_request_fields_default_to_stable_values() -> None:
    request = ClassificationRequest(text="hello")

    assert request.language == "en"
    assert request.policy_version == "v1"
    assert request.conversation_state_signature == ""
