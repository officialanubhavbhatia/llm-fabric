"""Ambiguity, multi-intent prompts, and prompts nobody can classify.

The interesting behaviour of a classifier is not what it does with "translate
this into German". It is what it does with a prompt that asks for two things at
once, a prompt that sits exactly between two intents, and a prompt that means
nothing at all. A classifier that answers all three confidently is worse than
one that admits the problem, because the router downstream has no way to tell
that the label it received was a guess.

These tests assert *behaviour*, not quality. Where the current implementation
handles a case badly, the test says so plainly rather than being weakened until
it passes.
"""

from __future__ import annotations

import pytest

from llm_fabric.intent.bootstrap import bootstrap_taxonomy
from llm_fabric.intent.cascade import CandidateBuffer
from llm_fabric.intent.classifiers.rules import DeterministicClassifier
from llm_fabric.intent.factory import build_offline_cascade
from llm_fabric.intent.schema import (
    AMBIGUITY_MARGIN,
    UNKNOWN_INTENT_ID,
    ClassificationRequest,
    ClassifierLayer,
    IntentClassification,
)
from llm_fabric.intent.taxonomy import IntentTaxonomy
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope

SCOPE = TenantScope(tenant_id="acme", user_id="alice")


@pytest.fixture
def taxonomy() -> IntentTaxonomy:
    return bootstrap_taxonomy()


@pytest.fixture
def rules() -> DeterministicClassifier:
    return DeterministicClassifier()


class TestMultiIntentPrompts:
    """Prompts that genuinely ask for two things."""

    MULTI_INTENT = [
        (
            "Summarise this article and then translate the summary into German",
            {"summarization", "translation"},
        ),
        (
            "Extract the figures from this report and tell me whether the trend is significant",
            {"extraction", "data_analysis"},
        ),
        ("Classify this image as a cat or a dog", {"classification", "vision"}),
    ]

    @pytest.mark.parametrize(("prompt", "constituents"), MULTI_INTENT)
    async def test_both_constituents_appear_in_the_ranking(
        self,
        rules: DeterministicClassifier,
        taxonomy: IntentTaxonomy,
        prompt: str,
        constituents: set[str],
    ) -> None:
        """The classifier must not simply lose one half of the request."""
        verdict = await rules.classify(ClassificationRequest(text=prompt), taxonomy)

        ranked = {verdict.intent_id} | {alt.intent_id for alt in verdict.alternatives}
        assert constituents <= ranked, f"ranking {ranked} dropped part of the request"

    @pytest.mark.parametrize(("prompt", "constituents"), MULTI_INTENT)
    async def test_split_evidence_suppresses_confidence(
        self,
        rules: DeterministicClassifier,
        taxonomy: IntentTaxonomy,
        prompt: str,
        constituents: set[str],
    ) -> None:
        """Two intents sharing the evidence must not produce a confident single label."""
        verdict = await rules.classify(ClassificationRequest(text=prompt), taxonomy)

        assert verdict.confidence < 0.7, (
            f"a two-intent prompt was labelled '{verdict.intent_id}' at "
            f"{verdict.confidence:.3f}, which is confident enough to route on"
        )

    async def test_adding_a_second_intent_lowers_confidence(
        self, rules: DeterministicClassifier, taxonomy: IntentTaxonomy
    ) -> None:
        """The invariant behind the behaviour above, isolated."""
        single = await rules.classify(
            ClassificationRequest(text="Summarise this article"), taxonomy
        )
        double = await rules.classify(
            ClassificationRequest(text="Summarise this article and translate it into German"),
            taxonomy,
        )

        assert single.confidence > double.confidence

    async def test_a_multi_intent_prompt_reaches_abstention_in_the_offline_cascade(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        engine = build_offline_cascade(taxonomy, TenantScopedCache())

        decision = await engine.classify(
            SCOPE,
            ClassificationRequest(
                text="Summarise this article and then translate the summary into German"
            ),
        )
        assert decision.abstained is True

    async def test_a_sequential_request_inside_one_domain_is_still_answered(
        self, rules: DeterministicClassifier, taxonomy: IntentTaxonomy
    ) -> None:
        """A known weakness, recorded rather than hidden.

        "Review this diff and fix any bugs" asks for two things, but both are
        coding, so the evidence reinforces one branch instead of splitting. The
        classifier answers confidently. That is defensible — every candidate
        label is in the same domain, so the routing consequence is small — but
        it means the confidence signal does not detect this shape of ambiguity.
        """
        verdict = await rules.classify(
            ClassificationRequest(text="Review this diff and fix any bugs you find"),
            taxonomy,
        )

        assert verdict.intent_id.startswith("coding")
        assert verdict.confidence > 0.7


class TestAmbiguity:
    """Prompts that sit between two intents rather than asking for both."""

    async def test_a_near_tie_is_flagged_as_ambiguous(
        self, rules: DeterministicClassifier, taxonomy: IntentTaxonomy
    ) -> None:
        verdict = await rules.classify(
            ClassificationRequest(text="Classify this image as a cat or a dog"), taxonomy
        )

        assert verdict.margin < AMBIGUITY_MARGIN

    async def test_the_ambiguity_flag_survives_onto_the_classification(self) -> None:
        engine = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())

        decision = await engine.classify(
            SCOPE, ClassificationRequest(text="Classify this image as a cat or a dog")
        )
        assert decision.trace_attributes()["intent.ambiguous"] is (
            decision.classification.is_ambiguous
        )

    async def test_a_clear_winner_is_not_ambiguous(
        self, rules: DeterministicClassifier, taxonomy: IntentTaxonomy
    ) -> None:
        verdict = await rules.classify(
            ClassificationRequest(text="Translate this paragraph into Japanese"), taxonomy
        )

        assert verdict.margin >= AMBIGUITY_MARGIN

    async def test_ambiguous_results_are_offered_to_the_learning_loop(self) -> None:
        buffer = CandidateBuffer()
        engine = build_offline_cascade(
            bootstrap_taxonomy(), TenantScopedCache(), candidate_sink=buffer
        )

        await engine.classify(
            SCOPE, ClassificationRequest(text="Classify this image as a cat or a dog")
        )
        assert buffer.items, "an ambiguous prompt is exactly what the loop wants to see"


class TestUnknownIntents:
    """Prompts with no classifiable content."""

    NONSENSE = ["asdkjh qwe zxcvb", "42", "###", "the the the and and", "   "]

    @pytest.mark.parametrize("prompt", NONSENSE)
    async def test_the_rules_layer_holds_no_opinion(
        self, rules: DeterministicClassifier, taxonomy: IntentTaxonomy, prompt: str
    ) -> None:
        verdict = await rules.classify(ClassificationRequest(text=prompt), taxonomy)

        assert verdict.has_opinion is False
        assert verdict.confidence == 0.0

    @pytest.mark.parametrize("prompt", NONSENSE)
    async def test_the_cascade_abstains(self, prompt: str) -> None:
        engine = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())

        decision = await engine.classify(SCOPE, ClassificationRequest(text=prompt))

        assert decision.abstained is True
        assert decision.classification.intent_id == UNKNOWN_INTENT_ID
        assert decision.layer is ClassifierLayer.ABSTAIN

    async def test_a_referential_prompt_abstains_because_context_is_absent(self) -> None:
        """The intent is real, but it lives in history this classifier cannot see."""
        engine = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())

        decision = await engine.classify(
            SCOPE, ClassificationRequest(text="Do the thing we discussed earlier")
        )
        assert decision.abstained is True

    async def test_an_abstention_is_a_well_formed_result_not_an_error(self) -> None:
        engine = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())

        decision = await engine.classify(SCOPE, ClassificationRequest(text="zzz qqq"))
        classification = decision.classification

        assert isinstance(classification, IntentClassification)
        assert classification.taxonomy_version == bootstrap_taxonomy().version
        assert classification.classifier_version
        assert classification.as_dict()["abstain"] is True

    async def test_abstention_is_not_reached_by_exception(self) -> None:
        """Every layer failing must still produce a result, never raise."""
        engine = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())

        decision = await engine.classify(SCOPE, ClassificationRequest(text="\x00\x01"))
        assert decision.abstained is True

    async def test_unknown_never_leaks_a_taxonomy_intent_id(self) -> None:
        engine = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())

        decision = await engine.classify(SCOPE, ClassificationRequest(text="###"))
        assert decision.classification.domain == UNKNOWN_INTENT_ID
        assert decision.classification.task is None
