"""Each classifier layer on its own, away from the cascade."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from llm_fabric.errors import ProviderUnavailableError
from llm_fabric.intent.bootstrap import bootstrap_taxonomy
from llm_fabric.intent.classifiers.base import ClassifierVerdict, materialise
from llm_fabric.intent.classifiers.embedding import EmbeddingClassifier
from llm_fabric.intent.classifiers.rules import DeterministicClassifier, Rule
from llm_fabric.intent.classifiers.structured import (
    ClassifierPricing,
    StructuredIntentClassifier,
)
from llm_fabric.intent.embeddings import HashingEmbedder, centroid, cosine_similarity
from llm_fabric.intent.schema import (
    ClassificationRequest,
    ClassifierLayer,
    IntentAlternative,
)
from llm_fabric.intent.taxonomy import IntentTaxonomy
from llm_fabric.serving.base import (
    InferenceRequest,
    Provider,
    ProviderResult,
    StreamEvent,
)


@pytest.fixture
def taxonomy() -> IntentTaxonomy:
    return bootstrap_taxonomy()


def request_for(text: str) -> ClassificationRequest:
    return ClassificationRequest(text=text)


class ScriptedProvider(Provider):
    """Returns whatever text the test hands it, and records what it was asked."""

    name = "scripted"

    def __init__(
        self,
        reply: str = "{}",
        *,
        error: Exception | None = None,
        prompt_tokens: int = 100,
        completion_tokens: int = 20,
    ) -> None:
        self.reply = reply
        self.error = error
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.requests: list[InferenceRequest] = []

    async def generate(self, request: InferenceRequest) -> ProviderResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return ProviderResult(
            text=self.reply,
            finish_reason="stop",
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )

    def stream(self, request: InferenceRequest) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError("the intent classifier never streams")


class TestEmbeddings:
    def test_the_hashing_embedder_is_deterministic(self) -> None:
        embedder = HashingEmbedder(dimensions=128)

        assert embedder.embed_one("hello world") == embedder.embed_one("hello world")

    def test_identical_text_is_maximally_similar(self) -> None:
        vector = HashingEmbedder(dimensions=128).embed_one("summarise this article")

        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_it_measures_lexical_overlap_not_meaning(self) -> None:
        """Documents the limitation rather than pretending it is not there."""
        embedder = HashingEmbedder(dimensions=512)

        lexical = cosine_similarity(
            embedder.embed_one("the car is red"), embedder.embed_one("the car is blue")
        )
        semantic = cosine_similarity(
            embedder.embed_one("the car is red"), embedder.embed_one("the automobile is crimson")
        )
        assert lexical > semantic

    def test_an_empty_string_yields_a_zero_vector(self) -> None:
        embedder = HashingEmbedder(dimensions=64)

        assert set(embedder.embed_one("")) == {0.0}
        assert cosine_similarity(embedder.embed_one(""), embedder.embed_one("hello")) == 0.0

    def test_comparing_different_dimensions_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="different dimensions"):
            cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0))

    def test_a_centroid_of_nothing_is_empty(self) -> None:
        assert centroid([]) == ()


class TestDeterministicClassifier:
    async def test_a_clear_prompt_is_recognised(self, taxonomy: IntentTaxonomy) -> None:
        verdict = await DeterministicClassifier().classify(
            request_for("Summarise this article in three bullet points"), taxonomy
        )

        assert verdict.intent_id == "summarization"
        assert verdict.confidence > 0.7

    async def test_an_unmatched_prompt_yields_no_opinion(self, taxonomy: IntentTaxonomy) -> None:
        verdict = await DeterministicClassifier().classify(request_for("zzz qqq vvv"), taxonomy)

        assert verdict.has_opinion is False
        assert verdict.confidence == 0.0

    async def test_an_empty_prompt_yields_no_opinion(self, taxonomy: IntentTaxonomy) -> None:
        verdict = await DeterministicClassifier().classify(request_for("   "), taxonomy)

        assert verdict.has_opinion is False

    async def test_a_negative_rule_suppresses_a_superficial_match(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        """The strongest translation cue in the language, on a coding task."""
        verdict = await DeterministicClassifier().classify(
            request_for("Translate this Python code into Rust"), taxonomy
        )

        assert verdict.intent_id != "translation"

    async def test_a_specific_child_beats_its_parent_when_its_own_evidence_fires(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        verdict = await DeterministicClassifier().classify(
            request_for("This Python test fails in CI, here is the stack trace"), taxonomy
        )

        assert verdict.intent_id == "coding.debug"

    async def test_generic_evidence_alone_stays_at_the_parent(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        verdict = await DeterministicClassifier().classify(
            request_for("Refactor this class to use dependency injection"), taxonomy
        )

        assert verdict.intent_id == "coding"

    async def test_rules_for_intents_outside_the_taxonomy_are_ignored(self) -> None:
        classifier = DeterministicClassifier(rules=[Rule.build("nonexistent", r"anything", 10.0)])
        small = bootstrap_taxonomy()

        verdict = await classifier.classify(request_for("anything at all"), small)
        assert verdict.intent_id != "nonexistent"

    async def test_scanning_is_bounded(self, taxonomy: IntentTaxonomy) -> None:
        classifier = DeterministicClassifier(max_scan_chars=20)
        padded = "x" * 500 + " summarise this article"

        verdict = await classifier.classify(request_for(padded), taxonomy)
        assert verdict.has_opinion is False

    async def test_alternatives_are_reported_and_bounded(self, taxonomy: IntentTaxonomy) -> None:
        verdict = await DeterministicClassifier().classify(
            request_for("Summarise this article and translate it into German"), taxonomy
        )

        assert len(verdict.alternatives) <= 3
        assert all(alt.confidence <= verdict.confidence for alt in verdict.alternatives)


class TestEmbeddingClassifier:
    async def test_it_assigns_a_prompt_to_the_nearest_intent(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        classifier = EmbeddingClassifier(HashingEmbedder(dimensions=512))

        verdict = await classifier.classify(
            request_for("Summarise this article in three bullet points"), taxonomy
        )
        assert verdict.intent_id == "summarization"

    async def test_preparation_is_idempotent(self, taxonomy: IntentTaxonomy) -> None:
        classifier = EmbeddingClassifier(HashingEmbedder(dimensions=128))
        await classifier.prepare(taxonomy)
        first = classifier.score(await classifier.embed_prompt("translate this into French"))
        await classifier.prepare(taxonomy)
        second = classifier.score(await classifier.embed_prompt("translate this into French"))

        assert first.intent_id == second.intent_id
        assert first.confidence == pytest.approx(second.confidence)

    async def test_scoring_before_preparation_yields_no_opinion(self) -> None:
        classifier = EmbeddingClassifier(HashingEmbedder(dimensions=64))

        assert classifier.score((0.0, 1.0)).has_opinion is False

    async def test_a_prompt_unlike_every_intent_scores_low(self, taxonomy: IntentTaxonomy) -> None:
        classifier = EmbeddingClassifier(HashingEmbedder(dimensions=512))

        verdict = await classifier.classify(request_for("qqqq wwww eeee rrrr"), taxonomy)
        assert verdict.confidence < 0.5

    async def test_the_embedding_model_is_part_of_the_version(self) -> None:
        small = EmbeddingClassifier(HashingEmbedder(dimensions=64))
        large = EmbeddingClassifier(HashingEmbedder(dimensions=512))

        assert small.version != large.version

    async def test_an_intent_without_examples_gets_no_centroid(self) -> None:
        from llm_fabric.intent.taxonomy import IntentNode

        taxonomy = IntentTaxonomy(
            "v1",
            [
                IntentNode(
                    intent_id="described_only",
                    name="Described only",
                    description="has a description but no examples",
                ),
                IntentNode(
                    intent_id="exemplified",
                    name="Exemplified",
                    description="has examples",
                    examples=("summarise this document",),
                ),
            ],
        )
        classifier = EmbeddingClassifier(HashingEmbedder(dimensions=256))

        verdict = await classifier.classify(request_for("summarise this document"), taxonomy)
        assert verdict.intent_id == "exemplified"


class TestStructuredClassifier:
    async def test_a_valid_reply_is_accepted(self, taxonomy: IntentTaxonomy) -> None:
        provider = ScriptedProvider(
            '{"intent_id": "coding", "confidence": 0.88, '
            '"alternatives": [{"intent_id": "coding.debug", "confidence": 0.3}], '
            '"reasoning": "asks for code"}'
        )
        classifier = StructuredIntentClassifier(provider, "small-model")

        verdict = await classifier.classify(request_for("write me a parser"), taxonomy)
        assert verdict.intent_id == "coding"
        assert verdict.confidence == pytest.approx(0.88)
        assert verdict.alternatives[0].intent_id == "coding.debug"

    async def test_code_fences_are_tolerated(self, taxonomy: IntentTaxonomy) -> None:
        provider = ScriptedProvider('```json\n{"intent_id": "writing", "confidence": 0.7}\n```')

        verdict = await StructuredIntentClassifier(provider, "m").classify(
            request_for("draft an email"), taxonomy
        )
        assert verdict.intent_id == "writing"

    async def test_a_hallucinated_intent_is_discarded(self, taxonomy: IntentTaxonomy) -> None:
        provider = ScriptedProvider('{"intent_id": "telepathy", "confidence": 0.99}')

        verdict = await StructuredIntentClassifier(provider, "m").classify(
            request_for("anything"), taxonomy
        )
        assert verdict.has_opinion is False
        assert "not a candidate" in verdict.rationale

    async def test_an_explicit_abstention_is_passed_through(self, taxonomy: IntentTaxonomy) -> None:
        provider = ScriptedProvider(
            '{"intent_id": "unknown", "confidence": 0.2, "reasoning": "too vague"}'
        )

        verdict = await StructuredIntentClassifier(provider, "m").classify(
            request_for("do the thing"), taxonomy
        )
        assert verdict.has_opinion is False
        assert "abstained" in verdict.rationale

    async def test_abstain_true_is_honoured_even_with_a_known_id(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        provider = ScriptedProvider(
            '{"intent_id": "coding", "confidence": 0.4, "abstain": true, "reasoning": "unsure"}'
        )
        verdict = await StructuredIntentClassifier(provider, "m").classify(
            request_for("maybe code?"), taxonomy
        )
        assert verdict.has_opinion is False
        assert "abstained" in verdict.rationale

    @pytest.mark.parametrize(
        "reply",
        [
            "not json at all",
            '{"intent_id": "coding"}',
            '{"intent_id": "coding", "confidence": 4}',
            "[]",
            "",
        ],
    )
    async def test_malformed_replies_are_not_repaired(
        self, taxonomy: IntentTaxonomy, reply: str
    ) -> None:
        provider = ScriptedProvider(reply)

        verdict = await StructuredIntentClassifier(provider, "m").classify(
            request_for("anything"), taxonomy
        )
        assert verdict.has_opinion is False
        assert len(provider.requests) == 1, "a failed parse must not trigger a retry"

    async def test_a_provider_outage_becomes_an_absent_opinion(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        provider = ScriptedProvider(error=ProviderUnavailableError("down"))

        verdict = await StructuredIntentClassifier(provider, "m").classify(
            request_for("summarise this"), taxonomy
        )
        assert verdict.has_opinion is False
        assert "provider failed" in verdict.rationale

    async def test_cost_is_computed_from_registry_rates(self, taxonomy: IntentTaxonomy) -> None:
        provider = ScriptedProvider(
            '{"intent_id": "coding", "confidence": 0.9}',
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        classifier = StructuredIntentClassifier(
            provider,
            "m",
            pricing=ClassifierPricing(input_cost_per_mtok=2.0, output_cost_per_mtok=8.0),
        )

        verdict = await classifier.classify(request_for("write code"), taxonomy)
        assert verdict.cost_usd == pytest.approx(10.0)

    async def test_a_failed_call_still_reports_what_it_cost(self, taxonomy: IntentTaxonomy) -> None:
        provider = ScriptedProvider("garbage", prompt_tokens=1_000_000, completion_tokens=0)
        classifier = StructuredIntentClassifier(
            provider, "m", pricing=ClassifierPricing(input_cost_per_mtok=3.0)
        )

        verdict = await classifier.classify(request_for("anything"), taxonomy)
        assert verdict.has_opinion is False
        assert verdict.cost_usd == pytest.approx(3.0)

    async def test_the_output_budget_is_bounded(self, taxonomy: IntentTaxonomy) -> None:
        provider = ScriptedProvider('{"intent_id": "coding", "confidence": 0.9}')

        await StructuredIntentClassifier(provider, "m").classify(
            request_for("write code"), taxonomy
        )
        sent = provider.requests[0]
        assert sent.max_tokens is not None and sent.max_tokens <= 256
        assert sent.temperature == 0.0

    async def test_a_long_prompt_is_truncated_before_being_sent(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        provider = ScriptedProvider('{"intent_id": "coding", "confidence": 0.9}')
        classifier = StructuredIntentClassifier(provider, "m", max_prompt_chars=50)

        await classifier.classify(request_for("summarise " * 500), taxonomy)
        user_message = provider.requests[0].messages[-1].content
        assert len(user_message) < 200

    async def test_a_shortlist_narrows_the_candidates(self, taxonomy: IntentTaxonomy) -> None:
        provider = ScriptedProvider('{"intent_id": "coding", "confidence": 0.9}')
        classifier = StructuredIntentClassifier(provider, "m")
        request = request_for("write code")
        request.metadata["intent_shortlist"] = ["coding", "writing"]

        await classifier.classify(request, taxonomy)
        catalogue = provider.requests[0].messages[0].content
        assert "coding.debug" not in catalogue
        assert "coding:" in catalogue

    async def test_it_refuses_to_serve_a_layer_it_is_not_meant_for(self) -> None:
        with pytest.raises(ValueError, match="L4 and L5"):
            StructuredIntentClassifier(ScriptedProvider(), "m", layer=ClassifierLayer.L2_RULES)


class TestVerdict:
    def test_an_opinionless_verdict_cannot_carry_confidence(self) -> None:
        with pytest.raises(ValueError, match="zero confidence"):
            ClassifierVerdict(intent_id=None, confidence=0.5)

    def test_the_margin_is_the_gap_to_the_runner_up(self) -> None:
        verdict = ClassifierVerdict(
            intent_id="coding",
            confidence=0.8,
            alternatives=(IntentAlternative("writing", 0.5),),
        )

        assert verdict.margin == pytest.approx(0.3)
        assert ClassifierVerdict(intent_id="coding", confidence=0.8).margin == 0.8

    def test_materialising_fills_the_shape_from_the_taxonomy(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        verdict = ClassifierVerdict(intent_id="agent", confidence=0.9)

        result = materialise(
            verdict,
            taxonomy=taxonomy,
            layer=ClassifierLayer.L2_RULES,
            classifier_version="test-1",
            latency_ms=1.5,
        )
        node = taxonomy.require("agent")
        assert result.agent_required is node.profile.agent_required is True
        assert result.risk_class is node.profile.risk_class
        assert result.taxonomy_version == taxonomy.version
        assert result.abstain is False

    def test_materialising_an_opinionless_verdict_is_an_error(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        with pytest.raises(ValueError, match="holds no opinion"):
            materialise(
                ClassifierVerdict.no_opinion(),
                taxonomy=taxonomy,
                layer=ClassifierLayer.L2_RULES,
                classifier_version="test-1",
                latency_ms=0.0,
            )
