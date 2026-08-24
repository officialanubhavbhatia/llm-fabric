"""The cascade: ordering, thresholds, caching, tracing and cost."""

from __future__ import annotations

import pytest

from llm_fabric.intent.cache import (
    ExactIntentCache,
    SemanticCachePolicy,
    SemanticIntentCache,
)
from llm_fabric.intent.cascade import (
    CandidateBuffer,
    CascadeThresholds,
    IntentCascade,
)
from llm_fabric.intent.classifiers.base import ClassifierVerdict
from llm_fabric.intent.classifiers.embedding import EmbeddingClassifier
from llm_fabric.intent.classifiers.rules import DeterministicClassifier
from llm_fabric.intent.embeddings import HashingEmbedder
from llm_fabric.intent.factory import build_offline_cascade
from llm_fabric.intent.schema import (
    UNKNOWN_INTENT_ID,
    ClassificationRequest,
    ClassifierLayer,
    IntentAlternative,
)
from llm_fabric.intent.taxonomy import IntentNode, IntentTaxonomy
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope

SCOPE = TenantScope(tenant_id="acme", user_id="alice")


class StubClassifier:
    """A classifier that says exactly what the test tells it to."""

    def __init__(
        self,
        layer: ClassifierLayer,
        verdict: ClassifierVerdict,
        *,
        version: str = "stub-1",
    ) -> None:
        self._layer = layer
        self._verdict = verdict
        self._version = version
        self.calls = 0

    @property
    def layer(self) -> ClassifierLayer:
        return self._layer

    @property
    def version(self) -> str:
        return self._version

    async def classify(
        self, request: ClassificationRequest, taxonomy: IntentTaxonomy
    ) -> ClassifierVerdict:
        self.calls += 1
        return self._verdict


@pytest.fixture
def taxonomy() -> IntentTaxonomy:
    return IntentTaxonomy(
        "test-v1",
        [
            IntentNode(
                intent_id="coding",
                name="Coding",
                description="code",
                examples=("write a function",),
            ),
            IntentNode(
                intent_id="writing",
                name="Writing",
                description="prose",
                examples=("draft an email",),
            ),
        ],
    )


def cascade_with(taxonomy: IntentTaxonomy, **kwargs: object) -> IntentCascade:
    defaults: dict[str, object] = {
        "taxonomy": taxonomy,
        "exact_cache": ExactIntentCache(TenantScopedCache()),
    }
    defaults.update(kwargs)
    return IntentCascade(**defaults)  # type: ignore[arg-type]


class TestThresholdGating:
    async def test_a_confident_layer_stops_the_cascade(self, taxonomy: IntentTaxonomy) -> None:
        rules = StubClassifier(
            ClassifierLayer.L2_RULES, ClassifierVerdict(intent_id="coding", confidence=0.95)
        )
        structured = StubClassifier(
            ClassifierLayer.L4_STRUCTURED_LLM,
            ClassifierVerdict(intent_id="writing", confidence=0.99),
        )
        cascade = cascade_with(taxonomy, rules=rules, structured=structured)

        decision = await cascade.classify(SCOPE, ClassificationRequest(text="anything"))

        assert decision.classification.intent_id == "coding"
        assert decision.layer is ClassifierLayer.L2_RULES
        assert structured.calls == 0, "an accepted layer must not run the expensive ones"

    async def test_an_unconfident_layer_falls_through(self, taxonomy: IntentTaxonomy) -> None:
        rules = StubClassifier(
            ClassifierLayer.L2_RULES, ClassifierVerdict(intent_id="coding", confidence=0.30)
        )
        structured = StubClassifier(
            ClassifierLayer.L4_STRUCTURED_LLM,
            ClassifierVerdict(intent_id="writing", confidence=0.80),
        )
        cascade = cascade_with(taxonomy, rules=rules, structured=structured)

        decision = await cascade.classify(SCOPE, ClassificationRequest(text="anything"))

        assert decision.classification.intent_id == "writing"
        assert decision.layer is ClassifierLayer.L4_STRUCTURED_LLM
        assert rules.calls == 1

    async def test_thresholds_fall_as_the_cascade_deepens(self) -> None:
        """A cheap layer must be more certain than an expensive one to short-circuit."""
        thresholds = CascadeThresholds()

        assert thresholds.rules > thresholds.embedding
        assert thresholds.embedding > thresholds.structured
        assert thresholds.structured > thresholds.escalation

    async def test_an_out_of_range_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            CascadeThresholds(rules=1.5)

    async def test_a_verdict_naming_an_intent_outside_the_taxonomy_is_ignored(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        rules = StubClassifier(
            ClassifierLayer.L2_RULES, ClassifierVerdict(intent_id="ghost", confidence=0.99)
        )
        cascade = cascade_with(taxonomy, rules=rules)

        decision = await cascade.classify(SCOPE, ClassificationRequest(text="anything"))
        assert decision.abstained is True


class TestAbstention:
    async def test_nothing_clearing_its_bar_produces_unknown(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        rules = StubClassifier(
            ClassifierLayer.L2_RULES, ClassifierVerdict(intent_id="coding", confidence=0.2)
        )
        cascade = cascade_with(taxonomy, rules=rules)

        decision = await cascade.classify(SCOPE, ClassificationRequest(text="anything"))

        assert decision.abstained is True
        assert decision.classification.intent_id == UNKNOWN_INTENT_ID
        assert decision.layer is ClassifierLayer.ABSTAIN

    async def test_the_rejected_best_guess_survives_on_the_abstention(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        rules = StubClassifier(
            ClassifierLayer.L2_RULES, ClassifierVerdict(intent_id="coding", confidence=0.65)
        )
        cascade = cascade_with(taxonomy, rules=rules)

        decision = await cascade.classify(SCOPE, ClassificationRequest(text="anything"))

        assert decision.classification.confidence == pytest.approx(0.65)
        assert decision.classification.alternatives[0].intent_id == "coding"

    async def test_a_cascade_with_no_classifiers_abstains_rather_than_failing(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        decision = await cascade_with(taxonomy).classify(
            SCOPE, ClassificationRequest(text="anything")
        )

        assert decision.abstained is True

    async def test_an_abstention_is_never_cached(self, taxonomy: IntentTaxonomy) -> None:
        rules = StubClassifier(
            ClassifierLayer.L2_RULES, ClassifierVerdict(intent_id="coding", confidence=0.1)
        )
        cascade = cascade_with(taxonomy, rules=rules)
        request = ClassificationRequest(text="anything")

        await cascade.classify(SCOPE, request)
        second = await cascade.classify(SCOPE, request)

        assert second.layer is ClassifierLayer.ABSTAIN
        assert second.classification.cache_hit is False


class TestCaching:
    async def test_an_identical_prompt_is_served_from_the_exact_cache(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        rules = StubClassifier(
            ClassifierLayer.L2_RULES, ClassifierVerdict(intent_id="coding", confidence=0.95)
        )
        cascade = cascade_with(taxonomy, rules=rules)
        request = ClassificationRequest(text="write me a parser")

        first = await cascade.classify(SCOPE, request)
        second = await cascade.classify(SCOPE, request)

        assert first.layer is ClassifierLayer.L2_RULES
        assert second.layer is ClassifierLayer.L0_EXACT_CACHE
        assert second.classification.cache_hit is True
        assert second.classification.intent_id == "coding"
        assert rules.calls == 1

    async def test_a_cache_hit_reports_this_request_not_the_cached_one(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        rules = StubClassifier(
            ClassifierLayer.L2_RULES,
            ClassifierVerdict(intent_id="coding", confidence=0.95, cost_usd=0.02),
        )
        cascade = cascade_with(taxonomy, rules=rules)
        request = ClassificationRequest(text="write me a parser")

        await cascade.classify(SCOPE, request)
        second = await cascade.classify(SCOPE, request)

        assert second.classification.cost_usd == 0.0, "a cache hit costs nothing"
        assert second.total_cost_usd == 0.0

    async def test_a_changed_layer_version_invalidates_the_cache(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        exact = ExactIntentCache(TenantScopedCache())
        request = ClassificationRequest(text="write me a parser")
        verdict = ClassifierVerdict(intent_id="coding", confidence=0.95)

        old = cascade_with(
            taxonomy,
            exact_cache=exact,
            rules=StubClassifier(ClassifierLayer.L2_RULES, verdict, version="rules-1"),
        )
        new = cascade_with(
            taxonomy,
            exact_cache=exact,
            rules=StubClassifier(ClassifierLayer.L2_RULES, verdict, version="rules-2"),
        )

        await old.classify(SCOPE, request)
        decision = await new.classify(SCOPE, request)

        assert old.version != new.version
        assert decision.classification.cache_hit is False

    async def test_the_semantic_cache_serves_a_near_identical_prompt(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        embedding = EmbeddingClassifier(HashingEmbedder(dimensions=256))
        cascade = cascade_with(
            taxonomy,
            rules=StubClassifier(
                ClassifierLayer.L2_RULES,
                ClassifierVerdict(intent_id="coding", confidence=0.95),
            ),
            embedding=embedding,
            semantic_cache=SemanticIntentCache(
                policy=SemanticCachePolicy(similarity_threshold=0.5, confidence_threshold=0.5)
            ),
        )

        await cascade.classify(SCOPE, ClassificationRequest(text="write me a parser"))
        decision = await cascade.classify(
            SCOPE, ClassificationRequest(text="write me a parser please")
        )

        assert decision.layer is ClassifierLayer.L1_SEMANTIC_CACHE
        assert decision.classification.cache_hit is True

    async def test_a_semantic_cache_without_an_embedder_is_refused(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        with pytest.raises(ValueError, match="needs an embedding classifier"):
            cascade_with(taxonomy, semantic_cache=SemanticIntentCache())

    async def test_a_different_conversation_state_does_not_reuse_the_answer(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        rules = StubClassifier(
            ClassifierLayer.L2_RULES, ClassifierVerdict(intent_id="coding", confidence=0.95)
        )
        cascade = cascade_with(taxonomy, rules=rules)

        await cascade.classify(SCOPE, ClassificationRequest(text="carry on"))
        decision = await cascade.classify(
            SCOPE,
            ClassificationRequest(text="carry on", conversation_state_signature="turn-7"),
        )

        assert decision.classification.cache_hit is False


class TestTracing:
    async def test_every_layer_that_ran_is_recorded(self, taxonomy: IntentTaxonomy) -> None:
        cascade = cascade_with(
            taxonomy,
            rules=StubClassifier(
                ClassifierLayer.L2_RULES,
                ClassifierVerdict(intent_id="coding", confidence=0.2),
            ),
            structured=StubClassifier(
                ClassifierLayer.L4_STRUCTURED_LLM,
                ClassifierVerdict(intent_id="writing", confidence=0.9),
            ),
        )

        decision = await cascade.classify(SCOPE, ClassificationRequest(text="anything"))

        layers = [attempt.layer for attempt in decision.attempts]
        assert layers == [
            ClassifierLayer.L0_EXACT_CACHE,
            ClassifierLayer.L2_RULES,
            ClassifierLayer.L4_STRUCTURED_LLM,
        ]
        assert [attempt.accepted for attempt in decision.attempts] == [False, False, True]

    async def test_the_trace_carries_each_layer_threshold(self, taxonomy: IntentTaxonomy) -> None:
        cascade = cascade_with(
            taxonomy,
            rules=StubClassifier(
                ClassifierLayer.L2_RULES,
                ClassifierVerdict(intent_id="coding", confidence=0.2),
            ),
        )

        decision = await cascade.classify(SCOPE, ClassificationRequest(text="anything"))
        rules_attempt = next(a for a in decision.attempts if a.layer is ClassifierLayer.L2_RULES)
        assert rules_attempt.threshold == CascadeThresholds().rules

    async def test_trace_attributes_are_bounded_and_prompt_free(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        cascade = cascade_with(
            taxonomy,
            rules=StubClassifier(
                ClassifierLayer.L2_RULES,
                ClassifierVerdict(intent_id="coding", confidence=0.9),
            ),
        )

        decision = await cascade.classify(
            SCOPE, ClassificationRequest(text="secret customer prompt")
        )
        attributes = decision.trace_attributes()

        assert "secret customer prompt" not in str(attributes)
        assert attributes["intent.layers_tried"] == 2
        assert attributes["intent.id"] == "coding"

    async def test_cost_accumulates_across_every_layer_that_ran(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        cascade = cascade_with(
            taxonomy,
            structured=StubClassifier(
                ClassifierLayer.L4_STRUCTURED_LLM,
                ClassifierVerdict(intent_id=None, confidence=0.0, cost_usd=0.01),
            ),
            escalation=StubClassifier(
                ClassifierLayer.L5_ESCALATION,
                ClassifierVerdict(intent_id="coding", confidence=0.9, cost_usd=0.05),
            ),
        )

        decision = await cascade.classify(SCOPE, ClassificationRequest(text="anything"))
        assert decision.total_cost_usd == pytest.approx(0.06)


class TestLearningLoopCandidates:
    async def test_an_abstention_is_offered_as_a_candidate(self, taxonomy: IntentTaxonomy) -> None:
        buffer = CandidateBuffer()
        cascade = cascade_with(
            taxonomy,
            rules=StubClassifier(
                ClassifierLayer.L2_RULES,
                ClassifierVerdict(intent_id="coding", confidence=0.1),
            ),
            candidate_sink=buffer,
        )

        await cascade.classify(SCOPE, ClassificationRequest(text="mystery prompt"))

        assert [c.reason for c in buffer.items] == ["abstained"]
        assert buffer.items[0].tenant_id == "acme"

    async def test_disagreement_between_layers_is_offered_as_a_candidate(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        buffer = CandidateBuffer()
        cascade = cascade_with(
            taxonomy,
            rules=StubClassifier(
                ClassifierLayer.L2_RULES,
                ClassifierVerdict(intent_id="coding", confidence=0.3),
            ),
            structured=StubClassifier(
                ClassifierLayer.L4_STRUCTURED_LLM,
                ClassifierVerdict(intent_id="writing", confidence=0.9),
            ),
            candidate_sink=buffer,
        )

        await cascade.classify(SCOPE, ClassificationRequest(text="borderline prompt"))

        assert [c.reason for c in buffer.items] == ["layer_disagreement"]

    async def test_a_confident_agreed_classification_is_not_a_candidate(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        buffer = CandidateBuffer()
        cascade = cascade_with(
            taxonomy,
            rules=StubClassifier(
                ClassifierLayer.L2_RULES,
                ClassifierVerdict(intent_id="coding", confidence=0.99),
            ),
            candidate_sink=buffer,
        )

        await cascade.classify(SCOPE, ClassificationRequest(text="write a parser"))
        assert buffer.items == []

    async def test_the_candidate_buffer_is_bounded_and_says_what_it_dropped(
        self, taxonomy: IntentTaxonomy
    ) -> None:
        buffer = CandidateBuffer(capacity=2)
        cascade = cascade_with(
            taxonomy,
            rules=StubClassifier(
                ClassifierLayer.L2_RULES,
                ClassifierVerdict(intent_id="coding", confidence=0.1),
            ),
            candidate_sink=buffer,
        )

        for index in range(5):
            await cascade.classify(SCOPE, ClassificationRequest(text=f"mystery {index}"))

        assert len(buffer.items) == 2
        assert buffer.dropped == 3
        assert buffer.drain() and buffer.items == []


class TestMetrics:
    async def test_the_answering_layer_is_counted(self, taxonomy: IntentTaxonomy) -> None:
        cascade = cascade_with(
            taxonomy,
            rules=StubClassifier(
                ClassifierLayer.L2_RULES,
                ClassifierVerdict(intent_id="coding", confidence=0.9),
            ),
        )
        request = ClassificationRequest(text="write a parser")

        await cascade.classify(SCOPE, request)
        await cascade.classify(SCOPE, request)

        snapshot = cascade.metrics.snapshot()
        assert snapshot["by_layer"]["l2_rules"] == 1
        assert snapshot["by_layer"]["l0_exact_cache"] == 1
        assert snapshot["cache_hits"]["exact"] == 1
        assert snapshot["classifications"] == 2

    async def test_abstentions_are_counted(self, taxonomy: IntentTaxonomy) -> None:
        cascade = cascade_with(taxonomy)

        await cascade.classify(SCOPE, ClassificationRequest(text="anything"))
        assert cascade.metrics.abstention_rate == 1.0

    async def test_ambiguity_is_counted(self, taxonomy: IntentTaxonomy) -> None:
        cascade = cascade_with(
            taxonomy,
            rules=StubClassifier(
                ClassifierLayer.L2_RULES,
                ClassifierVerdict(
                    intent_id="coding",
                    confidence=0.9,
                    alternatives=(IntentAlternative("writing", 0.88),),
                ),
            ),
        )

        await cascade.classify(SCOPE, ClassificationRequest(text="anything"))
        assert cascade.metrics.snapshot()["ambiguous"] == 1

    async def test_metric_labels_stay_bounded(self, taxonomy: IntentTaxonomy) -> None:
        snapshot = cascade_with(taxonomy).metrics.snapshot()

        assert set(snapshot["by_layer"]) <= {layer.value for layer in ClassifierLayer}
        assert "tenant" not in str(snapshot)


class TestAssembledCascade:
    async def test_the_offline_cascade_needs_no_provider(self) -> None:
        from llm_fabric.intent.bootstrap import bootstrap_taxonomy

        cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())

        decision = await cascade.classify(
            SCOPE, ClassificationRequest(text="Summarise this article in three bullets")
        )
        assert decision.classification.intent_id == "summarization"
        assert decision.total_cost_usd == 0.0

    async def test_the_real_rules_layer_drives_the_real_taxonomy(self) -> None:
        from llm_fabric.intent.bootstrap import bootstrap_taxonomy

        taxonomy = bootstrap_taxonomy()
        cascade = cascade_with(taxonomy, rules=DeterministicClassifier())

        decision = await cascade.classify(
            SCOPE, ClassificationRequest(text="Translate this paragraph into Japanese")
        )
        assert decision.classification.intent_id == "translation"
        assert decision.classification.required_capabilities >= {"multilingual"}
