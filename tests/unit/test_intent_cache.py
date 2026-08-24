"""The two intent caches, and the discriminators that keep them safe."""

from __future__ import annotations

import pytest

from llm_fabric.intent.cache import (
    ExactIntentCache,
    IntentCacheDiscriminators,
    SemanticCachePolicy,
    SemanticIntentCache,
    cache_report,
    normalise_prompt,
)
from llm_fabric.intent.embeddings import HashingEmbedder
from llm_fabric.intent.schema import (
    ClassificationRequest,
    Complexity,
    ContextClass,
    CostClass,
    IntentClassification,
    LatencyClass,
    Modality,
    QualityClass,
    ReasoningLevel,
    RiskClass,
)
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


BASE = IntentCacheDiscriminators(
    taxonomy_version="tax-1",
    classifier_version="clf-1",
    policy_version="v1",
    language="en",
)


def result(intent_id: str = "coding", confidence: float = 0.9) -> IntentClassification:
    return IntentClassification(
        intent_id=intent_id,
        domain=intent_id.split(".")[0],
        complexity=Complexity.MODERATE,
        reasoning_level=ReasoningLevel.LIGHT,
        modality=Modality.TEXT,
        context_class=ContextClass.SHORT,
        risk_class=RiskClass.LOW,
        latency_class=LatencyClass.INTERACTIVE,
        quality_class=QualityClass.STANDARD,
        cost_class=CostClass.LOW,
        confidence=confidence,
        classifier_version="clf-1",
        taxonomy_version="tax-1",
    )


@pytest.fixture
def scope() -> TenantScope:
    return TenantScope(tenant_id="acme", user_id="alice")


class TestDiscriminators:
    def test_every_mandated_component_is_part_of_the_key(self) -> None:
        parts = BASE.as_parts()

        for required in (
            "taxonomy_version",
            "classifier_version",
            "policy_version",
            "language",
            "conversation_state_signature",
        ):
            assert required in parts

    @pytest.mark.parametrize(
        "field",
        [
            "taxonomy_version",
            "classifier_version",
            "policy_version",
            "language",
            "conversation_state_signature",
        ],
    )
    def test_changing_any_component_changes_the_signature(self, field: str) -> None:
        changed = IntentCacheDiscriminators(**{**BASE.as_parts(), field: "different"})

        assert changed.signature != BASE.signature

    def test_it_is_built_from_the_request_and_the_versions_in_play(self) -> None:
        request = ClassificationRequest(
            text="hi", language="de", policy_version="v7", conversation_state_signature="abc"
        )

        built = IntentCacheDiscriminators.build(
            request, taxonomy_version="tax-9", classifier_version="clf-9"
        )

        assert built.language == "de"
        assert built.policy_version == "v7"
        assert built.conversation_state_signature == "abc"
        assert built.taxonomy_version == "tax-9"


class TestExactCache:
    def test_a_classification_round_trips(self, scope: TenantScope) -> None:
        cache = ExactIntentCache(TenantScopedCache())
        cache.put(scope, "Summarise this", BASE, result("summarization"))

        found = cache.get(scope, "Summarise this", BASE)
        assert found is not None
        assert found.intent_id == "summarization"

    def test_case_and_surrounding_whitespace_do_not_fragment_the_cache(
        self, scope: TenantScope
    ) -> None:
        cache = ExactIntentCache(TenantScopedCache())
        cache.put(scope, "Summarise This", BASE, result("summarization"))

        assert cache.get(scope, "  summarise this  ", BASE) is not None

    def test_internal_punctuation_is_preserved(self) -> None:
        assert normalise_prompt("Delete the user?") != normalise_prompt("Delete the user!")

    @pytest.mark.parametrize(
        "field", ["taxonomy_version", "classifier_version", "policy_version", "language"]
    )
    def test_a_changed_discriminator_misses(self, scope: TenantScope, field: str) -> None:
        cache = ExactIntentCache(TenantScopedCache())
        cache.put(scope, "Summarise this", BASE, result("summarization"))

        other = IntentCacheDiscriminators(**{**BASE.as_parts(), field: "changed"})
        assert cache.get(scope, "Summarise this", other) is None

    def test_hits_and_misses_are_counted(self, scope: TenantScope) -> None:
        cache = ExactIntentCache(TenantScopedCache())
        cache.get(scope, "miss", BASE)
        cache.put(scope, "hit", BASE, result())
        cache.get(scope, "hit", BASE)

        stats = cache.stats
        assert (stats.hits, stats.misses, stats.writes) == (1, 1, 1)
        assert stats.hit_rate == 0.5


class TestSemanticCache:
    @pytest.fixture
    def embedder(self) -> HashingEmbedder:
        return HashingEmbedder(dimensions=256)

    def test_a_near_identical_prompt_hits(
        self, scope: TenantScope, embedder: HashingEmbedder
    ) -> None:
        cache = SemanticIntentCache(
            policy=SemanticCachePolicy(similarity_threshold=0.5, confidence_threshold=0.5)
        )
        cache.admit(
            scope,
            "summarise this article",
            embedder.embed_one("summarise this article"),
            BASE,
            result("summarization"),
        )

        match = cache.lookup(scope, embedder.embed_one("summarise this article"), BASE)
        assert match is not None
        assert match.entry.classification.intent_id == "summarization"
        assert match.similarity == pytest.approx(1.0)

    def test_an_unrelated_prompt_misses(
        self, scope: TenantScope, embedder: HashingEmbedder
    ) -> None:
        cache = SemanticIntentCache(
            policy=SemanticCachePolicy(similarity_threshold=0.5, confidence_threshold=0.5)
        )
        cache.admit(
            scope,
            "summarise this article",
            embedder.embed_one("summarise this article"),
            BASE,
            result("summarization"),
        )

        assert cache.lookup(scope, embedder.embed_one("integrate x squared"), BASE) is None

    def test_similarity_alone_is_not_enough(
        self, scope: TenantScope, embedder: HashingEmbedder
    ) -> None:
        """An identical prompt still misses if the cached answer was unconfident."""
        cache = SemanticIntentCache(
            policy=SemanticCachePolicy(similarity_threshold=0.5, confidence_threshold=0.9)
        )
        vector = embedder.embed_one("summarise this")

        assert cache.admit(scope, "summarise this", vector, BASE, result(confidence=0.6)) is False
        assert cache.lookup(scope, vector, BASE) is None

    def test_an_abstention_is_never_admitted(
        self, scope: TenantScope, embedder: HashingEmbedder
    ) -> None:
        cache = SemanticIntentCache()
        abstention = IntentClassification.unknown(
            classifier_version="clf-1", taxonomy_version="tax-1", confidence=0.99
        )

        admitted = cache.admit(
            scope, "who knows", embedder.embed_one("who knows"), BASE, abstention
        )
        assert admitted is False

    def test_entries_do_not_survive_a_discriminator_change(
        self, scope: TenantScope, embedder: HashingEmbedder
    ) -> None:
        """Similarity must never be compared across taxonomy versions."""
        cache = SemanticIntentCache(
            policy=SemanticCachePolicy(similarity_threshold=0.5, confidence_threshold=0.5)
        )
        vector = embedder.embed_one("summarise this")
        cache.admit(scope, "summarise this", vector, BASE, result("summarization"))

        newer = IntentCacheDiscriminators(**{**BASE.as_parts(), "taxonomy_version": "tax-2"})
        assert cache.lookup(scope, vector, newer) is None

    def test_entries_expire(self, scope: TenantScope, embedder: HashingEmbedder) -> None:
        clock = FakeClock()
        cache = SemanticIntentCache(
            policy=SemanticCachePolicy(
                similarity_threshold=0.5, confidence_threshold=0.5, ttl_seconds=10.0
            ),
            clock=clock,
        )
        vector = embedder.embed_one("summarise this")
        cache.admit(scope, "summarise this", vector, BASE, result("summarization"))

        clock.advance(9.0)
        assert cache.lookup(scope, vector, BASE) is not None

        clock.advance(2.0)
        assert cache.lookup(scope, vector, BASE) is None
        assert cache.stats.expired == 1

    def test_the_index_is_bounded_per_signature(
        self, scope: TenantScope, embedder: HashingEmbedder
    ) -> None:
        cache = SemanticIntentCache(
            policy=SemanticCachePolicy(
                similarity_threshold=0.5, confidence_threshold=0.5, capacity_per_signature=5
            )
        )
        for index in range(50):
            prompt = f"summarise document number {index}"
            cache.admit(scope, prompt, embedder.embed_one(prompt), BASE, result("summarization"))

        assert cache.size(scope, BASE) == 5

    def test_the_false_hit_rate_is_unknown_until_hits_are_reviewed(self) -> None:
        cache = SemanticIntentCache()

        assert cache.stats.false_hit_rate is None

        cache.report_false_hit(reviewed=10, wrong=3)
        assert cache.stats.false_hit_rate == pytest.approx(0.3)

    def test_more_wrong_than_reviewed_is_refused(self) -> None:
        with pytest.raises(ValueError, match="more wrong"):
            SemanticIntentCache().report_false_hit(reviewed=1, wrong=2)


def test_the_report_keeps_the_two_caches_separate(scope: TenantScope) -> None:
    exact = ExactIntentCache(TenantScopedCache())
    semantic = SemanticIntentCache()
    exact.get(scope, "anything", BASE)

    report = cache_report(exact, semantic)

    assert set(report) == {"exact", "semantic"}
    assert report["exact"]["misses"] == 1
    assert "similarity_threshold" in report["semantic"]
