"""Adversarial: one tenant's intent classifications must be invisible to another.

An intent cache is a side channel. If tenant B can provoke a hit on an entry
tenant A populated, B learns that A asked something similar — and B's request is
answered using A's classification, which may carry A's routing policy and
capability requirements. The semantic cache makes this worse, because B does not
even need A's exact prompt.

The same two lines of defence as the rest of the fabric apply: keys are
namespaced by tenant, and stored records carry their owner so a namespacing bug
is caught rather than exploited. These tests attack both.

A failure here fails CI.
"""

from __future__ import annotations

import pytest

from llm_fabric.intent.cache import (
    ExactIntentCache,
    IntentCacheDiscriminators,
    SemanticCachePolicy,
    SemanticIntentCache,
)
from llm_fabric.intent.embeddings import CachingEmbedder, HashingEmbedder
from llm_fabric.intent.factory import build_offline_cascade
from llm_fabric.intent.schema import (
    ClassificationRequest,
    ClassifierLayer,
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
from llm_fabric.intent.taxonomy import IntentNode, IntentTaxonomy
from llm_fabric.tenancy.cache import CacheNamespace, TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope

pytestmark = pytest.mark.isolation

DISCRIMINATORS = IntentCacheDiscriminators(
    taxonomy_version="tax-1",
    classifier_version="clf-1",
    policy_version="v1",
    language="en",
)

VICTIM_PROMPT = "Summarise the acquisition memo for Project Halcyon"


def secret_classification() -> IntentClassification:
    """A classification carrying tenant-specific routing consequences."""
    return IntentClassification(
        intent_id="summarization",
        domain="summarization",
        complexity=Complexity.COMPLEX,
        reasoning_level=ReasoningLevel.DEEP,
        modality=Modality.TEXT,
        context_class=ContextClass.VERY_LONG,
        risk_class=RiskClass.CRITICAL,
        latency_class=LatencyClass.BATCH,
        quality_class=QualityClass.MAXIMUM,
        cost_class=CostClass.PREMIUM,
        confidence=0.97,
        classifier_version="clf-1",
        taxonomy_version="tax-1",
        required_capabilities=frozenset({"victim-only-capability"}),
    )


class TestExactIntentCache:
    def test_an_attacker_replaying_the_exact_prompt_misses(
        self, victim: TenantScope, attacker: TenantScope
    ) -> None:
        cache = ExactIntentCache(TenantScopedCache())
        cache.put(victim, VICTIM_PROMPT, DISCRIMINATORS, secret_classification())

        assert cache.get(attacker, VICTIM_PROMPT, DISCRIMINATORS) is None

    def test_sharing_a_user_id_does_not_help_the_attacker(
        self, victim: TenantScope, attacker_same_user_id: TenantScope
    ) -> None:
        cache = ExactIntentCache(TenantScopedCache())
        cache.put(victim, VICTIM_PROMPT, DISCRIMINATORS, secret_classification())

        assert cache.get(attacker_same_user_id, VICTIM_PROMPT, DISCRIMINATORS) is None

    def test_an_attacker_cannot_invalidate_a_victim_entry(
        self, victim: TenantScope, attacker: TenantScope
    ) -> None:
        cache = ExactIntentCache(TenantScopedCache())
        cache.put(victim, VICTIM_PROMPT, DISCRIMINATORS, secret_classification())

        assert cache.invalidate(attacker, VICTIM_PROMPT, DISCRIMINATORS) is False
        assert cache.invalidate_tenant(attacker) == 0
        assert cache.get(victim, VICTIM_PROMPT, DISCRIMINATORS) is not None

    def test_an_attacker_cannot_overwrite_a_victim_entry(
        self, victim: TenantScope, attacker: TenantScope
    ) -> None:
        cache = ExactIntentCache(TenantScopedCache())
        cache.put(victim, VICTIM_PROMPT, DISCRIMINATORS, secret_classification())

        poisoned = IntentClassification(
            intent_id="general_conversation",
            domain="general_conversation",
            complexity=Complexity.TRIVIAL,
            reasoning_level=ReasoningLevel.NONE,
            modality=Modality.TEXT,
            context_class=ContextClass.TINY,
            risk_class=RiskClass.LOW,
            latency_class=LatencyClass.REALTIME,
            quality_class=QualityClass.DRAFT,
            cost_class=CostClass.MINIMAL,
            confidence=0.99,
            classifier_version="clf-1",
            taxonomy_version="tax-1",
        )
        cache.put(attacker, VICTIM_PROMPT, DISCRIMINATORS, poisoned)

        survived = cache.get(victim, VICTIM_PROMPT, DISCRIMINATORS)
        assert survived is not None
        assert survived.intent_id == "summarization"
        assert survived.risk_class is RiskClass.CRITICAL


class TestSemanticIntentCache:
    @pytest.fixture
    def permissive(self) -> SemanticIntentCache:
        """Thresholds low enough that only isolation can prevent a hit."""
        return SemanticIntentCache(
            policy=SemanticCachePolicy(similarity_threshold=0.0, confidence_threshold=0.0)
        )

    def test_an_identical_vector_from_another_tenant_misses(
        self,
        victim: TenantScope,
        attacker: TenantScope,
        permissive: SemanticIntentCache,
    ) -> None:
        embedder = HashingEmbedder(dimensions=256)
        vector = embedder.embed_one(VICTIM_PROMPT)
        permissive.admit(victim, VICTIM_PROMPT, vector, DISCRIMINATORS, secret_classification())

        assert permissive.lookup(attacker, vector, DISCRIMINATORS) is None
        assert permissive.lookup(victim, vector, DISCRIMINATORS) is not None

    def test_a_probing_attacker_learns_nothing_from_near_neighbours(
        self,
        victim: TenantScope,
        attacker: TenantScope,
        permissive: SemanticIntentCache,
    ) -> None:
        """The realistic attack: guess at the topic and watch for a hit."""
        embedder = HashingEmbedder(dimensions=256)
        permissive.admit(
            victim,
            VICTIM_PROMPT,
            embedder.embed_one(VICTIM_PROMPT),
            DISCRIMINATORS,
            secret_classification(),
        )

        for probe in (
            "Summarise the acquisition memo",
            "acquisition memo Project Halcyon",
            "Summarise the memo for Project Halcyon",
        ):
            assert permissive.lookup(attacker, embedder.embed_one(probe), DISCRIMINATORS) is None

    def test_clearing_one_tenant_leaves_the_other_intact(
        self,
        victim: TenantScope,
        attacker: TenantScope,
        permissive: SemanticIntentCache,
    ) -> None:
        embedder = HashingEmbedder(dimensions=256)
        vector = embedder.embed_one(VICTIM_PROMPT)
        permissive.admit(victim, VICTIM_PROMPT, vector, DISCRIMINATORS, secret_classification())

        assert permissive.invalidate_tenant(attacker) == 0
        assert permissive.lookup(victim, vector, DISCRIMINATORS) is not None


class TestEmbeddingCache:
    async def test_two_tenants_embedding_the_same_text_do_not_share_an_entry(
        self, victim: TenantScope, attacker: TenantScope
    ) -> None:
        cache = TenantScopedCache()
        inner = HashingEmbedder(dimensions=128)

        await CachingEmbedder(inner, cache, victim).embed([VICTIM_PROMPT])

        attacker_view = cache.get(
            attacker,
            CacheNamespace.EMBEDDING,
            {"model": inner.model_id, "text": VICTIM_PROMPT},
        )
        assert attacker_view is None


class TestEndToEndCascade:
    """The whole engine, with two tenants sharing one process."""

    @pytest.fixture
    def taxonomy(self) -> IntentTaxonomy:
        return IntentTaxonomy(
            "tax-1",
            [
                IntentNode(
                    intent_id="summarization",
                    name="Summarisation",
                    description="condense",
                    examples=("summarise this document",),
                )
            ],
        )

    async def test_a_victim_classification_never_answers_an_attacker(
        self, taxonomy: IntentTaxonomy, victim: TenantScope, attacker: TenantScope
    ) -> None:
        engine = build_offline_cascade(
            taxonomy,
            TenantScopedCache(),
            semantic_policy=SemanticCachePolicy(similarity_threshold=0.0, confidence_threshold=0.0),
        )
        request = ClassificationRequest(text="summarise this document")

        first = await engine.classify(victim, request)
        replay = await engine.classify(attacker, request)

        assert first.classification.cache_hit is False
        assert replay.classification.cache_hit is False, (
            "the attacker was served the victim's cached classification"
        )
        assert replay.layer not in (
            ClassifierLayer.L0_EXACT_CACHE,
            ClassifierLayer.L1_SEMANTIC_CACHE,
        )

    async def test_each_tenant_warms_its_own_cache_independently(
        self, taxonomy: IntentTaxonomy, victim: TenantScope, attacker: TenantScope
    ) -> None:
        engine = build_offline_cascade(taxonomy, TenantScopedCache())
        request = ClassificationRequest(text="summarise this document")

        await engine.classify(victim, request)
        await engine.classify(attacker, request)

        assert (await engine.classify(victim, request)).classification.cache_hit is True
        assert (await engine.classify(attacker, request)).classification.cache_hit is True

    async def test_candidate_examples_carry_their_owning_tenant(
        self, taxonomy: IntentTaxonomy, victim: TenantScope, attacker: TenantScope
    ) -> None:
        """Candidates feed the learning loop, so they must not be pooled blindly."""
        from llm_fabric.intent.cascade import CandidateBuffer

        buffer = CandidateBuffer()
        engine = build_offline_cascade(taxonomy, TenantScopedCache(), candidate_sink=buffer)

        await engine.classify(victim, ClassificationRequest(text="zzz qqq vvv"))
        await engine.classify(attacker, ClassificationRequest(text="xxx yyy www"))

        owners = {candidate.tenant_id for candidate in buffer.items}
        assert owners == {victim.tenant_id, attacker.tenant_id}
        for candidate in buffer.items:
            assert candidate.tenant_id in (victim.tenant_id, attacker.tenant_id)


class TestClassificationRecordsAndExamples:
    def test_classification_records_are_tenant_scoped(
        self, victim: TenantScope, attacker: TenantScope
    ) -> None:
        from llm_fabric.intent.persist import IntentClassificationRepository

        repo = IntentClassificationRepository()
        row = repo.record(victim, secret_classification(), request_id="r1", prompt_hash="deadbeef")
        assert repo.get(attacker, row.record_id) is None
        assert repo.list(attacker) == []
        assert repo.get(victim, row.record_id) is not None

    def test_hard_examples_are_tenant_scoped(
        self, victim: TenantScope, attacker: TenantScope
    ) -> None:
        from llm_fabric.intent.cascade import CandidateExample, IntentDecision, LayerAttempt
        from llm_fabric.intent.learning import HardExampleStore
        from llm_fabric.intent.schema import ClassifierLayer

        store = HardExampleStore()
        decision = IntentDecision(
            classification=secret_classification(),
            attempts=(
                LayerAttempt(
                    layer=ClassifierLayer.L2_RULES,
                    intent_id="summarization",
                    confidence=0.4,
                    threshold=0.7,
                    accepted=False,
                    latency_ms=1.0,
                ),
            ),
        )
        victim_row = store.ingest(
            victim,
            CandidateExample(
                tenant_id=victim.tenant_id,
                text=VICTIM_PROMPT,
                reason="abstained",
                decision=decision,
            ),
        )
        assert victim_row is not None
        assert store.get(attacker, victim_row.example_id) is None
        assert store.list(attacker) == []
