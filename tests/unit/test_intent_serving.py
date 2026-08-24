"""Serving-path IntentOS coverage: every provider call carries an IntentResult."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.intent.bootstrap import bootstrap_taxonomy
from llm_fabric.intent.cache import (
    IntentCacheDiscriminators,
    SemanticIntentCache,
)
from llm_fabric.intent.cascade import IntentCascade
from llm_fabric.intent.embeddings import HashingEmbedder
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
    ServingClassificationState,
)
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.observability.usage_event import (
    UsageOperation,
    provider_invocations_without_intent,
)
from llm_fabric.serving.adapters.mock import MockProvider
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope


def _known(intent_id: str = "translation", *, confidence: float = 0.95) -> IntentClassification:
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


def _chat(client: TestClient, text: str = "translate this sentence into French"):
    return client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": text}]},
    )


def test_default_test_chat_still_carries_a_safe_fallback_intent(client: TestClient) -> None:
    response = _chat(client)
    assert response.status_code == 200
    assert response.headers["x-fabric-intent"] == "unknown"
    assert response.headers["x-fabric-intent-state"] == "safe_fallback"
    assert response.headers["x-fabric-intent-result-id"]
    assert response.headers["x-fabric-taxonomy-version"]
    assert response.headers["x-fabric-classifier-version"]


def test_provider_invocations_without_intent_is_zero(registry, settings, providers) -> None:
    meter = InMemoryMeter()
    app = create_app(
        settings=settings,
        registry=registry,
        provider_overrides=providers,
        meter=meter,
    )
    with TestClient(app) as client:
        assert _chat(client).status_code == 200
        assert _chat(client, "summarise this paragraph").status_code == 200
        streamed = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert streamed.status_code == 200
        list(streamed.iter_lines())
    events = meter.recent_events(limit=200)
    user_events = [
        event for event in events if event.operation == UsageOperation.USER_RESPONSE.value
    ]
    assert user_events
    assert provider_invocations_without_intent(events) == 0
    assert all(event.intent_result_id for event in user_events)
    assert all(event.taxonomy_version for event in user_events)
    assert all(event.classifier_version for event in user_events)


def test_enabled_classification_routes_on_the_intent_result(registry) -> None:
    meter = InMemoryMeter()
    app = create_app(
        settings=Settings(
            environment="test",
            api_keys=[],
            intent_classification_enabled=True,
        ),
        registry=registry,
        provider_overrides={"mock": MockProvider()},
        meter=meter,
    )
    with TestClient(app) as client:
        response = _chat(client)
    assert response.status_code == 200
    assert response.headers["x-fabric-intent"] == "translation"
    assert response.headers["x-fabric-intent-state"] == "known"
    assert response.headers["x-fabric-intent-result-id"]
    events = meter.recent_events(limit=50)
    assert provider_invocations_without_intent(events) == 0


@pytest.mark.asyncio
async def test_cascade_exception_degrades_to_safe_fallback() -> None:
    cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())

    async def boom(_scope: TenantScope, _request: ClassificationRequest):
        raise RuntimeError("redis unavailable")

    cascade._classify_unchecked = boom  # type: ignore[method-assign]
    decision = await cascade.classify(
        TenantScope(tenant_id="acme"),
        ClassificationRequest(text="debug this python traceback"),
    )
    assert decision.classification.serving_state is ServingClassificationState.SAFE_FALLBACK
    assert decision.classification.intent_id == "unknown"


@pytest.mark.asyncio
async def test_layer_failure_continues_the_cascade() -> None:
    class Exploding:
        layer = ClassifierLayer.L4_STRUCTURED_LLM
        version = "boom"

        async def classify(self, request, taxonomy):
            raise TimeoutError("l4 timed out")

    cascade = IntentCascade(
        taxonomy=bootstrap_taxonomy(),
        exact_cache=build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache()).exact_cache,
        rules=build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())._rules,
        embedding=None,
        structured=Exploding(),  # type: ignore[arg-type]
        semantic_cache=None,
    )
    decision = await cascade.classify(
        TenantScope(tenant_id="acme"),
        ClassificationRequest(text="translate this sentence into French"),
    )
    assert decision.classification.serving_state is not ServingClassificationState.SAFE_FALLBACK
    assert decision.classification.intent_id in {"translation", "unknown"}


@pytest.mark.asyncio
async def test_embedding_failure_continues_to_rules() -> None:
    class ExplodingEmbed:
        layer = ClassifierLayer.L3_EMBEDDING
        version = "boom"

        async def embed_prompt(self, text: str):
            del text
            raise RuntimeError("embedding service down")

        async def classify(self, request, taxonomy):
            raise AssertionError("classify must not run after embed_prompt failed")

        async def prepare(self, taxonomy) -> None:
            del taxonomy

    base = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())
    cascade = IntentCascade(
        taxonomy=bootstrap_taxonomy(),
        exact_cache=base.exact_cache,
        rules=base._rules,
        embedding=ExplodingEmbed(),  # type: ignore[arg-type]
        semantic_cache=None,
        structured=None,
        escalation=None,
    )
    decision = await cascade.classify(
        TenantScope(tenant_id="acme"),
        ClassificationRequest(text="translate this sentence into French"),
    )
    assert decision.classification.serving_state is not ServingClassificationState.SAFE_FALLBACK
    assert decision.classification.intent_id in {"translation", "unknown"}


@pytest.mark.asyncio
async def test_escalation_unavailable_does_not_skip_intentos() -> None:
    cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())
    assert cascade._escalation is None
    decision = await cascade.classify(
        TenantScope(tenant_id="acme"),
        ClassificationRequest(text="translate this sentence into French"),
    )
    assert decision.classification.intent_result_id
    assert decision.classification.serving_state is not ServingClassificationState.SAFE_FALLBACK


@pytest.mark.asyncio
async def test_distributed_semantic_cache_is_tenant_isolated() -> None:
    shared = TenantScopedCache()
    worker_a = SemanticIntentCache(cache=shared)
    worker_b = SemanticIntentCache(cache=shared)
    embedder = HashingEmbedder()
    vector = (await embedder.embed(["translate this sentence into French"]))[0]
    discriminators = IntentCacheDiscriminators(
        taxonomy_version="tax-1",
        classifier_version="clf-1",
        policy_version="v1",
        language="en",
    )
    acme = TenantScope(tenant_id="acme")
    globex = TenantScope(tenant_id="globex")
    known = _known()
    admitted = worker_a.admit(
        acme, "translate this sentence into French", vector, discriminators, known
    )
    assert admitted
    hit = worker_b.lookup(acme, vector, discriminators)
    assert hit is not None
    assert hit.entry.classification.intent_id == "translation"
    assert worker_b.lookup(globex, vector, discriminators) is None


@pytest.mark.asyncio
async def test_taxonomy_and_classifier_versions_do_not_cross_cache() -> None:
    cache = SemanticIntentCache(cache=TenantScopedCache())
    embedder = HashingEmbedder()
    vector = (await embedder.embed(["hello"]))[0]
    scope = TenantScope(tenant_id="acme")
    known = _known("coding")
    first = IntentCacheDiscriminators(
        taxonomy_version="t1",
        classifier_version="a",
        policy_version="v1",
        language="en",
    )
    second = IntentCacheDiscriminators(
        taxonomy_version="t2",
        classifier_version="a",
        policy_version="v1",
        language="en",
    )
    assert cache.admit(scope, "hello", vector, first, known)
    assert cache.lookup(scope, vector, first) is not None
    assert cache.lookup(scope, vector, second) is None
