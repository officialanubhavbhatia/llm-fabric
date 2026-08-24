"""Assembling a cascade from parts.

Kept out of `cascade.py` so the engine has no opinion about which classifiers
exist. Two arrangements are provided: one that needs nothing but CPU, and one
that adds the model-backed layers when a provider is available.
"""

from __future__ import annotations

from collections.abc import Callable

from llm_fabric.intent.cache import ExactIntentCache, SemanticCachePolicy, SemanticIntentCache
from llm_fabric.intent.cascade import CandidateExample, CascadeThresholds, IntentCascade
from llm_fabric.intent.classifiers.embedding import EmbeddingClassifier, PrototypeKind
from llm_fabric.intent.classifiers.rerank import LocalRerankClassifier
from llm_fabric.intent.classifiers.rules import DeterministicClassifier
from llm_fabric.intent.classifiers.structured import (
    ClassifierPricing,
    StructuredIntentClassifier,
)
from llm_fabric.intent.embeddings import EmbeddingProvider, HashingEmbedder
from llm_fabric.intent.metrics import IntentMetrics
from llm_fabric.intent.schema import ClassifierLayer
from llm_fabric.intent.taxonomy import IntentTaxonomy
from llm_fabric.serving.base import Provider
from llm_fabric.tenancy.cache import TenantScopedCache


def build_offline_cascade(
    taxonomy: IntentTaxonomy,
    cache: TenantScopedCache,
    *,
    embedder: EmbeddingProvider | None = None,
    thresholds: CascadeThresholds | None = None,
    metrics: IntentMetrics | None = None,
    semantic_policy: SemanticCachePolicy | None = None,
    candidate_sink: Callable[[CandidateExample], None] | None = None,
    prototype: PrototypeKind = PrototypeKind.EXAMPLES,
    l4_rerank: bool = False,
    hn_lambda: float | None = None,
    cx_lambda: float | None = None,
) -> IntentCascade:
    """L0 through L3 only, plus an optional local L4 rerank.

    The default embedder is the lexical `HashingEmbedder`, so L1 and L3 here
    measure word overlap rather than meaning. That is a deliberate trade for
    determinism in tests and local runs, not a claim that it classifies well.
    L5 is never attached here.
    """
    resolved_embedder = embedder or HashingEmbedder()
    embedding = EmbeddingClassifier(
        resolved_embedder,
        prototype=prototype,
        hn_lambda=0.35 if hn_lambda is None else hn_lambda,
        cx_lambda=0.0 if cx_lambda is None else cx_lambda,
    )
    rerank = (
        LocalRerankClassifier(
            resolved_embedder,
            hn_lambda=0.35 if hn_lambda is None else hn_lambda,
            cx_lambda=0.0 if cx_lambda is None else cx_lambda,
        )
        if l4_rerank
        else None
    )

    return IntentCascade(
        taxonomy=taxonomy,
        exact_cache=ExactIntentCache(cache),
        semantic_cache=SemanticIntentCache(policy=semantic_policy, cache=cache),
        rules=DeterministicClassifier(),
        embedding=embedding,
        structured=rerank,
        thresholds=thresholds,
        metrics=metrics,
        candidate_sink=candidate_sink,
    )


def build_full_cascade(
    taxonomy: IntentTaxonomy,
    cache: TenantScopedCache,
    *,
    provider: Provider,
    structured_model: str,
    escalation_model: str | None = None,
    structured_pricing: ClassifierPricing | None = None,
    escalation_pricing: ClassifierPricing | None = None,
    embedder: EmbeddingProvider | None = None,
    thresholds: CascadeThresholds | None = None,
    metrics: IntentMetrics | None = None,
    semantic_policy: SemanticCachePolicy | None = None,
    candidate_sink: Callable[[CandidateExample], None] | None = None,
) -> IntentCascade:
    """Every layer, including the model-backed ones.

    `escalation_model` should be a stronger model than `structured_model`;
    pointing both at the same model gives the cascade a second identical
    opinion, which costs twice as much and learns nothing.
    """
    resolved_embedder = embedder or HashingEmbedder()

    escalation = None
    if escalation_model is not None:
        escalation = StructuredIntentClassifier(
            provider,
            escalation_model,
            layer=ClassifierLayer.L5_ESCALATION,
            pricing=escalation_pricing,
        )

    return IntentCascade(
        taxonomy=taxonomy,
        exact_cache=ExactIntentCache(cache),
        semantic_cache=SemanticIntentCache(policy=semantic_policy, cache=cache),
        rules=DeterministicClassifier(),
        embedding=EmbeddingClassifier(resolved_embedder, prototype=PrototypeKind.EXAMPLES),
        structured=StructuredIntentClassifier(
            provider, structured_model, pricing=structured_pricing
        ),
        escalation=escalation,
        thresholds=thresholds,
        metrics=metrics,
        candidate_sink=candidate_sink,
    )
