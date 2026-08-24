"""Durable, append-only storage for published taxonomy versions and classifications.

Published taxonomy versions are immutable. A put for an existing version is
refused rather than overwritten. Classification records store a prompt *hash*,
never the prompt itself, unless a caller explicitly opts in — production traffic
must not become a training corpus by accident.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from llm_fabric.errors import InvalidRequestError
from llm_fabric.intent.schema import IntentClassification
from llm_fabric.intent.taxonomy import IntentTaxonomy, TaxonomyRegistry
from llm_fabric.tenancy.scope import TenantScope
from llm_fabric.tenancy.store import IsolationAudit, TenantScopedStore

#: Global published taxonomies are not tenant content. They live under a
#: reserved scope so RLS still has an owner, and no real tenant can read or
#: write them through the ordinary tenant path.
FABRIC_SCOPE = TenantScope(tenant_id="_fabric", user_id="system")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:20]}"


@dataclass(frozen=True, slots=True)
class TaxonomySnapshot:
    tenant_id: str
    taxonomy_version: str
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class IntentClassificationRecord:
    """What was decided, without the prompt that produced it."""

    tenant_id: str
    request_id: str
    intent_id: str
    confidence: float
    layer: str
    taxonomy_version: str
    classifier_version: str
    record_id: str = field(default_factory=lambda: _new_id("icls"))
    abstain: bool = False
    cache_source: str | None = None
    latency_ms: float = 0.0
    prompt_hash: str = ""
    embedding_model_version: str | None = None
    prompt_version: str | None = None
    policy_version: str = "v1"
    created_at: float = field(default_factory=time.time)


class PublishedTaxonomyStore:
    """Append-only snapshots of published taxonomy versions."""

    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        store: TenantScopedStore[TaxonomySnapshot] | None = None,
    ) -> None:
        self._store = store or TenantScopedStore(
            "intent_taxonomy", max_records_per_tenant=256, audit=audit
        )

    def publish(self, taxonomy: IntentTaxonomy) -> TaxonomySnapshot:
        existing = self._store.get(FABRIC_SCOPE, taxonomy.version)
        if existing is not None:
            raise InvalidRequestError(
                f"taxonomy version '{taxonomy.version}' is published and immutable"
            )
        snapshot = TaxonomySnapshot(
            tenant_id=FABRIC_SCOPE.tenant_id,
            taxonomy_version=taxonomy.version,
            payload=taxonomy.to_dict(),
        )
        return self._store.put(FABRIC_SCOPE, taxonomy.version, snapshot)

    def get(self, version: str) -> TaxonomySnapshot | None:
        return self._store.get(FABRIC_SCOPE, version)

    def versions(self) -> tuple[str, ...]:
        return tuple(item.taxonomy_version for item in self._store.list(FABRIC_SCOPE, limit=256))


class IntentClassificationRepository:
    """Tenant-scoped classification records. Isolation is the store's job."""

    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 50_000,
        store: TenantScopedStore[IntentClassificationRecord] | None = None,
    ) -> None:
        self._store = store or TenantScopedStore(
            "intent_classification", max_records_per_tenant=max_per_tenant, audit=audit
        )

    def record(
        self,
        scope: TenantScope,
        classification: IntentClassification,
        *,
        request_id: str,
        prompt_hash: str = "",
    ) -> IntentClassificationRecord:
        row = IntentClassificationRecord(
            tenant_id=scope.tenant_id,
            request_id=request_id,
            intent_id=classification.intent_id,
            confidence=classification.confidence,
            layer=classification.layer.value,
            taxonomy_version=classification.taxonomy_version,
            classifier_version=classification.classifier_version,
            abstain=classification.abstain,
            cache_source=classification.cache_source,
            latency_ms=classification.latency_ms,
            prompt_hash=prompt_hash,
            embedding_model_version=classification.embedding_model_version,
            prompt_version=classification.prompt_version,
            policy_version=classification.policy_version,
        )
        return self._store.put(scope, row.record_id, row)

    def get(self, scope: TenantScope, record_id: str) -> IntentClassificationRecord | None:
        return self._store.get(scope, record_id)

    def list(self, scope: TenantScope, *, limit: int = 100) -> list[IntentClassificationRecord]:
        return self._store.list(scope, limit=limit)


def registry_from_published(
    store: PublishedTaxonomyStore, *taxonomies: IntentTaxonomy
) -> TaxonomyRegistry:
    """In-memory registry seeded from published snapshots plus live objects."""
    registry = TaxonomyRegistry()
    for taxonomy in taxonomies:
        if store.get(taxonomy.version) is None:
            store.publish(taxonomy)
        registry.register(taxonomy)
    return registry
