"""Tenant-aware caching.

The constitution forbids calling every cache "the cache". Each namespace here is
a distinct cache with its own TTL, its own bound, its own invalidation and its
own counters. They share only the isolation guarantee, which they inherit by
being built on `TenantScopedStore`.

Cache keys are the classic cross-tenant leak: two tenants send the same prompt,
one gets the other's answer. So the tenant is part of both the bucket and the
key fingerprint, and a key is assembled from named discriminators rather than
string concatenation, which is where "oh, we forgot to include the policy
version" bugs come from.

The provider prefix/KV cache named in the constitution is deliberately absent:
it lives inside vLLM or Ollama, the fabric does not control its keys, and
modelling it here would imply a guarantee this process cannot make.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from llm_fabric.tenancy.scope import TenantScope
from llm_fabric.tenancy.store import IsolationAudit, TenantScopedStore


class CacheNamespace(StrEnum):
    """Logically distinct caches. Never collapse these into one."""

    EXACT_RESPONSE = "exact_response"
    SEMANTIC_RESPONSE = "semantic_response"
    INTENT = "intent"
    SEMANTIC_INTENT = "semantic_intent"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    PROMPT = "prompt"
    CONTEXT_ARTIFACT = "context_artifact"


@dataclass(frozen=True, slots=True)
class CachePolicy:
    ttl_seconds: float
    max_entries_per_tenant: int = 1_000
    enabled: bool = True


DEFAULT_POLICIES: Mapping[CacheNamespace, CachePolicy] = {
    # Exact hits are safe for longer because the input matched byte for byte.
    CacheNamespace.EXACT_RESPONSE: CachePolicy(ttl_seconds=300.0, max_entries_per_tenant=2_000),
    # Semantic hits are approximate, so they expire sooner by design.
    CacheNamespace.SEMANTIC_RESPONSE: CachePolicy(ttl_seconds=120.0, max_entries_per_tenant=1_000),
    CacheNamespace.INTENT: CachePolicy(ttl_seconds=600.0, max_entries_per_tenant=5_000),
    CacheNamespace.SEMANTIC_INTENT: CachePolicy(ttl_seconds=120.0, max_entries_per_tenant=2_000),
    CacheNamespace.EMBEDDING: CachePolicy(ttl_seconds=3_600.0, max_entries_per_tenant=5_000),
    CacheNamespace.RETRIEVAL: CachePolicy(ttl_seconds=180.0, max_entries_per_tenant=1_000),
    CacheNamespace.PROMPT: CachePolicy(ttl_seconds=3_600.0, max_entries_per_tenant=500),
    CacheNamespace.CONTEXT_ARTIFACT: CachePolicy(ttl_seconds=300.0, max_entries_per_tenant=1_000),
}


@dataclass(frozen=True, slots=True)
class CacheKey:
    namespace: CacheNamespace
    tenant_id: str
    fingerprint: str

    @classmethod
    def build(
        cls,
        namespace: CacheNamespace,
        scope: TenantScope,
        parts: Mapping[str, Any],
    ) -> CacheKey:
        """Derive a key from named discriminators.

        The tenant is mixed into the digest as well as the bucket, so even an
        identical prompt from two tenants produces two different fingerprints.
        """
        material = {"__tenant__": scope.tenant_id, **{k: parts[k] for k in sorted(parts)}}
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return cls(namespace=namespace, tenant_id=scope.tenant_id, fingerprint=digest)

    def as_storage_key(self) -> str:
        return f"{self.namespace.value}/{self.tenant_id}/{self.fingerprint}"


@dataclass(slots=True)
class _CacheEntry:
    tenant_id: str
    value: Any
    expires_at: float

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    expired: int = 0
    writes: int = 0
    invalidations: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        """`None` rather than 0.0 when nothing has been looked up yet.

        A hit rate of zero and "no data" are different facts, and reporting the
        former for the latter would be a fabricated metric.
        """
        return self.hits / self.lookups if self.lookups else None


class DistributedCache(Protocol):
    """Optional Redis (or compatible) backend. Tenant is part of every key."""

    def get(self, tenant_id: str, namespace: str, fingerprint: str) -> Any | None: ...

    def put(
        self,
        tenant_id: str,
        namespace: str,
        fingerprint: str,
        value: Any,
        *,
        ttl_seconds: float,
    ) -> None: ...

    def invalidate(self, tenant_id: str, namespace: str, fingerprint: str) -> bool: ...


class TenantScopedCache:
    """A family of independently-configured, tenant-isolated caches."""

    def __init__(
        self,
        policies: Mapping[CacheNamespace, CachePolicy] | None = None,
        *,
        audit: IsolationAudit | None = None,
        clock: Any = time.monotonic,
        redis_cache: DistributedCache | None = None,
    ) -> None:
        self._policies = dict(DEFAULT_POLICIES)
        if policies:
            self._policies.update(policies)
        self._audit = audit or IsolationAudit()
        self._clock = clock
        self._redis = redis_cache
        self._stores: dict[CacheNamespace, TenantScopedStore[_CacheEntry]] = {
            namespace: TenantScopedStore(
                f"cache:{namespace.value}",
                max_records_per_tenant=policy.max_entries_per_tenant,
                audit=self._audit,
            )
            for namespace, policy in self._policies.items()
        }
        self._stats: dict[CacheNamespace, CacheStats] = {
            namespace: CacheStats() for namespace in self._policies
        }
        self._lock = threading.Lock()

    @property
    def audit(self) -> IsolationAudit:
        return self._audit

    def policy(self, namespace: CacheNamespace) -> CachePolicy:
        return self._policies[namespace]

    def stats(self, namespace: CacheNamespace) -> CacheStats:
        with self._lock:
            current = self._stats[namespace]
            return CacheStats(
                hits=current.hits,
                misses=current.misses,
                expired=current.expired,
                writes=current.writes,
                invalidations=current.invalidations,
            )

    def get(
        self,
        scope: TenantScope,
        namespace: CacheNamespace,
        parts: Mapping[str, Any],
    ) -> Any | None:
        policy = self._policies[namespace]
        if not policy.enabled:
            return None

        key = CacheKey.build(namespace, scope, parts)
        if self._redis is not None:
            value = self._redis.get(scope.tenant_id, namespace.value, key.fingerprint)
            if value is None:
                self._bump(namespace, "misses")
                return None
            self._bump(namespace, "hits")
            return value

        store = self._stores[namespace]
        entry = store.get(scope, key.as_storage_key())

        if entry is None:
            self._bump(namespace, "misses")
            return None
        if entry.is_expired(self._clock()):
            store.delete(scope, key.as_storage_key())
            self._bump(namespace, "expired")
            self._bump(namespace, "misses")
            return None

        self._bump(namespace, "hits")
        return entry.value

    def put(
        self,
        scope: TenantScope,
        namespace: CacheNamespace,
        parts: Mapping[str, Any],
        value: Any,
        *,
        ttl_seconds: float | None = None,
    ) -> None:
        policy = self._policies[namespace]
        if not policy.enabled:
            return

        ttl = policy.ttl_seconds if ttl_seconds is None else ttl_seconds
        key = CacheKey.build(namespace, scope, parts)
        if self._redis is not None:
            self._redis.put(
                scope.tenant_id,
                namespace.value,
                key.fingerprint,
                value,
                ttl_seconds=ttl,
            )
            self._bump(namespace, "writes")
            return

        self._stores[namespace].put(
            scope,
            key.as_storage_key(),
            _CacheEntry(
                tenant_id=scope.tenant_id,
                value=value,
                expires_at=self._clock() + ttl,
            ),
        )
        self._bump(namespace, "writes")

    def invalidate(
        self,
        scope: TenantScope,
        namespace: CacheNamespace,
        parts: Mapping[str, Any],
    ) -> bool:
        key = CacheKey.build(namespace, scope, parts)
        if self._redis is not None:
            removed = self._redis.invalidate(scope.tenant_id, namespace.value, key.fingerprint)
            if removed:
                self._bump(namespace, "invalidations")
            return removed
        removed = self._stores[namespace].delete(scope, key.as_storage_key())
        if removed:
            self._bump(namespace, "invalidations")
        return removed

    def invalidate_namespace(self, scope: TenantScope, namespace: CacheNamespace) -> int:
        removed = self._stores[namespace].clear_tenant(scope)
        if removed:
            with self._lock:
                self._stats[namespace].invalidations += removed
        return removed

    def invalidate_tenant(self, scope: TenantScope) -> int:
        return sum(self.invalidate_namespace(scope, namespace) for namespace in tuple(self._stores))

    def _bump(self, namespace: CacheNamespace, field_name: str) -> None:
        with self._lock:
            stats = self._stats[namespace]
            setattr(stats, field_name, getattr(stats, field_name) + 1)
