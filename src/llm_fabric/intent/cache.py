"""The two intent caches: L0 exact and L1 semantic.

They are separate caches with separate TTLs and separate risk profiles, and the
constitution requires them to stay that way. An exact hit is safe for as long as
the discriminators hold, because the input matched byte for byte. A semantic hit
is a *guess* that two different prompts want the same thing, and a wrong guess
silently routes a caller to the wrong intent.

That asymmetry drives every decision here:

- Semantic entries expire sooner than exact entries.
- A semantic hit must clear both a similarity threshold and a confidence
  threshold; either alone is insufficient.
- The semantic index is partitioned by an exact-match discriminator signature,
  so similarity is only ever compared *within* one taxonomy version, classifier
  version, policy version, language and conversation state. Comparing across
  them would let a stale taxonomy answer a current question.
- False hits are counted. They cannot be detected at read time — that requires
  ground truth — so the API is a reporting hook for evaluation and feedback to
  call, and the rate is honest about how many hits were actually reviewed.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from llm_fabric.intent.embeddings import Vector, cosine_similarity
from llm_fabric.intent.schema import ClassificationRequest, IntentClassification
from llm_fabric.storage.codec import decode, encode
from llm_fabric.tenancy.cache import CacheNamespace, TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope
from llm_fabric.tenancy.store import TenantScopedStore

#: Ceiling on how many semantic entries one tenant may hold per signature.
DEFAULT_SEMANTIC_CAPACITY = 2_000


@dataclass(frozen=True, slots=True)
class IntentCacheDiscriminators:
    """Everything besides the prompt that changes what the answer should be.

    The constitution names these exactly. Any one of them differing means a
    cached classification must not be reused, so they are a single value that
    travels together rather than six arguments that can be passed inconsistently.
    """

    taxonomy_version: str
    classifier_version: str
    policy_version: str
    language: str
    conversation_state_signature: str = ""

    def as_parts(self) -> dict[str, str]:
        return {
            "taxonomy_version": self.taxonomy_version,
            "classifier_version": self.classifier_version,
            "policy_version": self.policy_version,
            "language": self.language,
            "conversation_state_signature": self.conversation_state_signature,
        }

    @property
    def signature(self) -> str:
        """A stable digest of the discriminators, used to partition the index."""
        encoded = json.dumps(self.as_parts(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def build(
        cls,
        request: ClassificationRequest,
        *,
        taxonomy_version: str,
        classifier_version: str,
    ) -> IntentCacheDiscriminators:
        return cls(
            taxonomy_version=taxonomy_version,
            classifier_version=classifier_version,
            policy_version=request.policy_version,
            language=request.language,
            conversation_state_signature=request.conversation_state_signature,
        )


def normalise_prompt(text: str) -> str:
    """Fold away differences that never change the intent.

    Case and surrounding whitespace do not change what someone is asking for, so
    treating them as distinct would fragment the exact cache for no benefit.
    Internal punctuation is preserved: "delete the user?" and "delete the user!"
    are not obviously the same request, and the exact cache is the layer that is
    supposed to be certain.
    """
    return " ".join(text.split()).casefold()


@dataclass(slots=True)
class IntentCacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    expired: int = 0
    #: Hits later reported wrong by evaluation or user feedback.
    false_hits: int = 0
    #: Hits that were actually checked against ground truth.
    reviewed_hits: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.lookups if self.lookups else None

    @property
    def false_hit_rate(self) -> float | None:
        """Fraction of *reviewed* hits that were wrong.

        `None` when nothing has been reviewed. Dividing false hits by total hits
        would understate the rate by treating every unreviewed hit as correct,
        which is precisely the assumption a false-hit metric exists to test.
        """
        return self.false_hits / self.reviewed_hits if self.reviewed_hits else None

    def snapshot(self) -> IntentCacheStats:
        return IntentCacheStats(
            hits=self.hits,
            misses=self.misses,
            writes=self.writes,
            expired=self.expired,
            false_hits=self.false_hits,
            reviewed_hits=self.reviewed_hits,
        )


class ExactIntentCache:
    """L0. Byte-for-byte prompt match under identical discriminators."""

    def __init__(self, cache: TenantScopedCache) -> None:
        self._cache = cache
        self._stats = IntentCacheStats()
        self._lock = threading.Lock()

    @property
    def stats(self) -> IntentCacheStats:
        with self._lock:
            return self._stats.snapshot()

    def _parts(self, text: str, discriminators: IntentCacheDiscriminators) -> dict[str, str]:
        return {"prompt": normalise_prompt(text), **discriminators.as_parts()}

    def get(
        self,
        scope: TenantScope,
        text: str,
        discriminators: IntentCacheDiscriminators,
    ) -> IntentClassification | None:
        try:
            found = self._cache.get(scope, CacheNamespace.INTENT, self._parts(text, discriminators))
        except Exception:  # noqa: BLE001 - cache outage is a miss, not a skip of IntentOS
            found = None
        hydrated = _hydrate_classification(found)
        with self._lock:
            if hydrated is not None:
                self._stats.hits += 1
            else:
                self._stats.misses += 1
        return hydrated

    def put(
        self,
        scope: TenantScope,
        text: str,
        discriminators: IntentCacheDiscriminators,
        classification: IntentClassification,
        *,
        ttl_seconds: float | None = None,
    ) -> None:
        try:
            self._cache.put(
                scope,
                CacheNamespace.INTENT,
                self._parts(text, discriminators),
                encode(classification),
                ttl_seconds=ttl_seconds,
            )
        except Exception:  # noqa: BLE001 - a failed write must not fail classification
            return
        with self._lock:
            self._stats.writes += 1

    def invalidate(
        self,
        scope: TenantScope,
        text: str,
        discriminators: IntentCacheDiscriminators,
    ) -> bool:
        return self._cache.invalidate(
            scope, CacheNamespace.INTENT, self._parts(text, discriminators)
        )

    def invalidate_tenant(self, scope: TenantScope) -> int:
        return self._cache.invalidate_namespace(scope, CacheNamespace.INTENT)

    def report_false_hit(self, *, reviewed: int = 1, wrong: int = 1) -> None:
        with self._lock:
            self._stats.reviewed_hits += reviewed
            self._stats.false_hits += wrong


@dataclass(slots=True)
class SemanticEntry:
    """One cached classification, addressable by vector similarity."""

    tenant_id: str
    signature: str
    prompt: str
    vector: Vector
    classification: IntentClassification
    expires_at: float

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at


@dataclass(frozen=True, slots=True)
class SemanticMatch:
    entry: SemanticEntry
    similarity: float


@dataclass(frozen=True, slots=True)
class SemanticCachePolicy:
    """Both thresholds must be satisfied for a hit to be served."""

    similarity_threshold: float = 0.92
    confidence_threshold: float = 0.75
    ttl_seconds: float = 120.0
    capacity_per_signature: int = DEFAULT_SEMANTIC_CAPACITY

    def __post_init__(self) -> None:
        for name, value in (
            ("similarity_threshold", self.similarity_threshold),
            ("confidence_threshold", self.confidence_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.capacity_per_signature <= 0:
            raise ValueError("capacity_per_signature must be positive")


@dataclass(slots=True)
class _SignatureIndex:
    """The vectors for one tenant under one discriminator signature."""

    tenant_id: str
    entries: dict[str, SemanticEntry] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)


class SemanticIntentCache:
    """L1. Nearest-neighbour lookup over previously classified prompts.

    Brute-force cosine over a tenant-partitioned index. With `cache` unset that
    index is process-local. With a `TenantScopedCache` (Redis when configured)
    the same discriminator-partitioned entries are visible to every worker.
    Exact and semantic caches stay separate namespaces.
    """

    def __init__(
        self,
        *,
        policy: SemanticCachePolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        cache: TenantScopedCache | None = None,
    ) -> None:
        self._policy = policy or SemanticCachePolicy()
        self._clock = clock
        self._cache = cache
        self._store: TenantScopedStore[_SignatureIndex] = TenantScopedStore(
            "cache:semantic_intent", max_records_per_tenant=64
        )
        self._stats = IntentCacheStats()
        self._lock = threading.Lock()

    @property
    def policy(self) -> SemanticCachePolicy:
        return self._policy

    @property
    def stats(self) -> IntentCacheStats:
        with self._lock:
            return self._stats.snapshot()

    def lookup(
        self,
        scope: TenantScope,
        vector: Vector,
        discriminators: IntentCacheDiscriminators,
    ) -> SemanticMatch | None:
        """Return a usable neighbour, or `None`.

        A neighbour is usable only when it lives under the same discriminator
        signature, is unexpired, is similar enough, and was itself confident
        enough. A confident-but-dissimilar entry and a similar-but-unconfident
        entry are both refused.
        """
        try:
            index = self._load_index(scope, discriminators.signature)
        except Exception:  # noqa: BLE001 - semantic cache outage continues the cascade
            self._miss()
            return None
        if index is None:
            self._miss()
            return None

        now = self._now()
        best: SemanticMatch | None = None

        with self._lock:
            stale = [key for key, entry in index.entries.items() if entry.is_expired(now)]
            for key in stale:
                index.entries.pop(key, None)
            if stale:
                index.order = [key for key in index.order if key in index.entries]
                self._stats.expired += len(stale)

            candidates = list(index.entries.values())

        if stale and self._cache is not None:
            self._persist_index(scope, discriminators.signature, index)

        for entry in candidates:
            if entry.classification.confidence < self._policy.confidence_threshold:
                continue
            similarity = cosine_similarity(vector, entry.vector)
            if similarity < self._policy.similarity_threshold:
                continue
            if best is None or similarity > best.similarity:
                best = SemanticMatch(entry=entry, similarity=similarity)

        if best is None:
            self._miss()
            return None

        with self._lock:
            self._stats.hits += 1
        return best

    def admit(
        self,
        scope: TenantScope,
        prompt: str,
        vector: Vector,
        discriminators: IntentCacheDiscriminators,
        classification: IntentClassification,
    ) -> bool:
        """Offer a classification to the cache.

        Refused when the result is not confident enough to be worth serving to
        somebody else later. Admitting a weak answer converts one uncertain
        classification into many.
        """
        if classification.abstain:
            return False
        if classification.serving_state.value != "known":
            return False
        if classification.confidence < self._policy.confidence_threshold:
            return False

        signature = discriminators.signature
        try:
            index = self._load_index(scope, signature)
        except Exception:  # noqa: BLE001 - a failed semantic write must not fail classification
            return False
        if index is None:
            index = _SignatureIndex(tenant_id=scope.tenant_id)

        key = hashlib.sha256(normalise_prompt(prompt).encode("utf-8")).hexdigest()[:24]
        entry = SemanticEntry(
            tenant_id=scope.tenant_id,
            signature=signature,
            prompt=prompt,
            vector=vector,
            classification=classification,
            expires_at=self._now() + self._policy.ttl_seconds,
        )

        with self._lock:
            if key not in index.entries:
                index.order.append(key)
            index.entries[key] = entry
            while len(index.order) > self._policy.capacity_per_signature:
                evicted = index.order.pop(0)
                index.entries.pop(evicted, None)
            self._stats.writes += 1
        try:
            self._persist_index(scope, signature, index)
        except Exception:  # noqa: BLE001
            return False
        return True

    def invalidate_tenant(self, scope: TenantScope) -> int:
        removed = self._store.clear_tenant(scope)
        if self._cache is not None:
            removed += self._cache.invalidate_namespace(scope, CacheNamespace.SEMANTIC_INTENT)
        return removed

    def report_false_hit(self, *, reviewed: int = 1, wrong: int = 1) -> None:
        """Record the outcome of reviewing served hits against ground truth."""
        if reviewed < wrong:
            raise ValueError("cannot report more wrong hits than reviewed hits")
        with self._lock:
            self._stats.reviewed_hits += reviewed
            self._stats.false_hits += wrong

    def size(self, scope: TenantScope, discriminators: IntentCacheDiscriminators) -> int:
        index = self._load_index(scope, discriminators.signature)
        return len(index.entries) if index else 0

    def _now(self) -> float:
        return time.time() if self._cache is not None else self._clock()

    def _load_index(self, scope: TenantScope, signature: str) -> _SignatureIndex | None:
        if self._cache is None:
            return self._store.get(scope, signature)
        payload = self._cache.get(
            scope, CacheNamespace.SEMANTIC_INTENT, {"signature": signature}
        )
        if not isinstance(payload, dict):
            return None
        if payload.get("tenant_id") != scope.tenant_id:
            return None
        return _index_from_payload(payload)

    def _persist_index(self, scope: TenantScope, signature: str, index: _SignatureIndex) -> None:
        if self._cache is None:
            self._store.put(scope, signature, index)
            return
        self._cache.put(
            scope,
            CacheNamespace.SEMANTIC_INTENT,
            {"signature": signature},
            _index_to_payload(index),
            ttl_seconds=self._policy.ttl_seconds,
        )

    def _miss(self) -> None:
        with self._lock:
            self._stats.misses += 1


def _hydrate_classification(found: Any) -> IntentClassification | None:
    if isinstance(found, IntentClassification):
        return found
    if isinstance(found, dict):
        try:
            return decode(IntentClassification, found)
        except (TypeError, ValueError, KeyError):
            return None
    return None


def _index_to_payload(index: _SignatureIndex) -> dict[str, Any]:
    entries = []
    for key in index.order:
        entry = index.entries.get(key)
        if entry is None:
            continue
        entries.append(
            {
                "key": key,
                "tenant_id": entry.tenant_id,
                "signature": entry.signature,
                "prompt": entry.prompt,
                "vector": list(entry.vector),
                "classification": encode(entry.classification),
                "expires_at": entry.expires_at,
            }
        )
    return {"tenant_id": index.tenant_id, "entries": entries}


def _index_from_payload(payload: Mapping[str, Any]) -> _SignatureIndex:
    tenant_id = str(payload.get("tenant_id") or "")
    index = _SignatureIndex(tenant_id=tenant_id)
    for raw in payload.get("entries") or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("tenant_id") != tenant_id:
            continue
        classification = _hydrate_classification(raw.get("classification"))
        if classification is None:
            continue
        vector = raw.get("vector") or []
        key = str(raw.get("key") or "")
        if not key:
            continue
        index.entries[key] = SemanticEntry(
            tenant_id=tenant_id,
            signature=str(raw.get("signature") or ""),
            prompt=str(raw.get("prompt") or ""),
            vector=tuple(float(value) for value in vector),
            classification=classification,
            expires_at=float(raw.get("expires_at") or 0.0),
        )
        index.order.append(key)
    return index


def cache_report(exact: ExactIntentCache, semantic: SemanticIntentCache) -> Mapping[str, Any]:
    """A combined view, keeping the two caches visibly distinct."""
    exact_stats = exact.stats
    semantic_stats = semantic.stats
    return {
        "exact": {
            "hits": exact_stats.hits,
            "misses": exact_stats.misses,
            "writes": exact_stats.writes,
            "hit_rate": exact_stats.hit_rate,
            "false_hit_rate": exact_stats.false_hit_rate,
        },
        "semantic": {
            "hits": semantic_stats.hits,
            "misses": semantic_stats.misses,
            "writes": semantic_stats.writes,
            "expired": semantic_stats.expired,
            "hit_rate": semantic_stats.hit_rate,
            "false_hit_rate": semantic_stats.false_hit_rate,
            "similarity_threshold": semantic.policy.similarity_threshold,
            "confidence_threshold": semantic.policy.confidence_threshold,
        },
    }
