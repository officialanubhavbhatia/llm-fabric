"""Hard-example collection. No live self-training.

Production classifications that look interesting become *candidates*. A
candidate is redacted, hashed for dedup, and stored as draft. Promotion to a
labelled dataset — let alone a classifier — requires a human review step and a
passing evaluation. Nothing in this module trains or swaps a cascade.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

from llm_fabric.errors import InvalidRequestError
from llm_fabric.intent.cascade import CandidateExample
from llm_fabric.tenancy.scope import TenantScope
from llm_fabric.tenancy.store import IsolationAudit, TenantScopedStore

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b")
_TOKEN = re.compile(r"\b(?:sk-|rk-|key-|token-|bearer\s+)[A-Za-z0-9_\-]{8,}\b", re.IGNORECASE)


class CandidateStatus(StrEnum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class HardExample:
    tenant_id: str
    prompt_hash: str
    redacted_text: str
    reason: str
    predicted_intent: str
    confidence: float
    example_id: str = field(default_factory=lambda: f"hex_{uuid4().hex[:20]}")
    status: CandidateStatus = CandidateStatus.DRAFT
    taxonomy_version: str = ""
    classifier_version: str = ""
    created_at: float = field(default_factory=time.time)


def redact(text: str) -> str:
    """Strip the obvious secret-shaped spans. Not a complete privacy system."""
    cleaned = _EMAIL.sub("[email]", text)
    cleaned = _TOKEN.sub("[secret]", cleaned)
    return _PHONE.sub("[phone]", cleaned)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(redact(text).strip().lower().encode("utf-8")).hexdigest()[:24]


class HardExampleStore:
    """Tenant-scoped candidate examples. Deduped. Never auto-promoted."""

    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 10_000,
        store: TenantScopedStore[HardExample] | None = None,
    ) -> None:
        self._store = store or TenantScopedStore(
            "intent_hard_example", max_records_per_tenant=max_per_tenant, audit=audit
        )
        self._hashes: dict[tuple[str, str], str] = {}

    def ingest(self, scope: TenantScope, candidate: CandidateExample) -> HardExample | None:
        redacted = redact(candidate.text)
        digest = prompt_hash(redacted)
        key = (scope.tenant_id, digest)
        if key in self._hashes:
            return None
        example = HardExample(
            tenant_id=scope.tenant_id,
            prompt_hash=digest,
            redacted_text=redacted,
            reason=candidate.reason,
            predicted_intent=candidate.decision.classification.intent_id,
            confidence=candidate.decision.classification.confidence,
            taxonomy_version=candidate.decision.classification.taxonomy_version,
            classifier_version=candidate.decision.classification.classifier_version,
        )
        stored = self._store.put(scope, example.example_id, example)
        self._hashes[key] = stored.example_id
        return stored

    def get(self, scope: TenantScope, example_id: str) -> HardExample | None:
        return self._store.get(scope, example_id)

    def list(self, scope: TenantScope, *, limit: int = 100) -> list[HardExample]:
        return self._store.list(scope, limit=limit)

    def accept(self, scope: TenantScope, example_id: str) -> HardExample:
        existing = self._store.require(scope, example_id)
        if existing.status is CandidateStatus.REJECTED:
            raise InvalidRequestError("a rejected example cannot be accepted")
        updated = HardExample(
            tenant_id=existing.tenant_id,
            prompt_hash=existing.prompt_hash,
            redacted_text=existing.redacted_text,
            reason=existing.reason,
            predicted_intent=existing.predicted_intent,
            confidence=existing.confidence,
            example_id=existing.example_id,
            status=CandidateStatus.ACCEPTED,
            taxonomy_version=existing.taxonomy_version,
            classifier_version=existing.classifier_version,
            created_at=existing.created_at,
        )
        return self._store.put(scope, example_id, updated)


def promotion_blocked_reason(*, eval_passed: bool, reviewed: bool) -> str | None:
    """Refuse unsupervised promotion. Returns None when promotion is allowed."""
    if not reviewed:
        return "candidate has not been reviewed"
    if not eval_passed:
        return "evaluation gates have not passed"
    return None
