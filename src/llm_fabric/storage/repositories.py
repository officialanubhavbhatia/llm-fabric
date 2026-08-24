"""Tenant-scoped repositories.

Each repository is a thin, typed face over `TenantScopedStore`, which is where
the isolation guarantee actually lives. Repositories deliberately do not accept
a raw tenant id: they take a `TenantScope` built from an authenticated
principal, so there is no code path that reads a record without saying, in
types, whose record it is entitled to read.

Backed by the in-memory store. Postgres is not built; these classes are the
interface a SQLAlchemy implementation will satisfy.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from llm_fabric.errors import InvalidRequestError, ResourceNotFoundError
from llm_fabric.eval.store import (
    EvalComparisonRepository,
    EvalRunRepository,
    EvalSuiteRepository,
)
from llm_fabric.heal.store import (
    DriftBaselineRepository,
    IncidentRepository,
    LearningJobRepository,
    RemediationRepository,
)
from llm_fabric.storage.records import (
    AuditEvent,
    Conversation,
    ConversationMessage,
    EvalDataset,
    EvalExample,
    IntentExample,
    PromptDefinition,
    PromptStatus,
    TraceRecord,
)
from llm_fabric.tenancy.scope import TenantScope
from llm_fabric.tenancy.store import IsolationAudit, TenantScopedStore


class ConversationRepository:
    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 10_000,
        store: TenantScopedStore[Conversation] | None = None,
    ) -> None:
        self._store: TenantScopedStore[Conversation] = store or TenantScopedStore(
            "conversation", max_records_per_tenant=max_per_tenant, audit=audit
        )

    @property
    def store(self) -> TenantScopedStore[Conversation]:
        return self._store

    def create(
        self,
        scope: TenantScope,
        *,
        title: str | None = None,
        messages: tuple[ConversationMessage, ...] = (),
    ) -> Conversation:
        conversation = Conversation(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id or "-",
            project_id=scope.project_id,
            title=title,
            messages=messages,
        )
        return self._store.put(scope, conversation.conversation_id, conversation)

    def get(self, scope: TenantScope, conversation_id: str) -> Conversation | None:
        return self._store.get(scope, conversation_id)

    def require(self, scope: TenantScope, conversation_id: str) -> Conversation:
        return self._store.require(scope, conversation_id)

    def list(self, scope: TenantScope, *, limit: int = 50) -> list[Conversation]:
        return self._store.list(scope, limit=limit)

    def append_message(
        self, scope: TenantScope, conversation_id: str, message: ConversationMessage
    ) -> Conversation:
        existing = self._store.require(scope, conversation_id)
        updated = replace(
            existing,
            messages=(*existing.messages, message),
            updated_at=message.created_at,
        )
        return self._store.put(scope, conversation_id, updated)

    def delete(self, scope: TenantScope, conversation_id: str) -> bool:
        return self._store.delete(scope, conversation_id)


class TraceRepository:
    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 20_000,
        store: TenantScopedStore[TraceRecord] | None = None,
    ) -> None:
        self._store: TenantScopedStore[TraceRecord] = store or TenantScopedStore(
            "trace", max_records_per_tenant=max_per_tenant, audit=audit
        )

    @property
    def store(self) -> TenantScopedStore[TraceRecord]:
        return self._store

    def record(self, scope: TenantScope, trace: TraceRecord) -> TraceRecord:
        return self._store.put(scope, trace.trace_id, trace)

    def get(self, scope: TenantScope, trace_id: str) -> TraceRecord | None:
        return self._store.get(scope, trace_id)

    def require(self, scope: TenantScope, trace_id: str) -> TraceRecord:
        return self._store.require(scope, trace_id)

    def list(self, scope: TenantScope, *, limit: int = 50) -> list[TraceRecord]:
        return self._store.list(scope, limit=limit)


class IntentExampleRepository:
    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 50_000,
        store: TenantScopedStore[IntentExample] | None = None,
    ) -> None:
        self._store: TenantScopedStore[IntentExample] = store or TenantScopedStore(
            "intent_example", max_records_per_tenant=max_per_tenant, audit=audit
        )

    @property
    def store(self) -> TenantScopedStore[IntentExample]:
        return self._store

    def add(self, scope: TenantScope, example: IntentExample) -> IntentExample:
        return self._store.put(scope, example.example_id, example)

    def get(self, scope: TenantScope, example_id: str) -> IntentExample | None:
        return self._store.get(scope, example_id)

    def require(self, scope: TenantScope, example_id: str) -> IntentExample:
        return self._store.require(scope, example_id)

    def list(
        self,
        scope: TenantScope,
        *,
        intent_id: str | None = None,
        limit: int = 100,
    ) -> list[IntentExample]:
        predicate = None
        if intent_id is not None:
            predicate = lambda example: example.intent_id == intent_id  # noqa: E731
        return self._store.list(scope, limit=limit, predicate=predicate)


class PromptRepository:
    """Versioned prompts. A published version is never modified."""

    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 10_000,
        store: TenantScopedStore[PromptDefinition] | None = None,
    ) -> None:
        self._store: TenantScopedStore[PromptDefinition] = store or TenantScopedStore(
            "prompt", max_records_per_tenant=max_per_tenant, audit=audit
        )

    @property
    def store(self) -> TenantScopedStore[PromptDefinition]:
        return self._store

    def publish(self, scope: TenantScope, prompt: PromptDefinition) -> PromptDefinition:
        existing = self._store.get(scope, prompt.key)
        if existing is not None and existing.is_published:
            raise InvalidRequestError(
                f"prompt '{existing.key}' is published and cannot be modified; "
                "create a new version instead"
            )
        return self._store.put(scope, prompt.key, prompt)

    def promote(
        self, scope: TenantScope, prompt_id: str, version: int, status: PromptStatus
    ) -> PromptDefinition:
        """Advance a version's status without altering its content."""
        existing = self._store.require(scope, f"{prompt_id}@{version}")
        return self._store.put(scope, existing.key, replace(existing, status=status))

    def get(self, scope: TenantScope, prompt_id: str, version: int) -> PromptDefinition | None:
        return self._store.get(scope, f"{prompt_id}@{version}")

    def require(self, scope: TenantScope, prompt_id: str, version: int) -> PromptDefinition:
        key = f"{prompt_id}@{version}"
        prompt = self._store.get(scope, key)
        if prompt is None:
            raise ResourceNotFoundError(f"no such prompt: '{key}'")
        return prompt

    def list(self, scope: TenantScope, *, limit: int = 100) -> list[PromptDefinition]:
        return self._store.list(scope, limit=limit)


class EvalDatasetRepository:
    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 5_000,
        store: TenantScopedStore[EvalDataset] | None = None,
    ) -> None:
        self._store: TenantScopedStore[EvalDataset] = store or TenantScopedStore(
            "eval_dataset", max_records_per_tenant=max_per_tenant, audit=audit
        )

    @property
    def store(self) -> TenantScopedStore[EvalDataset]:
        return self._store

    def create(
        self,
        scope: TenantScope,
        *,
        name: str,
        description: str | None = None,
        examples: tuple[EvalExample, ...] = (),
    ) -> EvalDataset:
        dataset = EvalDataset(
            tenant_id=scope.tenant_id,
            name=name,
            description=description,
            examples=examples,
        )
        return self._store.put(scope, dataset.dataset_id, dataset)

    def get(self, scope: TenantScope, dataset_id: str) -> EvalDataset | None:
        return self._store.get(scope, dataset_id)

    def require(self, scope: TenantScope, dataset_id: str) -> EvalDataset:
        return self._store.require(scope, dataset_id)

    def list(self, scope: TenantScope, *, limit: int = 50) -> list[EvalDataset]:
        return self._store.list(scope, limit=limit)


_SECRET_KEYS = frozenset({"key", "secret", "password", "token", "credential", "api_key"})


def _strip_secrets(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        key: "[REDACTED]" if key.lower() in _SECRET_KEYS else value
        for key, value in payload.items()
    }


class AuditRepository:
    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 20_000,
        store: TenantScopedStore[AuditEvent] | None = None,
    ) -> None:
        self._store: TenantScopedStore[AuditEvent] = store or TenantScopedStore(
            "audit_event", max_records_per_tenant=max_per_tenant, audit=audit
        )

    def record(
        self,
        scope: TenantScope,
        *,
        actor: str,
        action: str,
        target: str,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        reason: str | None = None,
        request_id: str | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            tenant_id=scope.tenant_id,
            actor=actor,
            action=action,
            target=target,
            before=_strip_secrets(before),
            after=_strip_secrets(after),
            reason=reason,
            request_id=request_id,
        )
        return self._store.put(scope, event.event_id, event)

    def get(self, scope: TenantScope, event_id: str) -> AuditEvent | None:
        return self._store.get(scope, event_id)

    def list(self, scope: TenantScope, *, limit: int = 100) -> list[AuditEvent]:
        return self._store.list(scope, limit=limit)


class TenantStores:
    """Every tenant-scoped repository, sharing one isolation audit.

    Sharing the audit means a boundary violation anywhere is visible in one
    place, which is what the cross-tenant test suite asserts against.
    `engine` selects PostgreSQL; `None` keeps the in-memory backend.
    """

    def __init__(
        self, audit: IsolationAudit | None = None, *, engine: object | None = None
    ) -> None:
        from sqlalchemy.engine import Engine

        from llm_fabric.eval.schema import EvalComparison, EvalRun, EvalSuite
        from llm_fabric.heal.schema import (
            DriftBaseline,
            Incident,
            LearningJob,
            RemediationRecord,
        )
        from llm_fabric.intent.learning import HardExample
        from llm_fabric.intent.persist import IntentClassificationRecord
        from llm_fabric.storage.factory import build_record_store

        self.audit = audit or IsolationAudit()
        pg = engine if isinstance(engine, Engine) else None
        self.engine = pg

        def store(name: str, record_type: type[Any], max_records: int) -> Any:
            return build_record_store(
                name, record_type, max_records=max_records, audit=self.audit, engine=pg
            )

        self.conversations = ConversationRepository(
            audit=self.audit, store=store("conversation", Conversation, 10_000)
        )
        self.traces = TraceRepository(audit=self.audit, store=store("trace", TraceRecord, 20_000))
        self.intent_examples = IntentExampleRepository(
            audit=self.audit, store=store("intent_example", IntentExample, 50_000)
        )
        self.prompts = PromptRepository(
            audit=self.audit, store=store("prompt", PromptDefinition, 10_000)
        )
        self.eval_datasets = EvalDatasetRepository(
            audit=self.audit, store=store("eval_dataset", EvalDataset, 5_000)
        )
        self.eval_suites = EvalSuiteRepository(
            audit=self.audit, store=store("eval_suite", EvalSuite, 1_000)
        )
        self.eval_runs = EvalRunRepository(
            audit=self.audit, store=store("eval_run", EvalRun, 5_000)
        )
        self.eval_comparisons = EvalComparisonRepository(
            audit=self.audit, store=store("eval_comparison", EvalComparison, 5_000)
        )
        self.incidents = IncidentRepository(
            audit=self.audit, store=store("incident", Incident, 5_000)
        )
        self.remediations = RemediationRepository(
            audit=self.audit, store=store("remediation", RemediationRecord, 5_000)
        )
        self.learning_jobs = LearningJobRepository(
            audit=self.audit, store=store("learning_job", LearningJob, 5_000)
        )
        self.drift_baselines = DriftBaselineRepository(
            audit=self.audit, store=store("drift_baseline", DriftBaseline, 5_000)
        )
        self.audit_events = AuditRepository(
            audit=self.audit, store=store("audit_event", AuditEvent, 20_000)
        )
        from llm_fabric.intent.learning import HardExampleStore
        from llm_fabric.intent.persist import IntentClassificationRepository

        self.intent_classifications = IntentClassificationRepository(
            audit=self.audit,
            store=store("intent_classification", IntentClassificationRecord, 50_000),
        )
        self.intent_hard_examples = HardExampleStore(
            audit=self.audit,
            store=store("intent_hard_example", HardExample, 10_000),
        )
