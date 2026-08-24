"""Tenant-scoped persistence for incidents, remediations, jobs and baselines."""

from __future__ import annotations

from llm_fabric.heal.schema import (
    DriftBaseline,
    Incident,
    LearningJob,
    RemediationRecord,
)
from llm_fabric.tenancy.scope import TenantScope
from llm_fabric.tenancy.store import IsolationAudit, TenantScopedStore


class IncidentRepository:
    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 5_000,
        store: TenantScopedStore[Incident] | None = None,
    ) -> None:
        self._store: TenantScopedStore[Incident] = store or TenantScopedStore(
            "incident", max_records_per_tenant=max_per_tenant, audit=audit
        )

    @property
    def store(self) -> TenantScopedStore[Incident]:
        return self._store

    def put(self, scope: TenantScope, incident: Incident) -> Incident:
        return self._store.put(scope, incident.incident_id, incident)

    def get(self, scope: TenantScope, incident_id: str) -> Incident | None:
        return self._store.get(scope, incident_id)

    def require(self, scope: TenantScope, incident_id: str) -> Incident:
        return self._store.require(scope, incident_id)

    def list(self, scope: TenantScope, *, limit: int = 50) -> list[Incident]:
        return self._store.list(scope, limit=limit)


class RemediationRepository:
    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 5_000,
        store: TenantScopedStore[RemediationRecord] | None = None,
    ) -> None:
        self._store: TenantScopedStore[RemediationRecord] = store or TenantScopedStore(
            "remediation", max_records_per_tenant=max_per_tenant, audit=audit
        )

    @property
    def store(self) -> TenantScopedStore[RemediationRecord]:
        return self._store

    def put(self, scope: TenantScope, record: RemediationRecord) -> RemediationRecord:
        return self._store.put(scope, record.record_id, record)

    def get(self, scope: TenantScope, record_id: str) -> RemediationRecord | None:
        return self._store.get(scope, record_id)

    def require(self, scope: TenantScope, record_id: str) -> RemediationRecord:
        return self._store.require(scope, record_id)

    def list(self, scope: TenantScope, *, limit: int = 50) -> list[RemediationRecord]:
        return self._store.list(scope, limit=limit)


class LearningJobRepository:
    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 5_000,
        store: TenantScopedStore[LearningJob] | None = None,
    ) -> None:
        self._store: TenantScopedStore[LearningJob] = store or TenantScopedStore(
            "learning_job", max_records_per_tenant=max_per_tenant, audit=audit
        )

    @property
    def store(self) -> TenantScopedStore[LearningJob]:
        return self._store

    def put(self, scope: TenantScope, job: LearningJob) -> LearningJob:
        return self._store.put(scope, job.job_id, job)

    def get(self, scope: TenantScope, job_id: str) -> LearningJob | None:
        return self._store.get(scope, job_id)

    def require(self, scope: TenantScope, job_id: str) -> LearningJob:
        return self._store.require(scope, job_id)

    def list(self, scope: TenantScope, *, limit: int = 50) -> list[LearningJob]:
        return self._store.list(scope, limit=limit)


class DriftBaselineRepository:
    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 1_000,
        store: TenantScopedStore[DriftBaseline] | None = None,
    ) -> None:
        self._store: TenantScopedStore[DriftBaseline] = store or TenantScopedStore(
            "drift_baseline", max_records_per_tenant=max_per_tenant, audit=audit
        )

    @property
    def store(self) -> TenantScopedStore[DriftBaseline]:
        return self._store

    def put(self, scope: TenantScope, baseline: DriftBaseline) -> DriftBaseline:
        return self._store.put(scope, baseline.baseline_id, baseline)

    def get(self, scope: TenantScope, baseline_id: str) -> DriftBaseline | None:
        return self._store.get(scope, baseline_id)

    def require(self, scope: TenantScope, baseline_id: str) -> DriftBaseline:
        return self._store.require(scope, baseline_id)

    def list(self, scope: TenantScope, *, limit: int = 50) -> list[DriftBaseline]:
        return self._store.list(scope, limit=limit)
