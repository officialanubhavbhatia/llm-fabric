"""Tenant-scoped persistence for evaluation runs and suites."""

from __future__ import annotations

from llm_fabric.eval.schema import EvalComparison, EvalRun, EvalSuite
from llm_fabric.tenancy.scope import TenantScope
from llm_fabric.tenancy.store import IsolationAudit, TenantScopedStore


class EvalSuiteRepository:
    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 1_000,
        store: TenantScopedStore[EvalSuite] | None = None,
    ) -> None:
        self._store: TenantScopedStore[EvalSuite] = store or TenantScopedStore(
            "eval_suite", max_records_per_tenant=max_per_tenant, audit=audit
        )

    @property
    def store(self) -> TenantScopedStore[EvalSuite]:
        return self._store

    def put(self, scope: TenantScope, suite: EvalSuite) -> EvalSuite:
        return self._store.put(scope, suite.suite_id, suite)

    def get(self, scope: TenantScope, suite_id: str) -> EvalSuite | None:
        return self._store.get(scope, suite_id)

    def require(self, scope: TenantScope, suite_id: str) -> EvalSuite:
        return self._store.require(scope, suite_id)

    def list(self, scope: TenantScope, *, limit: int = 50) -> list[EvalSuite]:
        return self._store.list(scope, limit=limit)


class EvalRunRepository:
    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 5_000,
        store: TenantScopedStore[EvalRun] | None = None,
    ) -> None:
        self._store: TenantScopedStore[EvalRun] = store or TenantScopedStore(
            "eval_run", max_records_per_tenant=max_per_tenant, audit=audit
        )

    @property
    def store(self) -> TenantScopedStore[EvalRun]:
        return self._store

    def put(self, scope: TenantScope, run: EvalRun) -> EvalRun:
        return self._store.put(scope, run.run_id, run)

    def get(self, scope: TenantScope, run_id: str) -> EvalRun | None:
        return self._store.get(scope, run_id)

    def require(self, scope: TenantScope, run_id: str) -> EvalRun:
        return self._store.require(scope, run_id)

    def list(self, scope: TenantScope, *, limit: int = 50) -> list[EvalRun]:
        return self._store.list(scope, limit=limit)


class EvalComparisonRepository:
    def __init__(
        self,
        *,
        audit: IsolationAudit | None = None,
        max_per_tenant: int = 5_000,
        store: TenantScopedStore[EvalComparison] | None = None,
    ) -> None:
        self._store: TenantScopedStore[EvalComparison] = store or TenantScopedStore(
            "eval_comparison", max_records_per_tenant=max_per_tenant, audit=audit
        )

    @property
    def store(self) -> TenantScopedStore[EvalComparison]:
        return self._store

    def put(self, scope: TenantScope, comparison: EvalComparison) -> EvalComparison:
        return self._store.put(scope, comparison.comparison_id, comparison)

    def get(self, scope: TenantScope, comparison_id: str) -> EvalComparison | None:
        return self._store.get(scope, comparison_id)

    def require(self, scope: TenantScope, comparison_id: str) -> EvalComparison:
        return self._store.require(scope, comparison_id)

    def list(self, scope: TenantScope, *, limit: int = 50) -> list[EvalComparison]:
        return self._store.list(scope, limit=limit)
