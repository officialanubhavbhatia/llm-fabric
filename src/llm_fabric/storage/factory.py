"""Build a tenant-scoped record store for the configured backend."""

from __future__ import annotations

from sqlalchemy.engine import Engine

from llm_fabric.storage.postgres import PostgresTenantStore
from llm_fabric.tenancy.store import IsolationAudit, TenantOwned, TenantScopedStore


def build_record_store[T: TenantOwned](
    name: str,
    record_type: type[T],
    *,
    max_records: int,
    audit: IsolationAudit | None = None,
    engine: Engine | None = None,
) -> TenantScopedStore[T]:
    if engine is None:
        return TenantScopedStore(name, max_records_per_tenant=max_records, audit=audit)
    return PostgresTenantStore(
        name,
        record_type,
        engine,
        max_records_per_tenant=max_records,
        audit=audit,
    )
