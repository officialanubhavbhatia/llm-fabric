"""Backup and restore of the durable tenant-record store.

A backup that has never been restored is not verified. This module dumps the
SQLAlchemy schema used by `tenant_records` (and sibling tables) to JSON and
reloads it into a fresh engine so the restore path is exercised in tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from llm_fabric.errors import ConfigurationError
from llm_fabric.storage.postgres import (
    AuditEventRow,
    TenantRecordRow,
    TenantRow,
    UserRow,
    create_database_engine,
    init_schema,
)


def dump_engine(engine: Any, destination: Path) -> dict[str, int]:
    """Write every durable row to `destination`. Returns per-table counts."""
    payload: dict[str, Any] = {"tenant_records": [], "tenants": [], "users": [], "audit_events": []}
    with Session(engine) as session:
        for row in session.execute(select(TenantRecordRow)).scalars():
            payload["tenant_records"].append(
                {
                    "store": row.store,
                    "tenant_id": row.tenant_id,
                    "key": row.key,
                    "payload": row.payload,
                    "updated_at": row.updated_at,
                }
            )
        for tenant in session.execute(select(TenantRow)).scalars():
            payload["tenants"].append(
                {
                    "tenant_id": tenant.tenant_id,
                    "name": tenant.name,
                    "created_at": tenant.created_at,
                }
            )
        for user in session.execute(select(UserRow)).scalars():
            payload["users"].append(
                {
                    "tenant_id": user.tenant_id,
                    "user_id": user.user_id,
                    "roles": user.roles,
                    "created_at": user.created_at,
                }
            )
        for event in session.execute(select(AuditEventRow)).scalars():
            payload["audit_events"].append(
                {
                    "event_id": event.event_id,
                    "tenant_id": event.tenant_id,
                    "actor": event.actor,
                    "action": event.action,
                    "target": event.target,
                    "before_json": event.before_json,
                    "after_json": event.after_json,
                    "reason": event.reason,
                    "request_id": event.request_id,
                    "created_at": event.created_at,
                }
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return {name: len(rows) for name, rows in payload.items()}


def restore_engine(source: Path, engine: Any) -> dict[str, int]:
    """Load a dump into `engine`. The schema is created if missing."""
    if not source.is_file():
        raise ConfigurationError(f"backup file does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    init_schema(engine)
    counts = {
        name: len(payload.get(name) or [])
        for name in ("tenant_records", "tenants", "users", "audit_events")
    }
    with Session(engine) as session:
        for row in payload.get("tenants") or []:
            session.merge(TenantRow(**row))
        for row in payload.get("users") or []:
            session.merge(UserRow(**row))
        for row in payload.get("audit_events") or []:
            session.merge(AuditEventRow(**row))
        for row in payload.get("tenant_records") or []:
            session.merge(TenantRecordRow(**row))
        session.commit()
    return counts


def clone_sqlite(source_url: str, destination_url: str, dump_path: Path) -> dict[str, int]:
    """Dump `source_url` and restore into `destination_url` via `dump_path`."""
    source = create_database_engine(source_url)
    try:
        counts = dump_engine(source, dump_path)
    finally:
        source.dispose()
    destination = create_database_engine(destination_url)
    try:
        restore_engine(dump_path, destination)
    finally:
        destination.dispose()
    return counts
