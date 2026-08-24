"""Durable tenant tables that used to be created by SQLAlchemy create_all.

Revision ID: 0002_tenant_schema
Revises: 0001_usage_events
Create Date: 2026-08-24

Idempotent on tables that already exist from the create_all era so an
existing cluster can `alembic upgrade head` without dropping data.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision: str = "0002_tenant_schema"
down_revision: str | Sequence[str] | None = "0001_usage_events"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ISOLATION = "tenant_id = current_setting('app.current_tenant', true)"


def _has_table(name: str) -> bool:
    return name in inspect(op.get_bind()).get_table_names()


def _grant_dml(*tables: str) -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    exists = bind.execute(sa.text("SELECT 1 FROM pg_roles WHERE rolname = 'fabric_app'")).scalar()
    if not exists:
        return
    for table in tables:
        op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {table} TO fabric_app")
    if _has_table("alembic_version"):
        op.execute("GRANT SELECT ON TABLE alembic_version TO fabric_app")


def _apply_rls(table: str, using: str, check: str) -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    me = bind.execute(sa.text("SELECT current_user")).scalar()
    owner = bind.execute(
        sa.text(
            "SELECT tableowner FROM pg_tables WHERE schemaname = 'public' AND tablename = :table"
        ),
        {"table": table},
    ).scalar()
    if owner is None or owner != me:
        return
    policy = f"{table}_isolation"
    op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
    op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")
    op.execute(f"CREATE POLICY {policy} ON {table} USING ({using}) WITH CHECK ({check})")


def upgrade() -> None:
    if not _has_table("tenant_records"):
        op.create_table(
            "tenant_records",
            sa.Column("store", sa.String(length=64), primary_key=True),
            sa.Column("tenant_id", sa.String(length=256), primary_key=True),
            sa.Column("key", sa.String(length=512), primary_key=True),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("updated_at", sa.Float(), nullable=False),
        )
    if not _has_table("tenants"):
        op.create_table(
            "tenants",
            sa.Column("tenant_id", sa.String(length=256), primary_key=True),
            sa.Column("name", sa.String(length=256), nullable=False),
            sa.Column("created_at", sa.Float(), nullable=False),
        )
    if not _has_table("users"):
        op.create_table(
            "users",
            sa.Column("tenant_id", sa.String(length=256), primary_key=True),
            sa.Column("user_id", sa.String(length=256), primary_key=True),
            sa.Column("roles", sa.Text(), nullable=False, server_default=""),
            sa.Column("created_at", sa.Float(), nullable=False),
        )
    if not _has_table("audit_events"):
        op.create_table(
            "audit_events",
            sa.Column("event_id", sa.String(length=64), primary_key=True),
            sa.Column("tenant_id", sa.String(length=256), nullable=False),
            sa.Column("actor", sa.String(length=256), nullable=False),
            sa.Column("action", sa.String(length=128), nullable=False),
            sa.Column("target", sa.String(length=512), nullable=False),
            sa.Column("before_json", sa.JSON(), nullable=True),
            sa.Column("after_json", sa.JSON(), nullable=True),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("request_id", sa.String(length=64), nullable=True),
            sa.Column("created_at", sa.Float(), nullable=False),
        )
        op.create_index("ix_audit_events_tenant_id", "audit_events", ["tenant_id"])
        op.create_index("ix_audit_events_created_at", "audit_events", ["created_at"])

    _apply_rls("tenant_records", _ISOLATION, _ISOLATION)
    _apply_rls("tenants", _ISOLATION, _ISOLATION)
    _apply_rls("users", _ISOLATION, _ISOLATION)
    _apply_rls("audit_events", _ISOLATION, _ISOLATION)
    _grant_dml("tenant_records", "tenants", "users", "audit_events", "usage_events")


def downgrade() -> None:
    op.drop_table("audit_events")
    op.drop_table("users")
    op.drop_table("tenants")
    op.drop_table("tenant_records")
