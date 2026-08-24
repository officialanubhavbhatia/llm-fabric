"""Create the usage_events ledger.

Revision ID: 0001_usage_events
Revises:
Create Date: 2026-08-24

Idempotent on a table that already exists from the SQLAlchemy create_all era.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision: str = "0001_usage_events"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_table(name: str) -> bool:
    return name in inspect(op.get_bind()).get_table_names()


def _has_index(name: str) -> bool:
    bind = op.get_bind()
    return any(index["name"] == name for index in inspect(bind).get_indexes("usage_events"))


_INDEXES: tuple[tuple[str, list[str]], ...] = (
    ("ix_usage_events_request_id", ["request_id"]),
    ("ix_usage_events_trace_id", ["trace_id"]),
    ("ix_usage_events_tenant_completed", ["tenant_id", "completed_at"]),
    ("ix_usage_events_user_completed", ["tenant_id", "user_id", "completed_at"]),
    ("ix_usage_events_project_completed", ["tenant_id", "project_id", "completed_at"]),
    ("ix_usage_events_provider_completed", ["provider", "completed_at"]),
    ("ix_usage_events_model_completed", ["model", "completed_at"]),
)


def upgrade() -> None:
    if not _has_table("usage_events"):
        op.create_table(
            "usage_events",
            sa.Column("event_id", sa.String(length=64), primary_key=True),
            sa.Column("invocation_id", sa.String(length=64), nullable=False),
            sa.Column("request_id", sa.String(length=64), nullable=False),
            sa.Column("trace_id", sa.String(length=64), nullable=True),
            sa.Column("tenant_id", sa.String(length=256), nullable=False),
            sa.Column("user_id", sa.String(length=256), nullable=True),
            sa.Column("project_id", sa.String(length=256), nullable=True),
            sa.Column("provider", sa.String(length=128), nullable=False),
            sa.Column("model", sa.String(length=256), nullable=False),
            sa.Column("requested_model", sa.String(length=256), nullable=True),
            sa.Column("policy", sa.String(length=64), nullable=True),
            sa.Column("deployment_id", sa.String(length=256), nullable=True),
            sa.Column("operation", sa.String(length=64), nullable=False),
            sa.Column("intent_id", sa.String(length=256), nullable=True),
            sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("cached_tokens", sa.Integer(), nullable=True),
            sa.Column("reasoning_tokens", sa.Integer(), nullable=True),
            sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("provider_cost_usd", sa.Float(), nullable=True),
            sa.Column("compute_cost_estimate_usd", sa.Float(), nullable=True),
            sa.Column("token_source", sa.String(length=32), nullable=False),
            sa.Column("started_at", sa.Float(), nullable=False),
            sa.Column("completed_at", sa.Float(), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("fallback_depth", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("attempt_number", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("streaming", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("error", sa.Text(), nullable=True),
            sa.UniqueConstraint("invocation_id", name="uq_usage_events_invocation_id"),
        )
        for name, columns in _INDEXES:
            op.create_index(name, "usage_events", columns)
    else:
        for name, columns in _INDEXES:
            if not _has_index(name):
                op.create_index(name, "usage_events", columns)

    bind = op.get_bind()
    if bind.dialect.name == "postgresql" and _has_table("usage_events"):
        op.execute("ALTER TABLE usage_events ENABLE ROW LEVEL SECURITY")
        op.execute("ALTER TABLE usage_events FORCE ROW LEVEL SECURITY")
        op.execute("DROP POLICY IF EXISTS usage_events_isolation ON usage_events")
        op.execute(
            """
            CREATE POLICY usage_events_isolation ON usage_events
              USING (
                tenant_id = current_setting('app.current_tenant', true)
                OR current_setting('app.fabric_observe', true) = 'on'
              )
              WITH CHECK (tenant_id = current_setting('app.current_tenant', true))
            """
        )
        exists = bind.execute(
            sa.text("SELECT 1 FROM pg_roles WHERE rolname = 'fabric_app'")
        ).scalar()
        if exists:
            op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE usage_events TO fabric_app")


def downgrade() -> None:
    op.drop_table("usage_events")
