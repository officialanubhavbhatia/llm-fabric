"""Add context_record_id to usage_events.

Revision ID: 0006_context_record
Revises: 0005_usage_intent
Create Date: 2026-08-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision: str = "0006_context_record"
down_revision: str | Sequence[str] | None = "0005_usage_intent"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = {column["name"] for column in inspect(bind).get_columns("usage_events")}
    if "context_record_id" not in existing:
        op.add_column("usage_events", sa.Column("context_record_id", sa.String(64), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    existing = {column["name"] for column in inspect(bind).get_columns("usage_events")}
    if "context_record_id" in existing:
        op.drop_column("usage_events", "context_record_id")
