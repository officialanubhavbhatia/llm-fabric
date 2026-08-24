"""Add topology identity columns to usage_events.

Revision ID: 0004_usage_topology
Revises: 0003_revoke_app_ddl
Create Date: 2026-08-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision: str = "0004_usage_topology"
down_revision: str | Sequence[str] | None = "0003_revoke_app_ddl"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS: tuple[tuple[str, sa.types.TypeEngine[object]], ...] = (
    ("provider_adapter", sa.String(64)),
    ("transport", sa.String(32)),
    ("runtime", sa.String(32)),
    ("grade", sa.String(32)),
    ("litellm_model", sa.String(256)),
    ("actual_served_model", sa.String(256)),
    ("route_id", sa.String(64)),
)


def upgrade() -> None:
    bind = op.get_bind()
    existing = {column["name"] for column in inspect(bind).get_columns("usage_events")}
    for name, column_type in _COLUMNS:
        if name in existing:
            continue
        op.add_column("usage_events", sa.Column(name, column_type, nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    existing = {column["name"] for column in inspect(bind).get_columns("usage_events")}
    for name, _column_type in reversed(_COLUMNS):
        if name in existing:
            op.drop_column("usage_events", name)
