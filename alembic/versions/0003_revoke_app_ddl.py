"""Revoke schema CREATE from the runtime application role.

Revision ID: 0003_revoke_app_ddl
Revises: 0002_tenant_schema
Create Date: 2026-08-24

Existing clusters granted CREATE on public to fabric_app so workers could
run create_all. Production workers no longer DDL; this revision removes the
privilege. The migration/table-owner role (Compose: fabric) keeps DDL.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_revoke_app_ddl"
down_revision: str | Sequence[str] | None = "0002_tenant_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    exists = bind.execute(sa.text("SELECT 1 FROM pg_roles WHERE rolname = 'fabric_app'")).scalar()
    if not exists:
        return
    database = bind.execute(sa.text("SELECT current_database()")).scalar()
    op.execute("GRANT USAGE ON SCHEMA public TO fabric_app")
    op.execute("REVOKE CREATE ON SCHEMA public FROM fabric_app")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO fabric_app")
    op.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO fabric_app")
    op.execute("GRANT SELECT ON TABLE alembic_version TO fabric_app")
    if database and str(database).replace("_", "").isalnum():
        op.execute(f"GRANT CONNECT ON DATABASE {database} TO fabric_app")


def downgrade() -> None:
    # Least privilege is the invariant. Do not put CREATE back on fabric_app.
    return
