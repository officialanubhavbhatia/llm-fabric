"""PostgreSQL-backed tenant-scoped records.

The in-memory store remains the default for development and unit tests. This
backend is the durable implementation: one table, tenant in the primary key,
application-layer filtering on every query, and PostgreSQL row-level security
when the dialect supports it.

ClickHouse is not used here. This module is on the request path for durable
records only.
"""

from __future__ import annotations

import builtins
import re
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    event,
    func,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from llm_fabric.errors import ConfigurationError, ResourceNotFoundError, TenantIsolationError
from llm_fabric.storage.codec import decode, encode
from llm_fabric.tenancy.scope import TenantScope
from llm_fabric.tenancy.store import IsolationAudit, TenantOwned, TenantScopedStore

_RLS_SETTING = "app.current_tenant"
_OBSERVE_SETTING = "app.fabric_observe"
APPLICATION_ROLE = "fabric_app"
_ROLE_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")


class Base(DeclarativeBase):
    pass


class TenantRecordRow(Base):
    __tablename__ = "tenant_records"

    store: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    key: Mapped[str] = mapped_column(String(512), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)


class TenantRow(Base):
    __tablename__ = "tenants"

    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)


class UserRow(Base):
    __tablename__ = "users"

    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    roles: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, nullable=False)


class AuditEventRow(Base):
    __tablename__ = "audit_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(256), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    target: Mapped[str] = mapped_column(String(512), nullable=False)
    before_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    after_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)


class UsageEventRow(Base):
    """Authoritative provider-invocation ledger. No prompt or response text."""

    __tablename__ = "usage_events"
    __table_args__ = (
        Index("ix_usage_events_tenant_completed", "tenant_id", "completed_at"),
        Index("ix_usage_events_user_completed", "tenant_id", "user_id", "completed_at"),
        Index("ix_usage_events_project_completed", "tenant_id", "project_id", "completed_at"),
        Index("ix_usage_events_provider_completed", "provider", "completed_at"),
        Index("ix_usage_events_model_completed", "model", "completed_at"),
    )

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    invocation_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(256), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    project_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    requested_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    policy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    deployment_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    route_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    intent_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    intent_result_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    taxonomy_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    classifier_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    context_record_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cached_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    provider_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    compute_cost_estimate_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    token_source: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[float] = mapped_column(Float, nullable=False)
    completed_at: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    fallback_depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    streaming: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider_adapter: Mapped[str | None] = mapped_column(String(64), nullable=True)
    transport: Mapped[str | None] = mapped_column(String(32), nullable=True)
    runtime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    grade: Mapped[str | None] = mapped_column(String(32), nullable=True)
    litellm_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    actual_served_model: Mapped[str | None] = mapped_column(String(256), nullable=True)


STARTUP_PROBE_TIMEOUT_S = 3


def create_database_engine(
    url: str,
    *,
    echo: bool = False,
    connect_timeout_s: float | None = None,
) -> Engine:
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    if url.startswith("postgresql://"):
        # The project depends on psycopg v3, not psycopg2. SQLAlchemy's default
        # `postgresql://` dialect still imports psycopg2.
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    if connect_timeout_s is not None and not url.startswith("sqlite"):
        # libpq/psycopg take an integer second floor. A probe must not hang.
        # sqlite rejects connect_timeout as a Connection() keyword.
        connect_args["connect_timeout"] = max(1, int(connect_timeout_s))
    engine = create_engine(
        url,
        echo=echo,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _sqlite_fk(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def init_schema(engine: Engine) -> None:
    """Create tables for development, tests, and sqlite convenience.

    Production workers must not call this. They assert an Alembic head via
    `assert_schema_revision` and then only issue DML.
    """
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        _enable_row_level_security(engine)
        provision_application_role(engine)


def current_role_bypasses_rls(engine: Engine) -> tuple[bool, str]:
    """True when the connected role is superuser or BYPASSRLS.

    FORCE ROW LEVEL SECURITY does not apply to those roles. The gateway must
    not use them at runtime or tenant isolation is application-filter only.
    """
    if engine.dialect.name != "postgresql":
        return False, engine.dialect.name
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT current_user, rolsuper, rolbypassrls "
                "FROM pg_roles WHERE rolname = current_user"
            )
        ).one()
    name, superuser, bypass = str(row[0]), bool(row[1]), bool(row[2])
    return superuser or bypass, name


def current_role_schema_privileges(engine: Engine) -> dict[str, bool | str]:
    """Inspect whether the connected role can mutate schema objects."""
    if engine.dialect.name != "postgresql":
        return {"dialect": engine.dialect.name, "create_on_public": False}
    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT current_user,
                       rolsuper,
                       rolcreatedb,
                       rolcreaterole,
                       has_schema_privilege(current_user, 'public', 'CREATE'),
                       has_schema_privilege(current_user, 'public', 'USAGE'),
                       has_database_privilege(
                           current_user, current_database(), 'CREATE'
                       )
                FROM pg_roles WHERE rolname = current_user
                """
            )
        ).one()
    return {
        "role": str(row[0]),
        "superuser": bool(row[1]),
        "createdb": bool(row[2]),
        "createrole": bool(row[3]),
        "create_on_public": bool(row[4]),
        "usage_on_public": bool(row[5]),
        "create_on_database": bool(row[6]),
    }


def provision_application_role(
    engine: Engine,
    *,
    role: str = APPLICATION_ROLE,
    password: str = "fabric",
) -> None:
    """Create a NOSUPERUSER / NOBYPASSRLS role the gateway can use.

    Docker `POSTGRES_USER` is a superuser and bypasses RLS even with FORCE.
    This is a no-op when the connected user cannot create roles.

    The application role receives DML plus schema USAGE. It must not receive
    CREATE on `public`; Alembic runs as the migration/table-owner role.
    """
    if engine.dialect.name != "postgresql":
        return
    if not _ROLE_NAME.fullmatch(role):
        raise ConfigurationError(f"invalid application role name '{role}'")
    with engine.begin() as connection:
        capable = connection.execute(
            text("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
        ).scalar()
        if not capable:
            return
        exists = connection.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :role"),
            {"role": role},
        ).scalar()
        if not exists:
            # PostgreSQL DDL does not accept a bound parameter for PASSWORD.
            # The role name is restricted to [_a-z0-9]; the password is escaped
            # as a string literal so a quote cannot break out of the statement.
            escaped = password.replace("'", "''")
            connection.execute(
                text(
                    f"CREATE ROLE {role} LOGIN PASSWORD '{escaped}' "
                    "NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS"
                )
            )
        database = connection.execute(text("SELECT current_database()")).scalar()
        if database and _ROLE_NAME.fullmatch(str(database)):
            connection.execute(text(f"GRANT CONNECT ON DATABASE {database} TO {role}"))
        connection.execute(text(f"GRANT USAGE ON SCHEMA public TO {role}"))
        connection.execute(text(f"REVOKE CREATE ON SCHEMA public FROM {role}"))
        connection.execute(
            text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {role}")
        )
        connection.execute(text(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}"))
        connection.execute(
            text(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, "
                f"INSERT, UPDATE, DELETE ON TABLES TO {role}"
            )
        )


def _enable_row_level_security(engine: Engine) -> None:
    isolation = "tenant_id = current_setting('app.current_tenant', true)"
    usage_using = f"({isolation} OR current_setting('app.fabric_observe', true) = 'on')"
    tables = (
        ("tenant_records", isolation, isolation),
        ("tenants", isolation, isolation),
        ("users", isolation, isolation),
        ("audit_events", isolation, isolation),
        ("usage_events", usage_using, isolation),
    )
    with engine.begin() as connection:
        me = connection.execute(text("SELECT current_user")).scalar()
        for table, using, check in tables:
            owner = connection.execute(
                text(
                    "SELECT tableowner FROM pg_tables "
                    "WHERE schemaname = 'public' AND tablename = :table"
                ),
                {"table": table},
            ).scalar()
            if owner is None or owner != me:
                # Missing table, or the application role is not the owner.
                # ENABLE/FORCE and policy DDL require ownership.
                continue
            policy = f"{table}_isolation"
            for statement in (
                f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
                f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",
                f"DROP POLICY IF EXISTS {policy} ON {table}",
                f"CREATE POLICY {policy} ON {table} USING ({using}) WITH CHECK ({check})",
            ):
                connection.execute(text(statement))


def probe_database(url: str, *, timeout_s: float = STARTUP_PROBE_TIMEOUT_S) -> None:
    """Fail closed when the durable store cannot be reached.

    The probe engine is created, used for `SELECT 1`, and disposed. It is not
    the serving pool. The error must not include the DSN.
    """
    engine = None
    try:
        engine = create_database_engine(url, connect_timeout_s=timeout_s)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except ConfigurationError:
        raise
    except Exception:
        raise ConfigurationError(
            "PostgreSQL is unreachable; production requires a reachable LLM_FABRIC_DATABASE_URL"
        ) from None
    finally:
        if engine is not None:
            engine.dispose()


def bind_isolation(session: Session, tenant_id: str, *, observe: bool = False) -> None:
    """Transaction-local tenant (and optional fleet-observe) GUCs for RLS."""
    dialect = session.get_bind().dialect.name
    if dialect != "postgresql":
        return
    session.execute(
        text("SELECT set_config(:setting, :tenant, true)"),
        {"setting": _RLS_SETTING, "tenant": tenant_id},
    )
    session.execute(
        text("SELECT set_config(:setting, :value, true)"),
        {
            "setting": _OBSERVE_SETTING,
            "value": "on" if observe else "off",
        },
    )


def _bind_tenant(session: Session, tenant_id: str) -> None:
    bind_isolation(session, tenant_id, observe=False)


class PostgresTenantStore[T: TenantOwned](TenantScopedStore[T]):
    """Durable `TenantScopedStore` over `tenant_records`.

    Isolation is enforced twice: the query always includes `tenant_id`, and
    decoded records are re-checked against the requesting scope. On PostgreSQL,
    RLS is a third line that the application cannot forget.
    """

    def __init__(
        self,
        name: str,
        record_type: type[T],
        engine: Engine,
        *,
        max_records_per_tenant: int = 10_000,
        audit: IsolationAudit | None = None,
    ) -> None:
        super().__init__(
            name,
            max_records_per_tenant=max_records_per_tenant,
            audit=audit,
        )
        self._record_type = record_type
        self._engine = engine
        # Drop the unused in-memory buckets so a bug cannot silently write there.
        self._buckets.clear()

    def put(self, scope: TenantScope, key: str, value: T) -> T:
        if value.tenant_id != scope.tenant_id:
            self._flag(scope.tenant_id, value.tenant_id, key)
            raise TenantIsolationError(
                f"refusing to write a record owned by another tenant into {self.name}"
            )
        payload = encode(value)
        now = time.time()
        with Session(self._engine) as session:
            _bind_tenant(session, scope.tenant_id)
            row = session.get(
                TenantRecordRow, {"store": self.name, "tenant_id": scope.tenant_id, "key": key}
            )
            if row is None:
                row = TenantRecordRow(
                    store=self.name,
                    tenant_id=scope.tenant_id,
                    key=key,
                    payload=payload,
                    updated_at=now,
                )
                session.add(row)
            else:
                if row.tenant_id != scope.tenant_id:
                    self._flag(scope.tenant_id, row.tenant_id, key)
                    raise TenantIsolationError(f"tenant boundary crossed inside {self.name}")
                row.payload = payload
                row.updated_at = now
            session.flush()
            self._evict_overflow(session, scope.tenant_id)
            session.commit()
        return value

    def delete(self, scope: TenantScope, key: str) -> bool:
        with Session(self._engine) as session:
            _bind_tenant(session, scope.tenant_id)
            result = session.execute(
                delete(TenantRecordRow).where(
                    TenantRecordRow.store == self.name,
                    TenantRecordRow.tenant_id == scope.tenant_id,
                    TenantRecordRow.key == key,
                )
            )
            session.commit()
            return bool(getattr(result, "rowcount", 0))

    def clear_tenant(self, scope: TenantScope) -> int:
        with Session(self._engine) as session:
            _bind_tenant(session, scope.tenant_id)
            result = session.execute(
                delete(TenantRecordRow).where(
                    TenantRecordRow.store == self.name,
                    TenantRecordRow.tenant_id == scope.tenant_id,
                )
            )
            session.commit()
            return int(getattr(result, "rowcount", 0) or 0)

    def get(self, scope: TenantScope, key: str) -> T | None:
        with Session(self._engine) as session:
            _bind_tenant(session, scope.tenant_id)
            row = session.execute(
                select(TenantRecordRow).where(
                    TenantRecordRow.store == self.name,
                    TenantRecordRow.tenant_id == scope.tenant_id,
                    TenantRecordRow.key == key,
                )
            ).scalar_one_or_none()
        if row is None:
            return None
        if row.tenant_id != scope.tenant_id:
            self._flag(scope.tenant_id, row.tenant_id, key)
            raise TenantIsolationError(f"tenant boundary crossed inside {self.name}")
        value = decode(self._record_type, row.payload)
        if value.tenant_id != scope.tenant_id:
            self._flag(scope.tenant_id, value.tenant_id, key)
            raise TenantIsolationError(f"tenant boundary crossed inside {self.name}")
        return value  # type: ignore[no-any-return]

    def require(self, scope: TenantScope, key: str) -> T:
        value = self.get(scope, key)
        if value is None:
            raise ResourceNotFoundError(f"no such {self.name}: '{key}'")
        return value

    def list(
        self,
        scope: TenantScope,
        *,
        limit: int = 100,
        predicate: Callable[[T], bool] | None = None,
    ) -> builtins.list[T]:
        if limit <= 0:
            return []
        with Session(self._engine) as session:
            _bind_tenant(session, scope.tenant_id)
            rows = (
                session.execute(
                    select(TenantRecordRow)
                    .where(
                        TenantRecordRow.store == self.name,
                        TenantRecordRow.tenant_id == scope.tenant_id,
                    )
                    .order_by(TenantRecordRow.updated_at.desc())
                    .limit(self._max_records)
                )
                .scalars()
                .all()
            )
        results: builtins.list[T] = []
        for row in rows:
            if row.tenant_id != scope.tenant_id:
                self._flag(scope.tenant_id, row.tenant_id, "<list>")
                raise TenantIsolationError(f"tenant boundary crossed inside {self.name}")
            value = decode(self._record_type, row.payload)
            if value.tenant_id != scope.tenant_id:
                self._flag(scope.tenant_id, value.tenant_id, "<list>")
                raise TenantIsolationError(f"tenant boundary crossed inside {self.name}")
            if predicate is not None and not predicate(value):
                continue
            results.append(value)
            if len(results) >= limit:
                break
        return results

    def keys(self, scope: TenantScope) -> builtins.list[str]:
        with Session(self._engine) as session:
            _bind_tenant(session, scope.tenant_id)
            rows = (
                session.execute(
                    select(TenantRecordRow.key).where(
                        TenantRecordRow.store == self.name,
                        TenantRecordRow.tenant_id == scope.tenant_id,
                    )
                )
                .scalars()
                .all()
            )
        return list(rows)

    def count(self, scope: TenantScope) -> int:
        with Session(self._engine) as session:
            _bind_tenant(session, scope.tenant_id)
            total = session.execute(
                select(func.count())
                .select_from(TenantRecordRow)
                .where(
                    TenantRecordRow.store == self.name,
                    TenantRecordRow.tenant_id == scope.tenant_id,
                )
            ).scalar_one()
            return int(total)

    def _evict_overflow(self, session: Session, tenant_id: str) -> None:
        rows = (
            session.execute(
                select(TenantRecordRow)
                .where(
                    TenantRecordRow.store == self.name,
                    TenantRecordRow.tenant_id == tenant_id,
                )
                .order_by(TenantRecordRow.updated_at.asc())
            )
            .scalars()
            .all()
        )
        extra = len(rows) - self._max_records
        for row in rows[: max(0, extra)]:
            session.delete(row)
