"""Assemble durable and distributed collaborators from settings.

Development and tests keep in-memory stores when URLs are unset. Production
refuses to start unless both PostgreSQL and Redis are configured and reachable.
"""

from __future__ import annotations

from typing import Any

from llm_fabric.config import Settings
from llm_fabric.errors import ConfigurationError
from llm_fabric.identity.revocation import InMemoryRevocationStore, TokenRevocationStore
from llm_fabric.observability.analytics import BufferedAnalyticsSink, NullAnalyticsSink
from llm_fabric.router.health import HealthTracker
from llm_fabric.storage.postgres import create_database_engine, init_schema, probe_database
from llm_fabric.storage.redis import (
    RedisHealthTracker,
    RedisQuotaLedger,
    RedisRevocationStore,
    connect_redis,
    probe_redis,
)
from llm_fabric.storage.repositories import TenantStores
from llm_fabric.storage.schema import assert_schema_revision
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.quota import QuotaLedger
from llm_fabric.tenancy.store import IsolationAudit


def probe_distributed_state(settings: Settings) -> None:
    """Reachability of the URLs the serving runtime will use.

    Configuration presence is `validate_startup`. This function talks to the
    network. A missing URL here is a probe failure, not a skipped check.
    Production also confirms the database is already at this build's Alembic
    head. Workers never run migrations.
    """
    if not settings.database_url:
        raise ConfigurationError(
            "production startup validation failed: PostgreSQL is not configured"
        )
    if not settings.redis_url:
        raise ConfigurationError("production startup validation failed: Redis is not configured")
    probe_database(settings.database_url)
    probe_redis(settings.redis_url)
    assert_schema_revision(settings.database_url)


def build_engine(settings: Settings) -> Any | None:
    if not settings.database_url:
        return None
    engine = create_database_engine(settings.database_url)
    if settings.environment == "production":
        # Alembic already ran in a migration job. Workers must not DDL.
        return engine
    init_schema(engine)
    return engine


def build_redis(settings: Settings) -> Any | None:
    if not settings.redis_url:
        return None
    timeout = settings.health_probe_timeout_s
    return connect_redis(
        settings.redis_url,
        socket_connect_timeout=timeout,
        socket_timeout=timeout,
    )


def build_stores(settings: Settings, *, engine: Any | None = None) -> TenantStores:
    return TenantStores(
        engine=engine
        if engine is not None
        else (build_engine(settings) if settings.database_url else None)
    )


def build_quota(
    settings: Settings, *, redis_client: Any | None = None
) -> QuotaLedger | RedisQuotaLedger:
    if redis_client is not None:
        return RedisQuotaLedger(
            redis_client,
            default_tenant_policy=settings.tenant_quota_policy,
            default_user_policy=settings.user_quota_policy,
        )
    return QuotaLedger(
        default_tenant_policy=settings.tenant_quota_policy,
        default_user_policy=settings.user_quota_policy,
    )


def build_health(settings: Settings, *, redis_client: Any | None = None, policy: Any) -> Any:
    if redis_client is not None:
        return RedisHealthTracker(redis_client, policy=policy)
    return HealthTracker(policy=policy)


def build_revocation(
    settings: Settings, *, redis_client: Any | None = None
) -> TokenRevocationStore:
    if redis_client is not None:
        return RedisRevocationStore(
            redis_client,
            fail_closed=settings.environment == "production",
        )
    return InMemoryRevocationStore()


def build_cache(audit: IsolationAudit) -> TenantScopedCache:
    return TenantScopedCache(audit=audit)


def build_analytics(settings: Settings) -> BufferedAnalyticsSink | NullAnalyticsSink:
    if settings.analytics_url:
        # A real ClickHouse drain is not on the request path. The buffer is
        # what serving uses; a separate process would ship events.
        return BufferedAnalyticsSink()
    return NullAnalyticsSink()
