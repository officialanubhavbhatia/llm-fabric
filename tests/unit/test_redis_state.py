"""Distributed quota, revocation and breaker state over fakeredis."""

from __future__ import annotations

import fakeredis
import pytest

from llm_fabric.errors import AuthenticationError, QuotaExceededError
from llm_fabric.identity.dev import DevIdentityProvider
from llm_fabric.identity.revocation import RevokingVerifier
from llm_fabric.storage.redis import RedisHealthTracker, RedisQuotaLedger, RedisRevocationStore
from llm_fabric.tenancy.quota import QuotaPolicy
from llm_fabric.tenancy.scope import TenantScope


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


def test_redis_quota_is_shared_across_ledger_instances(redis_client) -> None:
    policy = QuotaPolicy(requests_per_minute=3)
    first = RedisQuotaLedger(redis_client, default_tenant_policy=policy)
    second = RedisQuotaLedger(redis_client, default_tenant_policy=policy)
    scope = TenantScope(tenant_id="acme", user_id="alice")

    first.admit(scope)
    second.admit(scope)
    first.admit(scope)
    with pytest.raises(QuotaExceededError):
        second.admit(scope)


async def test_redis_revocation_is_visible_to_another_client(redis_client) -> None:
    issuer = DevIdentityProvider(secret="development-secret-that-is-long-enough")
    token = issuer.issue_token(tenant_id="acme", user_id="alice")
    principal = await issuer.verify(token)
    writer = RedisRevocationStore(redis_client)
    reader = RedisRevocationStore(redis_client)
    writer.revoke(token_id=principal.token_id)

    with pytest.raises(AuthenticationError, match="revoked"):
        await RevokingVerifier(issuer, reader).verify(token)


def test_redis_breaker_opens_for_every_client(redis_client) -> None:
    from llm_fabric.router.health import BreakerPolicy, BreakerState

    policy = BreakerPolicy(consecutive_failures=2, minimum_samples=1, open_duration_s=60)
    a = RedisHealthTracker(redis_client, policy=policy)
    b = RedisHealthTracker(redis_client, policy=policy)
    a.record_failure("gpt-x")
    a.record_failure("gpt-x")
    assert b.admits("gpt-x") is False
    assert b.snapshot("gpt-x").state is BreakerState.OPEN


def test_four_ledgers_share_one_quota(redis_client) -> None:
    policy = QuotaPolicy(requests_per_minute=10)
    ledgers = [RedisQuotaLedger(redis_client, default_tenant_policy=policy) for _ in range(4)]
    scope = TenantScope(tenant_id="acme", user_id="alice")
    for _ in range(10):
        ledgers[_ % 4].admit(scope)
    with pytest.raises(QuotaExceededError):
        ledgers[0].admit(scope)


def test_redis_cache_is_tenant_bound(redis_client) -> None:
    from llm_fabric.storage.redis import RedisCache
    from llm_fabric.tenancy.cache import CacheNamespace, TenantScopedCache

    backend = RedisCache(redis_client)
    cache = TenantScopedCache(redis_cache=backend)
    a = TenantScope(tenant_id="tenant-a", user_id="alice")
    b = TenantScope(tenant_id="tenant-b", user_id="bob")
    parts = {"prompt": "same"}
    cache.put(a, CacheNamespace.EXACT_RESPONSE, parts, {"answer": "a"})
    cache.put(b, CacheNamespace.EXACT_RESPONSE, parts, {"answer": "b"})
    assert cache.get(a, CacheNamespace.EXACT_RESPONSE, parts) == {"answer": "a"}
    assert cache.get(b, CacheNamespace.EXACT_RESPONSE, parts) == {"answer": "b"}
    assert cache.invalidate(b, CacheNamespace.EXACT_RESPONSE, parts) is True
    assert cache.get(a, CacheNamespace.EXACT_RESPONSE, parts) == {"answer": "a"}


def test_concurrency_is_shared(redis_client) -> None:
    policy = QuotaPolicy(max_concurrency=1)
    first = RedisQuotaLedger(redis_client, default_tenant_policy=policy)
    second = RedisQuotaLedger(redis_client, default_tenant_policy=policy)
    scope = TenantScope(tenant_id="acme", user_id="alice")
    first.acquire_concurrency(scope)
    with pytest.raises(QuotaExceededError):
        second.acquire_concurrency(scope)
    first.release_concurrency(scope)
    second.acquire_concurrency(scope)


def test_breaker_recovers_to_half_open(redis_client) -> None:
    from llm_fabric.router.health import BreakerPolicy, BreakerState

    clock = {"now": 0.0}

    def now() -> float:
        return clock["now"]

    policy = BreakerPolicy(consecutive_failures=1, minimum_samples=1, open_duration_s=1)
    tracker = RedisHealthTracker(redis_client, policy=policy, clock=now)
    tracker.record_failure("gpt-x")
    assert tracker.snapshot("gpt-x").state is BreakerState.OPEN
    clock["now"] = 2.0
    assert tracker.admits("gpt-x") is True
    assert tracker.snapshot("gpt-x").state is BreakerState.HALF_OPEN
