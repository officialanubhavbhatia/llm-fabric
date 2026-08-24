"""Cache behaviour: expiry, bounds, counters and key derivation."""

from __future__ import annotations

import pytest

from llm_fabric.tenancy.cache import (
    CacheKey,
    CacheNamespace,
    CachePolicy,
    TenantScopedCache,
)
from llm_fabric.tenancy.scope import TenantScope


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def scope() -> TenantScope:
    return TenantScope(tenant_id="acme", user_id="alice")


def test_a_value_round_trips(clock: FakeClock, scope: TenantScope) -> None:
    cache = TenantScopedCache(clock=clock)
    cache.put(scope, CacheNamespace.EXACT_RESPONSE, {"q": "hello"}, "world")

    assert cache.get(scope, CacheNamespace.EXACT_RESPONSE, {"q": "hello"}) == "world"


def test_an_entry_expires(clock: FakeClock, scope: TenantScope) -> None:
    cache = TenantScopedCache(
        {CacheNamespace.EXACT_RESPONSE: CachePolicy(ttl_seconds=10.0)}, clock=clock
    )
    cache.put(scope, CacheNamespace.EXACT_RESPONSE, {"q": "hello"}, "world")

    clock.advance(11)

    assert cache.get(scope, CacheNamespace.EXACT_RESPONSE, {"q": "hello"}) is None
    assert cache.stats(CacheNamespace.EXACT_RESPONSE).expired == 1


def test_namespaces_carry_independent_ttls(clock: FakeClock, scope: TenantScope) -> None:
    cache = TenantScopedCache(
        {
            CacheNamespace.EXACT_RESPONSE: CachePolicy(ttl_seconds=100.0),
            CacheNamespace.SEMANTIC_RESPONSE: CachePolicy(ttl_seconds=5.0),
        },
        clock=clock,
    )
    parts = {"q": "hello"}
    cache.put(scope, CacheNamespace.EXACT_RESPONSE, parts, "exact")
    cache.put(scope, CacheNamespace.SEMANTIC_RESPONSE, parts, "semantic")

    clock.advance(10)

    assert cache.get(scope, CacheNamespace.EXACT_RESPONSE, parts) == "exact"
    assert cache.get(scope, CacheNamespace.SEMANTIC_RESPONSE, parts) is None


def test_a_disabled_namespace_never_stores(clock: FakeClock, scope: TenantScope) -> None:
    cache = TenantScopedCache(
        {CacheNamespace.INTENT: CachePolicy(ttl_seconds=60.0, enabled=False)}, clock=clock
    )
    cache.put(scope, CacheNamespace.INTENT, {"q": "hello"}, "value")

    assert cache.get(scope, CacheNamespace.INTENT, {"q": "hello"}) is None


def test_entries_are_bounded_per_tenant(clock: FakeClock, scope: TenantScope) -> None:
    cache = TenantScopedCache(
        {CacheNamespace.EXACT_RESPONSE: CachePolicy(ttl_seconds=600.0, max_entries_per_tenant=5)},
        clock=clock,
    )
    for index in range(50):
        cache.put(scope, CacheNamespace.EXACT_RESPONSE, {"q": index}, index)

    # The oldest were evicted; the newest survive.
    assert cache.get(scope, CacheNamespace.EXACT_RESPONSE, {"q": 0}) is None
    assert cache.get(scope, CacheNamespace.EXACT_RESPONSE, {"q": 49}) == 49


def test_counters_track_hits_and_misses(clock: FakeClock, scope: TenantScope) -> None:
    cache = TenantScopedCache(clock=clock)
    cache.put(scope, CacheNamespace.EXACT_RESPONSE, {"q": "a"}, 1)

    cache.get(scope, CacheNamespace.EXACT_RESPONSE, {"q": "a"})
    cache.get(scope, CacheNamespace.EXACT_RESPONSE, {"q": "b"})

    stats = cache.stats(CacheNamespace.EXACT_RESPONSE)
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.writes == 1
    assert stats.hit_rate == pytest.approx(0.5)


def test_an_unused_cache_reports_no_hit_rate(clock: FakeClock) -> None:
    """`None`, not zero. "No data" and "never hits" are different claims."""
    cache = TenantScopedCache(clock=clock)

    assert cache.stats(CacheNamespace.EXACT_RESPONSE).hit_rate is None


def test_key_order_does_not_change_the_fingerprint(scope: TenantScope) -> None:
    first = CacheKey.build(CacheNamespace.INTENT, scope, {"a": 1, "b": 2})
    second = CacheKey.build(CacheNamespace.INTENT, scope, {"b": 2, "a": 1})

    assert first.fingerprint == second.fingerprint


def test_every_discriminator_changes_the_key(scope: TenantScope) -> None:
    """Cache keys must include everything that changes the answer."""
    base = {"prompt": "p", "taxonomy_version": "v1", "policy_version": "v1"}
    baseline = CacheKey.build(CacheNamespace.INTENT, scope, base)

    for field in base:
        altered = {**base, field: "changed"}
        assert CacheKey.build(CacheNamespace.INTENT, scope, altered).fingerprint != (
            baseline.fingerprint
        )


def test_invalidating_one_entry_leaves_the_rest(clock: FakeClock, scope: TenantScope) -> None:
    cache = TenantScopedCache(clock=clock)
    cache.put(scope, CacheNamespace.EXACT_RESPONSE, {"q": "a"}, 1)
    cache.put(scope, CacheNamespace.EXACT_RESPONSE, {"q": "b"}, 2)

    assert cache.invalidate(scope, CacheNamespace.EXACT_RESPONSE, {"q": "a"}) is True

    assert cache.get(scope, CacheNamespace.EXACT_RESPONSE, {"q": "a"}) is None
    assert cache.get(scope, CacheNamespace.EXACT_RESPONSE, {"q": "b"}) == 2
