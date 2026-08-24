"""Tenant-scoped store mechanics: bounds, ordering and scope validation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from llm_fabric.errors import ConfigurationError, ResourceNotFoundError
from llm_fabric.tenancy.scope import TenantScope
from llm_fabric.tenancy.store import TenantScopedStore


@dataclass(frozen=True, slots=True)
class Note:
    tenant_id: str
    body: str


@pytest.fixture
def store() -> TenantScopedStore[Note]:
    return TenantScopedStore("note", max_records_per_tenant=3)


@pytest.fixture
def scope() -> TenantScope:
    return TenantScope(tenant_id="acme", user_id="alice")


def test_a_scope_requires_a_tenant() -> None:
    with pytest.raises(ConfigurationError):
        TenantScope(tenant_id="")

    with pytest.raises(ConfigurationError):
        TenantScope(tenant_id="   ")


def test_the_user_key_is_namespaced_by_tenant() -> None:
    assert TenantScope(tenant_id="a", user_id="alice").user_key != (
        TenantScope(tenant_id="b", user_id="alice").user_key
    )


def test_records_round_trip(store: TenantScopedStore[Note], scope: TenantScope) -> None:
    store.put(scope, "n1", Note(tenant_id="acme", body="hello"))

    assert store.get(scope, "n1") == Note(tenant_id="acme", body="hello")
    assert store.count(scope) == 1


def test_require_raises_for_a_missing_record(
    store: TenantScopedStore[Note], scope: TenantScope
) -> None:
    with pytest.raises(ResourceNotFoundError):
        store.require(scope, "absent")


def test_records_are_bounded_per_tenant(store: TenantScopedStore[Note], scope: TenantScope) -> None:
    for index in range(10):
        store.put(scope, f"n{index}", Note(tenant_id="acme", body=str(index)))

    assert store.count(scope) == 3
    assert store.get(scope, "n0") is None
    assert store.get(scope, "n9") is not None


def test_one_tenant_cannot_evict_another(store: TenantScopedStore[Note]) -> None:
    """Bounds are per tenant, so a noisy tenant cannot flush a quiet one."""
    acme = TenantScope(tenant_id="acme")
    other = TenantScope(tenant_id="other")
    store.put(other, "keep", Note(tenant_id="other", body="precious"))

    for index in range(20):
        store.put(acme, f"n{index}", Note(tenant_id="acme", body=str(index)))

    assert store.get(other, "keep") is not None


def test_listing_is_newest_first(store: TenantScopedStore[Note], scope: TenantScope) -> None:
    store.put(scope, "a", Note(tenant_id="acme", body="first"))
    store.put(scope, "b", Note(tenant_id="acme", body="second"))

    assert [note.body for note in store.list(scope)] == ["second", "first"]


def test_listing_honours_the_limit(store: TenantScopedStore[Note], scope: TenantScope) -> None:
    for index in range(3):
        store.put(scope, f"n{index}", Note(tenant_id="acme", body=str(index)))

    assert len(store.list(scope, limit=2)) == 2
    assert store.list(scope, limit=0) == []


def test_listing_applies_a_predicate(store: TenantScopedStore[Note], scope: TenantScope) -> None:
    store.put(scope, "a", Note(tenant_id="acme", body="keep"))
    store.put(scope, "b", Note(tenant_id="acme", body="drop"))

    found = store.list(scope, predicate=lambda note: note.body == "keep")

    assert [note.body for note in found] == ["keep"]


def test_a_non_positive_bound_is_refused() -> None:
    with pytest.raises(ValueError):
        TenantScopedStore("note", max_records_per_tenant=0)
