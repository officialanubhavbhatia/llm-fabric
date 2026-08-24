"""Fixtures for the adversarial suite.

Two tenants throughout: `victim` owns the data, `attacker` tries to reach it.
Naming them for their role rather than "tenant_a"/"tenant_b" keeps each
assertion's intent legible at a glance.
"""

from __future__ import annotations

import pytest

from llm_fabric.storage.repositories import TenantStores
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope

VICTIM_TENANT = "tenant-a-acme"
ATTACKER_TENANT = "tenant-b-mallory"


@pytest.fixture
def stores() -> TenantStores:
    return TenantStores()


@pytest.fixture
def cache(stores: TenantStores) -> TenantScopedCache:
    return TenantScopedCache(audit=stores.audit)


@pytest.fixture
def victim() -> TenantScope:
    return TenantScope(tenant_id=VICTIM_TENANT, user_id="alice", project_id="proj-1")


@pytest.fixture
def attacker() -> TenantScope:
    return TenantScope(tenant_id=ATTACKER_TENANT, user_id="mallory", project_id="proj-9")


@pytest.fixture
def attacker_same_user_id() -> TenantScope:
    """An attacker who happens to share the victim's user id.

    Two tenants may legitimately contain a user called `alice`. Anything keyed on
    user id alone would collide here.
    """
    return TenantScope(tenant_id=ATTACKER_TENANT, user_id="alice", project_id="proj-1")
