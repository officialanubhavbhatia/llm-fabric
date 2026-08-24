"""Revocation denylist: a stolen token must stop working before it expires."""

from __future__ import annotations

import time

import pytest

from llm_fabric.errors import AuthenticationError
from llm_fabric.identity.apikey import ApiCredential, ApiKeyVerifier
from llm_fabric.identity.dev import DevIdentityProvider
from llm_fabric.identity.revocation import (
    InMemoryRevocationStore,
    RevokingVerifier,
    token_fingerprint,
)

SECRET = "development-secret-that-is-long-enough"


@pytest.fixture
def issuer() -> DevIdentityProvider:
    return DevIdentityProvider(secret=SECRET)


@pytest.fixture
def store() -> InMemoryRevocationStore:
    return InMemoryRevocationStore()


async def test_an_unrevoked_token_is_accepted(
    issuer: DevIdentityProvider, store: InMemoryRevocationStore
) -> None:
    verifier = RevokingVerifier(issuer, store)
    token = issuer.issue_token(tenant_id="acme", user_id="alice")

    principal = await verifier.verify(token)

    assert principal.tenant_id == "acme"


async def test_a_revoked_jti_is_refused(
    issuer: DevIdentityProvider, store: InMemoryRevocationStore
) -> None:
    token = issuer.issue_token(tenant_id="acme", user_id="alice")
    principal = await issuer.verify(token)
    assert principal.token_id is not None
    store.revoke(token_id=principal.token_id, expires_at=time.time() + 600)

    verifier = RevokingVerifier(issuer, store)

    with pytest.raises(AuthenticationError, match="revoked"):
        await verifier.verify(token)


async def test_a_revoked_fingerprint_is_refused(
    issuer: DevIdentityProvider, store: InMemoryRevocationStore
) -> None:
    token = issuer.issue_token(tenant_id="acme", user_id="alice")
    store.revoke(fingerprint=token_fingerprint(token), expires_at=time.time() + 600)

    verifier = RevokingVerifier(issuer, store)

    with pytest.raises(AuthenticationError, match="revoked"):
        await verifier.verify(token)


async def test_a_revoked_api_key_is_refused(store: InMemoryRevocationStore) -> None:
    key = "a-long-enough-api-key"
    inner = ApiKeyVerifier([ApiCredential(key=key, tenant_id="acme")])
    store.revoke(fingerprint=token_fingerprint(key))

    with pytest.raises(AuthenticationError, match="revoked"):
        await RevokingVerifier(inner, store).verify(key)


async def test_an_expired_revocation_stops_binding(
    issuer: DevIdentityProvider, store: InMemoryRevocationStore
) -> None:
    token = issuer.issue_token(tenant_id="acme", user_id="alice")
    principal = await issuer.verify(token)
    assert principal.token_id is not None
    store.revoke(token_id=principal.token_id, expires_at=time.time() - 1)

    principal = await RevokingVerifier(issuer, store).verify(token)

    assert principal.tenant_id == "acme"


async def test_insufficient_scope_is_refused(
    issuer: DevIdentityProvider, store: InMemoryRevocationStore
) -> None:
    token = issuer.issue_token(tenant_id="acme", user_id="alice", scopes=["chat:write"])
    verifier = RevokingVerifier(issuer, store, required_scopes=["fabric:admin"])

    with pytest.raises(AuthenticationError, match="required scope"):
        await verifier.verify(token)


def test_the_denylist_evicts_the_oldest_entry_at_its_cap() -> None:
    store = InMemoryRevocationStore(max_entries=2)
    store.revoke(token_id="first")
    store.revoke(token_id="second")
    store.revoke(token_id="third")

    assert store.is_revoked(token_id="first") is False
    assert store.is_revoked(token_id="second") is True
    assert store.is_revoked(token_id="third") is True


async def test_a_token_with_the_required_scope_is_accepted(
    issuer: DevIdentityProvider, store: InMemoryRevocationStore
) -> None:
    token = issuer.issue_token(tenant_id="acme", user_id="alice", scopes=["fabric:admin"])
    verifier = RevokingVerifier(issuer, store, required_scopes=["fabric:admin"])

    principal = await verifier.verify(token)

    assert principal.has_scope("fabric:admin")
