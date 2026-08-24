"""Adversarial tests against token validation and tenant resolution.

Isolation is only as strong as the identity it is derived from. If a caller can
forge, replay or reshape a token, the storage guarantees are irrelevant, so these
carry the same `isolation` marker and the same CI gate.
"""

from __future__ import annotations

import time

import jwt
import pytest

from llm_fabric.errors import AuthenticationError, ConfigurationError
from llm_fabric.identity.claims import DELEGATION_SCOPE, ClaimsMapping, Principal, RawClaims
from llm_fabric.identity.dev import DEV_AUDIENCE, DEV_ISSUER, DevIdentityProvider
from llm_fabric.identity.revocation import InMemoryRevocationStore, RevokingVerifier

pytestmark = pytest.mark.isolation

SECRET = "development-secret-that-is-long-enough"
OTHER_SECRET = "a-completely-different-secret-of-length"


@pytest.fixture
def provider() -> DevIdentityProvider:
    return DevIdentityProvider(secret=SECRET)


def _payload(**overrides: object) -> dict[str, object]:
    now = int(time.time())
    base: dict[str, object] = {
        "iss": DEV_ISSUER,
        "aud": DEV_AUDIENCE,
        "sub": "user-1",
        "iat": now,
        "nbf": now,
        "exp": now + 600,
        "tenant_id": "tenant-a-acme",
        "user_id": "alice",
    }
    base.update(overrides)
    return base


# -- forgery -----------------------------------------------------------------


async def test_a_token_signed_with_the_wrong_key_is_refused(
    provider: DevIdentityProvider,
) -> None:
    forged = jwt.encode(_payload(tenant_id="tenant-a-acme"), OTHER_SECRET, algorithm="HS256")

    with pytest.raises(AuthenticationError):
        await provider.verify(forged)


async def test_an_unsigned_token_is_refused(provider: DevIdentityProvider) -> None:
    """`alg: none` is the oldest JWT attack and must never be honoured."""
    unsigned = jwt.encode(_payload(), key="", algorithm="none")

    with pytest.raises(AuthenticationError):
        await provider.verify(unsigned)


async def test_a_tampered_payload_is_refused(provider: DevIdentityProvider) -> None:
    """Swapping the tenant in a valid token must invalidate the signature."""
    import base64
    import json

    token = provider.issue_token(tenant_id="tenant-a-acme", user_id="alice")
    header, payload, signature = token.split(".")

    decoded = json.loads(base64.urlsafe_b64decode(payload + "=="))
    decoded["tenant_id"] = "tenant-b-mallory"
    tampered_payload = base64.urlsafe_b64encode(json.dumps(decoded).encode()).rstrip(b"=").decode()

    with pytest.raises(AuthenticationError):
        await provider.verify(f"{header}.{tampered_payload}.{signature}")


async def test_garbage_is_refused_without_crashing(provider: DevIdentityProvider) -> None:
    for candidate in ("", "not-a-token", "a.b", "a.b.c", "..", "Bearer x"):
        with pytest.raises(AuthenticationError):
            await provider.verify(candidate)


# -- lifetime and audience ---------------------------------------------------


async def test_an_expired_token_is_refused(provider: DevIdentityProvider) -> None:
    expired = jwt.encode(
        _payload(exp=int(time.time()) - 3600, iat=int(time.time()) - 7200),
        SECRET,
        algorithm="HS256",
    )

    with pytest.raises(AuthenticationError):
        await provider.verify(expired)


async def test_a_token_for_another_audience_is_refused(provider: DevIdentityProvider) -> None:
    """A token minted for a different service must not be replayable here."""
    wrong_audience = jwt.encode(_payload(aud="some-other-service"), SECRET, algorithm="HS256")

    with pytest.raises(AuthenticationError):
        await provider.verify(wrong_audience)


async def test_a_token_from_another_issuer_is_refused(provider: DevIdentityProvider) -> None:
    foreign = jwt.encode(_payload(iss="https://evil.example"), SECRET, algorithm="HS256")

    with pytest.raises(AuthenticationError):
        await provider.verify(foreign)


async def test_a_token_used_before_nbf_is_refused(provider: DevIdentityProvider) -> None:
    future = int(time.time()) + 3600
    early = jwt.encode(
        _payload(nbf=future, iat=future, exp=future + 600),
        SECRET,
        algorithm="HS256",
    )

    with pytest.raises(AuthenticationError):
        await provider.verify(early)


async def test_a_revoked_token_is_refused(provider: DevIdentityProvider) -> None:
    token = provider.issue_token(tenant_id="tenant-a-acme", user_id="alice")
    principal = await provider.verify(token)
    store = InMemoryRevocationStore()
    store.revoke(token_id=principal.token_id)

    with pytest.raises(AuthenticationError, match="revoked"):
        await RevokingVerifier(provider, store).verify(token)


async def test_a_token_missing_a_required_scope_is_refused(
    provider: DevIdentityProvider,
) -> None:
    token = provider.issue_token(tenant_id="tenant-a-acme", user_id="alice", scopes=["chat:write"])
    verifier = RevokingVerifier(
        provider, InMemoryRevocationStore(), required_scopes=["fabric:admin"]
    )

    with pytest.raises(AuthenticationError, match="required scope"):
        await verifier.verify(token)


# -- claim integrity ---------------------------------------------------------


async def test_a_token_without_a_tenant_claim_is_refused(
    provider: DevIdentityProvider,
) -> None:
    """No tenant means no scope. Defaulting one would silently merge customers."""
    payload = _payload()
    del payload["tenant_id"]
    tokenless_tenant = jwt.encode(payload, SECRET, algorithm="HS256")

    with pytest.raises(AuthenticationError, match="tenant_id"):
        await provider.verify(tokenless_tenant)


async def test_an_empty_tenant_claim_is_refused(provider: DevIdentityProvider) -> None:
    blank = jwt.encode(_payload(tenant_id="   "), SECRET, algorithm="HS256")

    with pytest.raises(AuthenticationError):
        await provider.verify(blank)


async def test_a_valid_token_yields_the_expected_principal(
    provider: DevIdentityProvider,
) -> None:
    """Control: the refusals above are meaningless if nothing is ever accepted."""
    token = provider.issue_token(
        tenant_id="tenant-a-acme",
        user_id="alice",
        project_id="proj-1",
        roles=["admin"],
        scopes=["chat:write", "usage:read"],
    )

    principal = await provider.verify(token)

    assert principal.tenant_id == "tenant-a-acme"
    assert principal.user_id == "alice"
    assert principal.project_id == "proj-1"
    assert principal.has_role("admin")
    assert principal.has_scope("chat:write")
    assert not principal.has_scope("admin:everything")


# -- delegation --------------------------------------------------------------


def _principal(**overrides: object) -> Principal:
    base: dict[str, object] = {
        "tenant_id": "tenant-a-acme",
        "user_id": "alice",
        "subject": "user-1",
        "issuer": DEV_ISSUER,
    }
    base.update(overrides)
    return Principal(**base)  # type: ignore[arg-type]


def test_delegation_is_refused_without_the_scope() -> None:
    """The whole point: a header must not be able to change the tenant."""
    principal = _principal(scopes=frozenset({"chat:write"}))

    with pytest.raises(AuthenticationError):
        principal.delegate_to("tenant-b-mallory")


def test_delegation_requires_the_exact_scope() -> None:
    near_miss = _principal(scopes=frozenset({"fabric:delegate", "delegate_tenant"}))

    with pytest.raises(AuthenticationError):
        near_miss.delegate_to("tenant-b-mallory")


def test_delegation_records_the_original_tenant() -> None:
    principal = _principal(scopes=frozenset({DELEGATION_SCOPE}))

    delegated = principal.delegate_to("tenant-b-mallory")

    assert delegated.tenant_id == "tenant-b-mallory"
    assert delegated.delegated_from == "tenant-a-acme"
    assert delegated.audit_fields()["delegated_from"] == "tenant-a-acme"


def test_delegation_to_an_empty_tenant_is_refused() -> None:
    principal = _principal(scopes=frozenset({DELEGATION_SCOPE}))

    with pytest.raises(AuthenticationError):
        principal.delegate_to("")


# -- privacy -----------------------------------------------------------------


def test_audit_fields_never_carry_credentials() -> None:
    principal = _principal(token_id="abc123", scopes=frozenset({"chat:write"}))

    fields = principal.audit_fields()

    assert "scopes" not in fields
    assert "roles" not in fields
    assert set(fields) <= {
        "tenant_id",
        "user_id",
        "subject",
        "issuer",
        "project_id",
        "token_id",
        "delegated_from",
    }


# -- configuration guards ----------------------------------------------------


def test_dev_provider_refuses_a_short_secret() -> None:
    with pytest.raises(ConfigurationError, match="at least"):
        DevIdentityProvider(secret="short")


def test_scope_claim_accepts_both_oauth_shapes() -> None:
    """OAuth2 sends space-delimited scopes; many issuers send a list."""
    mapping = ClaimsMapping()
    spaced = RawClaims(
        payload=_payload(scope="a b c"), issuer=DEV_ISSUER, mapping=mapping
    ).to_principal()
    listed = RawClaims(
        payload=_payload(scope=["a", "b", "c"]), issuer=DEV_ISSUER, mapping=mapping
    ).to_principal()

    assert spaced.scopes == listed.scopes == frozenset({"a", "b", "c"})
