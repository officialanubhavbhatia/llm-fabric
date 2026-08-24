"""OIDC token validation against a stubbed identity provider.

An RSA key is generated per test module and served through an `httpx`
`MockTransport`, so the real verification path runs end to end with no network.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from llm_fabric.errors import AuthenticationError, ConfigurationError
from llm_fabric.identity.oidc import OIDCTokenVerifier

ISSUER = "https://issuer.example"
AUDIENCE = "myvista-llm-fabric"
KID = "test-key-1"


@pytest.fixture(scope="module")
def private_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def jwks(private_key: rsa.RSAPrivateKey) -> dict[str, Any]:
    import json

    jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk.update({"kid": KID, "alg": "RS256", "use": "sig"})
    return {"keys": [jwk]}


class Provider:
    """A stub issuer that counts how often its endpoints are hit."""

    def __init__(self, jwks: dict[str, Any]) -> None:
        self.jwks = jwks
        self.jwks_requests = 0
        self.discovery_requests = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/.well-known/openid-configuration"):
            self.discovery_requests += 1
            return httpx.Response(200, json={"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"})
        if request.url.path.endswith("/jwks"):
            self.jwks_requests += 1
            return httpx.Response(200, json=self.jwks)
        return httpx.Response(404)


@pytest.fixture
def provider(jwks: dict[str, Any]) -> Provider:
    return Provider(jwks)


@pytest.fixture
def verifier(provider: Provider) -> OIDCTokenVerifier:
    return OIDCTokenVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider.handler)),
        min_refresh_interval_seconds=0.0,
    )


def _token(private_key: rsa.RSAPrivateKey, **overrides: Any) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "user-1",
        "iat": now,
        "nbf": now,
        "jti": "token-1",
        "exp": now + 600,
        "tenant_id": "acme",
        "user_id": "alice",
        "scope": "chat:write",
        "roles": ["operator"],
    }
    payload.update(overrides)
    headers = {"kid": overrides.pop("kid", KID)}
    return jwt.encode(payload, private_key, algorithm="RS256", headers=headers)


# -- the happy path ----------------------------------------------------------


async def test_a_valid_token_is_accepted(
    verifier: OIDCTokenVerifier, private_key: rsa.RSAPrivateKey
) -> None:
    principal = await verifier.verify(_token(private_key))

    assert principal.tenant_id == "acme"
    assert principal.user_id == "alice"
    assert principal.issuer == ISSUER
    assert principal.has_scope("chat:write")
    assert principal.has_role("operator")


async def test_the_jwks_uri_is_discovered(
    verifier: OIDCTokenVerifier, provider: Provider, private_key: rsa.RSAPrivateKey
) -> None:
    await verifier.verify(_token(private_key))

    assert provider.discovery_requests == 1
    assert provider.jwks_requests == 1


async def test_signing_keys_are_cached(
    verifier: OIDCTokenVerifier, provider: Provider, private_key: rsa.RSAPrivateKey
) -> None:
    for _ in range(5):
        await verifier.verify(_token(private_key))

    assert provider.jwks_requests == 1


async def test_an_explicit_jwks_uri_skips_discovery(
    provider: Provider, private_key: rsa.RSAPrivateKey
) -> None:
    verifier = OIDCTokenVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_uri=f"{ISSUER}/jwks",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider.handler)),
    )

    await verifier.verify(_token(private_key))

    assert provider.discovery_requests == 0
    assert provider.jwks_requests == 1


# -- refusals ----------------------------------------------------------------


async def test_a_symmetric_token_is_refused(verifier: OIDCTokenVerifier) -> None:
    """Algorithm confusion.

    A JWKS is public. If the verifier accepted HS256, anyone could sign a token
    using the published modulus as the HMAC secret and mint any identity. The
    algorithm is checked before any key is fetched.
    """
    forged = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "attacker", "exp": int(time.time()) + 600},
        "public-key-material-used-as-a-secret",
        algorithm="HS256",
        headers={"kid": KID},
    )

    with pytest.raises(AuthenticationError, match="unsupported signing algorithm"):
        await verifier.verify(forged)


async def test_an_unsigned_token_is_refused(verifier: OIDCTokenVerifier) -> None:
    unsigned = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "attacker", "exp": int(time.time()) + 600},
        key="",
        algorithm="none",
    )

    with pytest.raises(AuthenticationError):
        await verifier.verify(unsigned)


async def test_a_token_signed_by_an_unknown_key_is_refused(
    verifier: OIDCTokenVerifier,
) -> None:
    stranger = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "attacker", "exp": int(time.time()) + 600},
        stranger,
        algorithm="RS256",
        headers={"kid": "some-other-kid"},
    )

    with pytest.raises(AuthenticationError):
        await verifier.verify(forged)


async def test_an_expired_token_is_refused(
    verifier: OIDCTokenVerifier, private_key: rsa.RSAPrivateKey
) -> None:
    with pytest.raises(AuthenticationError, match="expired"):
        await verifier.verify(_token(private_key, exp=int(time.time()) - 3600))


async def test_a_token_for_another_audience_is_refused(
    verifier: OIDCTokenVerifier, private_key: rsa.RSAPrivateKey
) -> None:
    with pytest.raises(AuthenticationError):
        await verifier.verify(_token(private_key, aud="another-service"))


async def test_a_token_from_another_issuer_is_refused(
    verifier: OIDCTokenVerifier, private_key: rsa.RSAPrivateKey
) -> None:
    with pytest.raises(AuthenticationError):
        await verifier.verify(_token(private_key, iss="https://evil.example"))


async def test_a_token_used_before_nbf_is_refused(
    verifier: OIDCTokenVerifier, private_key: rsa.RSAPrivateKey
) -> None:
    future = int(time.time()) + 3600
    with pytest.raises(AuthenticationError):
        await verifier.verify(_token(private_key, nbf=future, iat=future, exp=future + 600))


async def test_a_token_without_a_tenant_claim_is_refused(
    verifier: OIDCTokenVerifier, private_key: rsa.RSAPrivateKey
) -> None:
    token = _token(private_key)
    payload = jwt.decode(token, options={"verify_signature": False})
    del payload["tenant_id"]
    retokenized = jwt.encode(payload, private_key, algorithm="RS256", headers={"kid": KID})

    with pytest.raises(AuthenticationError, match="tenant_id"):
        await verifier.verify(retokenized)


async def test_malformed_input_is_refused(verifier: OIDCTokenVerifier) -> None:
    for candidate in ("", "nope", "a.b", "....", "null"):
        with pytest.raises(AuthenticationError):
            await verifier.verify(candidate)


# -- resilience --------------------------------------------------------------


async def test_jwks_refresh_is_rate_limited(
    provider: Provider, private_key: rsa.RSAPrivateKey
) -> None:
    """An unknown `kid` is attacker-controlled and must not amplify traffic."""
    verifier = OIDCTokenVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_uri=f"{ISSUER}/jwks",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider.handler)),
        min_refresh_interval_seconds=60.0,
    )
    stranger = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    for index in range(25):
        junk = jwt.encode(
            {"iss": ISSUER, "aud": AUDIENCE, "sub": "x", "exp": int(time.time()) + 600},
            stranger,
            algorithm="RS256",
            headers={"kid": f"unknown-{index}"},
        )
        with pytest.raises(AuthenticationError):
            await verifier.verify(junk)

    assert provider.jwks_requests == 1


async def test_an_unreachable_provider_is_a_refusal_not_a_crash(
    private_key: rsa.RSAPrivateKey,
) -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("provider is down")

    verifier = OIDCTokenVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_uri=f"{ISSUER}/jwks",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(unreachable)),
    )

    with pytest.raises(AuthenticationError, match="unreachable"):
        await verifier.verify(_token(private_key))


async def test_a_malformed_jwks_is_a_refusal(private_key: rsa.RSAPrivateKey) -> None:
    def malformed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    verifier = OIDCTokenVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_uri=f"{ISSUER}/jwks",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(malformed)),
    )

    with pytest.raises(AuthenticationError):
        await verifier.verify(_token(private_key))


def test_the_verifier_refuses_to_build_without_an_issuer() -> None:
    with pytest.raises(ConfigurationError):
        OIDCTokenVerifier(issuer="", audience=AUDIENCE)


def test_the_verifier_refuses_to_build_without_an_audience() -> None:
    """An unpinned audience makes any token from the issuer replayable here."""
    with pytest.raises(ConfigurationError):
        OIDCTokenVerifier(issuer=ISSUER, audience="")


async def test_warmup_fetches_signing_keys(verifier: OIDCTokenVerifier, provider: Provider) -> None:
    await verifier.warmup()
    assert provider.jwks_requests == 1


async def test_warmup_fails_when_the_provider_is_unreachable() -> None:
    def unreachable(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("provider is down")

    verifier = OIDCTokenVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_uri=f"{ISSUER}/jwks",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(unreachable)),
    )

    with pytest.raises(ConfigurationError, match="unreachable|no usable keys"):
        await verifier.warmup()


async def test_a_rotated_signing_key_is_accepted_after_refresh(
    provider: Provider, private_key: rsa.RSAPrivateKey, jwks: dict[str, Any]
) -> None:
    """Key rotation: a new `kid` must be honoured once JWKS is refreshed."""
    import json

    from jwt.algorithms import RSAAlgorithm

    rotated = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rotated_jwk = json.loads(RSAAlgorithm.to_jwk(rotated.public_key()))
    rotated_jwk.update({"kid": "rotated-key", "alg": "RS256", "use": "sig"})

    verifier = OIDCTokenVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_uri=f"{ISSUER}/jwks",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider.handler)),
        jwks_cache_seconds=0.0,
        min_refresh_interval_seconds=0.0,
    )

    await verifier.verify(_token(private_key))
    provider.jwks = {"keys": [rotated_jwk]}

    principal = await verifier.verify(_token(rotated, kid="rotated-key", jti="rotated"))

    assert principal.tenant_id == "acme"
    assert provider.jwks_requests >= 2
