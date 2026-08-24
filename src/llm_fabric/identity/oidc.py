"""OAuth2/OIDC bearer-token validation.

Signature verification is delegated to PyJWT. This module owns the parts that
are easy to get wrong and expensive to get wrong: refusing unexpected
algorithms, pinning issuer and audience, and fetching signing keys without
handing an attacker a way to make the gateway hammer the identity provider.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Final

import httpx
import jwt

from llm_fabric.errors import AuthenticationError, ConfigurationError
from llm_fabric.identity.claims import ClaimsMapping, Principal, RawClaims

#: Asymmetric algorithms only. A symmetric algorithm here would let anyone who
#: can read the public JWKS mint tokens, which is the classic `alg` confusion
#: attack, so HS* is refused outright rather than merely discouraged.
ALLOWED_ALGORITHMS: Final[tuple[str, ...]] = ("RS256", "RS384", "RS512", "ES256", "ES384")

_DISCOVERY_SUFFIX: Final = "/.well-known/openid-configuration"


class OIDCTokenVerifier:
    """Validates tokens against an OIDC issuer's published signing keys."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_uri: str | None = None,
        mapping: ClaimsMapping | None = None,
        jwks_cache_seconds: float = 300.0,
        min_refresh_interval_seconds: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
        request_timeout_s: float = 5.0,
        leeway_s: float = 30.0,
    ) -> None:
        if not issuer:
            raise ConfigurationError("OIDC verifier requires an issuer")
        if not audience:
            raise ConfigurationError("OIDC verifier requires an audience")

        self._issuer = issuer.rstrip("/")
        self._audience = audience
        self._jwks_uri = jwks_uri
        self._mapping = mapping or ClaimsMapping()
        self._jwks_cache_seconds = jwks_cache_seconds
        self._min_refresh_interval_seconds = min_refresh_interval_seconds
        self._leeway_s = leeway_s

        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout_s),
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )

        self._keys: dict[str, Any] = {}
        self._keys_fetched_at: float = 0.0
        self._last_refresh_attempt: float = 0.0
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def verify(self, token: str) -> Principal:
        header = _read_header(token)
        algorithm = header.get("alg")
        if algorithm not in ALLOWED_ALGORITHMS:
            # Refused before any key lookup so an attacker cannot steer the
            # gateway toward a key type of their choosing.
            raise AuthenticationError("token uses an unsupported signing algorithm")

        kid = header.get("kid")
        key = await self._signing_key(kid if isinstance(kid, str) else None)

        try:
            payload = jwt.decode(
                token,
                key=key,
                algorithms=list(ALLOWED_ALGORITHMS),
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway_s,
                options={
                    "require": ["exp", "iss", "aud", "sub"],
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_aud": True,
                    "verify_iss": True,
                },
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("token has expired") from exc
        except jwt.InvalidTokenError as exc:
            # The specific reason is deliberately not echoed to the caller.
            raise AuthenticationError("token is not valid") from exc

        return RawClaims(payload=payload, issuer=self._issuer, mapping=self._mapping).to_principal()

    async def warmup(self) -> None:
        """Fetch signing keys now, so a production process fails before serving.

        A first-request discovery failure would look like a caller problem
        (401) rather than a misconfigured gateway. Startup must refuse instead.
        """
        try:
            await self._refresh_keys(force=True)
        except AuthenticationError as exc:
            raise ConfigurationError(
                "OIDC identity provider is unreachable or published no usable keys"
            ) from exc

    async def _signing_key(self, kid: str | None) -> Any:
        key = self._cached_key(kid)
        if key is not None:
            return key

        await self._refresh_keys()

        # Look up without the TTL so a just-fetched set is usable even when
        # `jwks_cache_seconds` is 0 (no cache: refresh every miss, then use it).
        key = self._key_by_kid(kid)
        if key is None:
            raise AuthenticationError("token was signed by an unknown key")
        return key

    def _key_by_kid(self, kid: str | None) -> Any:
        if not self._keys:
            return None
        if kid is None:
            # Only unambiguous when the issuer publishes exactly one key.
            return next(iter(self._keys.values())) if len(self._keys) == 1 else None
        return self._keys.get(kid)

    def _cached_key(self, kid: str | None) -> Any:
        if not self._keys:
            return None
        if self._jwks_cache_seconds <= 0:
            return None
        if time.monotonic() - self._keys_fetched_at > self._jwks_cache_seconds:
            return None
        return self._key_by_kid(kid)

    async def _refresh_keys(self, *, force: bool = False) -> None:
        async with self._lock:
            now = time.monotonic()
            fresh = (
                self._jwks_cache_seconds > 0
                and bool(self._keys)
                and now - self._keys_fetched_at <= self._jwks_cache_seconds
            )
            if fresh and not force:
                return
            # An unknown `kid` is attacker-controlled, so refreshes are rate
            # limited: otherwise a stream of junk tokens becomes a traffic
            # amplifier pointed at the identity provider. Startup (`force`)
            # bypasses the limiter so an unreachable issuer fails closed.
            if not force and now - self._last_refresh_attempt < self._min_refresh_interval_seconds:
                return
            self._last_refresh_attempt = now

            jwks_uri = self._jwks_uri or await self._discover_jwks_uri()
            document = await self._fetch_json(jwks_uri)

            keys: dict[str, Any] = {}
            for entry in document.get("keys") or []:
                if not isinstance(entry, dict):
                    continue
                if entry.get("alg") and entry.get("alg") not in ALLOWED_ALGORITHMS:
                    continue
                key_id = entry.get("kid")
                if not isinstance(key_id, str):
                    continue
                try:
                    keys[key_id] = jwt.PyJWK(entry).key
                except (jwt.InvalidKeyError, jwt.PyJWKError, ValueError):
                    continue

            if not keys:
                raise AuthenticationError("identity provider published no usable signing keys")

            self._keys = keys
            self._keys_fetched_at = now

    async def _discover_jwks_uri(self) -> str:
        document = await self._fetch_json(f"{self._issuer}{_DISCOVERY_SUFFIX}")
        jwks_uri = document.get("jwks_uri")
        if not isinstance(jwks_uri, str) or not jwks_uri:
            raise ConfigurationError("OIDC discovery document does not advertise a jwks_uri")
        self._jwks_uri = jwks_uri
        return jwks_uri

    async def _fetch_json(self, url: str) -> dict[str, Any]:
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            document = response.json()
        except httpx.HTTPError as exc:
            raise AuthenticationError("identity provider is unreachable") from exc
        except ValueError as exc:
            raise AuthenticationError("identity provider returned a malformed document") from exc

        if not isinstance(document, dict):
            raise AuthenticationError("identity provider returned a malformed document")
        return document


def _read_header(token: str) -> dict[str, Any]:
    if not token or token.count(".") != 2:
        raise AuthenticationError("token is not a well-formed JWT")
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("token header is not readable") from exc
    if not isinstance(header, dict):
        raise AuthenticationError("token header is not readable")
    return header
