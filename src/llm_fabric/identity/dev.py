"""A local identity provider for development.

Production identity comes from a real OIDC issuer. Developers need something
that works offline, so this module both issues and verifies short-lived HS256
tokens carrying the same claim shape the OIDC path produces. Because the two
paths converge on `RawClaims.to_principal`, code downstream cannot tell them
apart, which is the point: local runs exercise the real authorization logic.

This provider is refused unless it is explicitly enabled, and it refuses a weak
secret outright. An accidentally-enabled dev issuer in production would let
anyone mint any tenant's identity.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable
from typing import Any, Final

import jwt

from llm_fabric.errors import AuthenticationError, ConfigurationError
from llm_fabric.identity.claims import ClaimsMapping, Principal, RawClaims

DEV_ISSUER: Final = "https://dev.myvista.local"
DEV_AUDIENCE: Final = "myvista-llm-fabric"
_ALGORITHM: Final = "HS256"
_MIN_SECRET_LENGTH: Final = 32


class DevIdentityProvider:
    """Issues and verifies local development tokens."""

    def __init__(
        self,
        *,
        secret: str,
        issuer: str = DEV_ISSUER,
        audience: str = DEV_AUDIENCE,
        mapping: ClaimsMapping | None = None,
        default_ttl_seconds: int = 3600,
        leeway_s: float = 5.0,
    ) -> None:
        if len(secret) < _MIN_SECRET_LENGTH:
            raise ConfigurationError(
                "development identity secret must be at least "
                f"{_MIN_SECRET_LENGTH} characters; refusing to start with a guessable secret"
            )
        self._secret = secret
        self._issuer = issuer
        self._audience = audience
        self._mapping = mapping or ClaimsMapping()
        self._default_ttl_seconds = default_ttl_seconds
        self._leeway_s = leeway_s

    async def aclose(self) -> None:
        return None

    def issue_token(
        self,
        *,
        tenant_id: str,
        user_id: str,
        subject: str | None = None,
        project_id: str | None = None,
        roles: Iterable[str] = (),
        scopes: Iterable[str] = (),
        ttl_seconds: int | None = None,
    ) -> str:
        if not tenant_id or not user_id:
            raise ConfigurationError("a development token needs both a tenant and a user")

        issued_at = int(time.time())
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        payload: dict[str, Any] = {
            "iss": self._issuer,
            "aud": self._audience,
            "sub": subject or f"{tenant_id}:{user_id}",
            "iat": issued_at,
            "nbf": issued_at,
            "exp": issued_at + ttl,
            "jti": uuid.uuid4().hex,
            self._mapping.tenant: tenant_id,
            self._mapping.user: user_id,
            self._mapping.roles: sorted(roles),
            self._mapping.scopes: " ".join(sorted(scopes)),
        }
        if project_id:
            payload[self._mapping.project] = project_id
        return jwt.encode(payload, self._secret, algorithm=_ALGORITHM)

    async def verify(self, token: str) -> Principal:
        try:
            payload = jwt.decode(
                token,
                self._secret,
                # Pinned to the symmetric algorithm this provider issues, so a
                # token claiming `alg: none` or an RS* header is refused.
                algorithms=[_ALGORITHM],
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
            raise AuthenticationError("token is not valid") from exc

        return RawClaims(payload=payload, issuer=self._issuer, mapping=self._mapping).to_principal()
