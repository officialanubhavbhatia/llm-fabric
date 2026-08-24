"""Static API-key credentials.

Machine callers frequently cannot hold an OIDC flow, so a server-configured key
remains a legitimate identity source: the tenant is bound to the key by the
operator, not asserted by the caller. That distinction is what makes this
acceptable where trusting a client-supplied tenant header would not be.

Keys are never logged or stored in a metering record; only a truncated SHA-256
fingerprint leaves this module.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from llm_fabric.errors import AuthenticationError, ConfigurationError
from llm_fabric.identity.claims import Principal

API_KEY_ISSUER = "urn:myvista:api-key"

#: A static key is a bearer credential with no expiry, so it has to carry enough
#: entropy on its own. Anything shorter is refused rather than warned about.
MIN_API_KEY_LENGTH = 16


class ApiCredential(BaseModel):
    """One configured key and the identity it grants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=MIN_API_KEY_LENGTH)
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(default="service-account", min_length=1)
    project_id: str | None = None
    roles: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        return fingerprint_key(self.key)


def fingerprint_key(key: str) -> str:
    """A short, stable, non-reversible label for a key, safe to put in logs."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def keys_match(presented: str, candidate: str) -> bool:
    """Constant-time compare that cannot raise on malformed input.

    `secrets.compare_digest` rejects non-ASCII strings with `TypeError`. A caller
    sending a malformed key must get 401, not a 500 that reveals the comparison
    blew up.
    """
    try:
        return secrets.compare_digest(presented, candidate)
    except (TypeError, ValueError):
        return False


class ApiKeyVerifier:
    """Resolves a presented key to the identity the operator bound to it."""

    def __init__(self, credentials: Iterable[ApiCredential]) -> None:
        self._credentials: Sequence[ApiCredential] = tuple(credentials)
        seen: set[str] = set()
        for credential in self._credentials:
            if credential.key in seen:
                raise ConfigurationError("duplicate API key in credential configuration")
            seen.add(credential.key)

    def __len__(self) -> int:
        return len(self._credentials)

    async def aclose(self) -> None:
        return None

    async def verify(self, token: str) -> Principal:
        matched: ApiCredential | None = None
        # Every credential is compared even after a hit, so the response time
        # does not reveal the position of the matching key.
        for credential in self._credentials:
            if keys_match(token, credential.key):
                matched = credential

        if matched is None:
            raise AuthenticationError("invalid API key")

        return Principal(
            tenant_id=matched.tenant_id,
            user_id=matched.user_id,
            subject=f"{API_KEY_ISSUER}:{matched.fingerprint}",
            issuer=API_KEY_ISSUER,
            roles=frozenset(matched.roles),
            scopes=frozenset(matched.scopes),
            project_id=matched.project_id,
            token_id=matched.fingerprint,
        )
