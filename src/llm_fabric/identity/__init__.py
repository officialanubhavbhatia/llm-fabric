"""Identity: turning credentials into a validated, typed caller."""

from llm_fabric.identity.apikey import (
    API_KEY_ISSUER,
    ApiCredential,
    ApiKeyVerifier,
    fingerprint_key,
    keys_match,
)
from llm_fabric.identity.claims import (
    DELEGATION_SCOPE,
    OBSERVE_ROLES,
    OBSERVE_SCOPE,
    ClaimsMapping,
    Principal,
    RawClaims,
)
from llm_fabric.identity.dev import DEV_AUDIENCE, DEV_ISSUER, DevIdentityProvider
from llm_fabric.identity.oidc import ALLOWED_ALGORITHMS, OIDCTokenVerifier
from llm_fabric.identity.revocation import (
    InMemoryRevocationStore,
    RevokingVerifier,
    TokenRevocationStore,
    token_fingerprint,
)
from llm_fabric.identity.verifier import TokenVerifier

__all__ = [
    "ALLOWED_ALGORITHMS",
    "API_KEY_ISSUER",
    "DELEGATION_SCOPE",
    "OBSERVE_ROLES",
    "OBSERVE_SCOPE",
    "DEV_AUDIENCE",
    "DEV_ISSUER",
    "ApiCredential",
    "ApiKeyVerifier",
    "ClaimsMapping",
    "DevIdentityProvider",
    "InMemoryRevocationStore",
    "OIDCTokenVerifier",
    "Principal",
    "RawClaims",
    "RevokingVerifier",
    "TokenRevocationStore",
    "TokenVerifier",
    "fingerprint_key",
    "keys_match",
    "token_fingerprint",
]
