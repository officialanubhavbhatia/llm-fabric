"""The typed identity the rest of the fabric is allowed to trust.

A `Principal` is only ever constructed from claims that a verifier has already
validated. Nothing in this module reads request bodies, query strings, or
unauthenticated headers, because a tenant id that a caller can choose is not an
identity — it is an access-control bypass with extra steps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_fabric.errors import AuthenticationError

#: Scope that permits a caller to act on behalf of another tenant. Delegation is
#: refused unless the validated token carries this scope.
DELEGATION_SCOPE = "fabric:delegate_tenant"

#: Scope that permits a caller to see fleet-wide Command Center views rather
#: than only their own tenant. Without it, observability is tenant-scoped.
OBSERVE_SCOPE = "fabric:observe"

#: Roles that also unlock fleet-wide observability. Kept small and named so a
#: token cannot stumble into operator visibility.
OBSERVE_ROLES = frozenset({"operator", "admin"})


@dataclass(frozen=True, slots=True)
class ClaimsMapping:
    """Where each piece of identity lives in a token.

    Issuers disagree about claim names, so the mapping is configuration rather
    than a hard-coded assumption about any one provider.
    """

    tenant: str = "tenant_id"
    user: str = "user_id"
    project: str = "project_id"
    roles: str = "roles"
    scopes: str = "scope"


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller.

    Every field originates in a validated token. `delegated_from` is set only
    when a caller holding `DELEGATION_SCOPE` explicitly acted for another
    tenant, so audit records can tell real identity from assumed identity.
    """

    tenant_id: str
    user_id: str
    subject: str
    issuer: str
    roles: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()
    project_id: str | None = None
    token_id: str | None = None
    expires_at: int | None = None
    delegated_from: str | None = None

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def has_any_role(self, roles: Sequence[str]) -> bool:
        return any(role in self.roles for role in roles)

    @property
    def may_delegate_tenant(self) -> bool:
        return DELEGATION_SCOPE in self.scopes

    @property
    def may_observe_fleet(self) -> bool:
        """True when this identity may see every tenant's telemetry.

        Tenant-scoped views are the default. Fleet-wide aggregation is an
        operator privilege, not a property of being authenticated.
        """
        return OBSERVE_SCOPE in self.scopes or bool(self.roles & OBSERVE_ROLES)

    def delegate_to(self, tenant_id: str) -> Principal:
        """Return a principal acting for `tenant_id`.

        Refused unless the token carries the delegation scope. The original
        tenant is preserved in `delegated_from` so the audit trail survives.
        """
        if not self.may_delegate_tenant:
            raise AuthenticationError("token is not permitted to act for another tenant")
        if not tenant_id:
            raise AuthenticationError("delegated tenant id must not be empty")
        if tenant_id == self.tenant_id:
            return self
        return Principal(
            tenant_id=tenant_id,
            user_id=self.user_id,
            subject=self.subject,
            issuer=self.issuer,
            roles=self.roles,
            scopes=self.scopes,
            project_id=self.project_id,
            token_id=self.token_id,
            expires_at=self.expires_at,
            delegated_from=self.tenant_id,
        )

    def audit_fields(self) -> dict[str, Any]:
        """Identity fields that are safe to put in a log or a span.

        The token itself is never included, and neither is any claim the fabric
        does not need. `subject` is an opaque issuer-scoped id, not a name.
        """
        fields: dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "subject": self.subject,
            "issuer": self.issuer,
            "project_id": self.project_id,
        }
        if self.delegated_from:
            fields["delegated_from"] = self.delegated_from
        if self.token_id:
            fields["token_id"] = self.token_id
        return fields


@dataclass(frozen=True, slots=True)
class RawClaims:
    """A verified token payload, before it is narrowed to a `Principal`."""

    payload: Mapping[str, Any]
    issuer: str
    mapping: ClaimsMapping = field(default_factory=ClaimsMapping)

    def to_principal(self) -> Principal:
        subject = _require_str(self.payload, "sub")
        tenant_id = _require_str(self.payload, self.mapping.tenant)
        # An issuer that authenticates a person but not a tenant is misconfigured
        # for this fabric; falling back to a default tenant would silently merge
        # customers, so it is an error instead.
        user_id = _optional_str(self.payload, self.mapping.user) or subject

        return Principal(
            tenant_id=tenant_id,
            user_id=user_id,
            subject=subject,
            issuer=self.issuer,
            roles=frozenset(_as_string_set(self.payload.get(self.mapping.roles))),
            scopes=frozenset(_as_string_set(self.payload.get(self.mapping.scopes))),
            project_id=_optional_str(self.payload, self.mapping.project),
            token_id=_optional_str(self.payload, "jti"),
            expires_at=_optional_int(self.payload, "exp"),
        )


def _require_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AuthenticationError(f"token is missing the required '{key}' claim")
    return value.strip()


def _optional_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_string_set(value: Any) -> set[str]:
    """Accept both OAuth2 space-delimited scopes and list-valued role claims."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {item for item in value.replace(",", " ").split() if item}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()
