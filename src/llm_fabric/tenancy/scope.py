"""The tenant scope every storage and cache operation must present.

Tenant isolation fails in practice when the tenant is an optional argument that
a caller can forget. So no repository in this package accepts a bare key: it
accepts a `TenantScope`, and the scope can only be built from an authenticated
`Principal` or explicitly in a test.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_fabric.errors import ConfigurationError
from llm_fabric.identity.claims import Principal


@dataclass(frozen=True, slots=True)
class TenantScope:
    """Who the caller is, for the purpose of reaching stored data."""

    tenant_id: str
    user_id: str | None = None
    project_id: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.tenant_id.strip():
            raise ConfigurationError("a tenant scope requires a non-empty tenant id")

    @classmethod
    def from_principal(cls, principal: Principal) -> TenantScope:
        return cls(
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            project_id=principal.project_id,
        )

    def owns(self, tenant_id: str) -> bool:
        return self.tenant_id == tenant_id

    @property
    def user_key(self) -> str:
        """Stable per-user identifier, namespaced by tenant.

        Two tenants may legitimately use the same user id, so a user-level quota
        or cache key that ignored the tenant would merge them.
        """
        return f"{self.tenant_id}/{self.user_id or '-'}"
