"""Audit events never persist secrets and stay tenant-bound."""

from __future__ import annotations

from llm_fabric.storage.repositories import TenantStores
from llm_fabric.tenancy.scope import TenantScope


def test_audit_redacts_secret_fields() -> None:
    stores = TenantStores()
    scope = TenantScope(tenant_id="acme", user_id="alice")
    event = stores.audit_events.record(
        scope,
        actor="alice",
        action="api_credential.changed",
        target="key-1",
        after={"key": "super-secret", "role": "admin"},
        reason="rotated",
        request_id="req-1",
    )
    assert event.after is not None
    assert event.after["key"] == "[REDACTED]"
    assert event.after["role"] == "admin"
    other = TenantScope(tenant_id="other", user_id="bob")
    assert stores.audit_events.get(other, event.event_id) is None
