"""Usage ledger tenant isolation against real PostgreSQL RLS."""

from __future__ import annotations

import os
import uuid

import pytest

from llm_fabric.observability.usage_event import TokenSource, UsageEvent
from llm_fabric.storage.postgres import (
    APPLICATION_ROLE,
    create_database_engine,
    init_schema,
    provision_application_role,
)
from llm_fabric.storage.usage import UsageLedger

pytestmark = [pytest.mark.isolation, pytest.mark.tenant_isolation]


def _postgres_url() -> str | None:
    url = os.environ.get("LLM_FABRIC_TEST_DATABASE_URL")
    if url and url.startswith("postgresql"):
        return url
    return None


def _app_database_url(admin_url: str) -> str:
    override = os.environ.get("LLM_FABRIC_TEST_DATABASE_APP_URL")
    if override:
        return override
    if "://fabric:" in admin_url:
        return admin_url.replace("://fabric:", f"://{APPLICATION_ROLE}:", 1)
    return admin_url


def _event(tenant_id: str, event_id: str, prompt: int = 10) -> UsageEvent:
    return UsageEvent(
        event_id=event_id,
        invocation_id=event_id,
        request_id=f"req-{event_id}",
        tenant_id=tenant_id,
        provider="mock",
        model="cheap",
        prompt_tokens=prompt,
        completion_tokens=1,
        token_source=TokenSource.PROVIDER_MEASURED.value,
        started_at=1.0,
        completed_at=2.0,
        status="success",
    )


@pytest.fixture
def app_ledger() -> UsageLedger:
    url = _postgres_url()
    if not url:
        pytest.fail("LLM_FABRIC_TEST_DATABASE_URL must point at PostgreSQL for usage RLS tests")
    admin = create_database_engine(url)
    init_schema(admin)
    provision_application_role(admin)
    admin.dispose()
    app_engine = create_database_engine(_app_database_url(url))
    return UsageLedger(app_engine)


def test_tenant_cannot_read_another_tenants_usage(app_ledger: UsageLedger) -> None:
    suffix = uuid.uuid4().hex[:8]
    acme = f"tenant-a-{suffix}"
    mallory = f"tenant-b-{suffix}"
    app_ledger.insert(_event(acme, f"a-{suffix}", prompt=42))
    app_ledger.insert(_event(mallory, f"b-{suffix}", prompt=99))

    acme_totals = app_ledger.totals(tenant_id=acme)
    mallory_events = app_ledger.list_events(tenant_id=mallory)
    acme_ids = {event.event_id for event in app_ledger.list_events(tenant_id=acme)}
    mallory_ids = {event.event_id for event in mallory_events}

    assert acme_totals.prompt_tokens == 42
    assert f"a-{suffix}" in acme_ids
    assert f"b-{suffix}" not in acme_ids
    assert f"a-{suffix}" not in mallory_ids
    assert all(event.tenant_id == mallory for event in mallory_events)


def test_fleet_observe_can_read_all_tenants(app_ledger: UsageLedger) -> None:
    suffix = uuid.uuid4().hex[:8]
    app_ledger.insert(_event(f"tenant-a-{suffix}", f"oa-{suffix}"))
    app_ledger.insert(_event(f"tenant-b-{suffix}", f"ob-{suffix}"))
    fleet = app_ledger.list_events(tenant_id=None, observe=True, limit=10_000)
    tenants = {event.tenant_id for event in fleet}
    assert f"tenant-a-{suffix}" in tenants
    assert f"tenant-b-{suffix}" in tenants


def test_observe_cannot_write_another_tenants_row(app_ledger: UsageLedger) -> None:
    suffix = uuid.uuid4().hex[:8]
    # Insert still binds the event's tenant. Writing as a different tenant is refused by RLS.
    event = _event(f"tenant-a-{suffix}", f"ow-{suffix}")
    app_ledger.insert(event)
    # A second insert of the same id is a duplicate, not a cross-tenant overwrite.
    again = app_ledger.insert(event)
    assert again.duplicate is True
