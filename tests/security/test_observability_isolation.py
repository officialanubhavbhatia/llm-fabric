"""Command Center views must not leak another tenant's telemetry."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.identity.claims import OBSERVE_SCOPE
from llm_fabric.identity.dev import DevIdentityProvider
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider

pytestmark = pytest.mark.isolation

SECRET = "development-secret-that-is-long-enough"
ACME = "tenant-a-acme"
MALLORY = "tenant-b-mallory"


@pytest.fixture
def issuer() -> DevIdentityProvider:
    return DevIdentityProvider(secret=SECRET)


def _client(registry: ModelRegistry) -> TestClient:
    app = create_app(
        settings=Settings(auth_mode="dev", dev_auth_secret=SECRET),
        registry=registry,
        provider_overrides={"mock": MockProvider()},
        meter=InMemoryMeter(),
    )
    return TestClient(app)


def test_dashboard_requests_are_tenant_scoped(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    acme = issuer.issue_token(tenant_id=ACME, user_id="alice")
    mallory = issuer.issue_token(tenant_id=MALLORY, user_id="mallory")

    with _client(registry) as client:
        assert (
            client.post(
                "/v1/chat/completions",
                json={"model": "cheap", "messages": [{"role": "user", "content": "ACME_SECRET"}]},
                headers={"Authorization": f"Bearer {acme}"},
            ).status_code
            == 200
        )
        mallory_view = client.get(
            "/v1/observability/dashboards/requests",
            headers={"Authorization": f"Bearer {mallory}"},
        ).json()
        acme_view = client.get(
            "/v1/observability/dashboards/users",
            headers={"Authorization": f"Bearer {acme}"},
        ).json()

    serialized = json.dumps(mallory_view)
    assert ACME not in serialized
    assert "ACME_SECRET" not in serialized
    assert mallory_view["available"] is True
    assert acme_view["data"]["users"]
    assert all(row["tenant_id"] == ACME for row in acme_view["data"]["users"])


def test_fleet_view_requires_observe_scope(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    operator = issuer.issue_token(tenant_id=ACME, user_id="ops", scopes=[OBSERVE_SCOPE])
    tenant = issuer.issue_token(tenant_id=ACME, user_id="alice")
    other = issuer.issue_token(tenant_id=MALLORY, user_id="mallory")

    with _client(registry) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {other}"},
        )
        tenant_view = client.get(
            "/v1/observability/dashboards/tenants",
            headers={"Authorization": f"Bearer {tenant}"},
        ).json()
        fleet_view = client.get(
            "/v1/observability/dashboards/tenants",
            headers={"Authorization": f"Bearer {operator}"},
        ).json()

    tenant_ids = {row["tenant_id"] for row in tenant_view["data"]["tenants"]}
    assert tenant_ids == {ACME}
    fleet_ids = {row["tenant_id"] for row in fleet_view["data"]["tenants"]}
    assert MALLORY in fleet_ids


def test_prometheus_metrics_are_public_and_carry_no_tenant(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    token = issuer.issue_token(tenant_id=ACME, user_id="alice")
    with _client(registry) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {token}"},
        )
        scrape = client.get("/metrics")
        assert scrape.status_code == 200
        body = scrape.text
    assert ACME not in body
    assert "alice" not in body
    assert "fabric_requests_total" in body


def test_a_trace_lookup_is_404_for_another_tenant(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    acme = issuer.issue_token(tenant_id=ACME, user_id="alice")
    mallory = issuer.issue_token(tenant_id=MALLORY, user_id="mallory")

    with _client(registry) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "cheap", "messages": [{"role": "user", "content": "secret"}]},
            headers={"Authorization": f"Bearer {acme}"},
        )
        traces = client.get(
            "/v1/observability/traces",
            headers={"Authorization": f"Bearer {acme}"},
        ).json()["traces"]
        assert traces
        trace_id = traces[0]["trace_id"]
        own = client.get(
            f"/v1/observability/traces/{trace_id}",
            headers={"Authorization": f"Bearer {acme}"},
        )
        other = client.get(
            f"/v1/observability/traces/{trace_id}",
            headers={"Authorization": f"Bearer {mallory}"},
        )
    assert own.status_code == 200
    assert other.status_code == 404
    assert "error" in other.json()
