"""Adversarial tests for the routing fabric.

Routing is a new surface that reads tenant configuration and reports internal
state, so it is a plausible way to reach across the tenant boundary. Three
claims are attacked here:

* the tenant a route is planned for comes from the token, never the request;
* a tenant's narrowing policy cannot be widened by anything the caller sends;
* preview and fleet health disclose no other tenant's policy or traffic.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.identity.dev import DevIdentityProvider
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.router.plan import TenantRoutingPolicies, TenantRoutingPolicy
from llm_fabric.router.policy import RoutePolicy
from llm_fabric.router.registry import Locality, ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider

pytestmark = pytest.mark.isolation

PREVIEW = "/v1/routes/preview"
HEALTH = "/v1/routes/health"
SECRET = "development-secret-that-is-long-enough"

ACME = "tenant-a-acme"
MALLORY = "tenant-b-mallory"


@pytest.fixture
def issuer() -> DevIdentityProvider:
    return DevIdentityProvider(secret=SECRET)


@pytest.fixture
def tenant_policies() -> TenantRoutingPolicies:
    """ACME is a regulated tenant pinned to in-region, in-house serving."""
    return TenantRoutingPolicies(
        [
            TenantRoutingPolicy(
                tenant_id=ACME,
                policy=RoutePolicy.PRIVATE_ONLY,
                allowed_localities=frozenset({Locality.LOCAL, Locality.PRIVATE}),
                denied_models=frozenset({"premium"}),
            )
        ]
    )


def _client(
    registry: ModelRegistry, tenant_policies: TenantRoutingPolicies | None = None
) -> TestClient:
    app = create_app(
        settings=Settings(auth_mode="dev", dev_auth_secret=SECRET),
        registry=registry,
        provider_overrides={"mock": MockProvider(), "failing": MockProvider(fail=True)},
        meter=InMemoryMeter(),
        tenant_routing=tenant_policies,
    )
    return TestClient(app)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _preview(client: TestClient, token: str, **body: object) -> dict:
    payload: dict[str, object] = {"model": "auto"}
    payload.update(body)
    response = client.post(PREVIEW, json=payload, headers=_auth(token))
    assert response.status_code == 200, response.text
    return response.json()


# -- authentication ----------------------------------------------------------


def test_preview_requires_a_caller_identity(registry: ModelRegistry) -> None:
    with _client(registry) as client:
        assert client.post(PREVIEW, json={"model": "auto"}).status_code == 401


def test_fleet_health_requires_a_caller_identity(registry: ModelRegistry) -> None:
    with _client(registry) as client:
        assert client.get(HEALTH).status_code == 401


def test_a_forged_token_cannot_preview(registry: ModelRegistry) -> None:
    forged = DevIdentityProvider(secret="a-different-secret-of-sufficient-length")
    token = forged.issue_token(tenant_id=ACME, user_id="mallory")
    with _client(registry) as client:
        response = client.post(PREVIEW, json={"model": "auto"}, headers=_auth(token))
    assert response.status_code == 401


# -- the tenant comes from the token -----------------------------------------


def test_the_tenant_is_taken_from_the_token_not_the_body(
    registry: ModelRegistry, issuer: DevIdentityProvider, tenant_policies
) -> None:
    mallory = issuer.issue_token(tenant_id=MALLORY, user_id="mallory")

    with _client(registry, tenant_policies) as client:
        body = _preview(client, mallory, tenant_id=ACME, tenant=ACME)

    # The claimed tenant is ignored entirely, including its routing policy.
    assert body["tenant_id"] == MALLORY
    assert body["tenant_policy"] is None


def test_an_attacker_cannot_read_another_tenants_policy(
    registry: ModelRegistry, issuer: DevIdentityProvider, tenant_policies
) -> None:
    acme = issuer.issue_token(tenant_id=ACME, user_id="alice")
    mallory = issuer.issue_token(tenant_id=MALLORY, user_id="mallory")

    with _client(registry, tenant_policies) as client:
        owner = _preview(client, acme)
        attacker = _preview(client, mallory)

    assert owner["tenant_policy"]["tenant_id"] == ACME
    assert owner["tenant_policy"]["denied_models"] == ["premium"]

    # Nothing about ACME's configuration appears in Mallory's view.
    assert attacker["tenant_policy"] is None
    assert "premium" not in str(attacker["excluded"])
    assert ACME not in str(attacker)


# -- a tenant policy narrows and cannot be widened ---------------------------


def test_a_tenant_policy_pin_survives_a_policy_override(
    registry: ModelRegistry, issuer: DevIdentityProvider, tenant_policies
) -> None:
    acme = issuer.issue_token(tenant_id=ACME, user_id="alice")

    with _client(registry, tenant_policies) as client:
        body = _preview(client, acme, policy="cost_first")

    assert body["policy"] == "private_only"
    assert any("pinned" in line for line in body["explanation"])


def test_a_denied_model_cannot_be_requested_directly(
    registry: ModelRegistry, issuer: DevIdentityProvider, tenant_policies
) -> None:
    acme = issuer.issue_token(tenant_id=ACME, user_id="alice")

    with _client(registry, tenant_policies) as client:
        body = _preview(client, acme, model="premium")

    assert body["selected"] is None
    denials = [item for item in body["excluded"] if item["rule"] == "denied_by_tenant_policy"]
    assert [item["model_id"] for item in denials] == ["premium"]
    assert denials[0]["detail"] == "on this tenant's deny list"


def test_a_locality_restriction_holds_against_every_lever(
    registry: ModelRegistry, issuer: DevIdentityProvider, tenant_policies
) -> None:
    """The mock fleet is external, so a tenant restricted to in-house gets nothing.

    Refusing to route is the correct outcome: the alternative is serving a
    regulated tenant from a deployment it is not allowed to use.
    """
    acme = issuer.issue_token(tenant_id=ACME, user_id="alice")

    with _client(registry, tenant_policies) as client:
        for lever in (
            {},
            {"policy": "cost_first"},
            {"policy": "quality_first"},
            {"model": "cheap"},
            {"minimum_grade": "Grade00"},
            {"budget_usd": 1000.0},
        ):
            body = _preview(client, acme, **lever)
            assert body["selected"] is None, lever
            assert all(
                item["rule"] in {"locality_not_permitted", "denied_by_tenant_policy"}
                for item in body["excluded"]
            ), lever


def test_the_tenant_restriction_also_holds_on_the_serving_path(
    registry: ModelRegistry, issuer: DevIdentityProvider, tenant_policies
) -> None:
    """Preview and execution must agree, or preview is not an explanation."""
    acme = issuer.issue_token(tenant_id=ACME, user_id="alice")
    mallory = issuer.issue_token(tenant_id=MALLORY, user_id="mallory")
    chat = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}

    with _client(registry, tenant_policies) as client:
        restricted = client.post("/v1/chat/completions", json=chat, headers=_auth(acme))
        unrestricted = client.post("/v1/chat/completions", json=chat, headers=_auth(mallory))

    assert restricted.status_code == 503
    assert restricted.json()["error"]["type"] == "no_candidate"
    assert unrestricted.status_code == 200


# -- disclosure --------------------------------------------------------------


def test_preview_discloses_no_other_tenants_traffic(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    acme = issuer.issue_token(tenant_id=ACME, user_id="alice")
    mallory = issuer.issue_token(tenant_id=MALLORY, user_id="mallory")

    with _client(registry) as client:
        for _ in range(3):
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "cheap",
                    "messages": [{"role": "user", "content": "ACME_SECRET_PROMPT"}],
                },
                headers=_auth(acme),
            )
        body = _preview(client, mallory, model="cheap")

    assert "ACME_SECRET_PROMPT" not in str(body)
    assert ACME not in str(body)


def test_fleet_health_attributes_no_request_to_any_tenant(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    acme = issuer.issue_token(tenant_id=ACME, user_id="alice")
    mallory = issuer.issue_token(tenant_id=MALLORY, user_id="mallory")

    with _client(registry) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth(acme),
        )
        body = client.get(HEALTH, headers=_auth(mallory)).json()

    # Health is a property of the backend. Counts are fleet-wide by construction,
    # and no tenant, user or prompt may appear beside them.
    assert body["observed"] >= 1
    rendered = str(body)
    assert ACME not in rendered
    assert "alice" not in rendered
    assert all("tenant" not in key for entry in body["deployments"] for key in entry)


def test_preview_is_free_and_leaves_no_usage_behind(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    """Preview must not be chargeable, or it becomes a way to bill a tenant."""
    acme = issuer.issue_token(tenant_id=ACME, user_id="alice")

    with _client(registry) as client:
        for _ in range(5):
            _preview(client, acme)
        usage = client.get("/v1/usage", headers=_auth(acme)).json()

    assert usage["totals"]["requests"] == 0
    assert usage["totals"]["cost_usd"] == 0
