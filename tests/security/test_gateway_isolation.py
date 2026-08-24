"""End-to-end cross-tenant tests through the real gateway.

The store-level tests prove the repositories hold the line. These prove the
wiring does too: real middleware, real token validation, real routes. A leak
introduced by a route that forgets to pass a scope would pass the unit tests and
fail here.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.identity.claims import DELEGATION_SCOPE
from llm_fabric.identity.dev import DevIdentityProvider
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider

pytestmark = pytest.mark.isolation

CHAT = "/v1/chat/completions"
SECRET = "development-secret-that-is-long-enough"

ACME = "tenant-a-acme"
MALLORY = "tenant-b-mallory"


@pytest.fixture
def issuer() -> DevIdentityProvider:
    return DevIdentityProvider(secret=SECRET)


def _build_client(registry: ModelRegistry, **setting_overrides: object) -> TestClient:
    app = create_app(
        settings=Settings(auth_mode="dev", dev_auth_secret=SECRET, **setting_overrides),
        registry=registry,
        provider_overrides={"mock": MockProvider(), "failing": MockProvider(fail=True)},
        meter=InMemoryMeter(),
    )
    return TestClient(app)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _chat(client: TestClient, token: str, content: str, model: str = "cheap"):
    return client.post(
        CHAT,
        json={"model": model, "messages": [{"role": "user", "content": content}]},
        headers=_auth(token),
    )


# -- usage -------------------------------------------------------------------


def test_usage_never_crosses_the_tenant_boundary(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    acme_token = issuer.issue_token(tenant_id=ACME, user_id="alice")
    mallory_token = issuer.issue_token(tenant_id=MALLORY, user_id="mallory")

    with _build_client(registry) as client:
        assert _chat(client, acme_token, "ACME_SECRET_PROMPT").status_code == 200
        assert _chat(client, mallory_token, "mallory prompt", model="premium").status_code == 200

        acme_usage = client.get("/v1/usage", headers=_auth(acme_token)).json()
        mallory_usage = client.get("/v1/usage", headers=_auth(mallory_token)).json()

    assert acme_usage["tenant_id"] == ACME
    assert acme_usage["totals"]["requests"] == 1
    assert [r["requested_model"] for r in acme_usage["recent"]] == ["cheap"]

    assert mallory_usage["totals"]["requests"] == 1
    assert [r["requested_model"] for r in mallory_usage["recent"]] == ["premium"]

    # Mallory's view must contain no trace of ACME at all: not the tenant id,
    # not the request id, not the model ACME chose.
    serialized = json.dumps(mallory_usage)
    assert ACME not in serialized
    assert "ACME_SECRET_PROMPT" not in serialized


def test_users_inside_one_tenant_share_that_tenant_usage(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    """The boundary is the tenant, not the user. Stated as a test, not assumed."""
    alice = issuer.issue_token(tenant_id=ACME, user_id="alice")
    bob = issuer.issue_token(tenant_id=ACME, user_id="bob")

    with _build_client(registry) as client:
        _chat(client, alice, "one")
        _chat(client, bob, "two")
        usage = client.get("/v1/usage", headers=_auth(alice)).json()

    assert usage["totals"]["requests"] == 2


# -- tenant spoofing ---------------------------------------------------------


def test_a_tenant_header_cannot_override_the_token(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    """The direct attack: claim another tenant via a header."""
    acme_token = issuer.issue_token(tenant_id=ACME, user_id="alice")
    mallory_token = issuer.issue_token(tenant_id=MALLORY, user_id="mallory")

    with _build_client(registry) as client:
        _chat(client, acme_token, "ACME_SECRET_PROMPT")

        spoofed = client.get(
            "/v1/usage",
            headers={**_auth(mallory_token), "X-Tenant-Id": ACME},
        )

    assert spoofed.status_code == 401
    assert ACME not in spoofed.text


def test_delegation_works_only_with_the_delegation_scope(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    acme_token = issuer.issue_token(tenant_id=ACME, user_id="alice")
    operator_token = issuer.issue_token(
        tenant_id="tenant-operator",
        user_id="ops",
        scopes=[DELEGATION_SCOPE],
    )

    with _build_client(registry) as client:
        _chat(client, acme_token, "ACME_SECRET_PROMPT")

        delegated = client.get(
            "/v1/usage",
            headers={**_auth(operator_token), "X-Tenant-Id": ACME},
        )
        undelegated = client.get("/v1/usage", headers=_auth(operator_token))

    assert delegated.status_code == 200
    assert delegated.json()["tenant_id"] == ACME
    assert delegated.json()["totals"]["requests"] == 1

    assert undelegated.json()["tenant_id"] == "tenant-operator"
    assert undelegated.json()["totals"]["requests"] == 0


def test_an_unauthenticated_caller_reaches_nothing(registry: ModelRegistry) -> None:
    with _build_client(registry) as client:
        for path in ("/v1/models", "/v1/usage", "/v1/observability/traces"):
            assert client.get(path).status_code == 401
        assert client.post(CHAT, json={"model": "cheap", "messages": []}).status_code == 401
        assert client.post("/v1/intents/classify", json={"input": "hi"}).status_code == 401
        assert client.post("/v1/evals/run", json={"suite": "ci"}).status_code == 401


def test_probes_stay_public_but_reveal_no_tenant_data(registry: ModelRegistry) -> None:
    with _build_client(registry) as client:
        health = client.get("/healthz")
        ready = client.get("/readyz")

    assert health.status_code == 200
    assert ready.status_code == 200
    assert "tenant" not in health.text.lower()


# -- quotas ------------------------------------------------------------------


def test_one_tenant_cannot_exhaust_another_tenants_quota(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    """Noisy-neighbour containment, which is an isolation property too."""
    acme_token = issuer.issue_token(tenant_id=ACME, user_id="alice")
    mallory_token = issuer.issue_token(tenant_id=MALLORY, user_id="mallory")

    with _build_client(registry, quota_tenant_requests_per_minute=2) as client:
        assert _chat(client, mallory_token, "one").status_code == 200
        assert _chat(client, mallory_token, "two").status_code == 200
        exhausted = _chat(client, mallory_token, "three")

        # ACME is untouched by Mallory burning their own allowance.
        assert _chat(client, acme_token, "one").status_code == 200

    assert exhausted.status_code == 429
    assert exhausted.json()["error"]["type"] == "quota_exceeded"
    assert int(exhausted.headers["retry-after"]) > 0


def test_one_user_cannot_exhaust_another_users_quota_inside_a_tenant(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    alice = issuer.issue_token(tenant_id=ACME, user_id="alice")
    bob = issuer.issue_token(tenant_id=ACME, user_id="bob")

    with _build_client(registry, quota_user_requests_per_minute=1) as client:
        assert _chat(client, alice, "one").status_code == 200
        assert _chat(client, alice, "two").status_code == 429
        assert _chat(client, bob, "one").status_code == 200


# -- trace metadata ----------------------------------------------------------


def test_responses_carry_correlation_and_trace_headers(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    token = issuer.issue_token(tenant_id=ACME, user_id="alice")

    with _build_client(registry) as client:
        response = client.get("/v1/models", headers=_auth(token))

    assert response.headers["x-fabric-request-id"]
    traceparent = response.headers["traceparent"]
    assert traceparent.startswith("00-")
    assert len(traceparent.split("-")) == 4


def test_an_inbound_traceparent_is_continued(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    token = issuer.issue_token(tenant_id=ACME, user_id="alice")
    inbound_trace = "4bf92f3577b34da6a3ce929d0e0e4736"

    with _build_client(registry) as client:
        response = client.get(
            "/v1/models",
            headers={**_auth(token), "traceparent": f"00-{inbound_trace}-00f067aa0ba902b7-01"},
        )

    assert response.headers["traceparent"].split("-")[1] == inbound_trace


def test_trace_headers_never_leak_the_tenant_identifier(
    registry: ModelRegistry, issuer: DevIdentityProvider
) -> None:
    """Trace *metadata* is tenant-aware internally; the wire format is not.

    Tenant ids belong in spans the operator can see, not in a header that
    downstream systems and proxies will log.
    """
    token = issuer.issue_token(tenant_id=ACME, user_id="alice")

    with _build_client(registry) as client:
        response = client.get("/v1/models", headers=_auth(token))

    assert ACME not in json.dumps(dict(response.headers))


# -- development issuer ------------------------------------------------------


def test_the_dev_token_endpoint_is_absent_unless_dev_mode_is_on(
    registry: ModelRegistry,
) -> None:
    """Outside dev mode there is no token mint, authenticated or not.

    Unauthenticated the answer is 401, because authentication runs before
    routing and an anonymous caller should not be able to map which routes
    exist. Authenticated, the answer is 404: the route was never mounted.
    """
    api_key = "a-long-enough-api-key"
    app = create_app(
        settings=Settings(api_keys=[api_key]),
        registry=registry,
        provider_overrides={"mock": MockProvider(), "failing": MockProvider(fail=True)},
        meter=InMemoryMeter(),
    )
    with TestClient(app) as client:
        anonymous = client.post("/v1/dev/token", json={"tenant_id": ACME})
        authenticated = client.post(
            "/v1/dev/token", json={"tenant_id": ACME}, headers=_auth(api_key)
        )

    assert anonymous.status_code == 401
    assert authenticated.status_code == 404


def test_a_dev_issued_token_authenticates_end_to_end(registry: ModelRegistry) -> None:
    with _build_client(registry) as client:
        issued = client.post(
            "/v1/dev/token",
            json={"tenant_id": ACME, "user_id": "alice", "scopes": ["chat:write"]},
        )
        token = issued.json()["access_token"]
        response = client.get("/v1/usage", headers=_auth(token))

    assert issued.status_code == 200
    assert response.status_code == 200
    assert response.json()["tenant_id"] == ACME
