"""Tests for model discovery, probes, usage reporting, and authentication."""

from __future__ import annotations

from fastapi.testclient import TestClient

from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider

CHAT = "/v1/chat/completions"


# -- discovery ---------------------------------------------------------------


def test_models_endpoint_lists_enabled_models_and_aliases(client: TestClient) -> None:
    payload = client.get("/v1/models").json()
    assert payload["object"] == "list"

    ids = {card["id"] for card in payload["data"]}
    assert {"cheap", "premium", "auto"} <= ids
    assert "retired" not in ids, "disabled models must not be advertised"


def test_model_detail_reports_owner_and_context_window(client: TestClient) -> None:
    card = client.get("/v1/models/cheap").json()
    assert card["owned_by"] == "mock"
    assert card["context_window"] == 8192


def test_alias_detail_is_owned_by_the_fabric(client: TestClient) -> None:
    assert client.get("/v1/models/auto").json()["owned_by"] == "llm-fabric"


def test_unknown_model_detail_is_a_client_error(client: TestClient) -> None:
    assert client.get("/v1/models/ghost").status_code == 400


# -- probes ------------------------------------------------------------------


def test_healthz_reports_ok(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"


def test_readyz_ready_when_models_are_enabled(client: TestClient) -> None:
    payload = client.get("/readyz").json()
    assert payload["status"] == "ready"
    assert payload["enabled_models"] > 0


def test_readyz_not_ready_without_enabled_models() -> None:
    registry = ModelRegistry.from_mapping(
        {"models": [{"id": "off", "provider": "mock", "enabled": False}]}
    )
    app = create_app(
        settings=Settings(api_keys=[]),
        registry=registry,
        provider_overrides={"mock": MockProvider()},
    )
    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "no_models_enabled"


# -- usage -------------------------------------------------------------------


def test_usage_starts_empty(client: TestClient) -> None:
    payload = client.get("/v1/usage").json()
    assert payload["totals"]["requests"] == 0
    assert payload["recent"] == []


def test_usage_states_that_it_is_not_durable(client: TestClient) -> None:
    assert "in-memory" in client.get("/v1/usage").json()["scope"]


def test_served_requests_appear_in_usage(client: TestClient) -> None:
    client.post(CHAT, json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    payload = client.get("/v1/usage").json()

    assert payload["totals"]["requests"] == 1
    record = payload["recent"][0]
    assert record["requested_model"] == "auto"
    assert record["served_model"] == "cheap"
    assert record["policy"] == "cheapest"


def test_estimated_cost_is_flagged(client: TestClient) -> None:
    """The mock provider reports no real usage, so its cost must be marked estimated."""
    client.post(CHAT, json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]})
    payload = client.get("/v1/usage").json()

    assert payload["recent"][0]["cost_is_estimated"] is True
    assert payload["totals"]["requests_with_estimated_cost"] == 1


def test_usage_records_every_attempt_including_failures(client: TestClient) -> None:
    client.post(CHAT, json={"model": "broken", "messages": [{"role": "user", "content": "hi"}]})
    record = client.get("/v1/usage").json()["recent"][0]

    assert record["failover_count"] == 1
    assert len(record["attempts"]) == 2
    assert record["attempts"][0]["error"] is not None
    assert record["attempts"][1]["error"] is None


def test_streamed_requests_are_metered_as_streamed(client: TestClient) -> None:
    client.post(
        CHAT,
        json={
            "model": "cheap",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert client.get("/v1/usage").json()["recent"][0]["streamed"] is True


# -- authentication ----------------------------------------------------------


def _authenticated_client(registry: ModelRegistry) -> TestClient:
    app = create_app(
        settings=Settings(api_keys=["secret-key"]),
        registry=registry,
        provider_overrides={"mock": MockProvider(), "failing": MockProvider(fail=True)},
        meter=InMemoryMeter(),
    )
    return TestClient(app)


def test_request_without_key_is_rejected(registry: ModelRegistry) -> None:
    with _authenticated_client(registry) as client:
        response = client.get("/v1/models")

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_wrong_key_is_rejected(registry: ModelRegistry) -> None:
    with _authenticated_client(registry) as client:
        response = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401


def test_bearer_token_is_accepted(registry: ModelRegistry) -> None:
    with _authenticated_client(registry) as client:
        response = client.get("/v1/models", headers={"Authorization": "Bearer secret-key"})

    assert response.status_code == 200


def test_x_api_key_header_is_accepted(registry: ModelRegistry) -> None:
    with _authenticated_client(registry) as client:
        response = client.get("/v1/models", headers={"x-api-key": "secret-key"})

    assert response.status_code == 200


def test_probes_do_not_require_authentication(registry: ModelRegistry) -> None:
    with _authenticated_client(registry) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200


def test_client_is_recorded_as_a_fingerprint_not_the_key(registry: ModelRegistry) -> None:
    with _authenticated_client(registry) as client:
        client.post(
            CHAT,
            json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer secret-key"},
        )
        record = client.get("/v1/usage", headers={"Authorization": "Bearer secret-key"}).json()[
            "recent"
        ][0]

    assert record["client_id"]
    assert "secret-key" not in record["client_id"]
