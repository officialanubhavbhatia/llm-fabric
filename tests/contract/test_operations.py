"""Tests for model discovery, probes, usage reporting, and authentication."""

from __future__ import annotations

import json

import pytest
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
    assert payload["ready"] is True
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
    assert record["policy"] == "balanced"


def test_estimated_cost_is_flagged(client: TestClient) -> None:
    """The mock provider reports no real usage, so its cost must be marked estimated."""
    client.post(CHAT, json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]})
    payload = client.get("/v1/usage").json()

    assert payload["recent"][0]["cost_is_estimated"] is True
    assert payload["totals"]["requests_with_estimated_cost"] == 1


def test_usage_records_every_attempt_including_failures(client: TestClient) -> None:
    client.post(CHAT, json={"model": "broken", "messages": [{"role": "user", "content": "hi"}]})
    payload = client.get("/v1/usage").json()
    record = payload["recent"][0]

    assert record["failover_count"] == 1
    assert len(record["attempts"]) == 2
    assert record["attempts"][0]["error"] is not None
    assert record["attempts"][1]["error"] is None
    assert payload["invocations"]["count"] == 2
    assert len(payload["recent_invocations"]) == 2
    assert all("token_source" in row for row in payload["recent_invocations"])
    assert "PROVIDER_MEASURED" not in {row["token_source"] for row in payload["recent_invocations"]}


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


VALID_KEY = "secret-key-0123456789"


def _authenticated_client(registry: ModelRegistry) -> TestClient:
    app = create_app(
        settings=Settings(api_keys=[VALID_KEY]),
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
    assert response.headers["www-authenticate"].startswith("Bearer")


def test_wrong_key_is_rejected(registry: ModelRegistry) -> None:
    with _authenticated_client(registry) as client:
        response = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401


def test_bearer_token_is_accepted(registry: ModelRegistry) -> None:
    with _authenticated_client(registry) as client:
        response = client.get("/v1/models", headers={"Authorization": f"Bearer {VALID_KEY}"})

    assert response.status_code == 200


def test_x_api_key_header_is_accepted(registry: ModelRegistry) -> None:
    with _authenticated_client(registry) as client:
        response = client.get("/v1/models", headers={"x-api-key": VALID_KEY})

    assert response.status_code == 200


def test_probes_do_not_require_authentication(registry: ModelRegistry) -> None:
    with _authenticated_client(registry) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200


def test_short_api_keys_are_refused_with_a_clear_error() -> None:
    from llm_fabric.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="shorter than"):
        Settings(api_keys=["tiny"]).resolved_credentials()


def test_usage_never_echoes_the_presented_key(registry: ModelRegistry) -> None:
    with _authenticated_client(registry) as client:
        client.post(
            CHAT,
            json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
        )
        payload = client.get("/v1/usage", headers={"Authorization": f"Bearer {VALID_KEY}"}).json()

    assert VALID_KEY not in json.dumps(payload)
    assert payload["recent"][0]["tenant_id"] == "default"


async def test_non_ascii_key_is_rejected_as_unauthorized() -> None:
    """`secrets.compare_digest` raises on non-ASCII; that must be a 401, not a 500."""
    from llm_fabric.errors import AuthenticationError
    from llm_fabric.identity.apikey import ApiCredential, ApiKeyVerifier

    verifier = ApiKeyVerifier([ApiCredential(key=VALID_KEY, tenant_id="t")])

    with pytest.raises(AuthenticationError):
        await verifier.verify("café-key-0123456789")


def test_failed_request_is_still_metered() -> None:
    registry = ModelRegistry.from_mapping({"models": [{"id": "only", "provider": "failing"}]})
    meter = InMemoryMeter()
    app = create_app(
        settings=Settings(api_keys=[]),
        registry=registry,
        provider_overrides={"failing": MockProvider(fail=True)},
        meter=meter,
    )
    with TestClient(app) as client:
        failed = client.post(
            CHAT, json={"model": "only", "messages": [{"role": "user", "content": "hi"}]}
        )
        usage = client.get("/v1/usage").json()

    assert failed.status_code == 502
    assert usage["totals"]["requests"] == 1
    assert usage["recent"][0]["error"]
    assert usage["recent"][0]["attempts"][0]["error"] is not None


def test_readyz_not_ready_when_enabled_provider_has_no_credentials() -> None:
    registry = ModelRegistry.from_mapping(
        {"models": [{"id": "gpt", "provider": "openai", "enabled": True}]}
    )
    app = create_app(
        settings=Settings(api_keys=[], openai_api_key=None),
        registry=registry,
    )
    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "no_servable_provider"
    assert response.json()["servable_models"] == 0
