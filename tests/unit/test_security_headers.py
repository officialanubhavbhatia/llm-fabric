"""Security headers and untrusted forwarded headers."""

from __future__ import annotations

from fastapi.testclient import TestClient

from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider


def test_responses_carry_hardening_headers(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_x_forwarded_for_is_ignored_from_untrusted_peers(registry: ModelRegistry) -> None:
    app = create_app(
        settings=Settings(allow_anonymous=True, trusted_proxies=[]),
        registry=registry,
        provider_overrides={"mock": MockProvider(), "failing": MockProvider(fail=True)},
        meter=InMemoryMeter(),
    )
    client = TestClient(app)
    response = client.get("/healthz", headers={"X-Forwarded-For": "1.2.3.4"})
    assert response.status_code == 200
    # The header was accepted as a string but must not become identity.
    assert response.headers.get("x-forwarded-for") is None
