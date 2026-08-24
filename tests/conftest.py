"""Shared fixtures.

Everything runs against injected mock providers. No test reaches the network, so
the suite is deterministic and needs no credentials.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider


@pytest.fixture(autouse=True)
def _explicit_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runtime has no implicit environment. Pytest names this one.

    Host `LLM_FABRIC_*` values (a local `.env`, a Compose shell, a previous
    live run) must not leak into unit tests: those are a different deployment.
    """
    for name in [key for key in os.environ if key.startswith("LLM_FABRIC_")]:
        if name.startswith("LLM_FABRIC_TEST_") or name.startswith("LLM_FABRIC_SYSTEM_"):
            continue
        if name == "LLM_FABRIC_SKIP_EVALS":
            continue
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_FABRIC_ENVIRONMENT", "test")


REGISTRY_DATA = {
    "models": [
        {
            "id": "cheap",
            "provider": "mock",
            "provider_model": "cheap-v1",
            "context_window": 8192,
            "input_cost_per_mtok": 0.1,
            "output_cost_per_mtok": 0.2,
            "capabilities": ["chat"],
            "fallbacks": ["premium"],
        },
        {
            "id": "premium",
            "provider": "mock",
            "provider_model": "premium-v1",
            "context_window": 32768,
            "input_cost_per_mtok": 3.0,
            "output_cost_per_mtok": 9.0,
            "capabilities": ["chat", "reasoning"],
        },
        {
            "id": "broken",
            "provider": "failing",
            "provider_model": "broken-v1",
            "input_cost_per_mtok": 0.01,
            "output_cost_per_mtok": 0.01,
            "capabilities": ["chat"],
            "fallbacks": ["cheap"],
        },
        {
            "id": "retired",
            "provider": "mock",
            "provider_model": "retired-v1",
            "enabled": False,
            "capabilities": ["chat"],
        },
    ],
    "aliases": [
        {"id": "auto", "policy": "cheapest", "candidates": ["premium", "cheap"]},
        {
            "id": "auto-reasoning",
            "policy": "cheapest",
            "requires": ["reasoning"],
            "candidates": ["cheap", "premium"],
        },
        {
            "id": "auto-failover",
            "policy": "declared",
            "candidates": ["broken", "cheap"],
        },
    ],
}


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.from_mapping(REGISTRY_DATA)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="test",
        api_keys=[],
        max_attempts=3,
        default_policy="cheapest",
    )


@pytest.fixture
def providers() -> dict[str, object]:
    return {"mock": MockProvider(), "failing": MockProvider(fail=True)}


@pytest.fixture
def meter() -> InMemoryMeter:
    return InMemoryMeter()


@pytest.fixture
def client(registry, settings, providers, meter) -> TestClient:
    app = create_app(
        settings=settings,
        registry=registry,
        provider_overrides=providers,
        meter=meter,
    )
    with TestClient(app) as test_client:
        yield test_client
