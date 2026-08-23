"""Shared fixtures.

Everything runs against injected mock providers. No test reaches the network, so
the suite is deterministic and needs no credentials.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider

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
    return Settings(api_keys=[], max_attempts=3, default_policy="cheapest")


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
