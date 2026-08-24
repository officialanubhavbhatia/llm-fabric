"""Optional live Ollama path. SKIPPED when the daemon is not running."""

from __future__ import annotations

import os

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.router.registry import ModelRegistry

pytestmark = pytest.mark.skipif(
    os.environ.get("LLM_FABRIC_LIVE_OLLAMA") != "1",
    reason="Ollama live inference not measured (set LLM_FABRIC_LIVE_OLLAMA=1)",
)

OLLAMA = "http://127.0.0.1:11434/v1"


def _ollama_up() -> bool:
    try:
        response = httpx.get(f"{OLLAMA}/models", timeout=1.0)
    except httpx.HTTPError:
        return False
    return response.status_code < 500


def _first_tag() -> str | None:
    try:
        payload = httpx.get(f"{OLLAMA}/models", timeout=5.0).json()
    except httpx.HTTPError:
        return None
    rows = payload.get("data") or []
    if not rows:
        return None
    return str(rows[0]["id"])


@pytest.mark.skipif(not _ollama_up(), reason="Ollama daemon not reachable")
def test_ollama_openai_compatible_lists_models() -> None:
    response = httpx.get(f"{OLLAMA}/models", timeout=5.0)
    assert response.status_code == 200
    assert response.json().get("object") == "list"


@pytest.mark.skipif(not _ollama_up(), reason="Ollama daemon not reachable")
def test_ollama_chat_completion() -> None:
    tag = _first_tag()
    if not tag:
        pytest.skip("Ollama has no pulled models")
    response = httpx.post(
        f"{OLLAMA}/chat/completions",
        json={
            "model": tag,
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 8,
        },
        timeout=60.0,
    )
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content


@pytest.mark.skipif(not _ollama_up(), reason="Ollama daemon not reachable")
def test_fabric_serves_ollama_and_sets_routing_headers() -> None:
    tag = _first_tag()
    if not tag:
        pytest.skip("Ollama has no pulled models")
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "local-small",
                    "provider": "ollama",
                    "provider_model": tag,
                    "grade": "L3",
                    "capabilities": ["chat", "streaming"],
                    "lifecycle": "approved",
                    "fallbacks": ["mock-small"],
                    "input_cost_per_mtok": 0.0,
                    "output_cost_per_mtok": 0.0,
                },
                {
                    "id": "mock-small",
                    "provider": "mock",
                    "grade": "L2",
                    "capabilities": ["chat"],
                    "lifecycle": "approved",
                },
            ],
            "aliases": [
                {
                    "id": "auto",
                    "policy": "cost_first",
                    "candidates": ["local-small", "mock-small"],
                }
            ],
        }
    )
    app = create_app(settings=Settings(environment="test"), registry=registry)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 8,
            },
        )
    assert response.status_code == 200
    assert response.headers.get("x-fabric-served-model") == "local-small"
    assert response.headers.get("x-fabric-provider") == "ollama"
    assert response.headers.get("x-fabric-selected-tier") == "L3"
    assert response.headers.get("x-fabric-transport") == "direct"
    assert response.headers.get("x-fabric-runtime") == "ollama"
    assert response.headers.get("x-fabric-provider-adapter") == "ollama"
    assert response.headers.get("x-fabric-route-id")
    assert response.headers.get("x-fabric-deployment-id") == "local-small"
    usage = response.json().get("usage") or {}
    assert int(usage.get("total_tokens") or 0) > 0


@pytest.mark.skipif(not _ollama_up(), reason="Ollama daemon not reachable")
def test_fabric_streams_ollama() -> None:
    tag = _first_tag()
    if not tag:
        pytest.skip("Ollama has no pulled models")
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "local-small",
                    "provider": "ollama",
                    "provider_model": tag,
                    "grade": "L3",
                    "capabilities": ["chat", "streaming"],
                    "lifecycle": "approved",
                }
            ]
        }
    )
    app = create_app(settings=Settings(environment="test"), registry=registry)
    with TestClient(app) as client, client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "local-small",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers.get("x-fabric-transport") == "direct"
        body = "".join(response.iter_text())
    assert "data:" in body
    assert "[DONE]" in body


@pytest.mark.skipif(not _ollama_up(), reason="Ollama daemon not reachable")
def test_ollama_unreachable_falls_back_to_mock() -> None:
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "local-small",
                    "provider": "ollama",
                    "provider_model": "llama3.2",
                    "capabilities": ["chat"],
                    "lifecycle": "approved",
                    "fallbacks": ["mock-small"],
                },
                {
                    "id": "mock-small",
                    "provider": "mock",
                    "capabilities": ["chat"],
                    "lifecycle": "approved",
                },
            ]
        }
    )
    app = create_app(
        settings=Settings(
            environment="test",
            ollama_base_url="http://127.0.0.1:9/v1",
        ),
        registry=registry,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "local-small", "messages": [{"role": "user", "content": "Hello"}]},
        )
    assert response.status_code == 200
    assert response.headers.get("x-fabric-served-model") == "mock-small"
    assert response.headers.get("x-fabric-provider") == "mock"
    assert int(response.headers.get("x-fabric-failovers") or "0") >= 1
