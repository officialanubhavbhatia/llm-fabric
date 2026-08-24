"""Optional live LiteLLM→Ollama path. SKIPPED unless explicitly enabled.

Never treat unit/fixture tests as this verification.
"""

from __future__ import annotations

import os

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.router.registry import ModelRegistry

pytestmark = pytest.mark.skipif(
    os.environ.get("LLM_FABRIC_LIVE_LITELLM") != "1",
    reason="LiteLLM live inference not measured (set LLM_FABRIC_LIVE_LITELLM=1)",
)

LITELLM = os.environ.get("LLM_FABRIC_LITELLM_BASE_URL", "http://127.0.0.1:4000/v1")
MODEL = os.environ.get("LLM_FABRIC_LIVE_LITELLM_MODEL", "smollm2-135m")


def _litellm_up() -> bool:
    try:
        response = httpx.get(f"{LITELLM.rstrip('/')}/models", timeout=1.0)
    except httpx.HTTPError:
        return False
    return response.status_code < 500


@pytest.mark.skipif(not _litellm_up(), reason="LiteLLM proxy not reachable")
def test_litellm_lists_models() -> None:
    response = httpx.get(f"{LITELLM.rstrip('/')}/models", timeout=5.0)
    assert response.status_code == 200


@pytest.mark.skipif(not _litellm_up(), reason="LiteLLM proxy not reachable")
def test_fabric_serves_litellm_ollama_and_sets_topology_headers() -> None:
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "litellm-ollama-live",
                    "provider": "litellm",
                    "provider_model": MODEL,
                    "transport": "litellm",
                    "runtime": "ollama",
                    "grade": "Grade00",
                    "capabilities": ["chat", "streaming"],
                    "lifecycle": "approved",
                    "enabled": True,
                }
            ]
        }
    )
    settings = Settings(
        _env_file=None,
        environment="test",
        allow_anonymous=True,
        litellm_base_url=LITELLM,
        litellm_num_retries=0,
        request_timeout_s=120.0,
    )
    with TestClient(create_app(settings=settings, registry=registry)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "litellm-ollama-live",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 8,
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"]
    assert response.headers.get("x-fabric-transport") == "litellm"
    assert response.headers.get("x-fabric-runtime") == "ollama"
    assert response.headers.get("x-fabric-provider-adapter") == "litellm"
    assert response.headers.get("x-fabric-route-id")
    assert response.headers.get("x-fabric-litellm-model") == MODEL
