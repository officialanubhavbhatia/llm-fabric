"""Gateway routes the SDK depends on."""

from __future__ import annotations

from fastapi.testclient import TestClient

from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.serving.adapters.mock import MockProvider


def test_classify_returns_a_decision(client: TestClient) -> None:
    response = client.post("/v1/intents/classify", json={"input": "debug this python traceback"})
    assert response.status_code == 200
    body = response.json()
    assert "classification" in body
    assert body["classification"]["intent_id"]
    assert "attempts" in body


def test_classify_rejects_an_empty_prompt(client: TestClient) -> None:
    response = client.post("/v1/intents/classify", json={"input": ""})
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_chat_with_intent_enabled_exposes_provenance_headers(registry) -> None:
    """Serving-path classification is off by default. When it is on, chat
    responses must carry intent provenance without changing authorization.
    """
    app = create_app(
        settings=Settings(
            environment="test",
            api_keys=[],
            intent_classification_enabled=True,
        ),
        registry=registry,
        provider_overrides={"mock": MockProvider()},
    )
    with TestClient(app) as client:
        classify = client.post(
            "/v1/intents/classify",
            json={"input": "translate this sentence into French"},
        )
        chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "translate this sentence into French"}],
            },
        )
    assert classify.status_code == 200
    intent = classify.json()["classification"]
    assert intent["intent_id"] == "translation"
    assert intent["taxonomy_version"]
    assert intent["classifier_version"]
    assert intent["layer"]
    assert chat.status_code == 200
    assert chat.headers["x-fabric-intent"] == "translation"
    assert chat.headers["x-fabric-intent-layer"]
    assert chat.headers["x-fabric-taxonomy-version"]
    assert chat.headers["x-fabric-classifier-version"]
    assert "authorization" not in chat.json()


def test_chat_shadow_classifies_without_changing_the_route(registry) -> None:
    app = create_app(
        settings=Settings(
            environment="test",
            api_keys=[],
            intent_shadow=True,
            intent_classification_enabled=False,
        ),
        registry=registry,
        provider_overrides={"mock": MockProvider()},
    )
    with TestClient(app) as client:
        chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "translate this sentence into French"}],
            },
        )
    assert chat.status_code == 200
    assert "x-fabric-intent-shadow" in chat.headers
    assert chat.headers["x-fabric-intent-shadow"] == "translation"
    assert chat.headers["x-fabric-intent"] == "unknown"
    assert chat.headers["x-fabric-intent-state"] == "safe_fallback"
    assert chat.headers["x-fabric-intent-result-id"]
    assert "authorization" not in chat.json()


def test_eval_ci_suite_returns_metrics() -> None:
    app = create_app(
        settings=Settings(api_keys=[]),
        provider_overrides={"mock": MockProvider()},
    )
    with TestClient(app) as client:
        response = client.post("/v1/evals/run", json={"suite": "ci"})
    assert response.status_code == 200
    body = response.json()
    assert body["suite_name"]
    assert "metrics" in body
    assert "provenance" in body


def test_unknown_eval_suite_is_400(client: TestClient) -> None:
    response = client.post("/v1/evals/run", json={"suite": "imaginary"})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
