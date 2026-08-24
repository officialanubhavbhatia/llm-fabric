"""Live socket tests. Skip unless LLM_FABRIC_SYSTEM_TEST=1 and a stack is up."""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.system

SKIP = pytest.mark.skipif(
    os.environ.get("LLM_FABRIC_SYSTEM_TEST") != "1",
    reason="live stack tests require LLM_FABRIC_SYSTEM_TEST=1 and a running compose stack",
)


@SKIP
def test_live_healthz() -> None:
    base = os.environ.get("LLM_FABRIC_SYSTEM_BASE_URL", "http://127.0.0.1:47317")
    response = httpx.get(f"{base}/healthz", timeout=5.0)
    assert response.status_code == 200
    assert response.json().get("status") == "ok"


@SKIP
def test_live_intent_classify_and_chat_headers() -> None:
    """SDK-shaped HTTP: classify, then chat with intent enabled on the stack.

    Requires a running compose stack with LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED=1.
    """
    base = os.environ.get("LLM_FABRIC_SYSTEM_BASE_URL", "http://127.0.0.1:47317")
    token = os.environ.get("LLM_FABRIC_SYSTEM_TOKEN", "")
    headers = {"authorization": f"Bearer {token}"} if token else {}
    classify = httpx.post(
        f"{base}/v1/intents/classify",
        json={"input": "translate this sentence into French"},
        headers=headers,
        timeout=10.0,
    )
    assert classify.status_code == 200, classify.text
    body = classify.json()
    intent = body["classification"]
    assert intent["taxonomy_version"]
    assert intent["classifier_version"]
    assert "layer" in intent
    chat = httpx.post(
        f"{base}/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "translate this sentence into French"}],
        },
        headers=headers,
        timeout=30.0,
    )
    assert chat.status_code == 200, chat.text
    if "x-fabric-intent" in chat.headers:
        assert chat.headers["x-fabric-intent-layer"]
        assert chat.headers["x-fabric-taxonomy-version"]
        assert chat.headers["x-fabric-classifier-version"]
