"""Observability overhead of required serving-path instrumentation.

Does not disable the compiler, IntentOS, or in-process OTEL to make a
faster number. Prometheus / vLLM / DCGM scrapes are off the request path
and are not included here.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from llm_fabric.context.compiler import compile_chat
from llm_fabric.contract.openai import ChatCompletionRequest, ChatMessage
from llm_fabric.gateway.app import create_app
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.serving.adapters.mock import MockProvider
from llm_fabric.tenancy.scope import TenantScope


def test_compiler_and_mock_chat_overhead_are_measurable(registry, settings) -> None:
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="hello " * 20)],
    )
    scope = TenantScope(tenant_id="acme")
    compile_chat(request, scope)
    started = time.perf_counter()
    for _ in range(50):
        compile_chat(request, scope)
    compile_ms = (time.perf_counter() - started) / 50 * 1000

    meter = InMemoryMeter()
    app = create_app(
        settings=settings,
        registry=registry,
        provider_overrides={"mock": MockProvider()},
        meter=meter,
    )
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "warmup"}]},
        )
        started = time.perf_counter()
        for _ in range(20):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
            )
            assert response.status_code == 200
        chat_ms = (time.perf_counter() - started) / 20 * 1000
    assert compile_ms >= 0
    assert chat_ms > 0
    # Required observability stays on. These are process-local timings, not a
    # published benchmark.
    print(f"context_compile_mean_ms={compile_ms:.3f} mock_chat_mean_ms={chat_ms:.3f}")
