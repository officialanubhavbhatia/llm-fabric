"""The MyVista SDK talks HTTP. Tests drive the app in-process, with no sockets."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.types import ASGIApp

from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider
from myvista import (
    AsyncMyVista,
    AuthenticationError,
    ChatCompletion,
    InvalidRequestError,
    ModelNotFoundError,
    MyVista,
    NotFoundError,
    UnsupportedError,
)


class _TestClientTransport(httpx.BaseTransport):
    """Drive the sync SDK against an in-process Starlette app."""

    def __init__(self, client: TestClient) -> None:
        self._client = client

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"
        response = self._client.request(
            request.method,
            path,
            headers=dict(request.headers),
            content=request.content,
        )
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=response.content,
            request=request,
        )


@pytest.fixture
def app(
    registry: ModelRegistry, settings: Settings, providers: dict[str, object], meter: InMemoryMeter
) -> ASGIApp:
    return create_app(
        settings=settings,
        registry=registry,
        provider_overrides=providers,
        meter=meter,
    )


@pytest.fixture
def sdk(app: ASGIApp) -> Iterator[MyVista]:
    with (
        TestClient(app) as starlette,
        MyVista(transport=_TestClientTransport(starlette), base_url="http://test") as client,
    ):
        yield client


def test_the_simple_path_is_one_call(sdk: MyVista) -> None:
    response = sdk.responses.create(input="Hello")
    assert response.output_text
    assert response.request_id
    assert response.fabric.served_model


def test_chat_matches_openai_shape_and_exposes_trace_id(sdk: MyVista) -> None:
    completion = sdk.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert isinstance(completion, ChatCompletion)
    assert completion.text
    assert completion.request_id
    assert completion.fabric.policy


def test_streaming_yields_deltas(sdk: MyVista) -> None:
    chunks = list(
        sdk.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
    )
    assert any(chunk.delta for chunk in chunks)


def test_unknown_model_is_typed(sdk: MyVista) -> None:
    with pytest.raises(ModelNotFoundError) as caught:
        sdk.chat.completions.create(
            model="no-such-model",
            messages=[{"role": "user", "content": "Hello"}],
        )
    assert caught.value.request_id
    assert caught.value.status_code == 400


def test_classify_uses_the_offline_cascade(sdk: MyVista) -> None:
    decision = sdk.intents.classify("debug this python traceback")
    assert "classification" in decision
    assert decision["classification"]["intent_id"]


def test_route_preview_explains_without_inference(sdk: MyVista) -> None:
    plan = sdk.routes.preview(model="auto", messages=[{"role": "user", "content": "Hello"}])
    assert plan["selected"]["model_id"]
    assert plan["explanation"]


def test_eval_runs_the_named_ci_suite() -> None:
    app = create_app(
        settings=Settings(api_keys=[]),
        provider_overrides={"mock": MockProvider()},
    )
    with (
        TestClient(app) as starlette,
        MyVista(transport=_TestClientTransport(starlette), base_url="http://test") as sdk,
    ):
        run = sdk.evals.run("ci")
    assert run["suite_name"]
    assert "metrics" in run


def test_unknown_eval_suite_is_a_client_error(sdk: MyVista) -> None:
    with pytest.raises(InvalidRequestError):
        sdk.evals.run("not-a-suite")


def test_traces_round_trip(sdk: MyVista) -> None:
    sdk.chat.completions.create(model="auto", messages=[{"role": "user", "content": "Hello"}])
    listed = sdk.traces.list()
    assert listed["traces"]
    trace_id = listed["traces"][0]["trace_id"]
    fetched = sdk.traces.get(trace_id)
    assert fetched["trace_id"] == trace_id


def test_missing_trace_is_404(sdk: MyVista) -> None:
    with pytest.raises(NotFoundError):
        sdk.traces.get("0" * 32)


def test_embeddings_and_agents_are_explicitly_unsupported(sdk: MyVista) -> None:
    with pytest.raises(UnsupportedError, match="embeddings"):
        sdk.embeddings.create(input="hello")
    with pytest.raises(UnsupportedError, match="agents"):
        sdk.agents.run(input="hello")


def test_api_key_is_sent_and_rejected_when_wrong(registry: ModelRegistry) -> None:
    locked = create_app(
        settings=Settings(api_keys=["sk-sdk-test-key-that-is-long"], auth_mode="api_key"),
        registry=registry,
        provider_overrides={"mock": MockProvider()},
        meter=InMemoryMeter(),
    )
    with TestClient(locked) as starlette:
        with (
            MyVista(
                api_key="wrong-key-that-is-long-enough",
                transport=_TestClientTransport(starlette),
            ) as client,
            pytest.raises(AuthenticationError),
        ):
            client.models.list()
        with MyVista(
            api_key="sk-sdk-test-key-that-is-long",
            transport=_TestClientTransport(starlette),
        ) as client:
            listed = client.models.list()
            assert listed["object"] == "list"


@pytest.mark.asyncio
async def test_async_chat_and_stream(app: ASGIApp) -> None:
    transport = httpx.ASGITransport(app=app)
    async with AsyncMyVista(transport=transport, base_url="http://test") as client:
        completion = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(completion, ChatCompletion)
        assert completion.text
        stream = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]
        assert any(chunk.delta for chunk in chunks)
