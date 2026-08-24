"""Tests for the public chat-completions surface.

These are the checks that matter to a caller: the response shape, the provenance
headers, the error envelope, and the SSE framing.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

PATH = "/v1/chat/completions"


def _body(model: str = "cheap", text: str = "hello", **extra: object) -> dict[str, object]:
    return {"model": model, "messages": [{"role": "user", "content": text}], **extra}


# -- buffered responses ------------------------------------------------------


def test_response_matches_openai_shape(client: TestClient) -> None:
    response = client.post(PATH, json=_body())
    assert response.status_code == 200

    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["id"].startswith("chatcmpl-")
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert isinstance(payload["choices"][0]["message"]["content"], str)


def test_usage_totals_are_self_consistent(client: TestClient) -> None:
    usage = client.post(PATH, json=_body()).json()["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_response_reports_the_model_that_served_it(client: TestClient) -> None:
    response = client.post(PATH, json=_body(model="auto"))
    payload = response.json()

    # 'auto' is an alias; the response must name the concrete model instead.
    assert payload["model"] == "cheap"
    assert response.headers["x-fabric-requested-model"] == "auto"
    assert response.headers["x-fabric-served-model"] == "cheap"
    assert response.headers["x-fabric-provider"] == "mock"
    assert response.headers["x-fabric-policy"] == "cost_first"


def test_failover_is_visible_to_the_caller(client: TestClient) -> None:
    response = client.post(PATH, json=_body(model="broken"))
    assert response.status_code == 200
    assert response.headers["x-fabric-served-model"] == "cheap"
    assert response.headers["x-fabric-failovers"] == "1"
    assert response.headers["x-fabric-invocations"] == "2"


def test_caller_request_id_is_echoed(client: TestClient) -> None:
    response = client.post(PATH, json=_body(), headers={"x-request-id": "trace-abc"})
    assert response.headers["x-fabric-request-id"] == "trace-abc"


def test_optional_parameters_are_forwarded_to_the_provider(registry, settings, meter) -> None:
    from llm_fabric.gateway.app import create_app
    from llm_fabric.serving.adapters.mock import MockProvider
    from llm_fabric.serving.base import InferenceRequest, ProviderResult

    seen: list[InferenceRequest] = []

    class RecordingProvider(MockProvider):
        async def generate(self, request: InferenceRequest) -> ProviderResult:
            seen.append(request)
            return await super().generate(request)

    app = create_app(
        settings=settings,
        registry=registry,
        provider_overrides={"mock": RecordingProvider(), "failing": MockProvider(fail=True)},
        meter=meter,
    )
    with TestClient(app) as client:
        response = client.post(
            PATH,
            json=_body(temperature=0.2, top_p=0.9, max_tokens=64, stop=["END"]),
        )

    assert response.status_code == 200
    assert len(seen) == 1
    assert seen[0].temperature == 0.2
    assert seen[0].top_p == 0.9
    assert seen[0].max_tokens == 64
    assert seen[0].stop == ["END"]


def test_unknown_sdk_fields_are_ignored(client: TestClient) -> None:
    response = client.post(
        PATH,
        json=_body(stream_options={"include_usage": True}, frequency_penalty=0.0),
    )
    assert response.status_code == 200


# -- errors ------------------------------------------------------------------


def test_unknown_model_is_a_client_error(client: TestClient) -> None:
    response = client.post(PATH, json=_body(model="no-such-model"))
    assert response.status_code == 400

    error = response.json()["error"]
    assert error["type"] == "model_not_found"
    assert "no-such-model" in error["message"]


def test_disabled_model_is_unavailable(client: TestClient) -> None:
    response = client.post(PATH, json=_body(model="retired"))
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "no_candidate"


def test_empty_messages_rejected(client: TestClient) -> None:
    response = client.post(PATH, json={"model": "cheap", "messages": []})
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_out_of_range_temperature_rejected(client: TestClient) -> None:
    response = client.post(PATH, json=_body(temperature=5.0))
    assert response.status_code == 422


def test_errors_share_one_envelope(client: TestClient) -> None:
    for payload, expected in [
        (_body(model="no-such-model"), 400),
        ({"model": "cheap", "messages": []}, 422),
    ]:
        error = client.post(PATH, json=payload).json()["error"]
        assert {"message", "type"} <= error.keys(), expected


# -- streaming ---------------------------------------------------------------


def _sse_events(raw: str) -> list[dict[str, object]]:
    events = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        if data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


def test_stream_is_sse_and_terminates_with_done(client: TestClient) -> None:
    response = client.post(PATH, json=_body(stream=True))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.rstrip().endswith("data: [DONE]")


def test_stream_chunks_have_chunk_object_type(client: TestClient) -> None:
    events = _sse_events(client.post(PATH, json=_body(stream=True)).text)
    assert events
    assert all(event["object"] == "chat.completion.chunk" for event in events)


def test_stream_shares_one_completion_id(client: TestClient) -> None:
    events = _sse_events(client.post(PATH, json=_body(stream=True)).text)
    assert len({event["id"] for event in events}) == 1


def test_first_chunk_announces_the_assistant_role(client: TestClient) -> None:
    events = _sse_events(client.post(PATH, json=_body(stream=True)).text)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"


def test_final_chunk_carries_finish_reason_and_usage(client: TestClient) -> None:
    final = _sse_events(client.post(PATH, json=_body(stream=True)).text)[-1]

    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["total_tokens"] > 0
    assert final["x_fabric"]["served_model"] == "cheap"


def test_streamed_content_reassembles(client: TestClient) -> None:
    events = _sse_events(client.post(PATH, json=_body(stream=True, text="round trip")).text)
    content = "".join(
        event["choices"][0]["delta"].get("content", "")
        for event in events
        if event["choices"][0]["delta"].get("content")
    )
    assert "round trip" in content


def test_stream_failover_is_reported_in_final_chunk(client: TestClient) -> None:
    final = _sse_events(client.post(PATH, json=_body(model="broken", stream=True)).text)[-1]
    assert final["x_fabric"]["served_model"] == "cheap"
    assert final["x_fabric"]["failovers"] == 1


def test_unknown_model_fails_before_streaming_starts(client: TestClient) -> None:
    """A bad model must be an HTTP error, not an error frame inside a 200 stream."""
    response = client.post(PATH, json=_body(model="no-such-model", stream=True))
    assert response.status_code == 400
    assert not response.headers["content-type"].startswith("text/event-stream")


def test_stream_failure_is_metered(meter, registry, settings) -> None:
    from llm_fabric.errors import ProviderUnavailableError
    from llm_fabric.gateway.app import create_app
    from llm_fabric.observability.metering import InMemoryMeter
    from llm_fabric.serving.adapters.mock import MockProvider
    from llm_fabric.serving.base import InferenceRequest, StreamDelta

    class DeltaThenFail(MockProvider):
        async def stream(self, request: InferenceRequest) -> object:
            yield StreamDelta(text="partial")
            raise ProviderUnavailableError("cut off")
            yield  # pragma: no cover

    isolated = InMemoryMeter()
    app = create_app(
        settings=settings,
        registry=registry,
        provider_overrides={"mock": MockProvider(), "failing": DeltaThenFail()},
        meter=isolated,
    )
    with TestClient(app) as client:
        response = client.post(PATH, json=_body(model="broken", stream=True))
        usage = client.get("/v1/usage").json()

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert any("error" in event for event in events)
    assert usage["totals"]["requests"] == 1
    assert usage["recent"][0]["error"]
    assert usage["recent"][0]["streamed"] is True
