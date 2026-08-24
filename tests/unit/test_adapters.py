"""HTTP adapters against a mock transport. No network."""

from __future__ import annotations

import json

import httpx
import pytest

from llm_fabric.contract.openai import ChatMessage
from llm_fabric.errors import ConfigurationError, InvalidRequestError, ProviderUnavailableError
from llm_fabric.serving.adapters.anthropic import AnthropicProvider, _anthropic_messages
from llm_fabric.serving.adapters.openai import OpenAIProvider
from llm_fabric.serving.base import InferenceRequest, StreamDelta, StreamEnd


def _request(text: str = "hello") -> InferenceRequest:
    return InferenceRequest(model="test-model", messages=[ChatMessage(role="user", content=text)])


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://example.test/v1",
    )


async def test_openai_refuses_to_build_without_a_key() -> None:
    with pytest.raises(ConfigurationError, match="API key"):
        OpenAIProvider(api_key=None)


async def test_openai_generate_reads_provider_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["temperature"] == 0.2
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi there"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    provider = OpenAIProvider(api_key="k", client=_client(handler))
    result = await provider.generate(
        InferenceRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0.2,
        )
    )
    assert result.text == "hi there"
    assert result.prompt_tokens == 4
    assert result.completion_tokens == 2
    assert result.usage_reported_by_provider is True


async def test_openai_generate_estimates_when_usage_missing() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "abcd"}, "finish_reason": "stop"}]},
        )

    result = await OpenAIProvider(api_key="k", client=_client(handler)).generate(_request())
    assert result.usage_reported_by_provider is False
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0


async def test_openai_generate_maps_429_to_retryable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    with pytest.raises(ProviderUnavailableError, match="slow down"):
        await OpenAIProvider(api_key="k", client=_client(handler)).generate(_request())


async def test_openai_generate_maps_400_to_caller_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad temperature"}})

    with pytest.raises(InvalidRequestError, match="bad temperature"):
        await OpenAIProvider(api_key="k", client=_client(handler)).generate(_request())


async def test_openai_stream_omits_stream_options_for_compatible_servers() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n',
        b'data: {"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n',
        b"data: [DONE]\n\n",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        assert "stream_options" not in body
        return httpx.Response(200, content=b"".join(chunks))

    events = [
        event
        async for event in OpenAIProvider(api_key="k", client=_client(handler)).stream(_request())
    ]
    deltas = [event.text for event in events if isinstance(event, StreamDelta)]
    end = events[-1]
    assert "".join(deltas) == "hello"
    assert isinstance(end, StreamEnd)
    assert end.prompt_tokens == 3
    assert end.completion_tokens == 2
    assert end.usage_reported_by_provider is True


async def test_openai_stream_estimates_when_usage_chunk_absent() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"abcd"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"".join(chunks))

    events = [
        event
        async for event in OpenAIProvider(api_key="k", client=_client(handler)).stream(_request())
    ]
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.usage_reported_by_provider is False
    assert end.completion_tokens > 0


async def test_anthropic_generate_maps_usage_and_stop_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["max_tokens"] == 1024
        assert body["system"] == "be brief"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 8, "output_tokens": 1},
            },
        )

    result = await AnthropicProvider(api_key="k", client=_client(handler)).generate(
        InferenceRequest(
            model="claude",
            messages=[
                ChatMessage(role="system", content="be brief"),
                ChatMessage(role="user", content="hello"),
            ],
        )
    )
    assert result.text == "ok"
    assert result.finish_reason == "length"
    assert result.prompt_tokens == 8
    assert result.completion_tokens == 1
    assert result.usage_reported_by_provider is True


async def test_anthropic_merges_consecutive_user_turns() -> None:
    mapped = _anthropic_messages(
        InferenceRequest(
            model="claude",
            messages=[
                ChatMessage(role="user", content="one"),
                ChatMessage(role="user", content="two"),
                ChatMessage(role="assistant", content="ok"),
            ],
        )
    )
    assert mapped == [
        {"role": "user", "content": "one\n\ntwo"},
        {"role": "assistant", "content": "ok"},
    ]


async def test_anthropic_prepends_user_turn_when_assistant_speaks_first() -> None:
    mapped = _anthropic_messages(
        InferenceRequest(
            model="claude",
            messages=[ChatMessage(role="assistant", content="hi")],
        )
    )
    assert mapped[0]["role"] == "user"
    assert mapped[1] == {"role": "assistant", "content": "hi"}


async def test_anthropic_stream_reads_message_events() -> None:
    chunks = [
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}\n\n',
        b'data: {"type":"content_block_delta","delta":{"text":"ab"}}\n\n',
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        b'"usage":{"output_tokens":2}}\n\n',
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"".join(chunks))

    events = [
        event
        async for event in AnthropicProvider(api_key="k", client=_client(handler)).stream(
            _request()
        )
    ]
    assert [e.text for e in events if isinstance(e, StreamDelta)] == ["ab"]
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.prompt_tokens == 5
    assert end.completion_tokens == 2
    assert end.usage_reported_by_provider is True


async def test_anthropic_stream_estimates_missing_output_tokens() -> None:
    chunks = [
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}\n\n',
        b'data: {"type":"content_block_delta","delta":{"text":"abcd"}}\n\n',
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"".join(chunks))

    events = [
        event
        async for event in AnthropicProvider(api_key="k", client=_client(handler)).stream(
            _request()
        )
    ]
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.usage_reported_by_provider is False
    assert end.prompt_tokens == 5
    assert end.completion_tokens > 0
