"""Anthropic messages adapter.

Two shape differences from the OpenAI dialect are handled here rather than
leaking upward: the system prompt is a top-level field instead of a message, and
`max_tokens` is required rather than optional.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_fabric.errors import ConfigurationError
from llm_fabric.serving.adapters._http import (
    build_client,
    iter_sse_data,
    raise_for_status,
    translate_transport_error,
)
from llm_fabric.serving.base import (
    InferenceRequest,
    Provider,
    ProviderResult,
    StreamDelta,
    StreamEnd,
    StreamEvent,
)
from llm_fabric.serving.tokens import (
    approximate_prompt_tokens,
    approximate_token_count,
    usage_from_provider,
)

_API_VERSION = "2023-06-01"

# Anthropic rejects requests without max_tokens; used when the caller omits it.
_DEFAULT_MAX_TOKENS = 1024

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def _anthropic_messages(request: InferenceRequest) -> list[dict[str, str]]:
    """Map the OpenAI message list onto Anthropic's alternating user/assistant shape.

    Consecutive same-role turns are merged. A conversation that starts with
    assistant, or that is only a system prompt, gets a leading user turn so the
    request is representable — Anthropic rejects anything else.
    """
    normalized: list[dict[str, str]] = []
    for message in request.non_system_messages():
        role = "assistant" if message.role == "assistant" else "user"
        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] = f"{normalized[-1]['content']}\n\n{message.content}"
        else:
            normalized.append({"role": role, "content": message.content})
    if not normalized:
        normalized.append({"role": "user", "content": " "})
    elif normalized[0]["role"] != "user":
        normalized.insert(0, {"role": "user", "content": " "})
    return normalized


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.anthropic.com/v1",
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        if not api_key:
            raise ConfigurationError(
                "Anthropic provider requires an API key "
                "(set LLM_FABRIC_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY)"
            )
        self._client = build_client(
            base_url,
            {
                "x-api-key": api_key,
                "anthropic-version": _API_VERSION,
                "Content-Type": "application/json",
            },
            timeout_s,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, request: InferenceRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens or _DEFAULT_MAX_TOKENS,
            "messages": _anthropic_messages(request),
            "stream": stream,
        }
        if system := request.system_prompt():
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop
        return payload

    async def generate(self, request: InferenceRequest) -> ProviderResult:
        try:
            response = await self._client.post(
                "/messages", json=self._payload(request, stream=False)
            )
        except httpx.HTTPError as exc:
            raise translate_transport_error(self.name, exc) from exc

        try:
            raise_for_status(self.name, response)
            body = response.json()
        finally:
            await response.aclose()

        text = "".join(
            block.get("text", "")
            for block in body.get("content") or []
            if block.get("type") == "text"
        )
        prompt_tokens, completion_tokens, reported = usage_from_provider(
            body.get("usage"),
            prompt_key="input_tokens",
            completion_key="output_tokens",
        )
        if not reported:
            prompt_tokens = approximate_prompt_tokens(request.messages)
            completion_tokens = approximate_token_count(text)
        return ProviderResult(
            text=text,
            finish_reason=_STOP_REASON_MAP.get(body.get("stop_reason") or "", "stop"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage_reported_by_provider=reported,
        )

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamEvent]:
        payload = self._payload(request, stream=True)
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0
        prompt_reported = False
        completion_reported = False
        emitted = ""

        try:
            async with self._client.stream("POST", "/messages", json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise_for_status(self.name, response)

                async for data in iter_sse_data(response):
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")

                    if event_type == "message_start":
                        parsed_prompt, _, parsed_any = usage_from_provider(
                            (event.get("message") or {}).get("usage"),
                            prompt_key="input_tokens",
                            completion_key="output_tokens",
                        )
                        if parsed_any:
                            prompt_tokens = parsed_prompt
                            prompt_reported = True
                    elif event_type == "content_block_delta":
                        text = (event.get("delta") or {}).get("text")
                        if text:
                            emitted += text
                            yield StreamDelta(text=text)
                    elif event_type == "message_delta":
                        if reason := (event.get("delta") or {}).get("stop_reason"):
                            finish_reason = _STOP_REASON_MAP.get(reason, "stop")
                        parsed_prompt, parsed_completion, parsed_any = usage_from_provider(
                            event.get("usage"),
                            prompt_key="input_tokens",
                            completion_key="output_tokens",
                        )
                        if parsed_any:
                            usage = event.get("usage") or {}
                            if "input_tokens" in usage:
                                prompt_tokens = parsed_prompt
                                prompt_reported = True
                            if "output_tokens" in usage:
                                completion_tokens = parsed_completion
                                completion_reported = True
        except httpx.HTTPError as exc:
            raise translate_transport_error(self.name, exc) from exc

        if not prompt_reported:
            prompt_tokens = approximate_prompt_tokens(request.messages)
        if not completion_reported:
            completion_tokens = approximate_token_count(emitted)

        yield StreamEnd(
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage_reported_by_provider=prompt_reported and completion_reported,
        )
