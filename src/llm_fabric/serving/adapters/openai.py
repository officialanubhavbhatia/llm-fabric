"""OpenAI chat-completions adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_fabric.errors import ConfigurationError, RetryableError
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
from llm_fabric.serving.tokens import approximate_prompt_tokens, approximate_token_count


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 60.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError("OpenAI provider requires an API key (set OPENAI_API_KEY)")
        self._client = build_client(
            base_url,
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout_s,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, request: InferenceRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if stream:
            # Without this the final chunk carries no usage and metering would
            # have to fall back to estimation.
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def generate(self, request: InferenceRequest) -> ProviderResult:
        try:
            response = await self._client.post(
                "/chat/completions", json=self._payload(request, stream=False)
            )
        except httpx.HTTPError as exc:
            raise translate_transport_error(self.name, exc) from exc

        raise_for_status(self.name, response)
        body = response.json()

        choices = body.get("choices") or []
        if not choices:
            raise RetryableError("openai: response contained no choices")

        choice = choices[0]
        usage = body.get("usage") or {}
        return ProviderResult(
            text=(choice.get("message") or {}).get("content") or "",
            finish_reason=choice.get("finish_reason") or "stop",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            usage_reported_by_provider=bool(usage),
        )

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamEvent]:
        payload = self._payload(request, stream=True)
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0
        reported = False
        emitted = ""

        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise_for_status(self.name, response)

                async for data in iter_sse_data(response):
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if usage := event.get("usage"):
                        prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                        completion_tokens = int(usage.get("completion_tokens", completion_tokens))
                        reported = True

                    for choice in event.get("choices") or []:
                        if reason := choice.get("finish_reason"):
                            finish_reason = reason
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            emitted += content
                            yield StreamDelta(text=content)
        except httpx.HTTPError as exc:
            raise translate_transport_error(self.name, exc) from exc

        if not reported:
            prompt_tokens = approximate_prompt_tokens(request.messages)
            completion_tokens = approximate_token_count(emitted)

        yield StreamEnd(
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage_reported_by_provider=reported,
        )
