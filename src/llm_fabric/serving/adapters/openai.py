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
from llm_fabric.serving.tokens import (
    approximate_prompt_tokens,
    approximate_token_count,
    usage_from_provider,
)


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        if not api_key:
            raise ConfigurationError(
                "OpenAI provider requires an API key "
                "(set LLM_FABRIC_OPENAI_API_KEY or OPENAI_API_KEY)"
            )
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
        if stream and _supports_stream_usage(self._client.base_url.host):
            # OpenAI's final chunk carries no usage unless asked. Local OpenAI-
            # compatible servers (Ollama) reject the field, so metering falls
            # back to estimation there rather than failing the stream.
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def generate(self, request: InferenceRequest) -> ProviderResult:
        try:
            response = await self._client.post(
                "/chat/completions", json=self._payload(request, stream=False)
            )
        except httpx.HTTPError as exc:
            raise translate_transport_error(self.name, exc) from exc

        try:
            raise_for_status(self.name, response)
            body = response.json()
        finally:
            await response.aclose()

        choices = body.get("choices") or []
        if not choices:
            raise RetryableError("openai: response contained no choices")

        choice = choices[0]
        text = (choice.get("message") or {}).get("content") or ""
        prompt_tokens, completion_tokens, reported = usage_from_provider(
            body.get("usage"),
            prompt_key="prompt_tokens",
            completion_key="completion_tokens",
        )
        if not reported:
            prompt_tokens = approximate_prompt_tokens(request.messages)
            completion_tokens = approximate_token_count(text)
        return ProviderResult(
            text=text,
            finish_reason=choice.get("finish_reason") or "stop",
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
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise_for_status(self.name, response)

                async for data in iter_sse_data(response):
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    parsed_prompt, parsed_completion, parsed_any = usage_from_provider(
                        event.get("usage"),
                        prompt_key="prompt_tokens",
                        completion_key="completion_tokens",
                    )
                    if parsed_any:
                        usage = event.get("usage") or {}
                        if "prompt_tokens" in usage:
                            prompt_tokens = parsed_prompt
                            prompt_reported = True
                        if "completion_tokens" in usage:
                            completion_tokens = parsed_completion
                            completion_reported = True

                    for choice in event.get("choices") or []:
                        if reason := choice.get("finish_reason"):
                            finish_reason = reason
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            emitted += content
                            yield StreamDelta(text=content)
        except httpx.HTTPError as exc:
            raise translate_transport_error(self.name, exc) from exc

        if not prompt_reported:
            prompt_tokens = approximate_prompt_tokens(request.messages)
        if not completion_reported:
            completion_tokens = approximate_token_count(emitted)
        reported = prompt_reported and completion_reported

        yield StreamEnd(
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage_reported_by_provider=reported,
        )


def _supports_stream_usage(host: str | None) -> bool:
    return host is None or "openai.com" in host
