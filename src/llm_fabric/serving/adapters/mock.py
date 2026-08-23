"""A local backend that needs no credentials.

Two jobs: let the fabric be run and exercised end to end without provider keys,
and give the test suite a backend whose output is deterministic. It performs no
inference — the text it returns is assembled from the request itself, and it is
never presented as a model response.

It can also be told to fail, which is how the router's fallback behaviour is
tested without depending on a real provider being down.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from llm_fabric.errors import ProviderUnavailableError
from llm_fabric.serving.base import (
    InferenceRequest,
    Provider,
    ProviderResult,
    StreamDelta,
    StreamEnd,
    StreamEvent,
)
from llm_fabric.serving.tokens import approximate_prompt_tokens, approximate_token_count


class MockProvider(Provider):
    name = "mock"

    def __init__(
        self,
        *,
        fail: bool = False,
        reply: str | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self._fail = fail
        self._reply = reply
        self._delay_s = delay_s

    def _compose(self, request: InferenceRequest) -> str:
        if self._reply is not None:
            return self._reply
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "",
        )
        return f"[mock:{request.model}] {last_user}"

    async def generate(self, request: InferenceRequest) -> ProviderResult:
        if self._fail:
            raise ProviderUnavailableError(f"mock provider configured to fail for {request.model}")
        if self._delay_s:
            await asyncio.sleep(self._delay_s)

        text = self._compose(request)
        return ProviderResult(
            text=text,
            finish_reason="stop",
            prompt_tokens=approximate_prompt_tokens(request.messages),
            completion_tokens=approximate_token_count(text),
            usage_reported_by_provider=False,
        )

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamEvent]:
        if self._fail:
            raise ProviderUnavailableError(f"mock provider configured to fail for {request.model}")

        text = self._compose(request)
        words = text.split(" ")
        for index, word in enumerate(words):
            if self._delay_s:
                await asyncio.sleep(self._delay_s)
            yield StreamDelta(text=word if index == 0 else f" {word}")

        yield StreamEnd(
            finish_reason="stop",
            prompt_tokens=approximate_prompt_tokens(request.messages),
            completion_tokens=approximate_token_count(text),
            usage_reported_by_provider=False,
        )
