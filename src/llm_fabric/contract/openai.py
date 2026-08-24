"""OpenAI-compatible request and response schema.

The fabric speaks the OpenAI chat-completions dialect so existing clients and
SDKs can point at it without code changes. Fields the fabric does not yet act on
are accepted and echoed where the dialect requires it, rather than rejected, so
that unmodified clients do not break; `docs/CONTRACT.md` records which fields
are honoured and which are currently inert.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["system", "user", "assistant", "tool"]


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatCompletionRequest(BaseModel):
    # Extra fields are ignored, not rejected: SDKs send keys the fabric does not
    # act on, and refusing those would break clients the dialect exists to serve.
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    stream: bool = False

    # Not honoured yet; accepted so OpenAI SDK clients do not fail validation.
    n: int | None = Field(default=None, ge=1, le=1)
    user: str | None = None

    @field_validator("messages")
    @classmethod
    def _require_messages(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if not value:
            raise ValueError("messages must contain at least one message")
        return value

    def stop_sequences(self) -> list[str]:
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop]
        return list(self.stop)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=_completion_id)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: Usage


class ChunkDelta(BaseModel):
    role: Role | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: ChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChunkChoice]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str
    context_window: int | None = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
