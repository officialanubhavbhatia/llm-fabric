"""The provider interface every backend implements.

Adding a backend means adding one `Provider` subclass. Nothing in the gateway or
the router changes, because neither of them knows what a provider is beyond this
interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from llm_fabric.contract.openai import ChatMessage


@dataclass(slots=True)
class InferenceRequest:
    """A request already resolved to a concrete backend model.

    `model` is the provider-native identifier, not the fabric-facing alias: the
    router has already translated it.
    """

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)

    def system_prompt(self) -> str | None:
        parts = [m.content for m in self.messages if m.role == "system"]
        return "\n\n".join(parts) if parts else None

    def non_system_messages(self) -> list[ChatMessage]:
        return [m for m in self.messages if m.role != "system"]


@dataclass(slots=True)
class ProviderResult:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int

    # False when token counts came from a heuristic rather than the backend, so
    # downstream cost figures can be labelled as estimates.
    usage_reported_by_provider: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class StreamDelta:
    """An incremental piece of generated text."""

    text: str


@dataclass(slots=True)
class StreamEnd:
    """Terminal stream event, carrying the usage the gateway needs to meter."""

    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    usage_reported_by_provider: bool = True


StreamEvent = StreamDelta | StreamEnd


class Provider(ABC):
    """A backend capable of serving inference requests."""

    #: Stable identifier used in the model registry and in metering records.
    name: str = "provider"

    @abstractmethod
    async def generate(self, request: InferenceRequest) -> ProviderResult:
        """Serve a request to completion."""

    @abstractmethod
    def stream(self, request: InferenceRequest) -> AsyncIterator[StreamEvent]:
        """Serve a request incrementally, ending with exactly one `StreamEnd`."""

    async def aclose(self) -> None:
        """Release any held resources. Overridden by providers holding clients."""
        return None
