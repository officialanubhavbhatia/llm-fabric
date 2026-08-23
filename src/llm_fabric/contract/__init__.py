"""The versioned public contract.

Kept separate from the gateway implementation so the surface callers depend on
can stay stable while everything behind it is refactored.
"""

from llm_fabric.contract.openai import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelCard,
    ModelList,
    Usage,
)

__all__ = [
    "ChatChoice",
    "ChatCompletionChunk",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "ModelCard",
    "ModelList",
    "Usage",
]
