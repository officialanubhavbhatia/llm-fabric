"""Token estimation for backends that do not report usage.

This is a heuristic, not a tokenizer. It exists so that requests served by a
backend which reports no usage still produce a metering record, and every such
record is flagged as an estimate. Cost figures derived from it are estimates too
and are labelled that way rather than presented as measured.
"""

from __future__ import annotations

from llm_fabric.contract.openai import ChatMessage

# Rough average across English text for common BPE vocabularies. Chosen for
# being cheap and dependency-free, not for accuracy.
_CHARS_PER_TOKEN = 4

# Per-message framing overhead in chat formats.
_MESSAGE_OVERHEAD_TOKENS = 4


def approximate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def approximate_prompt_tokens(messages: list[ChatMessage]) -> int:
    return sum(
        approximate_token_count(message.content) + _MESSAGE_OVERHEAD_TOKENS for message in messages
    )
