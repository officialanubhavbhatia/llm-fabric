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


def usage_from_provider(
    usage: object,
    *,
    prompt_key: str,
    completion_key: str,
) -> tuple[int, int, bool]:
    """Return (prompt, completion, reported).

    `reported` is true only when the backend actually sent one of the token
    fields. An empty object or a dict of unrelated keys is not usage.
    """
    if not isinstance(usage, dict) or not usage:
        return 0, 0, False
    has_prompt = prompt_key in usage
    has_completion = completion_key in usage
    if not (has_prompt or has_completion):
        return 0, 0, False
    return int(usage.get(prompt_key) or 0), int(usage.get(completion_key) or 0), True
