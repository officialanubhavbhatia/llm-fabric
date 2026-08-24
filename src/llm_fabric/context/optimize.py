"""Exact deduplication and budget truncation.

Semantic redundancy removal, compression and summarization are hooks. They do
not run unless a compressor is supplied. When they do not run, the related
token counters are 0 (nothing was transformed), not UNAVAILABLE.
"""

from __future__ import annotations

from llm_fabric.context.blocks import ContextBlock
from llm_fabric.errors import ContextTooLargeError


def deduplicate(blocks: tuple[ContextBlock, ...]) -> tuple[tuple[ContextBlock, ...], int]:
    """Drop later exact duplicates. Returns (kept, tokens_removed)."""
    seen: set[str] = set()
    kept: list[ContextBlock] = []
    removed = 0
    for block in blocks:
        fingerprint = block.fingerprint
        if fingerprint in seen:
            removed += block.tokens
            continue
        seen.add(fingerprint)
        kept.append(block)
    return tuple(kept), removed


def truncate_to_budget(
    blocks: tuple[ContextBlock, ...],
    *,
    available_tokens: int,
) -> tuple[tuple[ContextBlock, ...], int]:
    """Drop droppable blocks from the tail until the budget fits.

    REQUIRED blocks are never dropped. If they alone exceed the budget, this
    raises rather than shipping a prompt with policy cut out of it.
    """
    required = [block for block in blocks if not block.priority.is_droppable]
    required_tokens = sum(block.tokens for block in required)
    if required_tokens > available_tokens:
        raise ContextTooLargeError(
            f"required context is {required_tokens} tokens; "
            f"only {available_tokens} tokens remain after output reservation"
        )
    droppable = [block for block in blocks if block.priority.is_droppable]
    droppable.sort(key=lambda block: (block.priority.ordinal, -block.tokens))
    kept_droppable: list[ContextBlock] = []
    used = required_tokens
    dropped = 0
    for block in droppable:
        if used + block.tokens <= available_tokens:
            kept_droppable.append(block)
            used += block.tokens
        else:
            dropped += block.tokens
    required_ids = {block.block_id for block in required}
    ordered = tuple(
        block for block in blocks if block.block_id in required_ids or block in kept_droppable
    )
    return ordered, dropped
