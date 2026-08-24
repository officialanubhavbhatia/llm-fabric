"""Stable-prefix ordering for compiled context.

Provider prefix caches key on an exact leading byte sequence. Volatile blocks
(user turns, tool results) must sit after stable policy so a new question does
not invalidate the cached prefix. This labelling is a fabric claim about prompt
shape. It is not evidence that a runtime reused KV blocks.
"""

from __future__ import annotations

from llm_fabric.context.blocks import Cacheability, ContextBlock, TrustLevel


def order_blocks(blocks: tuple[ContextBlock, ...]) -> tuple[ContextBlock, ...]:
    """Stable and semi-stable first, then by trust, then original order."""
    indexed = list(enumerate(blocks))
    indexed.sort(
        key=lambda item: (
            item[1].cacheability.ordinal,
            item[1].trust_level.ordinal,
            item[0],
        )
    )
    return tuple(block for _, block in indexed)


def split_stable_prefix(
    blocks: tuple[ContextBlock, ...],
) -> tuple[tuple[ContextBlock, ...], tuple[ContextBlock, ...]]:
    """Leading run of STABLE/SEMI_STABLE blocks, then the rest."""
    cut = 0
    for block in blocks:
        if block.cacheability in (Cacheability.STABLE, Cacheability.SEMI_STABLE):
            cut += 1
            continue
        break
    return blocks[:cut], blocks[cut:]


def default_cacheability(trust: TrustLevel) -> Cacheability:
    if trust.is_authoritative:
        return Cacheability.STABLE
    return Cacheability.VOLATILE
