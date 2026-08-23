"""Tests for the token heuristic.

These assert the heuristic's contract — non-negative, monotonic, per-message
overhead — and deliberately do not assert accuracy against a real tokenizer,
because the heuristic makes no accuracy claim.
"""

from __future__ import annotations

from llm_fabric.contract.openai import ChatMessage
from llm_fabric.serving.tokens import approximate_prompt_tokens, approximate_token_count


def test_empty_text_costs_nothing() -> None:
    assert approximate_token_count("") == 0


def test_any_non_empty_text_costs_at_least_one() -> None:
    assert approximate_token_count("a") == 1


def test_longer_text_never_counts_fewer_tokens() -> None:
    short = approximate_token_count("hello")
    long = approximate_token_count("hello " * 100)
    assert long > short


def test_prompt_tokens_include_per_message_overhead() -> None:
    single = approximate_prompt_tokens([ChatMessage(role="user", content="hello there")])
    split = approximate_prompt_tokens(
        [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="user", content=" there"),
        ]
    )
    assert split > single


def test_no_messages_costs_nothing() -> None:
    assert approximate_prompt_tokens([]) == 0
