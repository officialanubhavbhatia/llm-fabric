"""Tests for the token heuristic.

These assert the heuristic's contract — non-negative, monotonic, per-message
overhead — and deliberately do not assert accuracy against a real tokenizer,
because the heuristic makes no accuracy claim.
"""

from __future__ import annotations

from llm_fabric.contract.openai import ChatMessage
from llm_fabric.serving.tokens import (
    approximate_prompt_tokens,
    approximate_token_count,
    usage_from_provider,
)


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


def test_usage_from_provider_requires_a_token_field() -> None:
    empty = usage_from_provider(
        None, prompt_key="prompt_tokens", completion_key="completion_tokens"
    )
    blank = usage_from_provider({}, prompt_key="prompt_tokens", completion_key="completion_tokens")
    unrelated = usage_from_provider(
        {"total_tokens": 9}, prompt_key="prompt_tokens", completion_key="completion_tokens"
    )
    present = usage_from_provider(
        {"prompt_tokens": 3, "completion_tokens": 2},
        prompt_key="prompt_tokens",
        completion_key="completion_tokens",
    )
    assert empty == (0, 0, False)
    assert blank == (0, 0, False)
    assert unrelated == (0, 0, False)
    assert present == (3, 2, True)
