"""Context compiler accounting, isolation, and provenance."""

from __future__ import annotations

import pytest

from llm_fabric.context.blocks import (
    Cacheability,
    ContextBlock,
    ContextType,
    Priority,
    Provenance,
    TrustLevel,
)
from llm_fabric.context.compiler import ContextCompiler, compile_chat
from llm_fabric.context.record import ContextRecord
from llm_fabric.contract.openai import ChatCompletionRequest, ChatMessage
from llm_fabric.errors import TenantIsolationError
from llm_fabric.observability.metric import CountProvenance, MetricHealth, provenance_missing
from llm_fabric.tenancy.scope import TenantScope


def _request(*messages: ChatMessage, max_tokens: int | None = 16) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="auto", messages=list(messages), max_tokens=max_tokens)


def test_absent_types_are_zero_not_unavailable() -> None:
    compiled = compile_chat(
        _request(ChatMessage(role="user", content="hello there")),
        TenantScope(tenant_id="acme"),
        request_id="r1",
    )
    record = compiled.record
    assert record.user_tokens.value and record.user_tokens.value > 0
    assert record.system_tokens.value == 0
    assert record.system_tokens.provenance is not CountProvenance.UNAVAILABLE
    assert record.retrieval_tokens.value == 0
    assert record.long_term_memory_tokens.value == 0
    assert record.tokens_compressed.value == 0
    assert record.tokens_summarized.value == 0
    assert "compressor" in (record.tokens_compressed.note or "")
    assert record.model_context_limit.health is MetricHealth.UNAVAILABLE
    assert record.model_context_limit.value is None
    assert record.context_utilization_percent.value is None
    assert provenance_missing(record.observations()) == 0
    assert all(block.tenant_id == "acme" for block in record.blocks)


def test_known_window_fills_utilization() -> None:
    compiled = compile_chat(
        _request(
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="hi"),
        ),
        TenantScope(tenant_id="acme"),
        context_window=2048,
    )
    assert compiled.record.model_context_limit.value == 2048
    assert compiled.record.context_utilization_percent.provenance is CountProvenance.DERIVED
    assert compiled.record.context_utilization_percent.value is not None
    prefix = compiled.record.stable_prefix_tokens.value
    assert prefix and prefix > 0
    assert "not a runtime KV hit" in (compiled.record.stable_prefix_tokens.note or "")


def test_exact_duplicate_is_deduplicated() -> None:
    compiled = compile_chat(
        _request(
            ChatMessage(role="system", content="policy"),
            ChatMessage(role="system", content="policy"),
            ChatMessage(role="user", content="ask"),
        ),
        TenantScope(tenant_id="acme"),
    )
    removed = compiled.record.tokens_deduplicated.value
    assert removed and removed > 0
    assert compiled.record.context_tokens_after_optimization.value < (
        compiled.record.context_tokens_before_optimization.value or 0
    )


def test_foreign_extra_block_is_rejected() -> None:
    foreign = ContextBlock(
        block_id="other-mem",
        content="secret",
        type=ContextType.LONG_TERM_MEMORY,
        tenant_id="other",
        provenance=Provenance(origin="memory"),
        trust_level=TrustLevel.USER,
        priority=Priority.LOW,
        cacheability=Cacheability.VOLATILE,
    )
    with pytest.raises(TenantIsolationError, match="foreign context"):
        ContextCompiler().compile(
            _request(ChatMessage(role="user", content="hi")),
            TenantScope(tenant_id="acme"),
            extra_blocks=(foreign,),
        )


def test_context_record_as_dict_withholds_prompt_text() -> None:
    record = ContextRecord(tenant_id="acme")
    payload = record.as_dict(include_content=False)
    assert payload["accounting"]["raw_input_tokens"]["provenance"]
    dumped = str(payload)
    assert "secret prompt" not in dumped
