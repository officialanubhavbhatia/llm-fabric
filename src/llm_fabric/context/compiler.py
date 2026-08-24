"""Compile chat messages into typed context blocks on the serving path.

The compiler always runs. Types that are absent from the request are counted as
0. A model window that is not yet known is UNAVAILABLE, never guessed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from llm_fabric.context.blocks import (
    Cacheability,
    ContextBlock,
    ContextType,
    Priority,
    Provenance,
    TrustLevel,
    Visibility,
)
from llm_fabric.context.budget import ApproximateTokenCounter, TokenCounter, plan_budget
from llm_fabric.context.optimize import deduplicate, truncate_to_budget
from llm_fabric.context.ordering import order_blocks, split_stable_prefix
from llm_fabric.context.record import ContextRecord, measured_tokens
from llm_fabric.contract.openai import ChatCompletionRequest, ChatMessage
from llm_fabric.errors import TenantIsolationError
from llm_fabric.observability.metric import CountProvenance, MetricScope, Observed
from llm_fabric.tenancy.scope import TenantScope

_TYPE_TRUST: dict[ContextType, TrustLevel] = {
    ContextType.SYSTEM: TrustLevel.SYSTEM,
    ContextType.DEVELOPER_POLICY: TrustLevel.DEVELOPER,
    ContextType.TENANT_POLICY: TrustLevel.TENANT,
    ContextType.CONVERSATION: TrustLevel.USER,
    ContextType.LONG_TERM_MEMORY: TrustLevel.USER,
    ContextType.RETRIEVAL: TrustLevel.RETRIEVED,
    ContextType.TOOL_RESULT: TrustLevel.TOOL,
    ContextType.AGENT_STATE: TrustLevel.USER,
    ContextType.USER: TrustLevel.USER,
    ContextType.OUTPUT_CONTRACT: TrustLevel.DEVELOPER,
}

_TYPE_FIELD: dict[ContextType, str] = {
    ContextType.SYSTEM: "system_tokens",
    ContextType.DEVELOPER_POLICY: "developer_policy_tokens",
    ContextType.TENANT_POLICY: "tenant_policy_tokens",
    ContextType.CONVERSATION: "conversation_tokens",
    ContextType.LONG_TERM_MEMORY: "long_term_memory_tokens",
    ContextType.RETRIEVAL: "retrieval_tokens",
    ContextType.TOOL_RESULT: "tool_result_tokens",
    ContextType.AGENT_STATE: "agent_state_tokens",
    ContextType.USER: "user_tokens",
    ContextType.OUTPUT_CONTRACT: "output_contract_tokens",
}

DEFAULT_RESERVED_OUTPUT = 256


@dataclass(frozen=True, slots=True)
class CompiledContext:
    """Compiler output the serving path must attach before routing."""

    record: ContextRecord
    messages: tuple[ChatMessage, ...]
    request: ChatCompletionRequest


class ContextCompiler:
    """Admit, count, order and budget context for one chat request."""

    def __init__(self, counter: TokenCounter | None = None) -> None:
        self._counter = counter or ApproximateTokenCounter()

    @property
    def provenance(self) -> CountProvenance:
        if self._counter.is_exact:
            return CountProvenance.TOKENIZER_MEASURED
        return CountProvenance.ESTIMATED

    def compile(
        self,
        request: ChatCompletionRequest,
        scope: TenantScope,
        *,
        request_id: str | None = None,
        context_window: int | None = None,
        reserved_output_tokens: int | None = None,
        extra_blocks: tuple[ContextBlock, ...] = (),
    ) -> CompiledContext:
        started = time.perf_counter()
        reserved = (
            reserved_output_tokens
            if reserved_output_tokens is not None
            else (request.max_tokens or DEFAULT_RESERVED_OUTPUT)
        )
        admitted = self._admit(request, scope, extra_blocks)
        counted = tuple(block.with_tokens(self._counter.count(block.content)) for block in admitted)
        before = sum(block.tokens for block in counted)
        deduped, removed_dup = deduplicate(counted)
        ordered = order_blocks(deduped)
        dropped = 0
        if context_window is not None:
            budget = plan_budget(
                context_window=context_window,
                reserved_output_tokens=reserved,
                counter=self._counter,
                block_count=len(ordered),
            )
            ordered, dropped = truncate_to_budget(ordered, available_tokens=budget.available_tokens)
            ordered = order_blocks(ordered)
        after = sum(block.tokens for block in ordered)
        stable, volatile = split_stable_prefix(ordered)
        provenance = self.provenance
        by_type = {field: 0 for field in _TYPE_FIELD.values()}
        for block in ordered:
            by_type[_TYPE_FIELD[block.type]] += block.tokens
        limit = (
            measured_tokens("model_context_limit", context_window, provenance=provenance)
            if context_window is not None
            else Observed.unavailable(
                "model_context_limit",
                scope=MetricScope.REQUEST,
                note="no deployment context window at compile time",
            )
        )
        utilization = Observed.unavailable(
            "context_utilization_percent",
            scope=MetricScope.REQUEST,
            note="requires a known model context limit",
        )
        if context_window is not None and context_window > 0:
            utilization = Observed(
                name="context_utilization_percent",
                value=round(100.0 * after / context_window, 4),
                provenance=CountProvenance.DERIVED,
                scope=MetricScope.REQUEST,
                unit="percent",
            )
        record = ContextRecord(
            tenant_id=scope.tenant_id,
            request_id=request_id,
            blocks=ordered,
            raw_input_tokens=measured_tokens("raw_input_tokens", before, provenance=provenance),
            system_tokens=measured_tokens(
                "system_tokens", by_type["system_tokens"], provenance=provenance
            ),
            developer_policy_tokens=measured_tokens(
                "developer_policy_tokens", by_type["developer_policy_tokens"], provenance=provenance
            ),
            tenant_policy_tokens=measured_tokens(
                "tenant_policy_tokens", by_type["tenant_policy_tokens"], provenance=provenance
            ),
            conversation_tokens=measured_tokens(
                "conversation_tokens", by_type["conversation_tokens"], provenance=provenance
            ),
            long_term_memory_tokens=measured_tokens(
                "long_term_memory_tokens",
                by_type["long_term_memory_tokens"],
                provenance=provenance,
            ),
            retrieval_tokens=measured_tokens(
                "retrieval_tokens", by_type["retrieval_tokens"], provenance=provenance
            ),
            tool_result_tokens=measured_tokens(
                "tool_result_tokens", by_type["tool_result_tokens"], provenance=provenance
            ),
            agent_state_tokens=measured_tokens(
                "agent_state_tokens", by_type["agent_state_tokens"], provenance=provenance
            ),
            user_tokens=measured_tokens(
                "user_tokens", by_type["user_tokens"], provenance=provenance
            ),
            output_contract_tokens=measured_tokens(
                "output_contract_tokens", by_type["output_contract_tokens"], provenance=provenance
            ),
            context_tokens_before_optimization=measured_tokens(
                "context_tokens_before_optimization", before, provenance=provenance
            ),
            context_tokens_after_optimization=measured_tokens(
                "context_tokens_after_optimization", after, provenance=provenance
            ),
            tokens_deduplicated=measured_tokens(
                "tokens_deduplicated", removed_dup, provenance=CountProvenance.DERIVED
            ),
            tokens_compressed=Observed.zero(
                "tokens_compressed",
                provenance=CountProvenance.DERIVED,
                scope=MetricScope.REQUEST,
                note="no compressor configured; nothing was compressed",
                unit="tokens",
            ),
            tokens_summarized=Observed.zero(
                "tokens_summarized",
                provenance=CountProvenance.DERIVED,
                scope=MetricScope.REQUEST,
                note="no summarizer configured; nothing was summarized",
                unit="tokens",
            ),
            tokens_dropped=measured_tokens(
                "tokens_dropped", dropped, provenance=CountProvenance.DERIVED
            ),
            stable_prefix_tokens=measured_tokens(
                "stable_prefix_tokens",
                sum(block.tokens for block in stable),
                provenance=CountProvenance.DERIVED,
                note="fabric prefix labelling; not a runtime KV hit",
            ),
            volatile_prompt_tokens=measured_tokens(
                "volatile_prompt_tokens",
                sum(block.tokens for block in volatile),
                provenance=CountProvenance.DERIVED,
            ),
            reserved_output_tokens=measured_tokens(
                "reserved_output_tokens", reserved, provenance=CountProvenance.ESTIMATED
            ),
            model_context_limit=limit,
            context_utilization_percent=utilization,
            compile_latency_ms=(time.perf_counter() - started) * 1000,
            counter_name=self._counter.name,
            token_provenance=provenance,
        )
        messages = tuple(_message_for(block) for block in ordered)
        compiled_request = request.model_copy(update={"messages": list(messages)})
        return CompiledContext(record=record, messages=messages, request=compiled_request)

    def bind_model_window(self, record: ContextRecord, context_window: int) -> ContextRecord:
        """Fill deployment window after the route planner selects a model."""
        after = int(record.context_tokens_after_optimization.value or 0)
        utilization = Observed(
            name="context_utilization_percent",
            value=round(100.0 * after / context_window, 4) if context_window else None,
            provenance=CountProvenance.DERIVED,
            scope=MetricScope.REQUEST,
            unit="percent",
        )
        if context_window <= 0:
            utilization = Observed.unavailable(
                "context_utilization_percent",
                scope=MetricScope.REQUEST,
                note="deployment context window is not positive",
            )
        return replace(
            record,
            model_context_limit=measured_tokens(
                "model_context_limit", context_window, provenance=CountProvenance.PROVIDER_MEASURED
            ),
            context_utilization_percent=utilization,
        )

    def _admit(
        self,
        request: ChatCompletionRequest,
        scope: TenantScope,
        extra_blocks: tuple[ContextBlock, ...],
    ) -> tuple[ContextBlock, ...]:
        blocks: list[ContextBlock] = []
        last_user = max(
            (index for index, message in enumerate(request.messages) if message.role == "user"),
            default=-1,
        )
        for index, message in enumerate(request.messages):
            kind, trust, priority, cacheability = _classify_message(message, index == last_user)
            blocks.append(
                ContextBlock(
                    block_id=f"msg-{index}",
                    content=message.content,
                    type=kind,
                    source=message.role,
                    tenant_id=scope.tenant_id,
                    provenance=Provenance(origin="chat.messages", producer="context-compiler"),
                    trust_level=trust,
                    priority=priority,
                    cacheability=cacheability,
                    visibility=Visibility.CALLER,
                )
            )
        for block in extra_blocks:
            if block.tenant_id != scope.tenant_id:
                raise TenantIsolationError(
                    f"refused foreign context block {block.block_id!r} "
                    f"from tenant {block.tenant_id!r}"
                )
            blocks.append(block)
        return tuple(blocks)


def _classify_message(
    message: ChatMessage, is_latest_user: bool
) -> tuple[ContextType, TrustLevel, Priority, Cacheability]:
    if message.role == "system":
        return (
            ContextType.SYSTEM,
            TrustLevel.SYSTEM,
            Priority.REQUIRED,
            Cacheability.STABLE,
        )
    if message.role == "tool":
        return (
            ContextType.TOOL_RESULT,
            TrustLevel.TOOL,
            Priority.NORMAL,
            Cacheability.VOLATILE,
        )
    if message.role == "assistant":
        return (
            ContextType.CONVERSATION,
            TrustLevel.USER,
            Priority.HIGH,
            Cacheability.VOLATILE,
        )
    if is_latest_user:
        return ContextType.USER, TrustLevel.USER, Priority.REQUIRED, Cacheability.VOLATILE
    return (
        ContextType.CONVERSATION,
        TrustLevel.USER,
        Priority.HIGH,
        Cacheability.VOLATILE,
    )


def _message_for(block: ContextBlock) -> ChatMessage:
    return ChatMessage(role=_role_for(block), content=block.content)


def _role_for(block: ContextBlock) -> str:
    if block.source in {"system", "user", "assistant", "tool"}:
        return block.source
    if block.type is ContextType.TOOL_RESULT:
        return "tool"
    if block.type is ContextType.USER:
        return "user"
    if block.trust_level.is_authoritative:
        return "system"
    return "user"


def compile_chat(
    request: ChatCompletionRequest,
    scope: TenantScope,
    *,
    request_id: str | None = None,
    context_window: int | None = None,
    reserved_output_tokens: int | None = None,
) -> CompiledContext:
    return ContextCompiler().compile(
        request,
        scope,
        request_id=request_id,
        context_window=context_window,
        reserved_output_tokens=reserved_output_tokens,
    )
