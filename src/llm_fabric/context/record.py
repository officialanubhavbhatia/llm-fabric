"""The ContextRecord: token accounting for one compiled prompt.

Unknown is not zero. A type that was not present on the request is recorded as
0 with ESTIMATED or TOKENIZER provenance because the compiler counted it. A
window the compiler does not know is UNAVAILABLE.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from llm_fabric.context.blocks import ContextBlock
from llm_fabric.observability.metric import CountProvenance, MetricScope, Observed


@dataclass(frozen=True, slots=True)
class ContextRecord:
    """What the compiler did, without the raw prompt text."""

    tenant_id: str
    request_id: str | None = None
    context_record_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    blocks: tuple[ContextBlock, ...] = ()
    raw_input_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "raw_input_tokens",
            provenance=CountProvenance.ESTIMATED,
            scope=MetricScope.REQUEST,
            unit="tokens",
        )
    )
    system_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "system_tokens", provenance=CountProvenance.ESTIMATED, scope=MetricScope.REQUEST
        )
    )
    developer_policy_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "developer_policy_tokens",
            provenance=CountProvenance.ESTIMATED,
            scope=MetricScope.REQUEST,
        )
    )
    tenant_policy_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "tenant_policy_tokens", provenance=CountProvenance.ESTIMATED, scope=MetricScope.REQUEST
        )
    )
    conversation_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "conversation_tokens", provenance=CountProvenance.ESTIMATED, scope=MetricScope.REQUEST
        )
    )
    long_term_memory_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "long_term_memory_tokens",
            provenance=CountProvenance.ESTIMATED,
            scope=MetricScope.REQUEST,
        )
    )
    retrieval_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "retrieval_tokens", provenance=CountProvenance.ESTIMATED, scope=MetricScope.REQUEST
        )
    )
    tool_result_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "tool_result_tokens", provenance=CountProvenance.ESTIMATED, scope=MetricScope.REQUEST
        )
    )
    agent_state_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "agent_state_tokens", provenance=CountProvenance.ESTIMATED, scope=MetricScope.REQUEST
        )
    )
    user_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "user_tokens", provenance=CountProvenance.ESTIMATED, scope=MetricScope.REQUEST
        )
    )
    output_contract_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "output_contract_tokens",
            provenance=CountProvenance.ESTIMATED,
            scope=MetricScope.REQUEST,
        )
    )
    context_tokens_before_optimization: Observed = field(
        default_factory=lambda: Observed.zero(
            "context_tokens_before_optimization",
            provenance=CountProvenance.ESTIMATED,
            scope=MetricScope.REQUEST,
        )
    )
    context_tokens_after_optimization: Observed = field(
        default_factory=lambda: Observed.zero(
            "context_tokens_after_optimization",
            provenance=CountProvenance.ESTIMATED,
            scope=MetricScope.REQUEST,
        )
    )
    tokens_deduplicated: Observed = field(
        default_factory=lambda: Observed.zero(
            "tokens_deduplicated", provenance=CountProvenance.DERIVED, scope=MetricScope.REQUEST
        )
    )
    tokens_compressed: Observed = field(
        default_factory=lambda: Observed.zero(
            "tokens_compressed",
            provenance=CountProvenance.DERIVED,
            scope=MetricScope.REQUEST,
            note="no compressor configured; nothing was compressed",
        )
    )
    tokens_summarized: Observed = field(
        default_factory=lambda: Observed.zero(
            "tokens_summarized",
            provenance=CountProvenance.DERIVED,
            scope=MetricScope.REQUEST,
            note="no summarizer configured; nothing was summarized",
        )
    )
    tokens_dropped: Observed = field(
        default_factory=lambda: Observed.zero(
            "tokens_dropped", provenance=CountProvenance.DERIVED, scope=MetricScope.REQUEST
        )
    )
    stable_prefix_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "stable_prefix_tokens", provenance=CountProvenance.DERIVED, scope=MetricScope.REQUEST
        )
    )
    volatile_prompt_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "volatile_prompt_tokens", provenance=CountProvenance.DERIVED, scope=MetricScope.REQUEST
        )
    )
    reserved_output_tokens: Observed = field(
        default_factory=lambda: Observed.zero(
            "reserved_output_tokens",
            provenance=CountProvenance.ESTIMATED,
            scope=MetricScope.REQUEST,
        )
    )
    model_context_limit: Observed = field(
        default_factory=lambda: Observed.unavailable(
            "model_context_limit",
            scope=MetricScope.REQUEST,
            note="no deployment selected at compile time",
        )
    )
    context_utilization_percent: Observed = field(
        default_factory=lambda: Observed.unavailable(
            "context_utilization_percent",
            scope=MetricScope.REQUEST,
            note="requires a known model context limit",
        )
    )
    compile_latency_ms: float = 0.0
    counter_name: str = "approximate-chars-v1"
    token_provenance: CountProvenance = CountProvenance.ESTIMATED

    def observations(self) -> list[Observed]:
        return [
            self.raw_input_tokens,
            self.system_tokens,
            self.developer_policy_tokens,
            self.tenant_policy_tokens,
            self.conversation_tokens,
            self.long_term_memory_tokens,
            self.retrieval_tokens,
            self.tool_result_tokens,
            self.agent_state_tokens,
            self.user_tokens,
            self.output_contract_tokens,
            self.context_tokens_before_optimization,
            self.context_tokens_after_optimization,
            self.tokens_deduplicated,
            self.tokens_compressed,
            self.tokens_summarized,
            self.tokens_dropped,
            self.stable_prefix_tokens,
            self.volatile_prompt_tokens,
            self.reserved_output_tokens,
            self.model_context_limit,
            self.context_utilization_percent,
        ]

    def as_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        return {
            "context_record_id": self.context_record_id,
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "counter_name": self.counter_name,
            "token_provenance": self.token_provenance.value,
            "compile_latency_ms": round(self.compile_latency_ms, 3),
            "blocks": [block.describe(include_content=include_content) for block in self.blocks],
            "accounting": {item.name: item.as_dict() for item in self.observations()},
        }


def measured_tokens(
    name: str, value: int, *, provenance: CountProvenance, note: str | None = None
) -> Observed:
    return Observed(
        name=name,
        value=int(value),
        provenance=provenance,
        scope=MetricScope.REQUEST,
        unit="tokens",
        note=note,
    )
