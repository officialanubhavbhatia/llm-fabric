"""Expected degraded behaviour when a collaborator disappears.

Chaos tests assert these outcomes rather than "no exception".
"""

from __future__ import annotations

from enum import StrEnum


class DegradedMode(StrEnum):
    FAIL_CLOSED = "fail_closed"
    FALLBACK = "fallback"
    RETRY_ONCE = "retry_once"
    REJECT = "reject"
    DEGRADE = "degrade"
    CONTINUE_WITHOUT_TELEMETRY = "continue_without_telemetry"


EXPECTED: dict[str, DegradedMode] = {
    "redis_unavailable_production_revocation": DegradedMode.FAIL_CLOSED,
    "redis_unavailable_development_revocation": DegradedMode.DEGRADE,
    "postgres_unavailable_production_startup": DegradedMode.FAIL_CLOSED,
    "postgres_unavailable_runtime_read": DegradedMode.FAIL_CLOSED,
    "otel_collector_unavailable": DegradedMode.CONTINUE_WITHOUT_TELEMETRY,
    "provider_timeout": DegradedMode.FALLBACK,
    "provider_429": DegradedMode.FALLBACK,
    "provider_500": DegradedMode.FALLBACK,
    "provider_malformed": DegradedMode.FALLBACK,
    "partial_stream_disconnect": DegradedMode.DEGRADE,
    "ollama_unavailable": DegradedMode.FALLBACK,
    "quota_store_unavailable": DegradedMode.FAIL_CLOSED,
    "cache_unavailable": DegradedMode.DEGRADE,
    "classifier_unavailable": DegradedMode.DEGRADE,
    "evaluator_unavailable": DegradedMode.REJECT,
    "context_overflow": DegradedMode.REJECT,
    "extreme_output_tokens": DegradedMode.REJECT,
    "tool_loop": DegradedMode.REJECT,
    "fallback_loop": DegradedMode.REJECT,
    "malformed_structured_output": DegradedMode.REJECT,
    "queue_saturation": DegradedMode.REJECT,
}
