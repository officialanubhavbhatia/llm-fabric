"""Immutable provider-invocation usage events.

Three grains are kept distinct:

* **Invocation** — one actual model/provider call (`UsageEvent`).
* **Request** — the OpenAI-compatible aggregate for the visible response
  (`UsageRecord` in `metering`). Fallback and internal calls are not folded
  into that object.
* **Rollup** — tenant/user/project/day counters. Redis holds a fast copy;
  PostgreSQL `usage_events` is authoritative.

`event_id` equals `invocation_id`. Replaying the same identifier inserts once.
A real retry constructs a new `Attempt` and therefore a new identifier, so it
is counted separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class TokenSource(StrEnum):
    """Where token counts came from. Estimates must never look measured."""

    PROVIDER_MEASURED = "PROVIDER_MEASURED"
    LOCAL_TOKENIZER_ESTIMATE = "LOCAL_TOKENIZER_ESTIMATE"
    DERIVED = "DERIVED"
    UNAVAILABLE = "UNAVAILABLE"


class UsageOperation(StrEnum):
    """Why this invocation ran. Lets economics include internal model calls."""

    USER_RESPONSE = "USER_RESPONSE"
    INTENT_CLASSIFIER = "INTENT_CLASSIFIER"
    GUARDRAIL = "GUARDRAIL"
    EVALUATOR = "EVALUATOR"
    ROUTER = "ROUTER"
    REPAIR = "REPAIR"
    AGENT = "AGENT"
    OTHER_INTERNAL = "OTHER_INTERNAL"


class UsageStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    TRUNCATED = "truncated"


def token_source_for_provider(*, reported: bool, tokens_known: bool) -> TokenSource:
    if reported and tokens_known:
        return TokenSource.PROVIDER_MEASURED
    if tokens_known:
        return TokenSource.LOCAL_TOKENIZER_ESTIMATE
    return TokenSource.UNAVAILABLE


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """One provider invocation. Never contains prompt or response text."""

    event_id: str
    request_id: str
    invocation_id: str
    tenant_id: str
    provider: str
    model: str
    requested_model: str | None = None
    policy: str | None = None
    operation: str = UsageOperation.USER_RESPONSE.value
    status: str = UsageStatus.SUCCESS.value
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    token_source: str = TokenSource.UNAVAILABLE.value
    started_at: float = 0.0
    completed_at: float = 0.0
    trace_id: str | None = None
    user_id: str | None = None
    project_id: str | None = None
    deployment_id: str | None = None
    intent_id: str | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    provider_cost_usd: float | None = None
    compute_cost_estimate_usd: float | None = None
    fallback_depth: int = 0
    attempt_number: int = 1
    streaming: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            object.__setattr__(
                self,
                "total_tokens",
                self.prompt_tokens + self.completion_tokens,
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PersistResult:
    """Outcome of recording one invocation event."""

    inserted: bool
    duplicate: bool = False
    dropped: bool = False
    deferred: bool = False


@dataclass(frozen=True, slots=True)
class InvocationTotals:
    """Authoritative sums over invocation events, not OpenAI response usage."""

    invocations: int
    requests: int
    prompt_tokens: int
    completion_tokens: int
    provider_cost_usd: float
    compute_cost_estimate_usd: float | None
    by_token_source: dict[str, int]
    estimated_invocations: int
    unavailable_invocations: int
    failovers: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
