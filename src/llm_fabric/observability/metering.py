"""Per-request usage accounting.

A routing decision is only defensible if it can be explained afterwards, so every
served request produces a record naming the model asked for, the model actually
used, the policy that chose it, and every attempt made along the way.

Cost is computed from registry prices. When a backend did not report token counts
the fabric estimates them, and the record says so via `cost_is_estimated` — an
estimated cost is never presented as a measured one.

The sink here keeps records in memory with a bounded buffer. That is sufficient
for local operation and for tests, and it is **not durable**: records are lost on
restart, and this process is the only one that can see them. A persistent sink is
not part of this phase.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

DEFAULT_BUFFER = 1000


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    model_id: str
    provider: str
    duration_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class UsageRecord:
    request_id: str
    requested_model: str
    served_model: str
    provider: str
    policy: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cost_is_estimated: bool
    latency_ms: float
    streamed: bool
    failover_count: int
    attempts: tuple[AttemptRecord, ...] = ()
    client_id: str | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MeteringSink(Protocol):
    def record(self, usage: UsageRecord) -> None: ...


@dataclass(frozen=True, slots=True)
class MeterTotals:
    requests: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    estimated_cost_requests: int
    failovers: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class InMemoryMeter:
    """Bounded, in-process metering sink. Not durable across restarts."""

    def __init__(self, buffer_size: int = DEFAULT_BUFFER) -> None:
        self._records: deque[UsageRecord] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()

    def record(self, usage: UsageRecord) -> None:
        with self._lock:
            self._records.append(usage)

    def recent(self, limit: int = 50) -> list[UsageRecord]:
        with self._lock:
            records = list(self._records)
        return records[-limit:][::-1]

    def totals(self) -> MeterTotals:
        with self._lock:
            records = list(self._records)
        return MeterTotals(
            requests=len(records),
            prompt_tokens=sum(r.prompt_tokens for r in records),
            completion_tokens=sum(r.completion_tokens for r in records),
            cost_usd=sum(r.cost_usd for r in records),
            estimated_cost_requests=sum(1 for r in records if r.cost_is_estimated),
            failovers=sum(r.failover_count for r in records),
        )
