"""Usage accounting: request records and invocation events.

A routing decision is only defensible if it can be explained afterwards, so every
served request produces a `UsageRecord` naming the model asked for, the model
actually used, the policy that chose it, and every attempt made along the way.

That record is the OpenAI-compatible *request* grain: tokens of the visible
response. Each provider invocation is a separate `UsageEvent` on the durable
ledger. Fallback attempts, retries, and (when they exist) internal model calls
are counted there even when the client only sees the last model.

Cost is computed from registry prices. When a backend did not report token counts
the fabric estimates them, and both the request record (`cost_is_estimated`) and
the invocation event (`token_source`) say so — an estimated figure is never
presented as a measured one.

Reads are tenant-filtered. Fleet aggregation requires `fabric:observe`.

`InMemoryMeter` is the test/dev sink: bounded, process-local, lost on restart.
`DurableMeter` writes invocation events to PostgreSQL (authoritative) and, after
a new insert, increments Redis fast counters. Worker identity does not affect
totals.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from llm_fabric.observability.usage_event import (
    InvocationTotals,
    PersistResult,
    TokenSource,
    UsageEvent,
    UsageStatus,
)
from llm_fabric.storage.usage import (
    RETRY_BUFFER,
    RedisUsageAggregates,
    UsageLedger,
    totals_from_events,
)
from llm_fabric.tenancy.scope import TenantScope

DEFAULT_BUFFER = 1000
_LOG = logging.getLogger("llm_fabric")


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
    tenant_id: str = "public"
    user_id: str | None = None
    error: str | None = None
    intent_id: str | None = None
    intent_layer: str | None = None
    intent_confidence: float | None = None
    intent_cache_hit: bool | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    trace_id: str | None = None
    context_record_id: str | None = None
    created_at: float = field(default_factory=time.time)
    invocation_count: int = 1
    ledger_prompt_tokens: int | None = None
    ledger_completion_tokens: int | None = None
    selected_tier: str | None = None

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


def records_from_events(events: Sequence[UsageEvent]) -> list[UsageRecord]:
    """Request-level rows derived from invocation events.

    OpenAI-compatible token fields come from the last successful (or last)
    invocation. `ledger_*` fields and `invocation_count` cover every attempt.
    """
    grouped: dict[str, list[UsageEvent]] = {}
    for event in events:
        grouped.setdefault(event.request_id, []).append(event)
    records: list[UsageRecord] = []
    for request_id, group in grouped.items():
        group = sorted(group, key=lambda item: (item.attempt_number, item.completed_at))
        visible = next((item for item in reversed(group) if item.status == "success"), group[-1])
        attempts = tuple(
            AttemptRecord(
                model_id=item.model,
                provider=item.provider,
                duration_ms=max(0.0, (item.completed_at - item.started_at) * 1000),
                error=item.error,
            )
            for item in group
        )
        records.append(
            UsageRecord(
                request_id=request_id,
                requested_model=visible.requested_model or visible.model,
                served_model=visible.model,
                provider=visible.provider,
                policy=visible.policy or "",
                prompt_tokens=visible.prompt_tokens,
                completion_tokens=visible.completion_tokens,
                cost_usd=sum(item.provider_cost_usd or 0.0 for item in group),
                cost_is_estimated=visible.token_source != TokenSource.PROVIDER_MEASURED,
                latency_ms=max(0.0, (group[-1].completed_at - group[0].started_at) * 1000),
                streamed=any(item.streaming for item in group),
                failover_count=max(0, len(group) - 1),
                attempts=attempts,
                tenant_id=visible.tenant_id,
                user_id=visible.user_id,
                error=group[-1].error if group[-1].status != "success" else None,
                intent_id=visible.intent_id,
                trace_id=visible.trace_id,
                created_at=group[-1].completed_at,
                invocation_count=len(group),
                ledger_prompt_tokens=sum(item.prompt_tokens for item in group),
                ledger_completion_tokens=sum(item.completion_tokens for item in group),
            )
        )
    records.sort(key=lambda item: item.created_at, reverse=True)
    return records


class UsageMeter:
    """Shared surface for process-local and durable meters."""

    durable: bool = False

    def record(self, usage: UsageRecord) -> None:
        raise NotImplementedError

    def record_events(self, events: Sequence[UsageEvent]) -> list[PersistResult]:
        return [PersistResult(inserted=True) for _ in events]

    def recent(
        self,
        limit: int = 50,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        observe: bool = False,
    ) -> list[UsageRecord]:
        raise NotImplementedError

    def totals(
        self, *, tenant_id: str | None = None, user_id: str | None = None, observe: bool = False
    ) -> MeterTotals:
        raise NotImplementedError

    def invocation_totals(
        self, *, tenant_id: str | None = None, user_id: str | None = None, observe: bool = False
    ) -> InvocationTotals:
        raise NotImplementedError

    def recent_events(
        self,
        limit: int = 200,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        observe: bool = False,
        request_id: str | None = None,
        trace_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[UsageEvent]:
        raise NotImplementedError

    def scope_note(self, *, fleet: bool, tenant_id: str) -> str:
        if fleet:
            return "fleet-wide, in-memory, this process only, lost on restart"
        return f"tenant '{tenant_id}', in-memory, this process only, lost on restart"


class InMemoryMeter(UsageMeter):
    """Bounded, in-process metering sink. Not durable across restarts."""

    def __init__(self, buffer_size: int = DEFAULT_BUFFER) -> None:
        self._records: deque[UsageRecord] = deque(maxlen=buffer_size)
        self._events: deque[UsageEvent] = deque(maxlen=buffer_size * 4)
        self._lock = threading.Lock()

    def record(self, usage: UsageRecord) -> None:
        with self._lock:
            self._records.append(usage)

    def record_events(self, events: Sequence[UsageEvent]) -> list[PersistResult]:
        results: list[PersistResult] = []
        with self._lock:
            known = {item.event_id for item in self._events}
            for event in events:
                if event.event_id in known:
                    results.append(PersistResult(inserted=False, duplicate=True))
                    continue
                self._events.append(event)
                known.add(event.event_id)
                results.append(PersistResult(inserted=True))
        return results

    def recent(
        self,
        limit: int = 50,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        observe: bool = False,
    ) -> list[UsageRecord]:
        del observe
        records = self._snapshot(tenant_id, user_id)
        return records[-limit:][::-1]

    def totals(
        self, *, tenant_id: str | None = None, user_id: str | None = None, observe: bool = False
    ) -> MeterTotals:
        del observe
        records = self._snapshot(tenant_id, user_id)
        return MeterTotals(
            requests=len(records),
            prompt_tokens=sum(r.prompt_tokens for r in records),
            completion_tokens=sum(r.completion_tokens for r in records),
            cost_usd=sum(r.cost_usd for r in records),
            estimated_cost_requests=sum(1 for r in records if r.cost_is_estimated),
            failovers=sum(r.failover_count for r in records),
        )

    def invocation_totals(
        self, *, tenant_id: str | None = None, user_id: str | None = None, observe: bool = False
    ) -> InvocationTotals:
        del observe
        events = self._event_snapshot(tenant_id, user_id)
        if events:
            return totals_from_events(events)
        records = self._snapshot(tenant_id, user_id)
        by_source: dict[str, int] = {}
        estimated = 0
        for record in records:
            source = (
                TokenSource.LOCAL_TOKENIZER_ESTIMATE.value
                if record.cost_is_estimated
                else TokenSource.PROVIDER_MEASURED.value
            )
            by_source[source] = by_source.get(source, 0) + 1
            if record.cost_is_estimated:
                estimated += 1
        return InvocationTotals(
            invocations=len(records),
            requests=len(records),
            prompt_tokens=sum(r.prompt_tokens for r in records),
            completion_tokens=sum(r.completion_tokens for r in records),
            provider_cost_usd=sum(r.cost_usd for r in records),
            compute_cost_estimate_usd=None,
            by_token_source=by_source,
            estimated_invocations=estimated,
            unavailable_invocations=0,
            failovers=sum(r.failover_count for r in records),
        )

    def recent_events(
        self,
        limit: int = 200,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        observe: bool = False,
        request_id: str | None = None,
        trace_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[UsageEvent]:
        del observe
        events = self._event_snapshot(tenant_id, user_id)
        if request_id is not None:
            events = [event for event in events if event.request_id == request_id]
        if trace_id is not None:
            events = [event for event in events if event.trace_id == trace_id]
        if provider is not None:
            events = [event for event in events if event.provider == provider]
        if model is not None:
            events = [event for event in events if event.model == model]
        return events[-limit:][::-1]

    def _snapshot(self, tenant_id: str | None, user_id: str | None = None) -> list[UsageRecord]:
        """Records visible to a caller.

        `tenant_id` of `None` means the process-wide buffer and is reserved for
        operators and tests; every request-serving path passes a tenant, because
        usage is one of the surfaces the constitution puts behind the tenant
        boundary.
        """
        with self._lock:
            records = list(self._records)
        if tenant_id is not None:
            records = [record for record in records if record.tenant_id == tenant_id]
        if user_id is not None:
            records = [record for record in records if record.user_id == user_id]
        return records

    def _event_snapshot(
        self, tenant_id: str | None, user_id: str | None = None
    ) -> list[UsageEvent]:
        with self._lock:
            events = list(self._events)
        if tenant_id is not None:
            events = [event for event in events if event.tenant_id == tenant_id]
        if user_id is not None:
            events = [event for event in events if event.user_id == user_id]
        return events


class DurableMeter(UsageMeter):
    """PostgreSQL ledger plus optional Redis fast counters.

    Persist sequence: provider result → durable insert → best-effort Redis INCR
    → return. Redis is never incremented for a duplicate event. There is no
    distributed transaction across provider, Postgres, Redis, and HTTP.
    """

    durable = True

    def __init__(
        self,
        engine: Any,
        *,
        redis_client: Any | None = None,
        retry_buffer: int = RETRY_BUFFER,
    ) -> None:
        self._ledger = UsageLedger(engine)
        self._aggregates = RedisUsageAggregates(redis_client) if redis_client is not None else None
        self._retry: deque[UsageEvent] = deque()
        self._retry_max = retry_buffer
        self._lock = threading.Lock()
        self.dropped_events = 0
        self.deferred_events = 0
        self.last_persist_error: str | None = None
        self.last_persist_ms: float | None = None
        self._dependency_health: Any | None = None

    def bind_dependency_health(self, health: Any) -> None:
        self._dependency_health = health

    @property
    def retry_depth(self) -> int:
        with self._lock:
            return len(self._retry)

    def scope_note(self, *, fleet: bool, tenant_id: str) -> str:
        if fleet:
            return "fleet-wide, postgres usage ledger"
        return f"tenant '{tenant_id}', postgres usage ledger"

    def record(self, usage: UsageRecord) -> None:
        # Request-level rows are derived from invocation events. `record` is
        # kept so heal/eval callers that only have a UsageRecord still compile.
        del usage

    def record_events(self, events: Sequence[UsageEvent]) -> list[PersistResult]:
        self._flush_retry()
        results: list[PersistResult] = []
        for event in events:
            results.append(self._persist_one(event))
        return results

    def _persist_one(self, event: UsageEvent) -> PersistResult:
        started = time.perf_counter()
        try:
            result = self._ledger.insert(event)
            self.last_persist_ms = (time.perf_counter() - started) * 1000
            self.last_persist_error = None
            if result.inserted and self._aggregates is not None:
                try:
                    self._aggregates.apply(event)
                except Exception as exc:  # noqa: BLE001 - Redis is best-effort
                    _LOG.warning("usage redis aggregate failed", extra={"error": str(exc)})
            return result
        except Exception as exc:  # noqa: BLE001 - generation already happened
            self.last_persist_error = str(exc)
            self.last_persist_ms = (time.perf_counter() - started) * 1000
            _LOG.error(
                "usage event persist failed",
                extra={
                    "event_id": event.event_id,
                    "request_id": event.request_id,
                    "error": str(exc),
                },
            )
            if self._is_postgres_unavailable(exc) and self._dependency_health is not None:
                self._dependency_health.observe_serving_failure(
                    "postgres", reason="serving_failure"
                )
            return self._defer(event)

    @staticmethod
    def _is_postgres_unavailable(exc: BaseException) -> bool:
        from sqlalchemy.exc import InterfaceError, OperationalError

        return isinstance(exc, (OperationalError, InterfaceError, TimeoutError, OSError))

    def _defer(self, event: UsageEvent) -> PersistResult:
        with self._lock:
            if len(self._retry) >= self._retry_max:
                self.dropped_events += 1
                _LOG.error(
                    "usage retry buffer full; event dropped",
                    extra={"event_id": event.event_id, "buffered": len(self._retry)},
                )
                return PersistResult(inserted=False, dropped=True)
            self._retry.append(event)
            self.deferred_events += 1
        return PersistResult(inserted=False, deferred=True)

    def _flush_retry(self) -> None:
        with self._lock:
            pending = list(self._retry)
            self._retry.clear()
        still: list[UsageEvent] = []
        for event in pending:
            try:
                result = self._ledger.insert(event)
                if result.inserted and self._aggregates is not None:
                    self._aggregates.apply(event)
            except Exception:
                still.append(event)
        if still:
            with self._lock:
                for event in still:
                    if len(self._retry) >= self._retry_max:
                        self.dropped_events += 1
                    else:
                        self._retry.append(event)

    def recent(
        self,
        limit: int = 50,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        observe: bool = False,
    ) -> list[UsageRecord]:
        events = self._ledger.list_events(
            tenant_id=tenant_id,
            observe=observe,
            user_id=user_id,
            limit=max(limit * 8, 200),
        )
        return records_from_events(events)[:limit]

    def totals(
        self, *, tenant_id: str | None = None, user_id: str | None = None, observe: bool = False
    ) -> MeterTotals:
        records = self.recent(
            limit=1_000_000, tenant_id=tenant_id, user_id=user_id, observe=observe
        )
        return MeterTotals(
            requests=len(records),
            prompt_tokens=sum(r.prompt_tokens for r in records),
            completion_tokens=sum(r.completion_tokens for r in records),
            cost_usd=sum(r.cost_usd for r in records),
            estimated_cost_requests=sum(1 for r in records if r.cost_is_estimated),
            failovers=sum(r.failover_count for r in records),
        )

    def invocation_totals(
        self, *, tenant_id: str | None = None, user_id: str | None = None, observe: bool = False
    ) -> InvocationTotals:
        return self._ledger.totals(tenant_id=tenant_id, observe=observe, user_id=user_id)

    def recent_events(
        self,
        limit: int = 200,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        observe: bool = False,
        request_id: str | None = None,
        trace_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[UsageEvent]:
        return self._ledger.list_events(
            tenant_id=tenant_id,
            observe=observe,
            user_id=user_id,
            request_id=request_id,
            trace_id=trace_id,
            provider=provider,
            model=model,
            limit=limit,
        )

    @property
    def ledger(self) -> UsageLedger:
        return self._ledger

    @property
    def aggregates(self) -> RedisUsageAggregates | None:
        return self._aggregates


def build_meter(
    *,
    engine: Any | None = None,
    redis_client: Any | None = None,
) -> UsageMeter:
    if engine is not None:
        return DurableMeter(engine, redis_client=redis_client)
    return InMemoryMeter()


def events_from_decision(
    decision: Any,
    *,
    request_id: str,
    scope: TenantScope,
    prompt_tokens: int,
    completion_tokens: int,
    token_source: TokenSource,
    streamed: bool,
    error: str | None,
    intent_id: str | None,
    intent_result_id: str | None = None,
    taxonomy_version: str | None = None,
    classifier_version: str | None = None,
    context_record_id: str | None = None,
    trace_id: str | None = None,
    spec: Any | None = None,
    now: float | None = None,
) -> list[UsageEvent]:
    """Build one invocation event per `Attempt`. Empty when no provider call ran."""
    del spec
    completed_at = now if now is not None else time.time()
    attempts = getattr(decision, "attempts", None) or []
    events: list[UsageEvent] = []
    for index, attempt in enumerate(attempts):
        is_last = index == len(attempts) - 1
        source = getattr(attempt, "token_source", TokenSource.UNAVAILABLE.value)
        prompt = int(getattr(attempt, "prompt_tokens", 0) or 0)
        completion = int(getattr(attempt, "completion_tokens", 0) or 0)
        status = UsageStatus.ERROR.value if attempt.error else UsageStatus.SUCCESS.value
        if is_last:
            prompt = prompt_tokens
            completion = completion_tokens
            source = (
                token_source.value if isinstance(token_source, TokenSource) else str(token_source)
            )
            if error:
                status = (
                    UsageStatus.CANCELLED.value if error == "cancelled" else UsageStatus.ERROR.value
                )
        invocation_id = getattr(attempt, "invocation_id", None) or f"{request_id}:{index + 1}"
        started = getattr(attempt, "started_at", None) or (
            completed_at - attempt.duration_ms / 1000
        )
        finished = getattr(attempt, "completed_at", None) or completed_at
        provider_cost = getattr(attempt, "provider_cost_usd", None)
        compute_cost = getattr(attempt, "compute_cost_estimate_usd", None)
        events.append(
            UsageEvent(
                event_id=invocation_id,
                invocation_id=invocation_id,
                request_id=request_id,
                trace_id=trace_id,
                tenant_id=scope.tenant_id,
                user_id=scope.user_id,
                project_id=scope.project_id,
                provider=attempt.provider,
                model=attempt.model_id,
                requested_model=getattr(decision, "requested_model", None),
                policy=getattr(decision, "policy", None),
                deployment_id=getattr(attempt, "deployment_id", None) or None,
                route_id=getattr(decision, "route_id", None) or request_id,
                operation=getattr(attempt, "operation", "USER_RESPONSE"),
                intent_id=intent_id,
                intent_result_id=intent_result_id,
                taxonomy_version=taxonomy_version,
                classifier_version=classifier_version,
                context_record_id=context_record_id,
                prompt_tokens=prompt,
                completion_tokens=completion,
                token_source=str(source),
                provider_cost_usd=provider_cost,
                compute_cost_estimate_usd=compute_cost,
                started_at=started,
                completed_at=finished,
                status=status,
                fallback_depth=int(getattr(attempt, "depth", 0) or 0),
                attempt_number=index + 1,
                streaming=streamed if is_last else False,
                error=attempt.error if not is_last else (error or attempt.error),
                provider_adapter=getattr(attempt, "provider_adapter", None) or None,
                transport=getattr(attempt, "transport", None) or None,
                runtime=getattr(attempt, "runtime", None) or None,
                grade=getattr(attempt, "grade", None),
                litellm_model=getattr(attempt, "litellm_model", None),
                actual_served_model=getattr(attempt, "actual_served_model", None),
            )
        )
    return events
