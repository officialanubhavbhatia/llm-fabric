"""Durable usage ledger and optional Redis fast aggregates.

PostgreSQL `usage_events` is the source of truth. Redis counters are a cache
updated only after a *new* insert. Losing Redis does not erase history.
Replaying an event with the same `event_id` is a no-op.
"""

from __future__ import annotations

import calendar
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from llm_fabric.observability.usage_event import (
    InvocationTotals,
    PersistResult,
    UsageEvent,
    UsageOperation,
)
from llm_fabric.storage.postgres import UsageEventRow, bind_isolation
from llm_fabric.tenancy.scope import TenantScope

RETRY_BUFFER = 256


def event_to_payload(event: UsageEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "invocation_id": event.invocation_id,
        "request_id": event.request_id,
        "trace_id": event.trace_id,
        "tenant_id": event.tenant_id,
        "user_id": event.user_id,
        "project_id": event.project_id,
        "provider": event.provider,
        "model": event.model,
        "requested_model": event.requested_model,
        "policy": event.policy,
        "deployment_id": event.deployment_id,
        "route_id": event.route_id,
        "operation": event.operation,
        "intent_id": event.intent_id,
        "intent_result_id": event.intent_result_id,
        "taxonomy_version": event.taxonomy_version,
        "classifier_version": event.classifier_version,
        "context_record_id": event.context_record_id,
        "prompt_tokens": event.prompt_tokens,
        "completion_tokens": event.completion_tokens,
        "cached_tokens": event.cached_tokens,
        "reasoning_tokens": event.reasoning_tokens,
        "total_tokens": event.total_tokens or (event.prompt_tokens + event.completion_tokens),
        "provider_cost_usd": event.provider_cost_usd,
        "compute_cost_estimate_usd": event.compute_cost_estimate_usd,
        "token_source": event.token_source,
        "started_at": event.started_at,
        "completed_at": event.completed_at,
        "status": event.status,
        "fallback_depth": event.fallback_depth,
        "attempt_number": event.attempt_number,
        "streaming": event.streaming,
        "error": event.error,
        "provider_adapter": event.provider_adapter,
        "transport": event.transport,
        "runtime": event.runtime,
        "grade": event.grade,
        "litellm_model": event.litellm_model,
        "actual_served_model": event.actual_served_model,
    }


def row_to_event(row: UsageEventRow) -> UsageEvent:
    return UsageEvent(
        event_id=row.event_id,
        request_id=row.request_id,
        invocation_id=row.invocation_id,
        tenant_id=row.tenant_id,
        provider=row.provider,
        model=row.model,
        requested_model=row.requested_model,
        policy=row.policy,
        operation=row.operation,
        status=row.status,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        total_tokens=row.total_tokens,
        token_source=row.token_source,
        started_at=row.started_at,
        completed_at=row.completed_at,
        trace_id=row.trace_id,
        user_id=row.user_id,
        project_id=row.project_id,
        deployment_id=row.deployment_id,
        route_id=row.route_id,
        intent_id=row.intent_id,
        intent_result_id=getattr(row, "intent_result_id", None),
        taxonomy_version=getattr(row, "taxonomy_version", None),
        classifier_version=getattr(row, "classifier_version", None),
        context_record_id=getattr(row, "context_record_id", None),
        cached_tokens=row.cached_tokens,
        reasoning_tokens=row.reasoning_tokens,
        provider_cost_usd=row.provider_cost_usd,
        compute_cost_estimate_usd=row.compute_cost_estimate_usd,
        fallback_depth=row.fallback_depth,
        attempt_number=row.attempt_number,
        streaming=row.streaming,
        error=row.error,
        provider_adapter=row.provider_adapter,
        transport=row.transport,
        runtime=row.runtime,
        grade=row.grade,
        litellm_model=row.litellm_model,
        actual_served_model=row.actual_served_model,
    )


class UsageLedger:
    """Idempotent inserts and tenant-scoped reads of `usage_events`."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert(self, event: UsageEvent) -> PersistResult:
        payload = event_to_payload(event)
        with Session(self._engine) as session:
            bind_isolation(session, event.tenant_id)
            existing = session.get(UsageEventRow, event.event_id)
            if existing is not None:
                return PersistResult(inserted=False, duplicate=True)
            session.add(UsageEventRow(**payload))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return PersistResult(inserted=False, duplicate=True)
        return PersistResult(inserted=True)

    def get(self, event_id: str, *, tenant_id: str) -> UsageEvent | None:
        with Session(self._engine) as session:
            bind_isolation(session, tenant_id)
            row = session.get(UsageEventRow, event_id)
        if row is None or row.tenant_id != tenant_id:
            return None
        return row_to_event(row)

    def list_events(
        self,
        *,
        tenant_id: str | None,
        observe: bool = False,
        user_id: str | None = None,
        project_id: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 1000,
    ) -> list[UsageEvent]:
        if tenant_id is None and not observe:
            raise PermissionError("fleet usage reads require explicit observe scope")
        bind_tenant = tenant_id or ""
        with Session(self._engine) as session:
            bind_isolation(session, bind_tenant, observe=observe)
            query = select(UsageEventRow)
            if tenant_id is not None and not observe:
                query = query.where(UsageEventRow.tenant_id == tenant_id)
            if user_id is not None:
                query = query.where(UsageEventRow.user_id == user_id)
            if project_id is not None:
                query = query.where(UsageEventRow.project_id == project_id)
            if request_id is not None:
                query = query.where(UsageEventRow.request_id == request_id)
            if trace_id is not None:
                query = query.where(UsageEventRow.trace_id == trace_id)
            if provider is not None:
                query = query.where(UsageEventRow.provider == provider)
            if model is not None:
                query = query.where(UsageEventRow.model == model)
            if since is not None:
                query = query.where(UsageEventRow.completed_at >= since)
            if until is not None:
                query = query.where(UsageEventRow.completed_at < until)
            query = query.order_by(UsageEventRow.completed_at.desc()).limit(limit)
            rows = session.execute(query).scalars().all()
        if tenant_id is not None and not observe:
            rows = [row for row in rows if row.tenant_id == tenant_id]
        return [row_to_event(row) for row in rows]

    def totals(
        self,
        *,
        tenant_id: str | None,
        observe: bool = False,
        user_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> InvocationTotals:
        events = self.list_events(
            tenant_id=tenant_id,
            observe=observe,
            user_id=user_id,
            since=since,
            until=until,
            limit=1_000_000,
        )
        return totals_from_events(events)

    def request_ids(self, *, tenant_id: str | None, observe: bool = False) -> set[str]:
        events = self.list_events(tenant_id=tenant_id, observe=observe, limit=1_000_000)
        return {event.request_id for event in events}

    def provider_invocations_without_intent(self, *, observe: bool = True) -> int:
        """Count USER_RESPONSE rows missing an IntentResult id. PASS is 0."""
        if not observe:
            raise PermissionError("fleet intent-coverage reads require explicit observe scope")
        with Session(self._engine) as session:
            bind_isolation(session, "", observe=True)
            query = (
                select(func.count())
                .select_from(UsageEventRow)
                .where(
                    UsageEventRow.operation == UsageOperation.USER_RESPONSE.value,
                    or_(
                        UsageEventRow.intent_result_id.is_(None),
                        UsageEventRow.intent_result_id == "",
                    ),
                )
            )
            return int(session.scalar(query) or 0)

    def provider_invocations_without_context_record(self, *, observe: bool = True) -> int:
        """Count USER_RESPONSE rows missing a ContextRecord id. PASS is 0."""
        if not observe:
            raise PermissionError("fleet context-coverage reads require explicit observe scope")
        with Session(self._engine) as session:
            bind_isolation(session, "", observe=True)
            query = (
                select(func.count())
                .select_from(UsageEventRow)
                .where(
                    UsageEventRow.operation == UsageOperation.USER_RESPONSE.value,
                    or_(
                        UsageEventRow.context_record_id.is_(None),
                        UsageEventRow.context_record_id == "",
                    ),
                )
            )
            return int(session.scalar(query) or 0)

    def day_buckets(self, *, tenant_id: str | None, observe: bool = False) -> list[dict[str, Any]]:
        """Ledger sums grouped by tenant and UTC day, for Redis reconciliation."""
        if tenant_id is None and not observe:
            raise PermissionError("fleet usage reads require explicit observe scope")
        bind_tenant = tenant_id or ""
        with Session(self._engine) as session:
            bind_isolation(session, bind_tenant, observe=observe)
            query = select(UsageEventRow)
            if tenant_id is not None and not observe:
                query = query.where(UsageEventRow.tenant_id == tenant_id)
            rows = session.execute(query).scalars().all()
        buckets: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            day = datetime.fromtimestamp(row.completed_at, tz=UTC).strftime("%Y%m%d")
            key = (row.tenant_id, day)
            bucket = buckets.setdefault(
                key,
                {
                    "tenant_id": row.tenant_id,
                    "day": day,
                    "invocations": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "request_ids": set(),
                },
            )
            bucket["invocations"] += 1
            bucket["prompt_tokens"] += row.prompt_tokens
            bucket["completion_tokens"] += row.completion_tokens
            bucket["request_ids"].add(row.request_id)
        out = []
        for bucket in buckets.values():
            ids = bucket.pop("request_ids")
            bucket["requests"] = len(ids)
            out.append(bucket)
        return out


def totals_from_events(events: Sequence[UsageEvent]) -> InvocationTotals:
    by_source: dict[str, int] = {}
    request_ids = {event.request_id for event in events}
    compute_values = [
        event.compute_cost_estimate_usd
        for event in events
        if event.compute_cost_estimate_usd is not None
    ]
    provider_cost = sum(event.provider_cost_usd or 0.0 for event in events)
    for event in events:
        by_source[event.token_source] = by_source.get(event.token_source, 0) + 1
    return InvocationTotals(
        invocations=len(events),
        requests=len(request_ids),
        prompt_tokens=sum(event.prompt_tokens for event in events),
        completion_tokens=sum(event.completion_tokens for event in events),
        provider_cost_usd=provider_cost,
        compute_cost_estimate_usd=sum(compute_values) if compute_values else None,
        by_token_source=by_source,
        estimated_invocations=sum(
            1 for event in events if event.token_source in {"LOCAL_TOKENIZER_ESTIMATE", "DERIVED"}
        ),
        unavailable_invocations=sum(1 for event in events if event.token_source == "UNAVAILABLE"),
        failovers=sum(
            1 for event in events if event.fallback_depth > 0 or event.attempt_number > 1
        ),
    )


def _day_ttl_s() -> int:
    now = datetime.now(tz=UTC)
    last = calendar.timegm((now.year, now.month, now.day, 23, 59, 59))
    return max(60, int(last - now.timestamp()) + 2 * 86_400)


class RedisUsageAggregates:
    """Atomic fast counters. Never the source of truth."""

    PREFIX = "fabric:usage:v1"

    def __init__(self, client: Any) -> None:
        self._client = client

    def _key(self, tenant_id: str, day: str) -> str:
        return f"{self.PREFIX}:{tenant_id}:d:{day}"

    def apply(self, event: UsageEvent) -> None:
        if self._client is None:
            return
        day = datetime.fromtimestamp(event.completed_at or event.started_at, tz=UTC).strftime(
            "%Y%m%d"
        )
        key = self._key(event.tenant_id, day)
        pipe = self._client.pipeline(transaction=True)
        pipe.hincrby(key, "invocations", 1)
        pipe.hincrby(key, "prompt_tokens", int(event.prompt_tokens))
        pipe.hincrby(key, "completion_tokens", int(event.completion_tokens))
        req_flag = f"{self.PREFIX}:req:{event.request_id}:{day}"
        pipe.set(req_flag, "1", nx=True, ex=_day_ttl_s())
        pipe.expire(key, _day_ttl_s())
        results = pipe.execute()
        # results[-2] is True when this request_id was first seen today.
        if results[-2]:
            self._client.hincrby(key, "requests", 1)

    def snapshot(self, tenant_id: str, day: str | None = None) -> dict[str, int]:
        if self._client is None:
            return {}
        day = day or datetime.now(tz=UTC).strftime("%Y%m%d")
        raw = self._client.hgetall(self._key(tenant_id, day)) or {}
        return {str(k): int(v) for k, v in raw.items()}

    def replace_day(self, tenant_id: str, day: str, values: dict[str, int]) -> None:
        """Overwrite a day's fast counters from the ledger. Never the reverse."""
        if self._client is None:
            return
        key = self._key(tenant_id, day)
        pipe = self._client.pipeline(transaction=True)
        pipe.delete(key)
        if values:
            mapping = {field: int(value) for field, value in values.items()}
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, _day_ttl_s())
        pipe.execute()


def scope_from_event(event: UsageEvent) -> TenantScope:
    return TenantScope(
        tenant_id=event.tenant_id, user_id=event.user_id, project_id=event.project_id
    )
