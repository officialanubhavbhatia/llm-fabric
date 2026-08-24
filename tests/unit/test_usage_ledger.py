from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from llm_fabric.observability.metering import DurableMeter, InMemoryMeter, events_from_decision
from llm_fabric.observability.usage_event import TokenSource, UsageEvent, UsageOperation
from llm_fabric.router.engine import Attempt, RouteDecision
from llm_fabric.storage.postgres import create_database_engine, init_schema
from llm_fabric.storage.usage import RedisUsageAggregates, UsageLedger
from llm_fabric.tenancy.scope import TenantScope

try:
    import fakeredis
except ImportError:  # pragma: no cover
    fakeredis = None


def _event(**overrides: object) -> UsageEvent:
    base: dict[str, object] = {
        "event_id": "inv-1",
        "invocation_id": "inv-1",
        "request_id": "req-1",
        "tenant_id": "acme",
        "provider": "mock",
        "model": "cheap",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "token_source": TokenSource.PROVIDER_MEASURED.value,
        "started_at": 1.0,
        "completed_at": 1.5,
        "status": "success",
        "operation": UsageOperation.USER_RESPONSE.value,
    }
    base.update(overrides)
    return UsageEvent(**base)  # type: ignore[arg-type]


def test_event_serialisation_round_trips() -> None:
    payload = _event(cached_tokens=2, reasoning_tokens=None).as_dict()
    assert payload["token_source"] == "PROVIDER_MEASURED"
    assert payload["prompt_tokens"] == 10
    assert "content" not in payload
    assert payload["total_tokens"] == 15


def test_estimated_token_source_is_not_measured() -> None:
    event = _event(token_source=TokenSource.LOCAL_TOKENIZER_ESTIMATE.value)
    assert event.token_source != TokenSource.PROVIDER_MEASURED.value


def test_in_memory_duplicate_event_is_idempotent() -> None:
    meter = InMemoryMeter()
    first, second = meter.record_events([_event(), _event()])
    assert first.inserted is True
    assert second.duplicate is True
    assert meter.invocation_totals().invocations == 1


def test_real_retry_is_counted_twice() -> None:
    meter = InMemoryMeter()
    meter.record_events(
        [
            _event(event_id="a", invocation_id="a", request_id="r"),
            _event(event_id="b", invocation_id="b", request_id="r", prompt_tokens=7),
        ]
    )
    totals = meter.invocation_totals()
    assert totals.invocations == 2
    assert totals.requests == 1
    assert totals.prompt_tokens == 17


def test_events_from_decision_cover_every_attempt() -> None:
    decision = RouteDecision(
        requested_model="broken",
        policy="declared",
        considered=["broken", "cheap"],
        attempts=[
            Attempt(model_id="broken", provider="failing", duration_ms=5, error="timeout", depth=0),
            Attempt(model_id="cheap", provider="mock", duration_ms=8, depth=1, prompt_tokens=3),
        ],
    )
    events = events_from_decision(
        decision,
        request_id="r1",
        scope=TenantScope(tenant_id="acme", user_id="alice"),
        prompt_tokens=12,
        completion_tokens=4,
        token_source=TokenSource.LOCAL_TOKENIZER_ESTIMATE,
        streamed=False,
        error=None,
        intent_id=None,
        trace_id="t1",
    )
    assert len(events) == 2
    assert events[0].model == "broken"
    assert events[0].status == "error"
    assert events[1].prompt_tokens == 12
    assert events[1].completion_tokens == 4
    assert events[1].token_source == "LOCAL_TOKENIZER_ESTIMATE"
    assert events[0].invocation_id != events[1].invocation_id


def test_sqlite_ledger_is_idempotent(tmp_path) -> None:
    engine = create_database_engine(f"sqlite:///{tmp_path / 'usage.db'}")
    init_schema(engine)
    ledger = UsageLedger(engine)
    first = ledger.insert(_event())
    second = ledger.insert(_event())
    assert first.inserted is True
    assert second.duplicate is True
    assert ledger.totals(tenant_id="acme").invocations == 1


def test_sqlite_ledger_isolates_tenants(tmp_path) -> None:
    engine = create_database_engine(f"sqlite:///{tmp_path / 'usage.db'}")
    init_schema(engine)
    ledger = UsageLedger(engine)
    ledger.insert(_event(tenant_id="acme"))
    ledger.insert(
        _event(event_id="inv-2", invocation_id="inv-2", tenant_id="other", prompt_tokens=99)
    )
    acme = ledger.totals(tenant_id="acme")
    assert acme.invocations == 1
    assert acme.prompt_tokens == 10
    assert ledger.totals(tenant_id="other").prompt_tokens == 99


def test_durable_meter_does_not_double_count_redis(tmp_path) -> None:
    if fakeredis is None:
        pytest.skip("fakeredis is required")
    engine = create_database_engine(f"sqlite:///{tmp_path / 'usage.db'}")
    init_schema(engine)
    client = fakeredis.FakeRedis(decode_responses=True)
    meter = DurableMeter(engine, redis_client=client)
    event = _event(completed_at=1_700_000_000.0)
    assert meter.record_events([event])[0].inserted is True
    assert meter.record_events([event])[0].duplicate is True
    snap = RedisUsageAggregates(client).snapshot("acme", "20231114")
    assert snap["invocations"] == 1
    assert snap["prompt_tokens"] == 10


def test_durable_meter_bounds_the_retry_buffer(tmp_path) -> None:
    engine = create_database_engine(f"sqlite:///{tmp_path / 'usage.db'}")
    init_schema(engine)
    meter = DurableMeter(engine, retry_buffer=2)
    meter._ledger.insert = lambda event: (_ for _ in ()).throw(RuntimeError("db down"))  # type: ignore[method-assign]
    results = meter.record_events(
        [_event(event_id=f"e{i}", invocation_id=f"e{i}") for i in range(5)]
    )
    assert sum(1 for item in results if item.deferred) == 2
    assert sum(1 for item in results if item.dropped) == 3
    assert meter.dropped_events == 3


def test_concurrent_distinct_inserts(tmp_path) -> None:
    engine = create_database_engine(f"sqlite:///{tmp_path / 'usage.db'}")
    init_schema(engine)
    ledger = UsageLedger(engine)

    def write(index: int) -> None:
        ledger.insert(_event(event_id=f"c{index}", invocation_id=f"c{index}", prompt_tokens=index))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(write, range(50)))
    assert ledger.totals(tenant_id="acme").invocations == 50


def test_concurrent_duplicate_inserts_count_once(tmp_path) -> None:
    engine = create_database_engine(f"sqlite:///{tmp_path / 'usage.db'}")
    init_schema(engine)
    ledger = UsageLedger(engine)
    event = _event()

    def write(_: int) -> None:
        ledger.insert(event)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(20)))
    assert ledger.totals(tenant_id="acme").invocations == 1


def test_unpriced_compute_cost_stays_unavailable() -> None:
    event = _event(provider_cost_usd=0.0, compute_cost_estimate_usd=None)
    assert event.provider_cost_usd == 0.0
    assert event.compute_cost_estimate_usd is None


def test_reconcile_reports_drift_then_repair(tmp_path) -> None:
    if fakeredis is None:
        pytest.skip("fakeredis is required")
    from llm_fabric.observability.cli import DRIFT, MATCH, REPAIRED, reconcile

    engine = create_database_engine(f"sqlite:///{tmp_path / 'usage.db'}")
    init_schema(engine)
    ledger = UsageLedger(engine)
    client = fakeredis.FakeRedis(decode_responses=True)
    aggregates = RedisUsageAggregates(client)
    event = _event(completed_at=1_700_000_000.0)
    ledger.insert(event)
    report = reconcile(ledger, aggregates, tenant_id="acme")
    assert report["status"] == DRIFT
    assert report["drifted"] == 1
    repaired = reconcile(ledger, aggregates, repair=True, tenant_id="acme")
    assert repaired["status"] == REPAIRED
    matched = reconcile(ledger, aggregates, tenant_id="acme")
    assert matched["status"] == MATCH
