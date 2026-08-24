from __future__ import annotations

import pytest

from llm_fabric.observability.metering import InMemoryMeter, UsageRecord


def _record(**overrides: object) -> UsageRecord:
    base: dict[str, object] = {
        "request_id": "r1",
        "requested_model": "auto",
        "served_model": "cheap",
        "provider": "mock",
        "policy": "cheapest",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost_usd": 0.001,
        "cost_is_estimated": False,
        "latency_ms": 12.5,
        "streamed": False,
        "failover_count": 0,
    }
    base.update(overrides)
    return UsageRecord(**base)  # type: ignore[arg-type]


def test_totals_aggregate_across_records() -> None:
    meter = InMemoryMeter()
    meter.record(_record())
    meter.record(_record(prompt_tokens=20, completion_tokens=10, cost_usd=0.002))

    totals = meter.totals()
    assert totals.requests == 2
    assert totals.prompt_tokens == 30
    assert totals.completion_tokens == 15
    assert totals.total_tokens == 45
    assert totals.cost_usd == pytest.approx(0.003)


def test_estimated_costs_are_counted_separately() -> None:
    meter = InMemoryMeter()
    meter.record(_record(cost_is_estimated=True))
    meter.record(_record(cost_is_estimated=False))

    assert meter.totals().estimated_cost_requests == 1


def test_failovers_are_summed() -> None:
    meter = InMemoryMeter()
    meter.record(_record(failover_count=2))
    meter.record(_record(failover_count=1))

    assert meter.totals().failovers == 3


def test_recent_returns_newest_first() -> None:
    meter = InMemoryMeter()
    meter.record(_record(request_id="first"))
    meter.record(_record(request_id="second"))

    assert [r.request_id for r in meter.recent()] == ["second", "first"]


def test_buffer_is_bounded() -> None:
    meter = InMemoryMeter(buffer_size=2)
    for index in range(5):
        meter.record(_record(request_id=str(index)))

    assert meter.totals().requests == 2
    assert [r.request_id for r in meter.recent()] == ["4", "3"]


def test_empty_meter_reports_zeroes() -> None:
    totals = InMemoryMeter().totals()
    assert totals.requests == 0
    assert totals.cost_usd == 0


def test_totals_and_recent_are_scoped_to_a_tenant() -> None:
    meter = InMemoryMeter()
    meter.record(_record(request_id="a", tenant_id="tenant-a"))
    meter.record(_record(request_id="b", tenant_id="tenant-b", prompt_tokens=99))

    assert meter.totals(tenant_id="tenant-a").requests == 1
    assert meter.totals(tenant_id="tenant-a").prompt_tokens == 10
    assert [r.request_id for r in meter.recent(tenant_id="tenant-b")] == ["b"]


def test_usage_can_be_narrowed_to_one_user_inside_a_tenant() -> None:
    meter = InMemoryMeter()
    meter.record(_record(request_id="a", tenant_id="acme", user_id="alice"))
    meter.record(_record(request_id="b", tenant_id="acme", user_id="bob"))

    assert meter.totals(tenant_id="acme").requests == 2
    assert [r.request_id for r in meter.recent(tenant_id="acme", user_id="alice")] == ["a"]
