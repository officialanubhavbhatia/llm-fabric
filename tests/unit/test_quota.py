"""Quota accounting, windows and bounds."""

from __future__ import annotations

import pytest

from llm_fabric.errors import QuotaExceededError
from llm_fabric.tenancy.quota import UNLIMITED, QuotaLedger, QuotaPolicy
from llm_fabric.tenancy.scope import TenantScope


class FakeClock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def scope() -> TenantScope:
    return TenantScope(tenant_id="acme", user_id="alice")


def test_an_unlimited_policy_never_refuses(clock: FakeClock, scope: TenantScope) -> None:
    ledger = QuotaLedger(clock=clock)
    for _ in range(1_000):
        ledger.admit(scope)


def test_requests_per_minute_is_enforced(clock: FakeClock, scope: TenantScope) -> None:
    ledger = QuotaLedger(default_tenant_policy=QuotaPolicy(requests_per_minute=2), clock=clock)
    ledger.admit(scope)
    ledger.admit(scope)

    with pytest.raises(QuotaExceededError):
        ledger.admit(scope)


def test_the_minute_window_rolls_over(clock: FakeClock, scope: TenantScope) -> None:
    ledger = QuotaLedger(default_tenant_policy=QuotaPolicy(requests_per_minute=1), clock=clock)
    ledger.admit(scope)

    clock.advance(61)
    ledger.admit(scope)


def test_a_daily_ceiling_survives_a_minute_rollover(clock: FakeClock, scope: TenantScope) -> None:
    ledger = QuotaLedger(default_tenant_policy=QuotaPolicy(requests_per_day=2), clock=clock)
    ledger.admit(scope)
    clock.advance(120)
    ledger.admit(scope)
    clock.advance(120)

    with pytest.raises(QuotaExceededError):
        ledger.admit(scope)


def test_a_refusal_reports_when_to_retry(clock: FakeClock, scope: TenantScope) -> None:
    ledger = QuotaLedger(default_tenant_policy=QuotaPolicy(requests_per_minute=1), clock=clock)
    ledger.admit(scope)

    with pytest.raises(QuotaExceededError) as caught:
        ledger.admit(scope)

    assert caught.value.retry_after_s is not None
    assert 0 < caught.value.retry_after_s <= 60


def test_token_and_cost_ceilings_apply_to_the_next_admission(
    clock: FakeClock, scope: TenantScope
) -> None:
    """A completed request is never retroactively refused."""
    ledger = QuotaLedger(default_tenant_policy=QuotaPolicy(tokens_per_day=100), clock=clock)
    ledger.admit(scope)
    ledger.record_usage(scope, tokens=150)

    with pytest.raises(QuotaExceededError, match="token"):
        ledger.admit(scope)


def test_spend_ceiling_is_enforced(clock: FakeClock, scope: TenantScope) -> None:
    ledger = QuotaLedger(default_tenant_policy=QuotaPolicy(cost_per_day_usd=1.0), clock=clock)
    ledger.admit(scope)
    ledger.record_usage(scope, cost_usd=1.5)

    with pytest.raises(QuotaExceededError, match="spend"):
        ledger.admit(scope)


def test_tenant_and_user_ceilings_are_independent(clock: FakeClock) -> None:
    ledger = QuotaLedger(
        default_tenant_policy=QuotaPolicy(requests_per_minute=10),
        default_user_policy=QuotaPolicy(requests_per_minute=1),
        clock=clock,
    )
    alice = TenantScope(tenant_id="acme", user_id="alice")
    bob = TenantScope(tenant_id="acme", user_id="bob")

    ledger.admit(alice)
    ledger.admit(bob)

    with pytest.raises(QuotaExceededError, match="user"):
        ledger.admit(alice)


def test_the_same_user_id_in_two_tenants_is_two_subjects(clock: FakeClock) -> None:
    ledger = QuotaLedger(default_user_policy=QuotaPolicy(requests_per_minute=1), clock=clock)
    acme_alice = TenantScope(tenant_id="acme", user_id="alice")
    other_alice = TenantScope(tenant_id="other", user_id="alice")

    ledger.admit(acme_alice)
    ledger.admit(other_alice)


def test_per_tenant_policies_override_the_default(clock: FakeClock) -> None:
    ledger = QuotaLedger(
        default_tenant_policy=QuotaPolicy(requests_per_minute=1),
        tenant_policies={"vip": UNLIMITED},
        clock=clock,
    )
    vip = TenantScope(tenant_id="vip", user_id="v")
    standard = TenantScope(tenant_id="standard", user_id="s")

    for _ in range(5):
        ledger.admit(vip)

    ledger.admit(standard)
    with pytest.raises(QuotaExceededError):
        ledger.admit(standard)


def test_tracked_subjects_are_bounded(clock: FakeClock) -> None:
    """A tenant id is attacker-influenced, so the ledger must not grow forever."""
    ledger = QuotaLedger(max_tracked_subjects=10, clock=clock)

    for index in range(500):
        ledger.admit(TenantScope(tenant_id=f"tenant-{index}", user_id="u"))

    assert len(ledger._counters) <= 10


def test_a_snapshot_reports_current_consumption(clock: FakeClock, scope: TenantScope) -> None:
    ledger = QuotaLedger(default_tenant_policy=QuotaPolicy(requests_per_day=10), clock=clock)
    ledger.admit(scope)
    ledger.record_usage(scope, tokens=42, cost_usd=0.5)

    snapshot = ledger.snapshot(scope, "tenant")

    assert snapshot.requests_today == 1
    assert snapshot.tokens_today == 42
    assert snapshot.cost_today_usd == pytest.approx(0.5)
    assert snapshot.policy.requests_per_day == 10


def test_negative_usage_is_refused(clock: FakeClock, scope: TenantScope) -> None:
    ledger = QuotaLedger(clock=clock)

    with pytest.raises(ValueError):
        ledger.record_usage(scope, tokens=-1)


def test_a_refused_request_is_not_counted(clock: FakeClock, scope: TenantScope) -> None:
    """A rejected admission must not consume the allowance it was refused."""
    ledger = QuotaLedger(default_tenant_policy=QuotaPolicy(requests_per_minute=1), clock=clock)
    ledger.admit(scope)

    for _ in range(5):
        with pytest.raises(QuotaExceededError):
            ledger.admit(scope)

    assert ledger.snapshot(scope, "tenant").requests_this_minute == 1


def test_a_user_refusal_does_not_charge_the_tenant(clock: FakeClock) -> None:
    """The tenant counter must not advance when the user check refuses first."""
    ledger = QuotaLedger(
        default_tenant_policy=QuotaPolicy(requests_per_minute=100),
        default_user_policy=QuotaPolicy(requests_per_minute=1),
        clock=clock,
    )
    alice = TenantScope(tenant_id="acme", user_id="alice")

    ledger.admit(alice)
    with pytest.raises(QuotaExceededError):
        ledger.admit(alice)

    assert ledger.snapshot(alice, "tenant").requests_this_minute == 1
