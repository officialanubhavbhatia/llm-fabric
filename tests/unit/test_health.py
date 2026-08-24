"""Observed health and circuit breakers.

The most important assertions here are the ones about *absence*: a deployment
nobody has called must report `None`, not a flattering default, because routing
treats absent as "no signal" and a default of 1.0 would let an untried backend
outrank a proven one.
"""

from __future__ import annotations

import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.router.health import BreakerPolicy, BreakerState, HealthTracker


class _Clock:
    """A hand-cranked clock, so cooldowns are tested without sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


def _tracker(clock: _Clock, **policy: object) -> HealthTracker:
    return HealthTracker(policy=BreakerPolicy(**policy), clock=clock)  # type: ignore[arg-type]


# -- absence -----------------------------------------------------------------


def test_an_untried_deployment_reports_no_signal() -> None:
    snapshot = HealthTracker().snapshot("never-called")
    assert snapshot.samples == 0
    assert not snapshot.has_signal
    assert snapshot.health_score is None
    assert snapshot.error_rate is None
    assert snapshot.ewma_latency_ms is None
    # It is still admitted: no evidence of trouble is not evidence of trouble.
    assert snapshot.state is BreakerState.CLOSED
    assert snapshot.admitting


def test_a_healthy_deployment_scores_one() -> None:
    tracker = HealthTracker()
    for _ in range(5):
        tracker.record_success("d", latency_ms=10.0)
    snapshot = tracker.snapshot("d")
    assert snapshot.health_score == pytest.approx(1.0)
    assert snapshot.error_rate == pytest.approx(0.0)
    assert snapshot.successes == 5


# -- EWMA --------------------------------------------------------------------


def test_latency_ewma_moves_towards_recent_samples() -> None:
    tracker = HealthTracker(alpha=0.5)
    tracker.record_success("d", latency_ms=100.0)
    assert tracker.snapshot("d").ewma_latency_ms == pytest.approx(100.0)

    tracker.record_success("d", latency_ms=200.0)
    assert tracker.snapshot("d").ewma_latency_ms == pytest.approx(150.0)

    tracker.record_success("d", latency_ms=200.0)
    assert tracker.snapshot("d").ewma_latency_ms == pytest.approx(175.0)


def test_one_slow_sample_does_not_evict_the_history() -> None:
    tracker = HealthTracker(alpha=0.2)
    for _ in range(10):
        tracker.record_success("d", latency_ms=10.0)
    tracker.record_success("d", latency_ms=1000.0)
    # Moved, but nowhere near the outlier.
    assert 100.0 < (tracker.snapshot("d").ewma_latency_ms or 0) < 250.0


def test_error_ewma_rises_with_failures() -> None:
    tracker = HealthTracker(alpha=0.5)
    tracker.record_success("d", latency_ms=1.0)
    assert tracker.snapshot("d").error_rate == pytest.approx(0.0)
    tracker.record_failure("d", error="boom")
    assert tracker.snapshot("d").error_rate == pytest.approx(0.5)
    assert tracker.snapshot("d").health_score == pytest.approx(0.5)


def test_alpha_must_be_a_proportion() -> None:
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ConfigurationError):
            HealthTracker(alpha=bad)


# -- breaker: opening --------------------------------------------------------


def test_consecutive_failures_trip_the_breaker(clock: _Clock) -> None:
    tracker = _tracker(clock, consecutive_failures=3)
    for _ in range(2):
        tracker.record_failure("d", error="boom")
    assert tracker.snapshot("d").state is BreakerState.CLOSED
    assert tracker.admits("d")

    tracker.record_failure("d", error="boom")
    assert tracker.snapshot("d").state is BreakerState.OPEN
    assert not tracker.admits("d")


def test_an_open_breaker_scores_zero_health(clock: _Clock) -> None:
    tracker = _tracker(clock, consecutive_failures=1)
    tracker.record_failure("d", error="boom")
    assert tracker.snapshot("d").health_score == 0.0


def test_a_single_failure_does_not_trip_the_rate_rule(clock: _Clock) -> None:
    # One failure is a 100% error rate on a sample of one. The minimum-sample
    # floor is what stops that from opening the breaker.
    tracker = _tracker(clock, consecutive_failures=99, error_rate=0.5, minimum_samples=10)
    tracker.record_failure("d", error="boom")
    assert tracker.snapshot("d").state is BreakerState.CLOSED


def test_a_sustained_error_rate_trips_the_breaker(clock: _Clock) -> None:
    tracker = HealthTracker(
        policy=BreakerPolicy(consecutive_failures=99, error_rate=0.5, minimum_samples=6),
        alpha=0.5,
        clock=clock,
    )
    for _ in range(3):
        tracker.record_success("d", latency_ms=1.0)
        tracker.record_failure("d", error="boom")
    assert tracker.snapshot("d").state is BreakerState.OPEN


def test_success_resets_the_failure_streak(clock: _Clock) -> None:
    tracker = _tracker(clock, consecutive_failures=3)
    tracker.record_failure("d", error="boom")
    tracker.record_failure("d", error="boom")
    tracker.record_success("d", latency_ms=1.0)
    tracker.record_failure("d", error="boom")
    tracker.record_failure("d", error="boom")
    assert tracker.snapshot("d").state is BreakerState.CLOSED


# -- breaker: recovering -----------------------------------------------------


def test_the_breaker_half_opens_after_the_cooldown(clock: _Clock) -> None:
    tracker = _tracker(clock, consecutive_failures=1, open_duration_s=30.0)
    tracker.record_failure("d", error="boom")
    assert not tracker.admits("d")

    clock.advance(29.0)
    assert not tracker.admits("d")

    clock.advance(2.0)
    assert tracker.admits("d")
    assert tracker.snapshot("d").state is BreakerState.HALF_OPEN


def test_probe_successes_close_the_breaker(clock: _Clock) -> None:
    tracker = _tracker(clock, consecutive_failures=1, open_duration_s=1.0, half_open_successes=2)
    tracker.record_failure("d", error="boom")
    clock.advance(2.0)
    assert tracker.admits("d")

    tracker.record_success("d", latency_ms=5.0)
    assert tracker.snapshot("d").state is BreakerState.HALF_OPEN
    tracker.record_success("d", latency_ms=5.0)
    assert tracker.snapshot("d").state is BreakerState.CLOSED


def test_a_failed_probe_reopens_immediately(clock: _Clock) -> None:
    tracker = _tracker(clock, consecutive_failures=5, open_duration_s=1.0)
    for _ in range(5):
        tracker.record_failure("d", error="boom")
    clock.advance(2.0)
    assert tracker.admits("d")
    assert tracker.snapshot("d").state is BreakerState.HALF_OPEN

    # Still broken. One failure must reopen without waiting for the streak again.
    tracker.record_failure("d", error="still broken")
    assert tracker.snapshot("d").state is BreakerState.OPEN
    assert not tracker.admits("d")


def test_half_open_admits_only_a_few_probes(clock: _Clock) -> None:
    tracker = _tracker(clock, consecutive_failures=1, open_duration_s=1.0, half_open_successes=1)
    tracker.record_failure("d", error="boom")
    clock.advance(2.0)

    # The real call order: admission first, which performs the open→half-open
    # transition, then the request is taken in flight.
    assert tracker.admits("d")
    with tracker.in_flight("d"):
        # One probe is running; a still-broken backend must not be handed more.
        assert not tracker.admits("d")
    assert tracker.admits("d")


# -- concurrency -------------------------------------------------------------


def test_queue_depth_tracks_in_flight_requests() -> None:
    tracker = HealthTracker()
    assert tracker.snapshot("d").queue_depth == 0
    with tracker.in_flight("d"):
        assert tracker.snapshot("d").queue_depth == 1
        with tracker.in_flight("d"):
            assert tracker.snapshot("d").queue_depth == 2
    assert tracker.snapshot("d").queue_depth == 0


def test_queue_depth_unwinds_even_when_the_call_raises() -> None:
    tracker = HealthTracker()
    with pytest.raises(RuntimeError), tracker.in_flight("d"):
        raise RuntimeError("boom")
    assert tracker.snapshot("d").queue_depth == 0


def test_concurrency_limit_sheds_load() -> None:
    tracker = HealthTracker(policy=BreakerPolicy(max_concurrency=2))
    with tracker.in_flight("d"), tracker.in_flight("d"):
        assert not tracker.admits("d")
    assert tracker.admits("d")


# -- policy ------------------------------------------------------------------


def test_a_deployment_can_have_its_own_thresholds(clock: _Clock) -> None:
    tracker = _tracker(clock, consecutive_failures=10)
    tracker.set_policy("fragile", BreakerPolicy(consecutive_failures=1))

    tracker.record_failure("fragile", error="boom")
    tracker.record_failure("sturdy", error="boom")

    assert tracker.snapshot("fragile").state is BreakerState.OPEN
    assert tracker.snapshot("sturdy").state is BreakerState.CLOSED


@pytest.mark.parametrize(
    "kwargs",
    [
        {"consecutive_failures": 0},
        {"error_rate": 0.0},
        {"error_rate": 1.5},
        {"minimum_samples": 0},
        {"open_duration_s": -1.0},
        {"half_open_successes": 0},
        {"max_concurrency": 0},
    ],
)
def test_nonsense_policies_are_refused(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        BreakerPolicy(**kwargs)  # type: ignore[arg-type]


def test_reset_forgets_observations() -> None:
    tracker = HealthTracker()
    tracker.record_success("a", latency_ms=1.0)
    tracker.record_success("b", latency_ms=1.0)

    tracker.reset("a")
    assert not tracker.snapshot("a").has_signal
    assert tracker.snapshot("b").has_signal

    tracker.reset()
    assert not tracker.snapshot("b").has_signal


def test_all_snapshots_covers_every_seen_deployment() -> None:
    tracker = HealthTracker()
    tracker.record_success("a", latency_ms=1.0)
    tracker.record_failure("b", error="boom")
    assert set(tracker.all_snapshots()) == {"a", "b"}
