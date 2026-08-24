"""Observed deployment health: latency, errors, queue pressure, circuit breakers.

Everything in this module is measured by this process from attempts it actually
made. Nothing here is configurable-as-a-value; the only configuration is *how* to
measure. That is the whole point of keeping it out of the registry, where a
number can be asserted into existence by typing it into YAML.

**Absent is not healthy.** A deployment nobody has called yet has no health
score, no error rate and no latency estimate, and this module returns `None` for
all three rather than an optimistic default. Routing treats `None` as "no
signal": it neither rewards nor penalises the deployment, and the route
explanation records the feature as `absent`. A default of 1.0 would let an
untried deployment outrank a proven one on evidence that does not exist.

The circuit breaker is the standard three states. `CLOSED` passes traffic.
`OPEN` refuses it, because a backend that has failed repeatedly will most likely
fail again and the fastest useful thing to do is stop asking. After a cooldown it
becomes `HALF_OPEN`, which admits a limited number of trial requests: enough to
learn whether recovery happened, few enough that a still-broken backend is not
handed the full load again.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from llm_fabric.errors import ConfigurationError

#: Weight given to the newest sample in every exponentially weighted average.
#: Low enough that one slow response does not evict the history, high enough that
#: a genuine regime change is visible within a handful of requests.
DEFAULT_EWMA_ALPHA = 0.2


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class BreakerPolicy:
    """When to stop sending traffic to a deployment, and when to try again."""

    #: Consecutive failures that trip the breaker regardless of rate.
    consecutive_failures: int = 5

    #: Error rate that trips the breaker, but only once `minimum_samples` have
    #: been seen. Without the floor, the first failed request would read as a
    #: 100% error rate and open the breaker on a sample of one.
    error_rate: float = 0.5
    minimum_samples: int = 10

    #: How long the breaker stays open before admitting a probe.
    open_duration_s: float = 30.0

    #: Consecutive probe successes needed to close again.
    half_open_successes: int = 2

    #: Concurrent in-flight requests allowed. `None` means unlimited.
    max_concurrency: int | None = None

    def __post_init__(self) -> None:
        if self.consecutive_failures < 1:
            raise ConfigurationError("consecutive_failures must be at least 1")
        if not 0.0 < self.error_rate <= 1.0:
            raise ConfigurationError("error_rate must lie in (0, 1]")
        if self.minimum_samples < 1:
            raise ConfigurationError("minimum_samples must be at least 1")
        if self.open_duration_s < 0:
            raise ConfigurationError("open_duration_s cannot be negative")
        if self.half_open_successes < 1:
            raise ConfigurationError("half_open_successes must be at least 1")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ConfigurationError("max_concurrency must be at least 1 when set")


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """A point-in-time read of one deployment's observed state.

    Every optional field is `None` when unmeasured. Consumers must distinguish
    that from a measured zero.
    """

    deployment_id: str
    state: BreakerState
    samples: int
    successes: int
    failures: int
    queue_depth: int
    ewma_latency_ms: float | None
    error_rate: float | None
    health_score: float | None
    consecutive_failures: int
    opened_at: float | None
    last_error: str | None
    held_open: bool = False

    @property
    def has_signal(self) -> bool:
        """True when any traffic has been observed, so the scores mean something."""
        return self.samples > 0

    @property
    def admitting(self) -> bool:
        return self.state is not BreakerState.OPEN

    def as_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "circuit_state": self.state.value,
            "samples": self.samples,
            "successes": self.successes,
            "failures": self.failures,
            "queue_depth": self.queue_depth,
            "ewma_latency_ms": (
                round(self.ewma_latency_ms, 3) if self.ewma_latency_ms is not None else None
            ),
            "error_rate": round(self.error_rate, 4) if self.error_rate is not None else None,
            "health_score": (
                round(self.health_score, 4) if self.health_score is not None else None
            ),
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "held_open": self.held_open,
        }


@dataclass(slots=True)
class _State:
    """Mutable per-deployment counters. Guarded by the tracker's lock."""

    samples: int = 0
    successes: int = 0
    failures: int = 0
    queue_depth: int = 0
    ewma_latency_ms: float | None = None
    ewma_error: float | None = None
    consecutive_failures: int = 0
    half_open_successes: int = 0
    state: BreakerState = BreakerState.CLOSED
    opened_at: float | None = None
    half_open_in_flight: int = 0
    last_error: str | None = None
    held_open: bool = False


class HealthTracker:
    """Observed health for every deployment the router has attempted.

    One instance is shared by the whole process, so it is guarded by a lock
    rather than assuming a single event loop.
    """

    def __init__(
        self,
        *,
        policy: BreakerPolicy | None = None,
        alpha: float = DEFAULT_EWMA_ALPHA,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ConfigurationError("EWMA alpha must lie in (0, 1]")
        self._policy = policy or BreakerPolicy()
        self._alpha = alpha
        self._clock = clock
        self._lock = threading.Lock()
        self._states: dict[str, _State] = {}
        self._overrides: dict[str, BreakerPolicy] = {}

    @property
    def policy(self) -> BreakerPolicy:
        return self._policy

    def set_policy(self, deployment_id: str, policy: BreakerPolicy) -> None:
        """Give one deployment its own thresholds."""
        with self._lock:
            self._overrides[deployment_id] = policy

    def _policy_for(self, deployment_id: str) -> BreakerPolicy:
        return self._overrides.get(deployment_id, self._policy)

    def _state(self, deployment_id: str) -> _State:
        state = self._states.get(deployment_id)
        if state is None:
            state = _State()
            self._states[deployment_id] = state
        return state

    # -- recording -----------------------------------------------------------

    def record_success(self, deployment_id: str, *, latency_ms: float) -> None:
        with self._lock:
            state = self._state(deployment_id)
            state.samples += 1
            state.successes += 1
            state.consecutive_failures = 0
            state.ewma_latency_ms = self._blend(state.ewma_latency_ms, max(0.0, latency_ms))
            state.ewma_error = self._blend(state.ewma_error, 0.0)

            if state.state is BreakerState.HALF_OPEN:
                state.half_open_successes += 1
                if state.half_open_successes >= self._policy_for(deployment_id).half_open_successes:
                    self._close(state)

    def record_failure(
        self, deployment_id: str, *, latency_ms: float = 0.0, error: str | None = None
    ) -> None:
        with self._lock:
            policy = self._policy_for(deployment_id)
            state = self._state(deployment_id)
            state.samples += 1
            state.failures += 1
            state.consecutive_failures += 1
            state.last_error = error
            if latency_ms > 0:
                state.ewma_latency_ms = self._blend(state.ewma_latency_ms, latency_ms)
            state.ewma_error = self._blend(state.ewma_error, 1.0)

            # A failure during a trial request means recovery has not happened.
            # Reopening immediately is the point of the half-open state.
            if state.state is BreakerState.HALF_OPEN:
                self._open(state)
                return

            tripped_by_streak = state.consecutive_failures >= policy.consecutive_failures
            tripped_by_rate = (
                state.samples >= policy.minimum_samples
                and state.ewma_error is not None
                and state.ewma_error >= policy.error_rate
            )
            if state.state is BreakerState.CLOSED and (tripped_by_streak or tripped_by_rate):
                self._open(state)

    def _blend(self, current: float | None, sample: float) -> float:
        return sample if current is None else self._alpha * sample + (1 - self._alpha) * current

    def _open(self, state: _State) -> None:
        state.state = BreakerState.OPEN
        state.opened_at = self._clock()
        state.half_open_successes = 0
        state.half_open_in_flight = 0

    def _close(self, state: _State) -> None:
        state.state = BreakerState.CLOSED
        state.opened_at = None
        state.half_open_successes = 0
        state.half_open_in_flight = 0
        state.consecutive_failures = 0
        state.held_open = False

    def force_open(
        self, deployment_id: str, *, reason: str | None = None, hold: bool = True
    ) -> None:
        """Open the breaker as an operator or remediation action.

        `hold=True` skips the cooldown so traffic does not leak back after
        `open_duration_s`. Recovery then requires `force_close`.
        """
        with self._lock:
            state = self._state(deployment_id)
            if reason:
                state.last_error = reason
            self._open(state)
            state.held_open = hold

    def force_close(self, deployment_id: str) -> None:
        with self._lock:
            self._close(self._state(deployment_id))

    # -- admission -----------------------------------------------------------

    def admits(self, deployment_id: str) -> bool:
        """True when the breaker and concurrency limit allow another request.

        Also performs the open→half-open transition, which is why it is not a
        pure predicate: the cooldown expiring is only observable by asking.
        """
        with self._lock:
            return self._admits_locked(deployment_id)

    def _admits_locked(self, deployment_id: str) -> bool:
        policy = self._policy_for(deployment_id)
        state = self._state(deployment_id)

        if policy.max_concurrency is not None and state.queue_depth >= policy.max_concurrency:
            return False

        if state.state is BreakerState.CLOSED:
            return True

        if state.state is BreakerState.OPEN:
            if state.held_open:
                return False
            opened_at = state.opened_at or 0.0
            if (self._clock() - opened_at) < policy.open_duration_s:
                return False
            state.state = BreakerState.HALF_OPEN
            state.half_open_successes = 0
            # `half_open_in_flight` is deliberately not reset. It is zero in any
            # flow that reached here through admission, and zeroing it would
            # discard a probe that is genuinely running.

        # Half-open: admit only as many probes as it takes to decide.
        return state.half_open_in_flight < policy.half_open_successes

    @contextmanager
    def in_flight(self, deployment_id: str) -> Iterator[None]:
        """Count a request against queue depth for as long as it is running."""
        with self._lock:
            state = self._state(deployment_id)
            state.queue_depth += 1
            if state.state is BreakerState.HALF_OPEN:
                state.half_open_in_flight += 1
        try:
            yield
        finally:
            with self._lock:
                state = self._state(deployment_id)
                state.queue_depth = max(0, state.queue_depth - 1)
                if state.half_open_in_flight > 0:
                    state.half_open_in_flight -= 1

    # -- reading -------------------------------------------------------------

    def snapshot(self, deployment_id: str) -> HealthSnapshot:
        with self._lock:
            state = self._state(deployment_id)
            return self._snapshot_locked(deployment_id, state)

    def _snapshot_locked(self, deployment_id: str, state: _State) -> HealthSnapshot:
        error_rate = state.ewma_error
        return HealthSnapshot(
            deployment_id=deployment_id,
            state=state.state,
            samples=state.samples,
            successes=state.successes,
            failures=state.failures,
            queue_depth=state.queue_depth,
            ewma_latency_ms=state.ewma_latency_ms,
            error_rate=error_rate,
            health_score=self._health_score(state, error_rate),
            consecutive_failures=state.consecutive_failures,
            opened_at=state.opened_at,
            last_error=state.last_error,
            held_open=state.held_open,
        )

    def _health_score(self, state: _State, error_rate: float | None) -> float | None:
        """Observed health in [0, 1], or `None` when nothing has been observed.

        An open breaker scores zero regardless of history: whatever the recent
        error rate says, the fabric has decided not to send traffic, and health
        that ignores that decision would be misleading.
        """
        if state.samples == 0:
            return None
        if state.state is BreakerState.OPEN:
            return 0.0
        if error_rate is None:
            return None
        return max(0.0, min(1.0, 1.0 - error_rate))

    def all_snapshots(self) -> dict[str, HealthSnapshot]:
        with self._lock:
            return {
                deployment_id: self._snapshot_locked(deployment_id, state)
                for deployment_id, state in self._states.items()
            }

    def reset(self, deployment_id: str | None = None) -> None:
        """Forget observations. Used between tests and by operational recovery."""
        with self._lock:
            if deployment_id is None:
                self._states.clear()
            else:
                self._states.pop(deployment_id, None)
