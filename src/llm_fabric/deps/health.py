"""Per-process serving-dependency health.

Classification is derived from how this process is actually wired, not from
whether a URL happens to be present in settings:

- PostgreSQL is MANDATORY_SERVING when this process has a database engine
  (DurableMeter / tenant stores). A production outage would otherwise admit
  generations whose authoritative usage_events row cannot be retained.
- Redis is MANDATORY_SERVING when this process has a Redis client. Production
  revocation is fail-closed; a gateway that cannot consult the denylist cannot
  safely serve ordinary authenticated traffic.
- OTEL is OPTIONAL_FAIL_SOFT when an exporter endpoint is configured. Spans
  already fail soft; an OTEL outage must not remove readiness.
- Prometheus and Grafana are not serving dependencies (scrape/UI only).
- Individual inference providers belong to routing, not this registry. A
  single unhealthy backend does not make the gateway NotReady.

Admission reads this cached state. It never opens a fresh Postgres or Redis
connection per request.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from llm_fabric.errors import DependencyUnavailableError

if TYPE_CHECKING:
    from llm_fabric.observability.prom import FabricMetrics

#: Closed set used as Prometheus labels. Anything else becomes `other`.
DEPENDENCY_NAMES = frozenset({"postgres", "redis", "telemetry"})

#: POST paths that start provider/model work. Admission is restricted to these
#: so a NotReady instance can still answer /healthz and /readyz.
INFERENCE_PATHS = frozenset({"/v1/chat/completions", "/v1/evals/run"})


class DependencyClass(StrEnum):
    MANDATORY_SERVING = "mandatory_serving"
    OPTIONAL_FAIL_SOFT = "optional_fail_soft"
    BACKGROUND_ONLY = "background_only"


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    SUSPECT = "suspect"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"


@dataclass(frozen=True, slots=True)
class DependencySnapshot:
    name: str
    classification: DependencyClass
    status: HealthStatus
    consecutive_failures: int
    consecutive_successes: int
    last_success_at: float | None
    last_failure_at: float | None
    reason: str | None

    @property
    def required(self) -> bool:
        return self.classification is DependencyClass.MANDATORY_SERVING

    def public_dict(self) -> dict[str, object]:
        """Bounded diagnostics. No DSN, password, host, or exception dump."""
        return {
            "required": self.required,
            "status": self.status.value,
        }


def detection_bound_s(*, interval_s: float, timeout_s: float, fail_threshold: int) -> float:
    """Worst-case probe-only detection window.

    Passive serving failures are faster (the failing request itself).
    """
    return fail_threshold * (interval_s + timeout_s)


def recovery_bound_s(*, interval_s: float, timeout_s: float, recovery_threshold: int) -> float:
    return recovery_threshold * (interval_s + timeout_s)


class DependencyHealth:
    """Mutable health for the dependencies this process actually uses.

    Safe for the asyncio event loop and for sync code that runs in a thread
    (usage persist, Redis PING from a worker thread). Each process has its own
    instance; workers converge independently within the probe bound.
    """

    def __init__(
        self,
        *,
        postgres: bool = False,
        redis: bool = False,
        telemetry: bool = False,
        fail_threshold: int = 2,
        recovery_threshold: int = 2,
        metrics: FabricMetrics | None = None,
    ) -> None:
        if fail_threshold < 1:
            raise ValueError("fail_threshold must be >= 1")
        if recovery_threshold < 1:
            raise ValueError("recovery_threshold must be >= 1")
        self._fail_threshold = fail_threshold
        self._recovery_threshold = recovery_threshold
        self._metrics = metrics
        self._lock = threading.Lock()
        self._states: dict[str, _LiveState] = {}
        self._serving_ready = True
        if postgres:
            self._states["postgres"] = _LiveState(
                name="postgres",
                classification=DependencyClass.MANDATORY_SERVING,
                status=HealthStatus.UNHEALTHY,
            )
        if redis:
            self._states["redis"] = _LiveState(
                name="redis",
                classification=DependencyClass.MANDATORY_SERVING,
                status=HealthStatus.UNHEALTHY,
            )
        if telemetry:
            # Fail-soft: do not block readiness on an unprobed exporter.
            self._states["telemetry"] = _LiveState(
                name="telemetry",
                classification=DependencyClass.OPTIONAL_FAIL_SOFT,
                status=HealthStatus.HEALTHY,
            )
        self._serving_ready = self._compute_serving_ready()

    def bind_metrics(self, metrics: FabricMetrics) -> None:
        self._metrics = metrics

    @property
    def fail_threshold(self) -> int:
        return self._fail_threshold

    @property
    def recovery_threshold(self) -> int:
        return self._recovery_threshold

    def snapshot(self, name: str) -> DependencySnapshot | None:
        with self._lock:
            live = self._states.get(name)
            return live.snapshot() if live is not None else None

    def snapshots(self) -> dict[str, DependencySnapshot]:
        with self._lock:
            return {name: live.snapshot() for name, live in self._states.items()}

    def public_dependencies(self) -> dict[str, dict[str, object]]:
        return {name: snap.public_dict() for name, snap in self.snapshots().items()}

    def serving_ready(self) -> bool:
        """True when every MANDATORY_SERVING dependency may admit new work.

        SUSPECT remains ready (one failed probe is not enough to flap).
        UNHEALTHY and RECOVERING are not ready.
        Optional dependencies never affect this bit.
        """
        with self._lock:
            return self._serving_ready

    def allows_new_serving(self) -> bool:
        return self.serving_ready()

    def refusal(self) -> DependencyUnavailableError | None:
        """Typed 503 if new inference must not start. None if admission allows."""
        with self._lock:
            if self._serving_ready:
                return None
            down = [
                name
                for name, live in self._states.items()
                if live.classification is DependencyClass.MANDATORY_SERVING
                and live.status in {HealthStatus.UNHEALTHY, HealthStatus.RECOVERING}
            ]
        names = ", ".join(down) if down else "mandatory serving dependency"
        return DependencyUnavailableError(f"mandatory serving dependency unavailable: {names}")

    def observe_probe_success(self, name: str) -> None:
        self._observe(name, success=True, serving=False)

    def observe_probe_failure(self, name: str, *, reason: str = "probe_failed") -> None:
        self._observe(name, success=False, serving=False, reason=reason)

    def observe_serving_failure(self, name: str, *, reason: str = "serving_failure") -> None:
        """A real request-path operation failed. Immediate UNHEALTHY."""
        self._observe(name, success=False, serving=True, reason=reason)

    def observe_serving_success(self, name: str) -> None:
        """A real request-path operation succeeded. Counts toward recovery."""
        self._observe(name, success=True, serving=True)

    def mark_optional_unhealthy(self, name: str, *, reason: str = "probe_failed") -> None:
        """OTEL and friends: visible as unhealthy, never removes readiness."""
        self._observe(name, success=False, serving=True, reason=reason)

    def _observe(
        self, name: str, *, success: bool, serving: bool, reason: str | None = None
    ) -> None:
        now = time.time()
        with self._lock:
            live = self._states.get(name)
            if live is None:
                return
            previous_status = live.status
            previous_ready = self._serving_ready
            if success:
                self._on_success(live, now)
            else:
                self._on_failure(live, now, serving=serving, reason=reason or "probe_failed")
            self._serving_ready = self._compute_serving_ready()
            new_status = live.status
            new_ready = self._serving_ready
            metrics = self._metrics
        if metrics is None:
            return
        label = name if name in DEPENDENCY_NAMES else "other"
        metrics.set_dependency_health(label, new_status is HealthStatus.HEALTHY)
        if not success:
            source = "serving" if serving else "probe"
            metrics.note_dependency_failure(label, source)
        if previous_status in {HealthStatus.UNHEALTHY, HealthStatus.RECOVERING} and (
            new_status is HealthStatus.HEALTHY
        ):
            metrics.note_dependency_recovery(label)
        if previous_ready != new_ready:
            metrics.note_readiness_transition(from_ready=previous_ready, to_ready=new_ready)

    def _on_success(self, live: _LiveState, now: float) -> None:
        live.last_success_at = now
        live.consecutive_failures = 0
        live.reason = None
        if live.status is HealthStatus.HEALTHY:
            live.consecutive_successes = 0
            return
        if live.status is HealthStatus.SUSPECT:
            live.status = HealthStatus.HEALTHY
            live.consecutive_successes = 0
            return
        # Initial UNHEALTHY (never observed a failure): one success is enough
        # so startup does not wait for recovery_threshold before serving.
        if live.status is HealthStatus.UNHEALTHY and live.last_failure_at is None:
            live.status = HealthStatus.HEALTHY
            live.consecutive_successes = 0
            return
        live.consecutive_successes += 1
        if live.consecutive_successes >= self._recovery_threshold:
            live.status = HealthStatus.HEALTHY
            live.consecutive_successes = 0
        else:
            live.status = HealthStatus.RECOVERING

    def _on_failure(self, live: _LiveState, now: float, *, serving: bool, reason: str) -> None:
        live.last_failure_at = now
        live.consecutive_successes = 0
        live.reason = _bounded_reason(reason)
        live.consecutive_failures += 1
        if serving or live.consecutive_failures >= self._fail_threshold:
            live.status = HealthStatus.UNHEALTHY
        elif live.status is HealthStatus.HEALTHY:
            live.status = HealthStatus.SUSPECT
        elif live.status is HealthStatus.RECOVERING:
            live.status = HealthStatus.UNHEALTHY

    def _compute_serving_ready(self) -> bool:
        for live in self._states.values():
            if live.classification is not DependencyClass.MANDATORY_SERVING:
                continue
            if live.status in {HealthStatus.UNHEALTHY, HealthStatus.RECOVERING}:
                return False
        return True


@dataclass
class _LiveState:
    name: str
    classification: DependencyClass
    status: HealthStatus
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_success_at: float | None = None
    last_failure_at: float | None = None
    reason: str | None = None

    def snapshot(self) -> DependencySnapshot:
        return DependencySnapshot(
            name=self.name,
            classification=self.classification,
            status=self.status,
            consecutive_failures=self.consecutive_failures,
            consecutive_successes=self.consecutive_successes,
            last_success_at=self.last_success_at,
            last_failure_at=self.last_failure_at,
            reason=self.reason,
        )


def _bounded_reason(reason: str) -> str:
    allowed = {
        "probe_failed",
        "probe_timeout",
        "serving_failure",
        "ping_failed",
    }
    if reason in allowed:
        return reason
    return "probe_failed"
