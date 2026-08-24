"""Per-tenant and per-user quotas.

Quotas exist for two different reasons and both matter. Commercially they bound
what a tenant can spend. Operationally they are the fabric's defence against
denial-of-wallet: an inference gateway with no ceiling converts a compromised
client key directly into an unbounded bill.

Both levels are enforced. A tenant-wide limit does not protect one user from
another inside the same tenant, and a per-user limit does not bound the tenant
in aggregate.

Counters are fixed-window. The in-memory ledger is the development default.
Production uses `RedisQuotaLedger` so every worker shares one ceiling.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from llm_fabric.errors import QuotaExceededError
from llm_fabric.tenancy.scope import TenantScope

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_DAY = 86_400
_SECONDS_PER_MONTH = 86_400 * 30

QuotaLevel = Literal["tenant", "user"]


@dataclass(frozen=True, slots=True)
class QuotaPolicy:
    """Ceilings for one subject. `None` means that dimension is unlimited."""

    requests_per_minute: int | None = None
    requests_per_day: int | None = None
    requests_per_month: int | None = None
    tokens_per_day: int | None = None
    cost_per_day_usd: float | None = None
    max_concurrency: int | None = None
    project_requests_per_minute: int | None = None
    provider_requests_per_minute: int | None = None
    model_requests_per_minute: int | None = None

    @property
    def is_unlimited(self) -> bool:
        return (
            self.requests_per_minute is None
            and self.requests_per_day is None
            and self.requests_per_month is None
            and self.tokens_per_day is None
            and self.cost_per_day_usd is None
            and self.max_concurrency is None
            and self.project_requests_per_minute is None
            and self.provider_requests_per_minute is None
            and self.model_requests_per_minute is None
        )


UNLIMITED = QuotaPolicy()


@dataclass(frozen=True, slots=True)
class QuotaSnapshot:
    """What a subject has consumed in the current windows."""

    subject: str
    level: QuotaLevel
    requests_this_minute: int
    requests_today: int
    tokens_today: int
    cost_today_usd: float
    policy: QuotaPolicy


@dataclass(slots=True)
class _Counters:
    minute_window: int = -1
    day_window: int = -1
    month_window: int = -1
    requests_this_minute: int = 0
    requests_today: int = 0
    requests_this_month: int = 0
    tokens_today: int = 0
    cost_today_usd: float = 0.0
    inflight: int = 0

    def roll(self, now: float) -> None:
        minute = int(now // _SECONDS_PER_MINUTE)
        day = int(now // _SECONDS_PER_DAY)
        month = int(now // _SECONDS_PER_MONTH)
        if minute != self.minute_window:
            self.minute_window = minute
            self.requests_this_minute = 0
        if day != self.day_window:
            self.day_window = day
            self.requests_today = 0
            self.tokens_today = 0
            self.cost_today_usd = 0.0
        if month != self.month_window:
            self.month_window = month
            self.requests_this_month = 0


class QuotaLedger:
    """Tracks and enforces consumption for tenants and users."""

    def __init__(
        self,
        *,
        default_tenant_policy: QuotaPolicy = UNLIMITED,
        default_user_policy: QuotaPolicy = UNLIMITED,
        tenant_policies: Mapping[str, QuotaPolicy] | None = None,
        user_policies: Mapping[str, QuotaPolicy] | None = None,
        max_tracked_subjects: int = 10_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_tracked_subjects <= 0:
            raise ValueError("max_tracked_subjects must be positive")
        self._default_tenant_policy = default_tenant_policy
        self._default_user_policy = default_user_policy
        self._tenant_policies: dict[str, QuotaPolicy] = dict(tenant_policies or {})
        self._user_policies: dict[str, QuotaPolicy] = dict(user_policies or {})
        self._max_tracked = max_tracked_subjects
        self._clock = clock
        self._counters: OrderedDict[tuple[QuotaLevel, str], _Counters] = OrderedDict()
        self._lock = threading.Lock()

    # -- policy administration ----------------------------------------------

    def set_tenant_policy(self, tenant_id: str, policy: QuotaPolicy) -> None:
        with self._lock:
            self._tenant_policies[tenant_id] = policy

    def set_user_policy(self, user_key: str, policy: QuotaPolicy) -> None:
        with self._lock:
            self._user_policies[user_key] = policy

    def tenant_policy(self, tenant_id: str) -> QuotaPolicy:
        return self._tenant_policies.get(tenant_id, self._default_tenant_policy)

    def user_policy(self, user_key: str) -> QuotaPolicy:
        return self._user_policies.get(user_key, self._default_user_policy)

    # -- enforcement ---------------------------------------------------------

    def admit(
        self,
        scope: TenantScope,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """Consume one request against both ceilings, or refuse the request.

        Tenant and user are checked and incremented together under one lock, so
        concurrent requests cannot both observe the last remaining unit.
        """
        now = self._clock()
        with self._lock:
            tenant_counters = self._counters_for("tenant", scope.tenant_id, now)
            user_counters = self._counters_for("user", scope.user_key, now)

            tenant_policy = self.tenant_policy(scope.tenant_id)
            user_policy = self.user_policy(scope.user_key)

            self._assert_request_headroom(
                tenant_counters, tenant_policy, "tenant", scope.tenant_id, now
            )
            self._assert_request_headroom(user_counters, user_policy, "user", scope.user_key, now)
            if tenant_policy.project_requests_per_minute is not None and scope.project_id:
                project = self._counters_for(
                    "user", f"project:{scope.tenant_id}/{scope.project_id}", now
                )
                if project.requests_this_minute >= tenant_policy.project_requests_per_minute:
                    raise QuotaExceededError(
                        "project request-per-minute quota exhausted",
                        retry_after_s=_seconds_until_next(now, _SECONDS_PER_MINUTE),
                    )
                project.requests_this_minute += 1
            if tenant_policy.provider_requests_per_minute is not None and provider:
                subject = self._counters_for("user", f"provider:{provider}", now)
                if subject.requests_this_minute >= tenant_policy.provider_requests_per_minute:
                    raise QuotaExceededError(
                        "provider request-per-minute quota exhausted",
                        retry_after_s=_seconds_until_next(now, _SECONDS_PER_MINUTE),
                    )
                subject.requests_this_minute += 1
            if tenant_policy.model_requests_per_minute is not None and model:
                subject = self._counters_for("user", f"model:{model}", now)
                if subject.requests_this_minute >= tenant_policy.model_requests_per_minute:
                    raise QuotaExceededError(
                        "model request-per-minute quota exhausted",
                        retry_after_s=_seconds_until_next(now, _SECONDS_PER_MINUTE),
                    )
                subject.requests_this_minute += 1

            tenant_counters.requests_this_minute += 1
            tenant_counters.requests_today += 1
            tenant_counters.requests_this_month += 1
            user_counters.requests_this_minute += 1
            user_counters.requests_today += 1
            user_counters.requests_this_month += 1

    def acquire_concurrency(self, scope: TenantScope) -> None:
        now = self._clock()
        with self._lock:
            tenant_policy = self.tenant_policy(scope.tenant_id)
            user_policy = self.user_policy(scope.user_key)
            tenant = self._counters_for("tenant", scope.tenant_id, now)
            user = self._counters_for("user", scope.user_key, now)
            if (
                tenant_policy.max_concurrency is not None
                and tenant.inflight >= tenant_policy.max_concurrency
            ):
                raise QuotaExceededError("tenant concurrency quota exhausted", retry_after_s=1)
            if (
                user_policy.max_concurrency is not None
                and user.inflight >= user_policy.max_concurrency
            ):
                raise QuotaExceededError("user concurrency quota exhausted", retry_after_s=1)
            tenant.inflight += 1
            user.inflight += 1

    def release_concurrency(self, scope: TenantScope) -> None:
        now = self._clock()
        with self._lock:
            tenant = self._counters_for("tenant", scope.tenant_id, now)
            user = self._counters_for("user", scope.user_key, now)
            tenant.inflight = max(0, tenant.inflight - 1)
            user.inflight = max(0, user.inflight - 1)

    def record_usage(
        self,
        scope: TenantScope,
        *,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record consumption observed after a request completed.

        Token and cost ceilings are enforced on the *next* admission rather than
        retroactively: a request that has already run cannot be un-run, and
        killing a completed response to enforce a ceiling would lose work the
        tenant has already paid for.
        """
        if tokens < 0 or cost_usd < 0:
            raise ValueError("recorded usage must not be negative")
        now = self._clock()
        with self._lock:
            for level, subject in (("tenant", scope.tenant_id), ("user", scope.user_key)):
                counters = self._counters_for(level, subject, now)  # type: ignore[arg-type]
                counters.tokens_today += tokens
                counters.cost_today_usd += cost_usd

    def snapshot(self, scope: TenantScope, level: QuotaLevel = "tenant") -> QuotaSnapshot:
        now = self._clock()
        subject = scope.tenant_id if level == "tenant" else scope.user_key
        policy = self.tenant_policy(subject) if level == "tenant" else self.user_policy(subject)
        with self._lock:
            counters = self._counters_for(level, subject, now)
            return QuotaSnapshot(
                subject=subject,
                level=level,
                requests_this_minute=counters.requests_this_minute,
                requests_today=counters.requests_today,
                tokens_today=counters.tokens_today,
                cost_today_usd=counters.cost_today_usd,
                policy=policy,
            )

    # -- internals -----------------------------------------------------------

    def _counters_for(self, level: QuotaLevel, subject: str, now: float) -> _Counters:
        key = (level, subject)
        counters = self._counters.get(key)
        if counters is None:
            counters = _Counters()
            self._counters[key] = counters
            while len(self._counters) > self._max_tracked:
                self._counters.popitem(last=False)
        self._counters.move_to_end(key)
        counters.roll(now)
        return counters

    def _assert_request_headroom(
        self,
        counters: _Counters,
        policy: QuotaPolicy,
        level: QuotaLevel,
        subject: str,
        now: float,
    ) -> None:
        if policy.is_unlimited:
            return

        if (
            policy.requests_per_minute is not None
            and counters.requests_this_minute >= policy.requests_per_minute
        ):
            raise QuotaExceededError(
                f"{level} request-per-minute quota exhausted",
                retry_after_s=_seconds_until_next(now, _SECONDS_PER_MINUTE),
            )
        if (
            policy.requests_per_day is not None
            and counters.requests_today >= policy.requests_per_day
        ):
            raise QuotaExceededError(
                f"{level} request-per-day quota exhausted",
                retry_after_s=_seconds_until_next(now, _SECONDS_PER_DAY),
            )
        if (
            policy.requests_per_month is not None
            and counters.requests_this_month >= policy.requests_per_month
        ):
            raise QuotaExceededError(
                f"{level} request-per-month quota exhausted",
                retry_after_s=_seconds_until_next(now, _SECONDS_PER_MONTH),
            )
        if policy.tokens_per_day is not None and counters.tokens_today >= policy.tokens_per_day:
            raise QuotaExceededError(
                f"{level} token-per-day quota exhausted",
                retry_after_s=_seconds_until_next(now, _SECONDS_PER_DAY),
            )
        if (
            policy.cost_per_day_usd is not None
            and counters.cost_today_usd >= policy.cost_per_day_usd
        ):
            raise QuotaExceededError(
                f"{level} daily spend ceiling reached",
                retry_after_s=_seconds_until_next(now, _SECONDS_PER_DAY),
            )
        del subject  # named for readability at the call site


def _seconds_until_next(now: float, window_seconds: int) -> int:
    elapsed = now % window_seconds
    return max(1, int(window_seconds - elapsed))
