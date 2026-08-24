"""Redis/Valkey client and distributed ephemeral state.

Durable records live in PostgreSQL. This module holds hot state that must be
shared across workers: revocation, quotas, circuit breakers, and cache entries.

All keys are prefixed `fabric:` and include the tenant where tenant-scoped.
Atomic quota admission uses a Lua script so concurrent workers cannot
GET/modify/SET past a ceiling.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Any

import redis

from llm_fabric.errors import ConfigurationError, DependencyUnavailableError, QuotaExceededError
from llm_fabric.tenancy.quota import (
    _SECONDS_PER_DAY,
    _SECONDS_PER_MINUTE,
    _SECONDS_PER_MONTH,
    UNLIMITED,
    QuotaLedger,
    QuotaLevel,
    QuotaPolicy,
    QuotaSnapshot,
    _seconds_until_next,
)
from llm_fabric.tenancy.scope import TenantScope

KEY_PREFIX = "fabric"


STARTUP_PROBE_TIMEOUT_S = 3


def connect_redis(
    url: str,
    *,
    socket_connect_timeout: float | None = None,
    socket_timeout: float | None = None,
) -> redis.Redis:
    kwargs: dict[str, Any] = {"decode_responses": True}
    if socket_connect_timeout is not None:
        kwargs["socket_connect_timeout"] = socket_connect_timeout
    if socket_timeout is not None:
        kwargs["socket_timeout"] = socket_timeout
    return redis.Redis.from_url(url, **kwargs)


def probe_redis(url: str, *, timeout_s: float = STARTUP_PROBE_TIMEOUT_S) -> None:
    """Fail closed when Redis cannot be reached. Disposes the probe client."""
    client = connect_redis(
        url,
        socket_connect_timeout=timeout_s,
        socket_timeout=timeout_s,
    )
    try:
        if client.ping() is not True:
            raise ConfigurationError("Redis PING did not return True")
    except ConfigurationError:
        raise
    except Exception:
        raise ConfigurationError(
            "Redis is unreachable; production requires a reachable LLM_FABRIC_REDIS_URL"
        ) from None
    finally:
        client.close()
        client.connection_pool.disconnect()


class RedisRevocationStore:
    """TTL denylist shared across workers.

    Stateless JWTs are still only as revoked as this store is reachable. If
    Redis disappears, `fail_closed` refuses credentials; `fail_open` (the
    default outside production) treats the store as empty so inference continues.
    """

    def __init__(
        self,
        client: redis.Redis,
        *,
        fail_closed: bool = False,
        default_ttl_s: int = 86_400,
    ) -> None:
        self._client = client
        self._fail_closed = fail_closed
        self._default_ttl_s = default_ttl_s
        self._dependency_health: Any | None = None

    def bind_dependency_health(self, health: Any) -> None:
        self._dependency_health = health

    def is_revoked(self, *, token_id: str | None = None, fingerprint: str | None = None) -> bool:
        keys = []
        if token_id:
            keys.append(f"{KEY_PREFIX}:revoke:jti:{token_id}")
        if fingerprint:
            keys.append(f"{KEY_PREFIX}:revoke:fp:{fingerprint}")
        if not keys:
            return False
        try:
            return any(self._client.exists(key) for key in keys)
        except redis.RedisError:
            if self._dependency_health is not None:
                self._dependency_health.observe_serving_failure("redis", reason="serving_failure")
            if self._fail_closed:
                raise DependencyUnavailableError("revocation store is unreachable") from None
            return False

    def revoke(
        self,
        *,
        token_id: str | None = None,
        fingerprint: str | None = None,
        expires_at: float | None = None,
    ) -> None:
        if token_id is None and fingerprint is None:
            raise ValueError("revoke requires a token_id or a fingerprint")
        ttl = self._ttl(expires_at)
        mapping = []
        if token_id:
            mapping.append(f"{KEY_PREFIX}:revoke:jti:{token_id}")
        if fingerprint:
            mapping.append(f"{KEY_PREFIX}:revoke:fp:{fingerprint}")
        for key in mapping:
            self._client.set(key, "1", ex=ttl)

    def _ttl(self, expires_at: float | None) -> int:
        if expires_at is None:
            return self._default_ttl_s
        remaining = int(expires_at - time.time())
        return max(1, remaining)


@contextmanager
def redis_lock(client: redis.Redis, name: str, *, timeout_s: float = 2.0) -> Iterator[None]:
    """SET NX mutex. Avoids Lua so the path works on fakeredis without lupa."""
    token = uuid.uuid4().hex
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if client.set(name, token, nx=True, ex=max(1, int(timeout_s))):
            try:
                yield
            finally:
                if client.get(name) == token:
                    client.delete(name)
            return
        time.sleep(0.005)
    raise ConfigurationError("redis lock timed out")


class RedisQuotaLedger(QuotaLedger):
    """Distributed request/token/spend ceilings. Limits apply fleet-wide."""

    def __init__(
        self,
        client: redis.Redis,
        *,
        default_tenant_policy: QuotaPolicy = UNLIMITED,
        default_user_policy: QuotaPolicy = UNLIMITED,
        tenant_policies: Mapping[str, QuotaPolicy] | None = None,
        user_policies: Mapping[str, QuotaPolicy] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(
            default_tenant_policy=default_tenant_policy,
            default_user_policy=default_user_policy,
            tenant_policies=tenant_policies,
            user_policies=user_policies,
            clock=clock,
        )
        self._client = client
        self._dependency_health: Any | None = None

    def bind_dependency_health(self, health: Any) -> None:
        self._dependency_health = health

    def _redis_unavailable(self) -> None:
        if self._dependency_health is not None:
            self._dependency_health.observe_serving_failure("redis", reason="serving_failure")
        raise DependencyUnavailableError("quota store is unreachable")

    def set_tenant_policy(self, tenant_id: str, policy: QuotaPolicy) -> None:
        self._tenant_policies[tenant_id] = policy

    def set_user_policy(self, user_key: str, policy: QuotaPolicy) -> None:
        self._user_policies[user_key] = policy

    def tenant_policy(self, tenant_id: str) -> QuotaPolicy:
        return self._tenant_policies.get(tenant_id, self._default_tenant_policy)

    def user_policy(self, user_key: str) -> QuotaPolicy:
        return self._user_policies.get(user_key, self._default_user_policy)

    def admit(
        self,
        scope: TenantScope,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        now = self._clock()
        minute = int(now // _SECONDS_PER_MINUTE)
        day = int(now // _SECONDS_PER_DAY)
        month = int(now // _SECONDS_PER_MONTH)
        tenant_policy = self.tenant_policy(scope.tenant_id)
        user_policy = self.user_policy(scope.user_key)
        rpm_ttl = _SECONDS_PER_MINUTE * 2
        rpd_ttl = _SECONDS_PER_DAY + 3600
        rpm_month_ttl = _SECONDS_PER_MONTH + 3600
        keys = {
            "t_rpm": f"{KEY_PREFIX}:quota:tenant:{scope.tenant_id}:rpm:{minute}",
            "t_rpd": f"{KEY_PREFIX}:quota:tenant:{scope.tenant_id}:rpd:{day}",
            "t_rpm_m": f"{KEY_PREFIX}:quota:tenant:{scope.tenant_id}:rpm_m:{month}",
            "u_rpm": f"{KEY_PREFIX}:quota:user:{scope.user_key}:rpm:{minute}",
            "u_rpd": f"{KEY_PREFIX}:quota:user:{scope.user_key}:rpd:{day}",
            "t_tok": f"{KEY_PREFIX}:quota:tenant:{scope.tenant_id}:tok:{day}",
            "t_cost": f"{KEY_PREFIX}:quota:tenant:{scope.tenant_id}:cost:{day}",
            "u_tok": f"{KEY_PREFIX}:quota:user:{scope.user_key}:tok:{day}",
            "u_cost": f"{KEY_PREFIX}:quota:user:{scope.user_key}:cost:{day}",
        }
        extras: list[tuple[str, int | None]] = []
        if tenant_policy.project_requests_per_minute is not None and scope.project_id:
            extras.append(
                (
                    f"{KEY_PREFIX}:quota:project:{scope.tenant_id}/{scope.project_id}:rpm:{minute}",
                    tenant_policy.project_requests_per_minute,
                )
            )
        if tenant_policy.provider_requests_per_minute is not None and provider:
            extras.append(
                (
                    f"{KEY_PREFIX}:quota:provider:{provider}:rpm:{minute}",
                    tenant_policy.provider_requests_per_minute,
                )
            )
        if tenant_policy.model_requests_per_minute is not None and model:
            extras.append(
                (
                    f"{KEY_PREFIX}:quota:model:{model}:rpm:{minute}",
                    tenant_policy.model_requests_per_minute,
                )
            )
        lock_name = f"{KEY_PREFIX}:quota:lock:{scope.tenant_id}:{scope.user_key}"
        bumped: list[str] = []
        try:
            with redis_lock(self._client, lock_name):
                t_rpm = self._bump(keys["t_rpm"], rpm_ttl)
                bumped.append(keys["t_rpm"])
                t_rpd = self._bump(keys["t_rpd"], rpd_ttl)
                bumped.append(keys["t_rpd"])
                t_month = self._bump(keys["t_rpm_m"], rpm_month_ttl)
                bumped.append(keys["t_rpm_m"])
                u_rpm = self._bump(keys["u_rpm"], rpm_ttl)
                bumped.append(keys["u_rpm"])
                u_rpd = self._bump(keys["u_rpd"], rpd_ttl)
                bumped.append(keys["u_rpd"])
                extra_values: list[int] = []
                for extra_key, _limit in extras:
                    extra_values.append(self._bump(extra_key, rpm_ttl))
                    bumped.append(extra_key)
                t_tok = float(self._client.get(keys["t_tok"]) or 0)
                t_cost = float(self._client.get(keys["t_cost"]) or 0)
                u_tok = float(self._client.get(keys["u_tok"]) or 0)
                u_cost = float(self._client.get(keys["u_cost"]) or 0)
                over = (
                    _exceeds(tenant_policy.requests_per_minute, t_rpm)
                    or _exceeds(tenant_policy.requests_per_day, t_rpd)
                    or _exceeds(tenant_policy.requests_per_month, t_month)
                    or _exceeds(user_policy.requests_per_minute, u_rpm)
                    or _exceeds(user_policy.requests_per_day, u_rpd)
                    or _exceeds(tenant_policy.tokens_per_day, t_tok)
                    or _exceeds(tenant_policy.cost_per_day_usd, t_cost)
                    or _exceeds(user_policy.tokens_per_day, u_tok)
                    or _exceeds(user_policy.cost_per_day_usd, u_cost)
                    or any(
                        _exceeds(limit, value)
                        for (_key, limit), value in zip(extras, extra_values, strict=True)
                    )
                )
                if over:
                    for key in bumped:
                        self._client.decr(key)
                    raise QuotaExceededError(
                        "distributed quota exhausted",
                        retry_after_s=_seconds_until_next(now, _SECONDS_PER_MINUTE),
                    )
        except QuotaExceededError:
            raise
        except (redis.RedisError, ConfigurationError):
            self._redis_unavailable()

    def acquire_concurrency(self, scope: TenantScope) -> None:
        tenant_policy = self.tenant_policy(scope.tenant_id)
        user_policy = self.user_policy(scope.user_key)
        pairs = (
            ("tenant", scope.tenant_id, tenant_policy.max_concurrency),
            ("user", scope.user_key, user_policy.max_concurrency),
        )
        if all(limit is None for _kind, _subject, limit in pairs):
            return
        lock_name = f"{KEY_PREFIX}:quota:clock:{scope.tenant_id}:{scope.user_key}"
        acquired: list[str] = []
        try:
            with redis_lock(self._client, lock_name):
                for kind, subject, limit in pairs:
                    if limit is None:
                        continue
                    key = f"{KEY_PREFIX}:quota:{kind}:{subject}:inflight"
                    current = int(self._client.get(key) or 0)
                    if current >= limit:
                        for held in acquired:
                            self._client.decr(held)
                        raise QuotaExceededError(
                            f"{kind} concurrency quota exhausted",
                            retry_after_s=1,
                        )
                    self._client.incr(key)
                    self._client.expire(key, 120)
                    acquired.append(key)
        except QuotaExceededError:
            raise
        except (redis.RedisError, ConfigurationError):
            self._redis_unavailable()

    def release_concurrency(self, scope: TenantScope) -> None:
        for kind, subject in (("tenant", scope.tenant_id), ("user", scope.user_key)):
            key = f"{KEY_PREFIX}:quota:{kind}:{subject}:inflight"
            try:
                value = int(self._client.decr(key))
                if value < 0:
                    self._client.set(key, 0)
            except redis.RedisError:
                return

    def _bump(self, key: str, ttl: int) -> int:
        value = int(self._client.incr(key))
        if value == 1:
            self._client.expire(key, ttl)
        return value

    def record_usage(
        self,
        scope: TenantScope,
        *,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        if tokens < 0 or cost_usd < 0:
            raise ValueError("recorded usage must not be negative")
        now = self._clock()
        day = int(now // _SECONDS_PER_DAY)
        pipe = self._client.pipeline()
        for kind, subject in (("tenant", scope.tenant_id), ("user", scope.user_key)):
            tok_key = f"{KEY_PREFIX}:quota:{kind}:{subject}:tok:{day}"
            cost_key = f"{KEY_PREFIX}:quota:{kind}:{subject}:cost:{day}"
            pipe.incrby(tok_key, tokens)
            pipe.incrbyfloat(cost_key, cost_usd)
            pipe.expire(tok_key, _SECONDS_PER_DAY + 3600)
            pipe.expire(cost_key, _SECONDS_PER_DAY + 3600)
        pipe.execute()

    def snapshot(self, scope: TenantScope, level: QuotaLevel = "tenant") -> QuotaSnapshot:
        now = self._clock()
        minute = int(now // _SECONDS_PER_MINUTE)
        day = int(now // _SECONDS_PER_DAY)
        subject = scope.tenant_id if level == "tenant" else scope.user_key
        kind = "tenant" if level == "tenant" else "user"
        policy = self.tenant_policy(subject) if level == "tenant" else self.user_policy(subject)
        rpm = int(self._client.get(f"{KEY_PREFIX}:quota:{kind}:{subject}:rpm:{minute}") or 0)
        rpd = int(self._client.get(f"{KEY_PREFIX}:quota:{kind}:{subject}:rpd:{day}") or 0)
        tokens = int(float(self._client.get(f"{KEY_PREFIX}:quota:{kind}:{subject}:tok:{day}") or 0))
        cost = float(self._client.get(f"{KEY_PREFIX}:quota:{kind}:{subject}:cost:{day}") or 0)
        return QuotaSnapshot(
            subject=subject,
            level=level,
            requests_this_minute=rpm,
            requests_today=rpd,
            tokens_today=tokens,
            cost_today_usd=cost,
            policy=policy,
        )


def _exceeds(limit: int | float | None, value: float) -> bool:
    return limit is not None and value > limit


class RedisHealthTracker:
    """Shared breaker state. A tripped provider is visible to every worker.

    Per-deployment JSON is mutated under a short Redis lock so two workers
    cannot GET/modify/SET the breaker past each other. In-flight concurrency
    remains process-local: it is a queue-depth signal, not fleet capacity.
    State hashes expire so a vanished worker cannot pin a breaker open forever.
    """

    def __init__(
        self,
        client: redis.Redis,
        *,
        policy: Any | None = None,
        ttl_s: int = 300,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        from llm_fabric.router.health import BreakerPolicy, BreakerState, HealthSnapshot

        self._client = client
        self._policy = policy or BreakerPolicy()
        self._ttl_s = ttl_s
        self._clock = clock
        self._BreakerState = BreakerState
        self._HealthSnapshot = HealthSnapshot
        self._local_depth: dict[str, int] = {}
        self._overrides: dict[str, Any] = {}

    def _key(self, deployment_id: str) -> str:
        return f"{KEY_PREFIX}:health:{deployment_id}"

    def _policy_for(self, deployment_id: str) -> Any:
        return self._overrides.get(deployment_id, self._policy)

    def set_policy(self, deployment_id: str, policy: Any) -> None:
        self._overrides[deployment_id] = policy

    def _empty(self) -> dict[str, Any]:
        return {
            "state": self._BreakerState.CLOSED.value,
            "samples": 0,
            "successes": 0,
            "failures": 0,
            "consecutive_failures": 0,
            "half_open_successes": 0,
            "half_open_in_flight": 0,
            "ewma_latency_ms": None,
            "ewma_error": None,
            "opened_at": None,
            "last_error": None,
            "held_open": False,
        }

    def _load(self, deployment_id: str) -> dict[str, Any]:
        raw = self._client.get(self._key(deployment_id))
        if not raw:
            return self._empty()
        data = json.loads(raw)
        return {**self._empty(), **data}

    def _save(self, deployment_id: str, data: dict[str, Any]) -> None:
        self._client.set(self._key(deployment_id), json.dumps(data), ex=self._ttl_s)

    def _locked(self, deployment_id: str) -> AbstractContextManager[None]:
        return redis_lock(self._client, f"{KEY_PREFIX}:healthlock:{deployment_id}")

    def record_success(self, deployment_id: str, *, latency_ms: float) -> None:
        with self._locked(deployment_id):
            data = self._load(deployment_id)
            data["samples"] += 1
            data["successes"] += 1
            data["consecutive_failures"] = 0
            data["ewma_latency_ms"] = (
                latency_ms
                if data["ewma_latency_ms"] is None
                else (0.2 * latency_ms + 0.8 * data["ewma_latency_ms"])
            )
            data["ewma_error"] = (
                0.0 if data["ewma_error"] is None else 0.2 * 0.0 + 0.8 * data["ewma_error"]
            )
            if data["state"] == self._BreakerState.HALF_OPEN.value:
                data["half_open_successes"] += 1
                needed = self._policy_for(deployment_id).half_open_successes
                if data["half_open_successes"] >= needed:
                    data["state"] = self._BreakerState.CLOSED.value
                    data["opened_at"] = None
                    data["half_open_successes"] = 0
                    data["held_open"] = False
            self._save(deployment_id, data)

    def record_failure(
        self,
        deployment_id: str,
        *,
        latency_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        with self._locked(deployment_id):
            policy = self._policy_for(deployment_id)
            data = self._load(deployment_id)
            data["samples"] += 1
            data["failures"] += 1
            data["consecutive_failures"] += 1
            data["last_error"] = error
            if latency_ms > 0:
                data["ewma_latency_ms"] = (
                    latency_ms
                    if data["ewma_latency_ms"] is None
                    else (0.2 * latency_ms + 0.8 * data["ewma_latency_ms"])
                )
            data["ewma_error"] = (
                1.0 if data["ewma_error"] is None else 0.2 * 1.0 + 0.8 * data["ewma_error"]
            )
            if data["state"] == self._BreakerState.HALF_OPEN.value:
                data["state"] = self._BreakerState.OPEN.value
                data["opened_at"] = self._clock()
                data["half_open_successes"] = 0
                self._save(deployment_id, data)
                return
            tripped = data["consecutive_failures"] >= policy.consecutive_failures or (
                data["samples"] >= policy.minimum_samples
                and data["ewma_error"] is not None
                and data["ewma_error"] >= policy.error_rate
            )
            if data["state"] == self._BreakerState.CLOSED.value and tripped:
                data["state"] = self._BreakerState.OPEN.value
                data["opened_at"] = self._clock()
            self._save(deployment_id, data)

    def force_open(
        self, deployment_id: str, *, reason: str | None = None, hold: bool = True
    ) -> None:
        with self._locked(deployment_id):
            data = self._load(deployment_id)
            if reason:
                data["last_error"] = reason
            data["state"] = self._BreakerState.OPEN.value
            data["opened_at"] = self._clock()
            data["held_open"] = hold
            self._save(deployment_id, data)

    def force_close(self, deployment_id: str) -> None:
        with self._locked(deployment_id):
            data = self._load(deployment_id)
            data["state"] = self._BreakerState.CLOSED.value
            data["opened_at"] = None
            data["held_open"] = False
            data["consecutive_failures"] = 0
            data["half_open_successes"] = 0
            self._save(deployment_id, data)

    def admits(self, deployment_id: str) -> bool:
        with self._locked(deployment_id):
            policy = self._policy_for(deployment_id)
            data = self._load(deployment_id)
            local_depth = self._local_depth.get(deployment_id, 0)
            if policy.max_concurrency is not None and local_depth >= policy.max_concurrency:
                return False
            state = data["state"]
            if state == self._BreakerState.CLOSED.value:
                return True
            if state == self._BreakerState.OPEN.value:
                if data["held_open"]:
                    return False
                opened_at = data["opened_at"] or 0.0
                if (self._clock() - opened_at) < policy.open_duration_s:
                    return False
                data["state"] = self._BreakerState.HALF_OPEN.value
                data["half_open_successes"] = 0
                self._save(deployment_id, data)
                state = self._BreakerState.HALF_OPEN.value
            return bool(data["half_open_in_flight"] < policy.half_open_successes)

    def in_flight(self, deployment_id: str) -> Any:
        from contextlib import contextmanager

        @contextmanager
        def _cm() -> Any:
            self._local_depth[deployment_id] = self._local_depth.get(deployment_id, 0) + 1
            try:
                yield
            finally:
                self._local_depth[deployment_id] = max(
                    0, self._local_depth.get(deployment_id, 0) - 1
                )

        return _cm()

    def snapshot(self, deployment_id: str) -> Any:
        data = self._load(deployment_id)
        state = self._BreakerState(data["state"])
        error_rate = data["ewma_error"]
        samples = data["samples"]
        health = None
        if samples:
            health = (
                0.0
                if state is self._BreakerState.OPEN
                else (None if error_rate is None else max(0.0, min(1.0, 1.0 - error_rate)))
            )
        return self._HealthSnapshot(
            deployment_id=deployment_id,
            state=state,
            samples=samples,
            successes=data["successes"],
            failures=data["failures"],
            queue_depth=self._local_depth.get(deployment_id, 0),
            ewma_latency_ms=data["ewma_latency_ms"],
            error_rate=error_rate,
            health_score=health,
            consecutive_failures=data["consecutive_failures"],
            opened_at=data["opened_at"],
            last_error=data["last_error"],
            held_open=bool(data["held_open"]),
        )

    def all_snapshots(self) -> dict[str, Any]:
        found: dict[str, Any] = {}
        for key in self._client.scan_iter(f"{KEY_PREFIX}:health:*"):
            name = str(key).rsplit(":", 1)[-1]
            found[name] = self.snapshot(name)
        return found

    def reset(self, deployment_id: str | None = None) -> None:
        if deployment_id is None:
            for key in list(self._client.scan_iter(f"{KEY_PREFIX}:health:*")):
                self._client.delete(key)
            return
        self._client.delete(self._key(deployment_id))


class RedisCache:
    """Tenant-scoped cache entries. Values are JSON. Missing Redis is a miss."""

    def __init__(self, client: redis.Redis, *, fail_soft: bool = True) -> None:
        self._client = client
        self._fail_soft = fail_soft

    def get(self, tenant_id: str, namespace: str, fingerprint: str) -> Any | None:
        key = f"{KEY_PREFIX}:cache:{namespace}:{tenant_id}:{fingerprint}"
        try:
            raw = self._client.get(key)
        except redis.RedisError:
            if self._fail_soft:
                return None
            raise
        if raw is None:
            return None
        envelope = json.loads(raw)
        if envelope.get("tenant_id") != tenant_id:
            return None
        return envelope.get("value")

    def put(
        self,
        tenant_id: str,
        namespace: str,
        fingerprint: str,
        value: Any,
        *,
        ttl_seconds: float,
    ) -> None:
        key = f"{KEY_PREFIX}:cache:{namespace}:{tenant_id}:{fingerprint}"
        envelope = json.dumps({"tenant_id": tenant_id, "value": value}, default=str)
        try:
            self._client.set(key, envelope, ex=max(1, int(ttl_seconds)))
        except redis.RedisError:
            if self._fail_soft:
                return
            raise

    def invalidate(self, tenant_id: str, namespace: str, fingerprint: str) -> bool:
        key = f"{KEY_PREFIX}:cache:{namespace}:{tenant_id}:{fingerprint}"
        try:
            return bool(self._client.delete(key))
        except redis.RedisError:
            if self._fail_soft:
                return False
            raise
