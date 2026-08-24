"""Tenant-scoped storage.

Isolation here is structural rather than conventional, and it is enforced twice
on purpose:

1. **Namespacing.** Records live in a per-tenant bucket, so one tenant's key
   simply does not exist in another tenant's namespace. There is no query that
   spans tenants because there is no shared keyspace to query.
2. **Ownership re-check.** Every record carries the tenant that wrote it, and
   every read asserts that it matches the requesting scope. If that assertion
   ever fires, the fabric has a bug: it raises `TenantIsolationError`, which is
   an internal error, not something a caller can provoke into a useful answer.

The second check is redundant while the first is correct. That is the reason to
keep it — the first is the one a future refactor can quietly break.

The in-memory backend is bounded and **not durable**. Postgres and Redis
backends are not built; `TenantScopedStore` is the interface they will implement.
"""

from __future__ import annotations

import builtins
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

from llm_fabric.errors import ResourceNotFoundError, TenantIsolationError
from llm_fabric.tenancy.scope import TenantScope

DEFAULT_MAX_RECORDS_PER_TENANT = 10_000


class TenantOwned(Protocol):
    """Anything stored here must be able to say which tenant owns it."""

    @property
    def tenant_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class IsolationViolation:
    """A recorded attempt to read across a tenant boundary."""

    store: str
    requesting_tenant: str
    owning_tenant: str
    key: str


@dataclass
class IsolationAudit:
    """Counts boundary violations so a breach is observable, not just fatal."""

    violations: list[IsolationViolation] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, violation: IsolationViolation) -> None:
        with self._lock:
            self.violations.append(violation)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.violations)

    def clear(self) -> None:
        with self._lock:
            self.violations.clear()


@dataclass(slots=True)
class _Entry[T: TenantOwned]:
    tenant_id: str
    value: T


class TenantScopedStore[T: TenantOwned]:
    """A bounded, tenant-partitioned record store."""

    def __init__(
        self,
        name: str,
        *,
        max_records_per_tenant: int = DEFAULT_MAX_RECORDS_PER_TENANT,
        audit: IsolationAudit | None = None,
    ) -> None:
        if max_records_per_tenant <= 0:
            raise ValueError("max_records_per_tenant must be positive")
        self._name = name
        self._max_records = max_records_per_tenant
        self._audit = audit or IsolationAudit()
        self._buckets: dict[str, OrderedDict[str, _Entry[T]]] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def audit(self) -> IsolationAudit:
        return self._audit

    # -- writes --------------------------------------------------------------

    def put(self, scope: TenantScope, key: str, value: T) -> T:
        """Store `value` under `key` inside `scope`'s namespace.

        A value whose own `tenant_id` disagrees with the scope is rejected: that
        is a caller trying to write a record it does not own.
        """
        if value.tenant_id != scope.tenant_id:
            self._flag(scope.tenant_id, value.tenant_id, key)
            raise TenantIsolationError(
                f"refusing to write a record owned by another tenant into {self._name}"
            )

        with self._lock:
            bucket = self._buckets.setdefault(scope.tenant_id, OrderedDict())
            bucket[key] = _Entry(tenant_id=scope.tenant_id, value=value)
            bucket.move_to_end(key)
            while len(bucket) > self._max_records:
                bucket.popitem(last=False)
        return value

    def delete(self, scope: TenantScope, key: str) -> bool:
        with self._lock:
            bucket = self._buckets.get(scope.tenant_id)
            if bucket is None:
                return False
            return bucket.pop(key, None) is not None

    def clear_tenant(self, scope: TenantScope) -> int:
        with self._lock:
            bucket = self._buckets.pop(scope.tenant_id, None)
            return len(bucket) if bucket else 0

    # -- reads ---------------------------------------------------------------

    def get(self, scope: TenantScope, key: str) -> T | None:
        """Return the record, or `None` when this tenant has no such record.

        A record belonging to another tenant is indistinguishable from absence,
        which is the intended answer: confirming existence would itself leak.
        """
        with self._lock:
            bucket = self._buckets.get(scope.tenant_id)
            entry = bucket.get(key) if bucket is not None else None

        if entry is None:
            return None
        if entry.tenant_id != scope.tenant_id or entry.value.tenant_id != scope.tenant_id:
            self._flag(scope.tenant_id, entry.tenant_id, key)
            raise TenantIsolationError(f"tenant boundary crossed inside {self._name}")
        return entry.value

    def require(self, scope: TenantScope, key: str) -> T:
        value = self.get(scope, key)
        if value is None:
            raise ResourceNotFoundError(f"no such {self._name}: '{key}'")
        return value

    def list(
        self,
        scope: TenantScope,
        *,
        limit: int = 100,
        predicate: Callable[[T], bool] | None = None,
    ) -> builtins.list[T]:
        # `builtins.list` because this method shadows the built-in inside the
        # class body, and `list` reads better than `list_records` at call sites.
        if limit <= 0:
            return []
        with self._lock:
            bucket = self._buckets.get(scope.tenant_id)
            entries = builtins.list(bucket.values()) if bucket is not None else []

        results: builtins.list[T] = []
        for entry in reversed(entries):
            if entry.tenant_id != scope.tenant_id:
                self._flag(scope.tenant_id, entry.tenant_id, "<list>")
                raise TenantIsolationError(f"tenant boundary crossed inside {self._name}")
            if predicate is not None and not predicate(entry.value):
                continue
            results.append(entry.value)
            if len(results) >= limit:
                break
        return results

    def keys(self, scope: TenantScope) -> builtins.list[str]:
        with self._lock:
            bucket = self._buckets.get(scope.tenant_id)
            return builtins.list(bucket.keys()) if bucket is not None else []

    def count(self, scope: TenantScope) -> int:
        with self._lock:
            bucket = self._buckets.get(scope.tenant_id)
            return len(bucket) if bucket is not None else 0

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._buckets))

    def _flag(self, requesting: str, owning: str, key: str) -> None:
        self._audit.record(
            IsolationViolation(
                store=self._name,
                requesting_tenant=requesting,
                owning_tenant=owning,
                key=key,
            )
        )
