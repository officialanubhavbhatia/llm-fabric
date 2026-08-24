"""Bounded background probes for serving dependencies.

`/readyz` and admission read cached state from `DependencyHealth`. This
monitor is the active source of that state. Serving-path failures are a
second, faster source (passive).

The loop is cancelled on application shutdown. Probes use the same
timeout-bounded helpers as production startup (`SELECT 1`, `PING`) and never
include a DSN in the error they record.
"""

from __future__ import annotations

import asyncio
import logging
import random
from contextlib import suppress
from typing import Any

from llm_fabric.deps.health import DependencyHealth
from llm_fabric.storage.postgres import probe_database
from llm_fabric.storage.redis import probe_redis

_LOG = logging.getLogger("llm_fabric.deps")


class DependencyMonitor:
    def __init__(
        self,
        health: DependencyHealth,
        *,
        database_url: str | None = None,
        redis_url: str | None = None,
        interval_s: float = 2.0,
        timeout_s: float = 1.0,
        jitter: bool = True,
    ) -> None:
        self._health = health
        self._database_url = database_url
        self._redis_url = redis_url
        self._interval_s = interval_s
        self._timeout_s = timeout_s
        self._jitter = jitter
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    def has_work(self) -> bool:
        return bool(self._database_url or self._redis_url)

    async def start(self) -> None:
        """Run one probe immediately, then the background loop if needed."""
        await self.probe_once()
        if not self.has_work() or self._stopping:
            return
        self._task = asyncio.create_task(self._run(), name="llm-fabric-dependency-monitor")

    async def stop(self) -> None:
        self._stopping = True
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def probe_once(self) -> None:
        if self._database_url:
            await self._probe("postgres", self._check_postgres)
        if self._redis_url:
            await self._probe("redis", self._check_redis)

    async def _run(self) -> None:
        try:
            while not self._stopping:
                await asyncio.sleep(self._sleep_s())
                if self._stopping:
                    return
                await self.probe_once()
        except asyncio.CancelledError:
            raise

    def _sleep_s(self) -> float:
        if not self._jitter:
            return self._interval_s
        # 75%–125% of the interval so replicas do not probe in lockstep.
        return self._interval_s * (0.75 + 0.5 * random.random())

    async def _probe(self, name: str, check: Any) -> None:
        try:
            await asyncio.wait_for(asyncio.to_thread(check), timeout=self._timeout_s + 0.25)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            _LOG.warning("dependency probe timed out", extra={"dependency": name})
            self._health.observe_probe_failure(name, reason="probe_timeout")
        except Exception:
            _LOG.warning("dependency probe failed", extra={"dependency": name})
            self._health.observe_probe_failure(name, reason="probe_failed")
        else:
            self._health.observe_probe_success(name)

    def _check_postgres(self) -> None:
        assert self._database_url is not None
        probe_database(self._database_url, timeout_s=self._timeout_s)

    def _check_redis(self) -> None:
        assert self._redis_url is not None
        probe_redis(self._redis_url, timeout_s=self._timeout_s)
