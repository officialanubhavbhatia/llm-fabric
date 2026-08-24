"""Analytical telemetry sink.

High-volume traces, token events and route events must not sit on the
synchronous request path. This module is a bounded, fail-soft buffer that
drops rather than blocking inference when the analytical store is absent or
slow. A ClickHouse adapter can drain the buffer; until one is configured the
sink records a drop metric and continues.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class AnalyticsEvent:
    kind: str
    tenant_id: str | None
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)


class AnalyticsSink(Protocol):
    def emit(self, event: AnalyticsEvent) -> None: ...

    def drain(self, max_items: int = 100) -> list[AnalyticsEvent]: ...

    @property
    def dropped(self) -> int: ...


class NullAnalyticsSink:
    """Configured absence. Emits nowhere. Not a ClickHouse client."""

    def emit(self, event: AnalyticsEvent) -> None:
        del event

    def drain(self, max_items: int = 100) -> list[AnalyticsEvent]:
        del max_items
        return []

    @property
    def dropped(self) -> int:
        return 0


class BufferedAnalyticsSink:
    """Bounded in-process buffer. Never blocks the caller.

    When the buffer is full the oldest event is dropped and `dropped` increases.
    That is visible telemetry-loss, not silent unbounded memory growth.
    """

    def __init__(self, *, max_events: int = 10_000) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self._buffer: deque[AnalyticsEvent] = deque(maxlen=max_events)
        self._dropped = 0
        self._lock = threading.Lock()
        self._max = max_events

    def emit(self, event: AnalyticsEvent) -> None:
        with self._lock:
            if len(self._buffer) >= self._max:
                self._buffer.popleft()
                self._dropped += 1
            self._buffer.append(event)

    def drain(self, max_items: int = 100) -> list[AnalyticsEvent]:
        taken: list[AnalyticsEvent] = []
        with self._lock:
            while self._buffer and len(taken) < max_items:
                taken.append(self._buffer.popleft())
        return taken

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def buffered(self) -> int:
        with self._lock:
            return len(self._buffer)
