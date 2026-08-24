"""In-process operational controls that remediations actually mutate.

These are the seams the request path reads. A control that nothing consults
would be a pretend remediation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from llm_fabric.intent.cascade import IntentCascade
from llm_fabric.router.registry import ModelSpec


@dataclass
class TrafficOverlay:
    """Models removed from the candidate set by a remediation, not by policy."""

    _excluded: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def exclude(self, model_id: str) -> None:
        with self._lock:
            self._excluded.add(model_id)

    def restore(self, model_id: str | None = None) -> None:
        with self._lock:
            if model_id is None:
                self._excluded.clear()
            else:
                self._excluded.discard(model_id)

    def excludes(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._excluded

    def excluded(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._excluded)


@dataclass
class ClassifierLedger:
    """Pinned cascade revisions. Rollback needs at least one earlier pin."""

    _pins: list[IntentCascade] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def pin(self, cascade: IntentCascade) -> None:
        with self._lock:
            if self._pins and self._pins[-1].version == cascade.version:
                return
            self._pins.append(cascade)

    def current(self) -> IntentCascade | None:
        with self._lock:
            return self._pins[-1] if self._pins else None

    def previous(self) -> IntentCascade | None:
        with self._lock:
            return self._pins[-2] if len(self._pins) >= 2 else None

    def rollback(self) -> IntentCascade | None:
        with self._lock:
            if len(self._pins) < 2:
                return None
            self._pins.pop()
            return self._pins[-1]


@dataclass
class ModelRevisionBook:
    """Prior `ModelSpec` snapshots so a rollback has something real to restore."""

    _history: dict[str, list[ModelSpec]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self, spec: ModelSpec) -> None:
        with self._lock:
            trail = self._history.setdefault(spec.id, [])
            if trail and trail[-1] == spec:
                return
            trail.append(spec)

    def take_previous(self, model_id: str) -> ModelSpec | None:
        """Pop the most recently remembered spec so it can be restored."""
        with self._lock:
            trail = self._history.get(model_id) or []
            return trail.pop() if trail else None


class OperationalControls:
    """Shared by the router, the chat path and the heal controller."""

    def __init__(self) -> None:
        self.traffic = TrafficOverlay()
        self.classifiers = ClassifierLedger()
        self.models = ModelRevisionBook()
        self._context_ceiling: int | None = None
        self._lock = threading.Lock()

    @property
    def context_ceiling_tokens(self) -> int | None:
        with self._lock:
            return self._context_ceiling

    def set_context_ceiling(self, tokens: int | None) -> None:
        if tokens is not None and tokens < 1:
            raise ValueError("context ceiling must be a positive token count")
        with self._lock:
            self._context_ceiling = tokens
