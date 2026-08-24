"""First-class evaluation objects named by the constitution.

A run is only as trustworthy as the provenance it carries. Every field below is
something that was known at evaluation time; missing pieces stay `None` rather
than being filled with a plausible default.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:20]}"


#: Version of the in-repo metric implementations. Bump when a formula changes
#: so two runs that look comparable are not silently using different maths.
METRIC_VERSION = "eval-metrics-v1"


class EvaluatorKind(StrEnum):
    DETERMINISTIC = "deterministic"
    CLASSIFICATION = "classification"
    ROUTING = "routing"
    JUDGE = "judge"
    DEEPEVAL = "deepeval"
    LM_EVAL = "lm_eval"


class GateDirection(StrEnum):
    """Whether a larger score is better or worse."""

    HIGHER = "higher"
    LOWER = "lower"


class GateKind(StrEnum):
    """Two distinct release questions. Do not collapse them into one number.

    `absolute` — is this score good enough for the product, regardless of
    yesterday's run. `regression` — did this score drop materially from the
    accepted baseline. A classifier can pass a weak absolute floor and still
    fail a regression tripwire, or the reverse.
    """

    ABSOLUTE = "absolute"
    REGRESSION = "regression"


@dataclass(frozen=True, slots=True)
class EvalProvenance:
    """What produced a run. The constitution's required field set."""

    dataset_version: str
    metric_version: str = METRIC_VERSION
    configuration: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    commit: str | None = None
    model: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    taxonomy_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvalMetricSpec:
    """One named measurement a suite has opted into.

    Suites list metrics explicitly. A metric that is not listed is not computed,
    which is how the constitution's "do not apply irrelevant metrics" rule is
    enforced rather than hoped for.
    """

    name: str
    direction: GateDirection = GateDirection.HIGHER
    description: str = ""


@dataclass(frozen=True, slots=True)
class EvalTask:
    name: str
    evaluator: EvaluatorKind
    metrics: tuple[str, ...]
    dataset_path: str | None = None
    dataset_id: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalSuite:
    name: str
    tasks: tuple[EvalTask, ...]
    tenant_id: str = "public"
    suite_id: str = field(default_factory=lambda: _new_id("evsu"))
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "description": self.description,
            "tenant_id": self.tenant_id,
            "tasks": [
                {
                    "name": task.name,
                    "evaluator": task.evaluator.value,
                    "metrics": list(task.metrics),
                    "dataset_path": task.dataset_path,
                    "dataset_id": task.dataset_id,
                    "options": task.options,
                }
                for task in self.tasks
            ],
        }


@dataclass(frozen=True, slots=True)
class ExampleResult:
    example_id: str
    scores: dict[str, float | None]
    output: str | None = None
    error: str | None = None
    notes: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "scores": self.scores,
            "output": self.output,
            "error": self.error,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Aggregated scores for one task. Per-example rows stay attached."""

    task: str
    evaluator: EvaluatorKind
    metrics: dict[str, float | None]
    examples: tuple[ExampleResult, ...] = ()
    unavailable: tuple[str, ...] = ()
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "evaluator": self.evaluator.value,
            "metrics": self.metrics,
            "unavailable": list(self.unavailable),
            "note": self.note,
            "examples": [row.as_dict() for row in self.examples],
        }


@dataclass(frozen=True, slots=True)
class EvalRun:
    tenant_id: str
    suite_name: str
    provenance: EvalProvenance
    results: tuple[EvalResult, ...]
    run_id: str = field(default_factory=lambda: _new_id("evrun"))
    created_at: float = field(default_factory=time.time)

    def metric(self, name: str) -> float | None:
        for result in self.results:
            if name in result.metrics:
                return result.metrics[name]
        return None

    def all_metrics(self) -> dict[str, float | None]:
        merged: dict[str, float | None] = {}
        for result in self.results:
            merged.update(result.metrics)
        return merged

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "suite_name": self.suite_name,
            "created_at": self.created_at,
            "provenance": self.provenance.as_dict(),
            "results": [result.as_dict() for result in self.results],
            "metrics": self.all_metrics(),
        }


@dataclass(frozen=True, slots=True)
class MetricDelta:
    name: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    direction: GateDirection

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "delta": self.delta,
            "direction": self.direction.value,
        }


@dataclass(frozen=True, slots=True)
class EvalComparison:
    baseline_run_id: str
    candidate_run_id: str
    deltas: tuple[MetricDelta, ...]
    tenant_id: str = "public"
    comparison_id: str = field(default_factory=lambda: _new_id("evcmp"))
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "tenant_id": self.tenant_id,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "deltas": [delta.as_dict() for delta in self.deltas],
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class EvalGate:
    metric: str
    direction: GateDirection = GateDirection.HIGHER
    kind: GateKind | None = None
    minimum: float | None = None
    maximum: float | None = None
    max_degradation: float | None = None
    critical: bool = True
    description: str = ""

    def resolved_kind(self) -> GateKind:
        if self.kind is not None:
            return self.kind
        if self.max_degradation is not None and self.minimum is None and self.maximum is None:
            return GateKind.REGRESSION
        return GateKind.ABSOLUTE


@dataclass(frozen=True, slots=True)
class GateVerdict:
    gate: EvalGate
    passed: bool
    value: float | None
    baseline: float | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.gate.metric,
            "kind": self.gate.resolved_kind().value,
            "critical": self.gate.critical,
            "passed": self.passed,
            "value": self.value,
            "baseline": self.baseline,
            "reason": self.reason,
        }


def dataset_version(examples: list[dict[str, Any]] | tuple[Any, ...]) -> str:
    """Stable hash of the labelled content. Order-sensitive on purpose."""
    payload = json.dumps(examples, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
