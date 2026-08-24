"""First-class objects for health scoring, drift and controlled remediation.

Missing measurements stay `None`. A zero that was never observed would look like
a healthy system, which is how unsupervised mutation starts.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:20]}"


class DriftKind(StrEnum):
    CLASSIFIER = "classifier"
    INTENT = "intent"
    ROUTING = "routing"
    LATENCY = "latency"
    ERROR = "error"
    COST = "cost"
    QUALITY = "quality"
    EMBEDDING = "embedding"
    CONTEXT_LENGTH = "context_length"
    SAFETY_BLOCKS = "safety_blocks"


class DriftSeverity(StrEnum):
    INSUFFICIENT = "insufficient"
    STABLE = "stable"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    UNAVAILABLE = "unavailable"


class RemediationKind(StrEnum):
    OPEN_CIRCUIT = "open_circuit_breaker"
    SHIFT_TRAFFIC = "shift_traffic"
    ROLLBACK_MODEL = "rollback_model"
    ROLLBACK_PROMPT = "rollback_prompt"
    ROLLBACK_CLASSIFIER = "rollback_classifier"
    REDUCE_CONTEXT = "reduce_context_limit"
    INVALIDATE_CACHE = "invalidate_cache"
    RAISE_INCIDENT = "raise_incident"
    LEARNING_JOB = "learning_job"


class RemediationClass(StrEnum):
    """How freely a remediation may run.

    Operational actions may apply under policy. Learning-related actions change
    what the fabric believes, and the constitution forbids promoting them
    without an evaluation. Forbidden actions are refused even when requested.
    """

    OPERATIONAL = "operational"
    LEARNING = "learning"
    FORBIDDEN = "forbidden"


LEARNING_KINDS = frozenset({RemediationKind.LEARNING_JOB})

#: Prompt rollback of a *new* version is learning. Rolling back to a previously
#: production version is operational recovery and is handled in the applicator.
#: Authorization and safety are never in the allow-list.
FORBIDDEN_TARGETS = frozenset(
    {
        "authorization",
        "authentication",
        "auth_mode",
        "safety",
        "guardrail",
        "guardrails",
        "tenant_isolation",
        "isolation",
    }
)

FORBIDDEN_KINDS = frozenset(
    {
        "mutate_authorization",
        "mutate_authentication",
        "mutate_safety",
        "disable_auth",
        "disable_guardrails",
        "disable_isolation",
    }
)


class IncidentSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class LearningJobStatus(StrEnum):
    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    PROMOTED = "promoted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    """Observed health of one model or one provider, or `None` if unmeasured."""

    id: str
    kind: str
    samples: int
    health_score: float | None
    error_rate: float | None
    ewma_latency_ms: float | None
    circuit_open: bool
    deployments: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "samples": self.samples,
            "health_score": (
                round(self.health_score, 4) if self.health_score is not None else None
            ),
            "error_rate": round(self.error_rate, 4) if self.error_rate is not None else None,
            "ewma_latency_ms": (
                round(self.ewma_latency_ms, 3) if self.ewma_latency_ms is not None else None
            ),
            "circuit_open": self.circuit_open,
            "deployments": list(self.deployments),
        }


@dataclass(frozen=True, slots=True)
class DriftSignal:
    """One named comparison. `value` is None when the comparison could not run."""

    kind: DriftKind
    metric: str
    severity: DriftSeverity
    value: float | None
    baseline: float | None
    current: float | None
    samples_baseline: int
    samples_current: int
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "metric": self.metric,
            "severity": self.severity.value,
            "value": round(value, 6) if (value := self.value) is not None else None,
            "baseline": self.baseline,
            "current": self.current,
            "samples_baseline": self.samples_baseline,
            "samples_current": self.samples_current,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class DriftReport:
    tenant_id: str
    signals: tuple[DriftSignal, ...]
    model_health: tuple[ComponentHealth, ...]
    provider_health: tuple[ComponentHealth, ...]
    created_at: float = field(default_factory=time.time)
    report_id: str = field(default_factory=lambda: _new_id("drift"))
    note: str | None = None

    def significant(self) -> tuple[DriftSignal, ...]:
        return tuple(s for s in self.signals if s.severity is DriftSeverity.SIGNIFICANT)

    def as_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at,
            "note": self.note,
            "signals": [signal.as_dict() for signal in self.signals],
            "model_health": [row.as_dict() for row in self.model_health],
            "provider_health": [row.as_dict() for row in self.provider_health],
        }


@dataclass(frozen=True, slots=True)
class RemediationProposal:
    kind: RemediationKind
    target: str
    reason: str
    classification: RemediationClass
    auto: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target": self.target,
            "reason": self.reason,
            "classification": self.classification.value,
            "auto": self.auto,
            "parameters": self.parameters,
        }


@dataclass(frozen=True, slots=True)
class RemediationRecord:
    tenant_id: str
    kind: RemediationKind
    target: str
    applied: bool
    reason: str
    classification: RemediationClass
    record_id: str = field(default_factory=lambda: _new_id("rem"))
    created_at: float = field(default_factory=time.time)
    note: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["classification"] = self.classification.value
        return payload


@dataclass(frozen=True, slots=True)
class Incident:
    tenant_id: str
    title: str
    severity: IncidentSeverity
    summary: str
    incident_id: str = field(default_factory=lambda: _new_id("inc"))
    created_at: float = field(default_factory=time.time)
    signals: tuple[str, ...] = ()
    remediation_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "tenant_id": self.tenant_id,
            "title": self.title,
            "severity": self.severity.value,
            "summary": self.summary,
            "created_at": self.created_at,
            "signals": list(self.signals),
            "remediation_ids": list(self.remediation_ids),
        }


@dataclass(frozen=True, slots=True)
class LearningJob:
    """A candidate for the intent learning loop. Never trains or promotes itself."""

    tenant_id: str
    reason: str
    status: LearningJobStatus = LearningJobStatus.PROPOSED
    job_id: str = field(default_factory=lambda: _new_id("learn"))
    created_at: float = field(default_factory=time.time)
    eval_run_id: str | None = None
    note: str | None = None
    candidate_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "reason": self.reason,
            "status": self.status.value,
            "created_at": self.created_at,
            "eval_run_id": self.eval_run_id,
            "note": self.note,
            "candidate_count": self.candidate_count,
        }


@dataclass(frozen=True, slots=True)
class DriftBaseline:
    """Captured window used as the left-hand side of a later comparison."""

    tenant_id: str
    samples: int
    intent_counts: dict[str, int] = field(default_factory=dict)
    route_counts: dict[str, int] = field(default_factory=dict)
    layer_counts: dict[str, int] = field(default_factory=dict)
    mean_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    error_rate: float | None = None
    mean_cost_usd: float | None = None
    cost_samples: int = 0
    mean_confidence: float | None = None
    unknown_rate: float | None = None
    fallback_rate: float | None = None
    mean_tokens: float | None = None
    classified_samples: int = 0
    baseline_id: str = field(default_factory=lambda: _new_id("base"))
    captured_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
