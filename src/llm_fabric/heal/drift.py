"""Compare a baseline window to a current window.

Categorical drift uses population stability index with Laplace smoothing.
Numeric drift uses the relative change in a measured mean or rate. Either
window below `min_samples` produces `insufficient`, not a fabricated 0.0.

Signals the serving path cannot produce — embedding distribution, compiler
context length, safety blocks — are recorded as unavailable rather than guessed.
Quality drift uses evaluation-run metrics when two runs exist; usage records
do not carry a quality score, and declared registry quality is not a measurement.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from llm_fabric.eval.schema import EvalRun
from llm_fabric.heal.schema import (
    DriftBaseline,
    DriftKind,
    DriftReport,
    DriftSeverity,
    DriftSignal,
)
from llm_fabric.heal.scoring import model_health, provider_health
from llm_fabric.intent.metrics import percentile
from llm_fabric.intent.schema import UNKNOWN_INTENT_ID
from llm_fabric.observability.metering import UsageRecord
from llm_fabric.router.health import HealthTracker
from llm_fabric.router.registry import ModelRegistry

#: Judgements, not fitted values. PSI bands follow the usual industry cutovers.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
RATE_MODERATE = 0.05
RATE_SIGNIFICANT = 0.10
RELATIVE_MODERATE = 0.15
RELATIVE_SIGNIFICANT = 0.25
DEFAULT_MIN_SAMPLES = 20


def capture_baseline(records: Sequence[UsageRecord], *, tenant_id: str) -> DriftBaseline:
    return _summarise(records, tenant_id=tenant_id)


def analyze(
    records: Sequence[UsageRecord],
    *,
    tenant_id: str,
    health: HealthTracker,
    registry: ModelRegistry,
    eval_runs: Sequence[EvalRun] = (),
    baseline: DriftBaseline | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> DriftReport:
    ordered = sorted(records, key=lambda row: row.created_at)
    current_rows: Sequence[UsageRecord]
    if baseline is None:
        left, right = _split(ordered, min_samples=min_samples)
        base = _summarise(left, tenant_id=tenant_id)
        current_rows = right
    else:
        base = baseline
        current_rows = ordered
    current = _summarise(current_rows, tenant_id=tenant_id)
    signals = _compare(base, current, min_samples=min_samples)
    signals = signals + _quality_signals(eval_runs)
    signals = signals + _unavailable_backends()
    note = (
        "Compared an explicit baseline to the current buffer."
        if baseline is not None
        else "Split this process's usage buffer into an older and a newer window."
    )
    return DriftReport(
        tenant_id=tenant_id,
        signals=signals,
        model_health=model_health(health, registry),
        provider_health=provider_health(health, registry),
        note=note,
    )


def _split(
    records: Sequence[UsageRecord], *, min_samples: int
) -> tuple[tuple[UsageRecord, ...], tuple[UsageRecord, ...]]:
    if len(records) < 2 * min_samples:
        return (), tuple(records)
    cut = len(records) // 2
    return tuple(records[:cut]), tuple(records[cut:])


def _summarise(records: Sequence[UsageRecord], *, tenant_id: str) -> DriftBaseline:
    if not records:
        return DriftBaseline(tenant_id=tenant_id, samples=0)
    latencies = [row.latency_ms for row in records]
    errors = sum(1 for row in records if row.error is not None)
    failovers = sum(1 for row in records if row.failover_count > 0)
    priced = [row.cost_usd for row in records if not row.cost_is_estimated]
    classified = [row for row in records if row.intent_id is not None]
    confidences = [row.intent_confidence for row in classified if row.intent_confidence is not None]
    unknowns = sum(1 for row in classified if row.intent_id == UNKNOWN_INTENT_ID)
    return DriftBaseline(
        tenant_id=tenant_id,
        samples=len(records),
        intent_counts=dict(Counter(row.intent_id for row in classified if row.intent_id)),
        route_counts=dict(Counter(row.served_model for row in records)),
        layer_counts=dict(Counter(row.intent_layer for row in classified if row.intent_layer)),
        mean_latency_ms=_mean(latencies),
        p95_latency_ms=percentile(latencies, 0.95),
        error_rate=errors / len(records),
        mean_cost_usd=_mean(priced),
        cost_samples=len(priced),
        mean_confidence=_mean(confidences),
        unknown_rate=(unknowns / len(classified)) if classified else None,
        fallback_rate=failovers / len(records),
        mean_tokens=_mean([float(row.total_tokens) for row in records]),
        classified_samples=len(classified),
    )


def _compare(
    baseline: DriftBaseline | None,
    current: DriftBaseline | None,
    *,
    min_samples: int,
) -> tuple[DriftSignal, ...]:
    signals: list[DriftSignal] = []
    base_n = baseline.samples if baseline else 0
    curr_n = current.samples if current else 0

    signals.append(
        _categorical(
            DriftKind.INTENT,
            "intent_distribution",
            baseline.intent_counts if baseline else {},
            current.intent_counts if current else {},
            baseline.classified_samples if baseline else 0,
            current.classified_samples if current else 0,
            min_samples,
            missing="No classified traffic; intent classification is off or produced no ids.",
        )
    )
    signals.append(
        _categorical(
            DriftKind.ROUTING,
            "route_distribution",
            baseline.route_counts if baseline else {},
            current.route_counts if current else {},
            base_n,
            curr_n,
            min_samples,
            missing="Not enough routed requests to compare model mix.",
        )
    )
    signals.append(
        _categorical(
            DriftKind.CLASSIFIER,
            "classifier_layer_mix",
            baseline.layer_counts if baseline else {},
            current.layer_counts if current else {},
            baseline.classified_samples if baseline else 0,
            current.classified_samples if current else 0,
            min_samples,
            missing="No classifier layer labels on usage records.",
        )
    )
    signals.append(
        _relative(
            DriftKind.CLASSIFIER,
            "mean_confidence",
            baseline.mean_confidence if baseline else None,
            current.mean_confidence if current else None,
            baseline.classified_samples if baseline else 0,
            current.classified_samples if current else 0,
            min_samples,
            invert=True,
            missing="Classifier confidence was not recorded on these requests.",
        )
    )
    signals.append(
        _rate(
            DriftKind.CLASSIFIER,
            "unknown_intent_rate",
            baseline.unknown_rate if baseline else None,
            current.unknown_rate if current else None,
            baseline.classified_samples if baseline else 0,
            current.classified_samples if current else 0,
            min_samples,
            missing="Unknown-intent rate needs classified traffic.",
        )
    )
    signals.append(
        _relative(
            DriftKind.LATENCY,
            "p95_latency_ms",
            baseline.p95_latency_ms if baseline else None,
            current.p95_latency_ms if current else None,
            base_n,
            curr_n,
            min_samples,
            missing="Latency was not measured.",
        )
    )
    signals.append(
        _rate(
            DriftKind.ERROR,
            "error_rate",
            baseline.error_rate if baseline else None,
            current.error_rate if current else None,
            base_n,
            curr_n,
            min_samples,
            missing="Not enough requests to compute an error rate.",
        )
    )
    cost_base_n = baseline.cost_samples if baseline else 0
    cost_curr_n = current.cost_samples if current else 0
    signals.append(
        _relative(
            DriftKind.COST,
            "mean_cost_usd",
            baseline.mean_cost_usd if baseline else None,
            current.mean_cost_usd if current else None,
            cost_base_n,
            cost_curr_n,
            min_samples,
            missing=(
                "Cost drift uses only records whose cost is not estimated. None were available."
            ),
        )
    )
    signals.append(
        _rate(
            DriftKind.ROUTING,
            "fallback_rate",
            baseline.fallback_rate if baseline else None,
            current.fallback_rate if current else None,
            base_n,
            curr_n,
            min_samples,
            missing="Not enough requests to compute a fallback rate.",
        )
    )
    return tuple(signals)


def _quality_signals(runs: Sequence[EvalRun]) -> tuple[DriftSignal, ...]:
    if len(runs) < 2:
        return (
            DriftSignal(
                kind=DriftKind.QUALITY,
                metric="eval_metrics",
                severity=DriftSeverity.UNAVAILABLE,
                value=None,
                baseline=None,
                current=None,
                samples_baseline=len(runs),
                samples_current=0,
                note=(
                    "Quality drift compares two evaluation runs. Usage records "
                    "do not carry a quality score, and declared registry quality "
                    "is not a measurement."
                ),
            ),
        )
    older, newer = runs[-2], runs[-1]
    shared = sorted(set(older.all_metrics()) & set(newer.all_metrics()))
    measured = [
        name for name in shared if older.metric(name) is not None and newer.metric(name) is not None
    ]
    if not measured:
        return (
            DriftSignal(
                kind=DriftKind.QUALITY,
                metric="eval_metrics",
                severity=DriftSeverity.UNAVAILABLE,
                value=None,
                baseline=None,
                current=None,
                samples_baseline=1,
                samples_current=1,
                note="Two evaluation runs exist but share no measured metric.",
            ),
        )
    drops = []
    for name in measured:
        left = older.metric(name)
        right = newer.metric(name)
        if left is None or right is None or left == 0:
            continue
        drops.append((left - right) / abs(left))
    if not drops:
        return (
            DriftSignal(
                kind=DriftKind.QUALITY,
                metric="eval_metrics",
                severity=DriftSeverity.STABLE,
                value=0.0,
                baseline=None,
                current=None,
                samples_baseline=1,
                samples_current=1,
                note=f"Compared {', '.join(measured)} across two eval runs.",
            ),
        )
    worst = max(drops)
    return (
        DriftSignal(
            kind=DriftKind.QUALITY,
            metric="eval_metrics",
            severity=_relative_severity(worst),
            value=worst,
            baseline=None,
            current=None,
            samples_baseline=1,
            samples_current=1,
            note=f"Largest relative drop among {', '.join(measured)}.",
        ),
    )


def _unavailable_backends() -> tuple[DriftSignal, ...]:
    notes = {
        DriftKind.EMBEDDING: (
            "Embedding vectors are not stored on usage records, so embedding "
            "distribution drift cannot be computed."
        ),
        DriftKind.CONTEXT_LENGTH: (
            "The context compiler is on the serving path. Compiler "
            "context-length drift is not computed as a fleet baseline yet; "
            "per-request ContextRecords exist in-process."
        ),
        DriftKind.SAFETY_BLOCKS: (
            "The guardrail engine is not built, so safety-block frequency is unmeasured."
        ),
    }
    return tuple(
        DriftSignal(
            kind=kind,
            metric=kind.value,
            severity=DriftSeverity.UNAVAILABLE,
            value=None,
            baseline=None,
            current=None,
            samples_baseline=0,
            samples_current=0,
            note=note,
        )
        for kind, note in notes.items()
    )


def _categorical(
    kind: DriftKind,
    metric: str,
    baseline: dict[str, int],
    current: dict[str, int],
    base_n: int,
    curr_n: int,
    min_samples: int,
    *,
    missing: str,
) -> DriftSignal:
    if base_n < min_samples or curr_n < min_samples:
        return DriftSignal(
            kind=kind,
            metric=metric,
            severity=DriftSeverity.INSUFFICIENT,
            value=None,
            baseline=None,
            current=None,
            samples_baseline=base_n,
            samples_current=curr_n,
            note=missing
            if base_n == 0 or curr_n == 0
            else (f"Need {min_samples} samples on each side; had {base_n} and {curr_n}."),
        )
    psi = population_stability_index(baseline, current)
    return DriftSignal(
        kind=kind,
        metric=metric,
        severity=_psi_severity(psi),
        value=psi,
        baseline=None,
        current=None,
        samples_baseline=base_n,
        samples_current=curr_n,
    )


def _relative(
    kind: DriftKind,
    metric: str,
    baseline: float | None,
    current: float | None,
    base_n: int,
    curr_n: int,
    min_samples: int,
    *,
    missing: str,
    invert: bool = False,
) -> DriftSignal:
    if base_n < min_samples or curr_n < min_samples:
        return DriftSignal(
            kind=kind,
            metric=metric,
            severity=DriftSeverity.INSUFFICIENT,
            value=None,
            baseline=baseline,
            current=current,
            samples_baseline=base_n,
            samples_current=curr_n,
            note=f"Need {min_samples} samples on each side; had {base_n} and {curr_n}.",
        )
    if baseline is None or current is None:
        return DriftSignal(
            kind=kind,
            metric=metric,
            severity=DriftSeverity.UNAVAILABLE,
            value=None,
            baseline=baseline,
            current=current,
            samples_baseline=base_n,
            samples_current=curr_n,
            note=missing,
        )
    if baseline == 0:
        delta = 0.0 if current == 0 else None
        return DriftSignal(
            kind=kind,
            metric=metric,
            severity=DriftSeverity.UNAVAILABLE if delta is None else DriftSeverity.STABLE,
            value=delta,
            baseline=baseline,
            current=current,
            samples_baseline=base_n,
            samples_current=curr_n,
            note="Baseline was zero; a relative change is undefined." if delta is None else None,
        )
    delta = (current - baseline) / abs(baseline)
    watch = -delta if invert else delta
    return DriftSignal(
        kind=kind,
        metric=metric,
        severity=_relative_severity(watch),
        value=delta,
        baseline=baseline,
        current=current,
        samples_baseline=base_n,
        samples_current=curr_n,
    )


def _rate(
    kind: DriftKind,
    metric: str,
    baseline: float | None,
    current: float | None,
    base_n: int,
    curr_n: int,
    min_samples: int,
    *,
    missing: str,
) -> DriftSignal:
    if base_n < min_samples or curr_n < min_samples:
        return DriftSignal(
            kind=kind,
            metric=metric,
            severity=DriftSeverity.INSUFFICIENT,
            value=None,
            baseline=baseline,
            current=current,
            samples_baseline=base_n,
            samples_current=curr_n,
            note=f"Need {min_samples} samples on each side; had {base_n} and {curr_n}.",
        )
    if baseline is None or current is None:
        return DriftSignal(
            kind=kind,
            metric=metric,
            severity=DriftSeverity.UNAVAILABLE,
            value=None,
            baseline=baseline,
            current=current,
            samples_baseline=base_n,
            samples_current=curr_n,
            note=missing,
        )
    delta = current - baseline
    if delta <= RATE_MODERATE:
        severity = DriftSeverity.STABLE
    elif delta <= RATE_SIGNIFICANT:
        severity = DriftSeverity.MODERATE
    else:
        severity = DriftSeverity.SIGNIFICANT
    return DriftSignal(
        kind=kind,
        metric=metric,
        severity=severity,
        value=delta,
        baseline=baseline,
        current=current,
        samples_baseline=base_n,
        samples_current=curr_n,
    )


def population_stability_index(baseline: dict[str, int], current: dict[str, int]) -> float:
    """PSI with add-one smoothing so empty bins do not explode."""
    keys = sorted(set(baseline) | set(current))
    if not keys:
        return 0.0
    base_total = sum(baseline.values()) + len(keys)
    curr_total = sum(current.values()) + len(keys)
    score = 0.0
    for key in keys:
        left = (baseline.get(key, 0) + 1) / base_total
        right = (current.get(key, 0) + 1) / curr_total
        score += (right - left) * math.log(right / left)
    return score


def _psi_severity(psi: float) -> DriftSeverity:
    if psi < PSI_MODERATE:
        return DriftSeverity.STABLE
    if psi < PSI_SIGNIFICANT:
        return DriftSeverity.MODERATE
    return DriftSeverity.SIGNIFICANT


def _relative_severity(delta: float) -> DriftSeverity:
    if delta <= RELATIVE_MODERATE:
        return DriftSeverity.STABLE
    if delta <= RELATIVE_SIGNIFICANT:
        return DriftSeverity.MODERATE
    return DriftSeverity.SIGNIFICANT


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def usage_from_dicts(rows: Sequence[dict[str, Any]]) -> list[UsageRecord]:
    """Rehydrate usage rows from JSON. Unknown fields are ignored."""
    records: list[UsageRecord] = []
    for index, row in enumerate(rows):
        records.append(
            UsageRecord(
                request_id=str(row.get("request_id") or f"row-{index}"),
                requested_model=str(row.get("requested_model") or "auto"),
                served_model=str(row.get("served_model") or ""),
                provider=str(row.get("provider") or ""),
                policy=str(row.get("policy") or ""),
                prompt_tokens=int(row.get("prompt_tokens") or 0),
                completion_tokens=int(row.get("completion_tokens") or 0),
                cost_usd=float(row.get("cost_usd") or 0.0),
                cost_is_estimated=bool(row.get("cost_is_estimated", False)),
                latency_ms=float(row.get("latency_ms") or 0.0),
                streamed=bool(row.get("streamed", False)),
                failover_count=int(row.get("failover_count") or 0),
                tenant_id=str(row.get("tenant_id") or "public"),
                error=row.get("error"),
                intent_id=row.get("intent_id"),
                intent_layer=row.get("intent_layer"),
                intent_confidence=(
                    float(row["intent_confidence"])
                    if row.get("intent_confidence") is not None
                    else None
                ),
                created_at=float(row.get("created_at") or index),
            )
        )
    return records
