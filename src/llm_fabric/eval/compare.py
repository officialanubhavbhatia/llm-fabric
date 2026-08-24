"""Baseline vs candidate, and experiment-to-experiment comparison.

A metric present on only one side is a `None` delta, never a zero. Comparing
against a missing baseline would otherwise look like a huge improvement.
"""

from __future__ import annotations

from llm_fabric.eval.schema import (
    EvalComparison,
    EvalRun,
    GateDirection,
    MetricDelta,
)

_LOWER_IS_BETTER = frozenset(
    {
        "fallback_rate",
        "ece",
        "expected_calibration_error",
        "semantic_false_hit_rate",
        "cost_regret",
        "latency_regret",
        "quality_regret",
        "declared_quality_regret",
        "declared_cost_regret",
        "declared_latency_regret",
        "unnecessary_escalation_rate",
        "underpowered_rate",
        "overpowered_rate",
    }
)


def compare_runs(baseline: EvalRun, candidate: EvalRun) -> EvalComparison:
    names = sorted(set(baseline.all_metrics()) | set(candidate.all_metrics()))
    deltas: list[MetricDelta] = []
    for name in names:
        left = baseline.metric(name)
        right = candidate.metric(name)
        direction = GateDirection.LOWER if name in _LOWER_IS_BETTER else GateDirection.HIGHER
        delta = None if left is None or right is None else right - left
        deltas.append(
            MetricDelta(
                name=name,
                baseline=left,
                candidate=right,
                delta=delta,
                direction=direction,
            )
        )
    note = None
    if baseline.provenance.metric_version != candidate.provenance.metric_version:
        note = "metric_version differs; deltas are reported but the formulas may not be the same"
    if baseline.provenance.dataset_version != candidate.provenance.dataset_version:
        extra = "dataset_version differs; this is not a paired comparison"
        note = f"{note}; {extra}" if note else extra
    return EvalComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        deltas=tuple(deltas),
        tenant_id=candidate.tenant_id,
        note=note,
    )
