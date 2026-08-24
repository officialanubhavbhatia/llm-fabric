"""Deployment gates: reject a candidate that degrades a critical metric.

An unmeasured critical metric fails the gate. Passing by default because a
number is missing is how regressions hide.
"""

from __future__ import annotations

from llm_fabric.eval.schema import (
    EvalComparison,
    EvalGate,
    EvalRun,
    GateDirection,
    GateKind,
    GateVerdict,
)


def apply_gates(
    candidate: EvalRun,
    gates: list[EvalGate],
    *,
    comparison: EvalComparison | None = None,
) -> list[GateVerdict]:
    baseline_by_name = (
        {delta.name: delta.baseline for delta in comparison.deltas} if comparison else {}
    )
    verdicts: list[GateVerdict] = []
    for gate in gates:
        value = candidate.metric(gate.metric)
        baseline = baseline_by_name.get(gate.metric)
        verdicts.append(_one(gate, value, baseline))
    return verdicts


def critical_failures(verdicts: list[GateVerdict]) -> list[GateVerdict]:
    return [verdict for verdict in verdicts if verdict.gate.critical and not verdict.passed]


def _one(gate: EvalGate, value: float | None, baseline: float | None) -> GateVerdict:
    kind = gate.resolved_kind()
    if value is None:
        return GateVerdict(
            gate=gate,
            passed=False,
            value=None,
            baseline=baseline,
            reason=f"{gate.metric} was not measured; an unmeasured critical gate fails",
        )
    if kind is GateKind.ABSOLUTE:
        if gate.minimum is not None and value < gate.minimum:
            return GateVerdict(
                gate=gate,
                passed=False,
                value=value,
                baseline=baseline,
                reason=f"{gate.metric}={value} is below the absolute floor {gate.minimum}",
            )
        if gate.maximum is not None and value > gate.maximum:
            return GateVerdict(
                gate=gate,
                passed=False,
                value=value,
                baseline=baseline,
                reason=f"{gate.metric}={value} is above the absolute ceiling {gate.maximum}",
            )
        if gate.minimum is None and gate.maximum is None:
            return GateVerdict(
                gate=gate,
                passed=False,
                value=value,
                baseline=baseline,
                reason=f"{gate.metric} absolute gate has no floor or ceiling",
            )
        return GateVerdict(
            gate=gate,
            passed=True,
            value=value,
            baseline=baseline,
            reason="ok",
        )
    if kind is GateKind.REGRESSION:
        if gate.max_degradation is None:
            return GateVerdict(
                gate=gate,
                passed=False,
                value=value,
                baseline=baseline,
                reason=f"{gate.metric} regression gate has no max_degradation",
            )
        if baseline is None:
            return GateVerdict(
                gate=gate,
                passed=False,
                value=value,
                baseline=None,
                reason=(
                    f"{gate.metric} has a degradation ceiling but no baseline; "
                    "refusing to pass an unpaired comparison"
                ),
            )
        drop = baseline - value if gate.direction is GateDirection.HIGHER else value - baseline
        if drop > gate.max_degradation:
            return GateVerdict(
                gate=gate,
                passed=False,
                value=value,
                baseline=baseline,
                reason=(
                    f"{gate.metric} degraded by {drop} from baseline {baseline} "
                    f"(allowed {gate.max_degradation})"
                ),
            )
        return GateVerdict(
            gate=gate,
            passed=True,
            value=value,
            baseline=baseline,
            reason="ok",
        )
    return GateVerdict(
        gate=gate,
        passed=False,
        value=value,
        baseline=baseline,
        reason=f"{gate.metric} has an unknown gate kind",
    )
