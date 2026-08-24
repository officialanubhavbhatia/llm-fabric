"""Evaluation platform: suites, runs, comparisons, gates, adapters."""

from llm_fabric.eval.schema import (
    METRIC_VERSION,
    EvalComparison,
    EvalGate,
    EvalProvenance,
    EvalResult,
    EvalRun,
    EvalSuite,
    EvaluatorKind,
)

__all__ = [
    "METRIC_VERSION",
    "EvalComparison",
    "EvalGate",
    "EvalProvenance",
    "EvalResult",
    "EvalRun",
    "EvalSuite",
    "EvaluatorKind",
]
