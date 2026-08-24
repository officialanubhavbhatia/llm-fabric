"""Classification evaluator.

Wraps the existing intent benchmark. Metrics are taken from a real
`BenchmarkReport`; nothing here invents a score the benchmark did not compute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from llm_fabric.eval.schema import EvalResult, EvaluatorKind, ExampleResult
from llm_fabric.intent.benchmark import load_dataset, run_benchmark
from llm_fabric.intent.cascade import IntentCascade
from llm_fabric.tenancy.scope import TenantScope

#: Metrics this evaluator can emit. A suite that asks for others gets them as
#: unavailable rather than a silent zero.
SUPPORTED = frozenset(
    {
        "accuracy",
        "lenient_accuracy",
        "macro_f1",
        "micro_f1",
        "expected_calibration_error",
        "unknown_intent_recall",
        "abstention_accuracy",
        "semantic_false_hit_rate",
        "classifier_latency_ms",
        "classification_cost_usd",
        "high_confidence_precision",
        "brier_score",
    }
)


async def evaluate(
    *,
    task: str,
    dataset_path: Path,
    cascade: IntentCascade,
    metrics: tuple[str, ...],
    mode: Literal["classifier", "cache"] = "classifier",
    scope: TenantScope | None = None,
) -> EvalResult:
    cases = load_dataset(dataset_path)
    report = await run_benchmark(cascade, cases, scope=scope, mode=mode)
    data = report.as_dict()
    available: dict[str, float | None] = {
        "accuracy": data.get("accuracy"),
        "lenient_accuracy": data.get("lenient_accuracy"),
        "macro_f1": data.get("macro_f1"),
        "micro_f1": data.get("micro_f1"),
        "expected_calibration_error": data.get("expected_calibration_error"),
        "unknown_intent_recall": _nested(data, "abstention", "unknown_intent_recall"),
        "abstention_accuracy": _nested(data, "abstention", "accuracy"),
        "semantic_false_hit_rate": _nested(data, "cache", "semantic_false_hit_rate"),
        "classifier_latency_ms": _nested(data, "latency_ms", "mean"),
        "classification_cost_usd": _nested(data, "cost", "total_usd"),
        "high_confidence_precision": _high_conf(data, 0.90),
        "brier_score": data.get("brier_score"),
    }
    selected = {name: available.get(name) for name in metrics if name in SUPPORTED}
    unknown = tuple(name for name in metrics if name not in SUPPORTED)
    rows = tuple(
        ExampleResult(
            example_id=str(failure["id"]),
            scores={"accuracy": 0.0},
            output=str(failure.get("predicted")),
            notes=f"expected {failure.get('expected')}",
        )
        for failure in report.failures()[:50]
    )
    return EvalResult(
        task=task,
        evaluator=EvaluatorKind.CLASSIFICATION,
        metrics=selected,
        examples=rows,
        unavailable=unknown,
        note=(
            "Scores come from llm-fabric-bench against the labelled dataset. "
            "They are a regression tripwire, not a claim about production traffic."
        ),
    )


def _high_conf(data: dict[str, Any], threshold: float) -> float | None:
    rows = data.get("high_confidence_routing") or []
    for row in rows:
        if isinstance(row, dict) and row.get("threshold") == threshold:
            value = row.get("precision")
            return value if isinstance(value, int | float) else None
    return None


def _nested(data: dict[str, Any], *keys: str) -> float | None:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, int | float) else None
