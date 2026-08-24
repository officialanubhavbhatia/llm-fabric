"""Deterministic graders. No model, no network, same input same score.

Only the metrics a task names are computed. An example without an expected
label contributes `None` for that metric, not a zero that would look like a
failure.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from llm_fabric.eval.schema import EvalResult, EvaluatorKind, ExampleResult
from llm_fabric.storage.records import EvalExample

SUPPORTED = frozenset({"exact_match", "casefold_match", "contains", "json_valid", "regex"})


def score_example(
    example: EvalExample,
    output: str,
    metrics: tuple[str, ...],
) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for name in metrics:
        if name not in SUPPORTED:
            scores[name] = None
            continue
        scores[name] = _score(name, example, output)
    return scores


def evaluate(
    *,
    task: str,
    examples: list[EvalExample],
    outputs: Mapping[str, str],
    metrics: tuple[str, ...],
) -> EvalResult:
    rows: list[ExampleResult] = []
    requested = tuple(name for name in metrics if name in SUPPORTED)
    unknown = tuple(name for name in metrics if name not in SUPPORTED)
    for index, example in enumerate(examples):
        example_id = str(example.metadata.get("id") or index)
        output = outputs.get(example_id)
        if output is None:
            rows.append(
                ExampleResult(
                    example_id=example_id,
                    scores=dict.fromkeys(requested, None),
                    error="no output produced for this example",
                )
            )
            continue
        rows.append(
            ExampleResult(
                example_id=example_id,
                scores=score_example(example, output, requested),
                output=output,
            )
        )
    aggregates = _mean(rows, requested)
    return EvalResult(
        task=task,
        evaluator=EvaluatorKind.DETERMINISTIC,
        metrics=aggregates,
        examples=tuple(rows),
        unavailable=unknown,
        note=(
            "Metrics are means over examples that had a label and an output. "
            "Unlabelled examples do not pull the mean toward zero."
            + (f" Unknown metrics were skipped: {', '.join(unknown)}." if unknown else "")
        ),
    )


def _score(name: str, example: EvalExample, output: str) -> float | None:
    if name == "json_valid":
        candidate = output.strip()
        if not candidate.startswith(("{", "[")):
            return None
        try:
            json.loads(output)
        except json.JSONDecodeError:
            return 0.0
        return 1.0
    if example.expected is None:
        return None
    expected = example.expected
    if name == "exact_match":
        return 1.0 if output == expected else 0.0
    if name == "casefold_match":
        return 1.0 if output.casefold() == expected.casefold() else 0.0
    if name == "contains":
        return 1.0 if expected in output else 0.0
    if name == "regex":
        pattern = str(example.metadata.get("regex") or expected)
        try:
            return 1.0 if re.search(pattern, output) else 0.0
        except re.error:
            return None
    return None


def _mean(rows: list[ExampleResult], metrics: tuple[str, ...]) -> dict[str, float | None]:
    aggregates: dict[str, float | None] = {}
    for name in metrics:
        values: list[float] = []
        for row in rows:
            value = row.scores.get(name)
            if value is not None and row.error is None:
                values.append(value)
        aggregates[name] = sum(values) / len(values) if values else None
    return aggregates


def identity_outputs(examples: list[EvalExample]) -> dict[str, str]:
    """Treat `metadata.output` as the system output when the suite is offline.

    Used by CI suites that grade a recorded output, not a live model.
    """
    outputs: dict[str, str] = {}
    for index, example in enumerate(examples):
        example_id = str(example.metadata.get("id") or index)
        recorded = example.metadata.get("output")
        if isinstance(recorded, str):
            outputs[example_id] = recorded
    return outputs
