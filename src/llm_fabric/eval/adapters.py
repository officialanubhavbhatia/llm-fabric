"""Optional evaluation-framework adapters.

DeepEval and lm-evaluation-harness are integrations, not the system of record.
If the package is not installed the adapter says so and returns no scores.
A suite must name the metrics or tasks it wants; nothing is applied by default.
These packages are not inference-runtime dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llm_fabric.eval.schema import EvalResult, EvaluatorKind


class DeepEvalAdapter:
    """Runs named DeepEval metrics when the package is installed.

    `exact_match` is mapped onto DeepEval's metric protocol without calling an
    LLM. Metrics this adapter has not mapped stay unavailable rather than guessed.
    """

    kind = EvaluatorKind.DEEPEVAL
    mapped = frozenset({"exact_match"})

    def available(self) -> bool:
        try:
            import deepeval  # noqa: F401
        except ImportError:
            return False
        return True

    def evaluate(
        self,
        *,
        task: str,
        metrics: tuple[str, ...],
        cases: Sequence[dict[str, Any]] | None = None,
    ) -> EvalResult:
        if not metrics:
            return EvalResult(
                task=task,
                evaluator=self.kind,
                metrics={},
                unavailable=(),
                note="No DeepEval metrics were named. None were applied.",
            )
        if not self.available():
            return EvalResult(
                task=task,
                evaluator=self.kind,
                metrics=dict.fromkeys(metrics, None),
                unavailable=metrics,
                note="deepeval is not installed. Scores are not synthesized.",
            )
        unknown = tuple(name for name in metrics if name not in self.mapped)
        mapped = tuple(name for name in metrics if name in self.mapped)
        scores: dict[str, float | None] = dict.fromkeys(metrics, None)
        note_parts: list[str] = []
        if mapped:
            try:
                scores["exact_match"] = _deepeval_exact_match(cases or ())
                note_parts.append(
                    "exact_match ran through DeepEval's LLMTestCase/BaseMetric protocol."
                )
            except Exception as exc:  # noqa: BLE001 — adapter must not crash the suite
                note_parts.append(f"exact_match failed: {exc}")
        if unknown:
            note_parts.append(
                "Unmapped DeepEval metrics stayed unavailable rather than guessed: "
                + ", ".join(unknown)
            )
        return EvalResult(
            task=task,
            evaluator=self.kind,
            metrics=scores,
            unavailable=unknown if scores.get("exact_match") is not None else metrics,
            note=" ".join(note_parts) or None,
        )


class LMEvalHarnessAdapter:
    """lm-evaluation-harness adapter. Runs only the tasks a suite names."""

    kind = EvaluatorKind.LM_EVAL

    def available(self) -> bool:
        try:
            import lm_eval  # noqa: F401
        except ImportError:
            return False
        return True

    def evaluate(
        self,
        *,
        task: str,
        metrics: tuple[str, ...],
        harness_tasks: Sequence[str] = (),
        limit: int = 2,
    ) -> EvalResult:
        if not harness_tasks and not metrics:
            return EvalResult(
                task=task,
                evaluator=self.kind,
                metrics={},
                unavailable=(),
                note="No lm-evaluation-harness tasks were named. None were applied.",
            )
        named = tuple(harness_tasks) or metrics
        if not self.available():
            return EvalResult(
                task=task,
                evaluator=self.kind,
                metrics=dict.fromkeys(named, None),
                unavailable=named,
                note="lm_eval is not installed. Scores are not synthesized.",
            )
        if not harness_tasks:
            return EvalResult(
                task=task,
                evaluator=self.kind,
                metrics=dict.fromkeys(named, None),
                unavailable=named,
                note="lm_eval is installed but no harness_tasks were named.",
            )
        try:
            scores = _lm_eval_dummy(tuple(harness_tasks), limit=limit)
        except Exception as exc:  # noqa: BLE001 — adapter must not crash the suite
            return EvalResult(
                task=task,
                evaluator=self.kind,
                metrics=dict.fromkeys(named, None),
                unavailable=named,
                note=f"lm_eval run failed: {exc}",
            )
        selected = {name: scores.get(name) for name in named}
        missing = tuple(name for name, value in selected.items() if value is None)
        return EvalResult(
            task=task,
            evaluator=self.kind,
            metrics=selected,
            unavailable=missing,
            note=(
                f"lm_eval simple_evaluate ran tasks {list(harness_tasks)} "
                f"with the dummy model, limit={limit}."
            ),
        )


def _deepeval_exact_match(cases: Sequence[dict[str, Any]]) -> float:
    from deepeval.metrics.base_metric import BaseMetric
    from deepeval.test_case import LLMTestCase

    class ExactMatchMetric(BaseMetric):  # type: ignore[misc]
        def __init__(self) -> None:
            self.threshold = 1.0
            self.score = 0.0
            self.success = False
            self.reason = ""
            self.async_mode = False

        def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
            del args, kwargs
            expected = test_case.expected_output or ""
            actual = test_case.actual_output or ""
            self.score = 1.0 if expected == actual else 0.0
            self.success = self.score >= self.threshold
            self.reason = "exact match" if self.success else "mismatch"
            return self.score

        async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
            return self.measure(test_case, *args, **kwargs)

        def is_successful(self) -> bool:
            return bool(self.success)

    labelled = [
        case
        for case in cases
        if case.get("expected") is not None and case.get("actual") is not None
    ]
    if not labelled:
        raise RuntimeError("no labelled DeepEval cases with actual outputs")
    scores: list[float] = []
    for case in labelled:
        metric = ExactMatchMetric()
        test_case = LLMTestCase(
            input=str(case.get("input") or ""),
            actual_output=str(case.get("actual") or ""),
            expected_output=str(case.get("expected") or ""),
        )
        scores.append(float(metric.measure(test_case)))
    return sum(scores) / len(scores)


def _lm_eval_dummy(harness_tasks: tuple[str, ...], *, limit: int) -> dict[str, float]:
    from lm_eval import simple_evaluate

    payload = simple_evaluate(
        model="dummy",
        tasks=list(harness_tasks),
        limit=limit,
        bootstrap_iters=0,
        device="cpu",
    )
    results = payload.get("results") or {}
    scores: dict[str, float] = {}
    for task_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            scores[str(key)] = float(value)
            scores[f"{task_name}/{key}"] = float(value)
            if key == "acc,none" or key.startswith("acc"):
                scores.setdefault("acc", float(value))
                scores.setdefault(task_name, float(value))
    return scores
