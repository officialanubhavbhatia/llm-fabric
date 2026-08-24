"""Run an evaluation suite and persist the run with full provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from llm_fabric.eval.adapters import DeepEvalAdapter, LMEvalHarnessAdapter
from llm_fabric.eval.classification import evaluate as evaluate_classification
from llm_fabric.eval.deterministic import evaluate as evaluate_deterministic
from llm_fabric.eval.deterministic import identity_outputs
from llm_fabric.eval.judge import LLMJudge
from llm_fabric.eval.judge import evaluate as evaluate_judge
from llm_fabric.eval.provenance import build_provenance
from llm_fabric.eval.routing import evaluate as evaluate_routing
from llm_fabric.eval.routing import load_cases as load_routing_cases
from llm_fabric.eval.schema import (
    EvalResult,
    EvalRun,
    EvalSuite,
    EvalTask,
    EvaluatorKind,
)
from llm_fabric.eval.store import EvalRunRepository
from llm_fabric.intent.cascade import IntentCascade
from llm_fabric.router.engine import Router
from llm_fabric.storage.records import EvalExample
from llm_fabric.tenancy.scope import TenantScope


def load_suite(path: Path) -> EvalSuite:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tasks = tuple(
        EvalTask(
            name=str(item["name"]),
            evaluator=EvaluatorKind(item["evaluator"]),
            metrics=tuple(item.get("metrics") or ()),
            dataset_path=item.get("dataset_path"),
            dataset_id=item.get("dataset_id"),
            options=dict(item.get("options") or {}),
        )
        for item in raw.get("tasks") or ()
    )
    return EvalSuite(
        name=str(raw.get("name") or path.stem),
        description=str(raw.get("description") or ""),
        tenant_id=str(raw.get("tenant_id") or "public"),
        tasks=tasks,
    )


def load_examples(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        row = json.loads(text)
        examples.append(
            EvalExample(
                input=str(row.get("input") or row.get("text") or ""),
                expected=row.get("expected"),
                metadata={k: v for k, v in row.items() if k not in {"input", "text", "expected"}},
            )
        )
    return examples


async def run_suite(
    suite: EvalSuite,
    *,
    scope: TenantScope,
    router: Router | None = None,
    cascade: IntentCascade | None = None,
    judge: LLMJudge | None = None,
    store: EvalRunRepository | None = None,
    repo: Path | None = None,
) -> EvalRun:
    results: list[EvalResult] = []
    example_payloads: list[dict[str, Any]] = []
    model: str | None = None
    taxonomy_version: str | None = None

    for task in suite.tasks:
        result = await _run_task(task, router=router, cascade=cascade, judge=judge)
        results.append(result)
        if task.dataset_path:
            example_payloads.append({"path": task.dataset_path, "metrics": list(task.metrics)})
        if task.evaluator is EvaluatorKind.JUDGE and judge is not None:
            model = judge.model
        if task.evaluator is EvaluatorKind.CLASSIFICATION and cascade is not None:
            taxonomy_version = cascade.taxonomy.version

    run = EvalRun(
        tenant_id=scope.tenant_id,
        suite_name=suite.name,
        provenance=build_provenance(
            examples=example_payloads,
            configuration={
                "suite": suite.name,
                "tasks": [task.name for task in suite.tasks],
            },
            model=model,
            taxonomy_version=taxonomy_version,
            repo=repo,
        ),
        results=tuple(results),
    )
    if store is not None:
        store.put(scope, run)
    return run


async def _run_task(
    task: EvalTask,
    *,
    router: Router | None,
    cascade: IntentCascade | None,
    judge: LLMJudge | None,
) -> EvalResult:
    path = Path(task.dataset_path) if task.dataset_path else None
    if task.evaluator is EvaluatorKind.DETERMINISTIC:
        if path is None:
            return _missing_dataset(task)
        examples = load_examples(path)
        return evaluate_deterministic(
            task=task.name,
            examples=examples,
            outputs=identity_outputs(examples),
            metrics=task.metrics,
        )
    if task.evaluator is EvaluatorKind.CLASSIFICATION:
        if path is None:
            return _missing_dataset(task)
        if cascade is None:
            return EvalResult(
                task=task.name,
                evaluator=task.evaluator,
                metrics=dict.fromkeys(task.metrics, None),
                unavailable=task.metrics,
                note="No intent cascade is configured.",
            )
        return await evaluate_classification(
            task=task.name,
            dataset_path=path,
            cascade=cascade,
            metrics=task.metrics,
            mode="cache" if task.options.get("mode") == "cache" else "classifier",
        )
    if task.evaluator is EvaluatorKind.ROUTING:
        if path is None or router is None:
            return EvalResult(
                task=task.name,
                evaluator=task.evaluator,
                metrics=dict.fromkeys(task.metrics, None),
                unavailable=task.metrics,
                note="Routing eval needs a dataset and a router.",
            )
        return evaluate_routing(
            task=task.name,
            cases=load_routing_cases(path),
            router=router,
            metrics=task.metrics,
        )
    if task.evaluator is EvaluatorKind.JUDGE:
        if path is None:
            return _missing_dataset(task)
        examples = load_examples(path)
        return await evaluate_judge(
            task=task.name,
            examples=examples,
            outputs=identity_outputs(examples),
            metrics=task.metrics,
            judge=judge,
        )
    if task.evaluator is EvaluatorKind.DEEPEVAL:
        examples = load_examples(path) if path else []
        outputs = identity_outputs(examples)
        cases = []
        for index, example in enumerate(examples):
            example_id = str(example.metadata.get("id") or index)
            cases.append(
                {
                    "id": example_id,
                    "input": example.input,
                    "expected": example.expected,
                    "actual": outputs.get(example_id),
                }
            )
        return DeepEvalAdapter().evaluate(task=task.name, metrics=task.metrics, cases=cases)
    if task.evaluator is EvaluatorKind.LM_EVAL:
        return LMEvalHarnessAdapter().evaluate(
            task=task.name,
            metrics=task.metrics,
            harness_tasks=tuple(task.options.get("harness_tasks") or ()),
            limit=int(task.options.get("limit") or 2),
        )
    return EvalResult(
        task=task.name,
        evaluator=task.evaluator,
        metrics=dict.fromkeys(task.metrics, None),
        unavailable=task.metrics,
        note=f"evaluator '{task.evaluator}' is not implemented",
    )


def _missing_dataset(task: EvalTask) -> EvalResult:
    return EvalResult(
        task=task.name,
        evaluator=task.evaluator,
        metrics=dict.fromkeys(task.metrics, None),
        unavailable=task.metrics,
        note="No dataset_path was given.",
    )
