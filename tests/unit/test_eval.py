"""Evaluation platform: scores that exist, adapters that stay silent, gates that fail closed."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_fabric.eval.adapters import DeepEvalAdapter, LMEvalHarnessAdapter
from llm_fabric.eval.compare import compare_runs
from llm_fabric.eval.deterministic import evaluate as evaluate_deterministic
from llm_fabric.eval.deterministic import identity_outputs
from llm_fabric.eval.gates import apply_gates, critical_failures
from llm_fabric.eval.judge import _parse
from llm_fabric.eval.provenance import build_provenance
from llm_fabric.eval.runner import load_examples, load_suite, run_suite
from llm_fabric.eval.sampling import dataset_from_traces, sample_usage
from llm_fabric.eval.schema import (
    METRIC_VERSION,
    EvalGate,
    EvalProvenance,
    EvalResult,
    EvalRun,
    EvaluatorKind,
    GateKind,
)
from llm_fabric.gateway.app import create_app
from llm_fabric.intent.bootstrap import bootstrap_taxonomy
from llm_fabric.intent.factory import build_offline_cascade
from llm_fabric.observability.metering import InMemoryMeter, UsageRecord
from llm_fabric.storage.records import EvalExample
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope

REPO = Path(__file__).resolve().parents[2]


def _run(*, metrics: dict[str, float | None], run_id: str = "r") -> EvalRun:
    return EvalRun(
        tenant_id="public",
        suite_name="t",
        provenance=EvalProvenance(dataset_version="d", metric_version=METRIC_VERSION),
        results=(
            EvalResult(
                task="t",
                evaluator=EvaluatorKind.DETERMINISTIC,
                metrics=metrics,
            ),
        ),
        run_id=run_id,
    )


def test_deterministic_skips_unlabelled_and_inapplicable_json() -> None:
    examples = [
        EvalExample(input="a", expected="yes", metadata={"id": "1", "output": "yes"}),
        EvalExample(input="b", expected=None, metadata={"id": "2", "output": "no"}),
        EvalExample(input="c", expected="not-json", metadata={"id": "3", "output": "not-json"}),
    ]
    result = evaluate_deterministic(
        task="det",
        examples=examples,
        outputs=identity_outputs(examples),
        metrics=("exact_match", "json_valid"),
    )
    assert result.metrics["exact_match"] == 1.0
    assert result.metrics["json_valid"] is None


def test_compare_does_not_invent_a_delta_for_a_missing_side() -> None:
    comparison = compare_runs(
        _run(metrics={"exact_match": 1.0, "mystery": None}, run_id="b"),
        _run(metrics={"exact_match": 0.5}, run_id="c"),
    )
    by_name = {delta.name: delta for delta in comparison.deltas}
    assert by_name["exact_match"].delta == pytest.approx(-0.5)
    assert by_name["mystery"].delta is None


def test_critical_gate_fails_when_unmeasured() -> None:
    verdicts = apply_gates(
        _run(metrics={"exact_match": None}),
        [EvalGate(metric="exact_match", minimum=1.0, critical=True)],
    )
    assert critical_failures(verdicts)
    assert "not measured" in verdicts[0].reason


def test_degradation_gate_fails_on_a_material_drop() -> None:
    baseline = _run(metrics={"accuracy": 0.80}, run_id="b")
    candidate = _run(metrics={"accuracy": 0.70}, run_id="c")
    comparison = compare_runs(baseline, candidate)
    verdicts = apply_gates(
        candidate,
        [
            EvalGate(
                metric="accuracy",
                kind=GateKind.REGRESSION,
                max_degradation=0.05,
                critical=True,
            )
        ],
        comparison=comparison,
    )
    assert critical_failures(verdicts)


def test_absolute_and_regression_gates_are_distinct() -> None:
    baseline = _run(metrics={"accuracy": 0.66}, run_id="b")
    candidate = _run(metrics={"accuracy": 0.66}, run_id="c")
    comparison = compare_runs(baseline, candidate)
    verdicts = apply_gates(
        candidate,
        [
            EvalGate(
                metric="accuracy",
                kind=GateKind.REGRESSION,
                max_degradation=0.05,
                critical=True,
            ),
            EvalGate(
                metric="accuracy",
                kind=GateKind.ABSOLUTE,
                minimum=0.90,
                critical=True,
            ),
        ],
        comparison=comparison,
    )
    by_reason = [verdict.passed for verdict in verdicts]
    assert by_reason == [True, False]
    assert verdicts[0].gate.resolved_kind() is GateKind.REGRESSION
    assert verdicts[1].gate.resolved_kind() is GateKind.ABSOLUTE


def test_adapters_are_unavailable_without_their_packages() -> None:
    deep = DeepEvalAdapter().evaluate(task="d", metrics=("faithfulness",))
    harness = LMEvalHarnessAdapter().evaluate(task="h", metrics=("acc",), harness_tasks=("arc",))
    assert all(value is None for value in deep.metrics.values())
    assert "not synthesized" in (deep.note or "")
    assert all(value is None for value in harness.metrics.values())
    empty = DeepEvalAdapter().evaluate(task="d", metrics=())
    assert empty.metrics == {}


def test_judge_parse_rejects_out_of_range() -> None:
    assert _parse('{"score": 1.4}').get("score") is None
    assert _parse('{"score": 0.5, "rationale": "ok"}')["score"] == 0.5


def test_production_sample_does_not_invent_labels() -> None:
    meter = InMemoryMeter()
    meter.record(
        UsageRecord(
            request_id="r1",
            requested_model="auto",
            served_model="mock-small",
            provider="mock",
            policy="cost_first",
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=0.0,
            cost_is_estimated=True,
            latency_ms=1.0,
            streamed=False,
            failover_count=0,
        )
    )
    scope = TenantScope(tenant_id="acme", user_id="eval")
    dataset = sample_usage(meter.recent(limit=10), scope=scope, rate=1.0, limit=10)
    assert dataset.examples[0].expected is None
    traces = dataset_from_traces(
        [{"trace_id": "t1", "spans": [{"name": "request"}]}],
        scope=scope,
    )
    assert traces.examples[0].expected is None


@pytest.mark.asyncio
async def test_ci_suite_routing_and_deterministic_are_perfect() -> None:
    app = create_app()
    suite = load_suite(REPO / "datasets/eval/ci-suite.yaml")
    cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())
    run = await run_suite(
        suite,
        scope=TenantScope(tenant_id="public", user_id="eval"),
        router=app.state.router,
        cascade=cascade,
        repo=REPO,
    )
    assert run.metric("exact_match") == 1.0
    assert run.metric("route_match") == 1.0
    assert run.provenance.commit is None or len(run.provenance.commit) == 40
    assert run.provenance.metric_version == METRIC_VERSION
    assert run.provenance.dataset_version
    assert run.provenance.taxonomy_version
    deepeval = next(result for result in run.results if result.evaluator is EvaluatorKind.DEEPEVAL)
    assert deepeval.metrics == {}


def test_load_examples_keeps_recorded_output() -> None:
    examples = load_examples(REPO / "datasets/eval/deterministic.jsonl")
    assert identity_outputs(examples)["det-001"] == "pong"


def test_provenance_records_missing_commit_as_missing(tmp_path: Path) -> None:
    provenance = build_provenance(examples=[{"a": 1}], configuration={}, repo=tmp_path)
    assert provenance.commit is None


def test_eval_cli_imports_without_a_circular_import() -> None:
    from llm_fabric.eval.cli import main

    assert callable(main)
