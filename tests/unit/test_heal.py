"""Health scores, drift, and remediations that refuse forbidden mutations."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from llm_fabric.errors import ForbiddenRemediationError
from llm_fabric.eval.gates import apply_gates, critical_failures
from llm_fabric.eval.schema import EvalGate, EvalProvenance, EvalResult, EvalRun, EvaluatorKind
from llm_fabric.gateway.app import create_app
from llm_fabric.heal.controls import OperationalControls
from llm_fabric.heal.drift import analyze, capture_baseline, population_stability_index
from llm_fabric.heal.engine import HealController
from llm_fabric.heal.policies import is_forbidden, propose
from llm_fabric.heal.schema import (
    DriftKind,
    DriftSeverity,
    LearningJob,
    LearningJobStatus,
    RemediationClass,
    RemediationKind,
    RemediationProposal,
)
from llm_fabric.heal.scoring import model_health, provider_health
from llm_fabric.observability.metering import UsageRecord
from llm_fabric.router.health import BreakerState, HealthTracker
from llm_fabric.router.plan import ExclusionRule, RoutePlanner, RouteRequest
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.storage.records import PromptDefinition, PromptStatus
from llm_fabric.storage.repositories import TenantStores
from llm_fabric.tenancy.cache import CacheNamespace, TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope


def _registry() -> ModelRegistry:
    return ModelRegistry.from_mapping(
        {
            "models": [
                {"id": "cheap", "provider": "mock", "input_price_per_mtok": 0.1},
                {"id": "dear", "provider": "mock", "input_price_per_mtok": 1.0},
                {"id": "other", "provider": "elsewhere", "input_price_per_mtok": 0.5},
            ]
        }
    )


def _usage(**overrides: object) -> UsageRecord:
    base: dict[str, object] = {
        "request_id": "r1",
        "requested_model": "auto",
        "served_model": "cheap",
        "provider": "mock",
        "policy": "cost_first",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost_usd": 0.001,
        "cost_is_estimated": False,
        "latency_ms": 10.0,
        "streamed": False,
        "failover_count": 0,
        "tenant_id": "acme",
    }
    base.update(overrides)
    return UsageRecord(**base)  # type: ignore[arg-type]


def _eval_run(*, run_id: str, accuracy: float) -> EvalRun:
    return EvalRun(
        tenant_id="acme",
        suite_name="ci",
        provenance=EvalProvenance(dataset_version="d"),
        results=(
            EvalResult(
                task="t",
                evaluator=EvaluatorKind.DETERMINISTIC,
                metrics={"accuracy": accuracy},
            ),
        ),
        run_id=run_id,
    )


def test_untried_models_have_no_health_score() -> None:
    health = HealthTracker()
    registry = _registry()
    models = {row.id: row for row in model_health(health, registry)}
    assert models["cheap"].health_score is None
    assert models["cheap"].samples == 0
    providers = {row.id: row for row in provider_health(health, registry)}
    assert providers["mock"].health_score is None


def test_provider_score_is_sample_weighted() -> None:
    health = HealthTracker()
    for _ in range(9):
        health.record_success("cheap", latency_ms=10)
    health.record_failure("dear", latency_ms=10, error="boom")
    providers = {row.id: row for row in provider_health(health, _registry())}
    mock = providers["mock"]
    assert mock.samples == 10
    assert mock.health_score is not None
    assert mock.error_rate is not None
    # Nine successes and one failure: not a 50/50 average of the two deployments.
    assert mock.error_rate < 0.2


def test_insufficient_samples_are_not_scored_as_zero_drift() -> None:
    records = [_usage(request_id=f"r{i}", created_at=float(i)) for i in range(10)]
    report = analyze(
        records,
        tenant_id="acme",
        health=HealthTracker(),
        registry=_registry(),
        min_samples=20,
    )
    error = next(s for s in report.signals if s.metric == "error_rate")
    assert error.severity is DriftSeverity.INSUFFICIENT
    assert error.value is None


def test_psi_detects_a_route_shift() -> None:
    baseline = [
        _usage(request_id=f"b{i}", served_model="cheap", created_at=float(i)) for i in range(30)
    ]
    current = [
        _usage(request_id=f"c{i}", served_model="dear", created_at=100 + i) for i in range(30)
    ]
    report = analyze(
        baseline + current,
        tenant_id="acme",
        health=HealthTracker(),
        registry=_registry(),
        min_samples=20,
    )
    route = next(s for s in report.signals if s.metric == "route_distribution")
    assert route.severity is DriftSeverity.SIGNIFICANT
    assert route.value is not None and route.value > 0.25


def test_cost_drift_ignores_estimated_only_records() -> None:
    rows = [
        _usage(request_id=f"e{i}", cost_is_estimated=True, cost_usd=9.0, created_at=float(i))
        for i in range(50)
    ]
    report = analyze(
        rows,
        tenant_id="acme",
        health=HealthTracker(),
        registry=_registry(),
        min_samples=20,
    )
    cost = next(s for s in report.signals if s.metric == "mean_cost_usd")
    assert cost.value is None
    assert cost.severity in {DriftSeverity.UNAVAILABLE, DriftSeverity.INSUFFICIENT}


def test_quality_and_embedding_and_safety_stay_unavailable_without_backends() -> None:
    report = analyze(
        [_usage()],
        tenant_id="acme",
        health=HealthTracker(),
        registry=_registry(),
        min_samples=20,
    )
    by_kind = {s.kind: s for s in report.signals}
    assert by_kind[DriftKind.QUALITY].severity is DriftSeverity.UNAVAILABLE
    assert by_kind[DriftKind.EMBEDDING].severity is DriftSeverity.UNAVAILABLE
    assert by_kind[DriftKind.CONTEXT_LENGTH].severity is DriftSeverity.UNAVAILABLE
    assert by_kind[DriftKind.SAFETY_BLOCKS].severity is DriftSeverity.UNAVAILABLE


def test_quality_drift_uses_eval_runs_not_declared_scores() -> None:
    report = analyze(
        [_usage()],
        tenant_id="acme",
        health=HealthTracker(),
        registry=_registry(),
        eval_runs=(_eval_run(run_id="a", accuracy=0.9), _eval_run(run_id="b", accuracy=0.4)),
        min_samples=20,
    )
    quality = next(s for s in report.signals if s.kind is DriftKind.QUALITY)
    assert quality.severity is DriftSeverity.SIGNIFICANT


def test_authorization_and_safety_mutations_are_forbidden() -> None:
    assert is_forbidden("mutate_authorization", "auth_mode")
    assert is_forbidden("open_circuit_breaker", "safety")
    assert is_forbidden("shift_traffic", "guardrails")
    assert not is_forbidden("open_circuit_breaker", "cheap")


def test_applicator_refuses_forbidden_targets() -> None:
    stores = TenantStores()
    controller = HealController(
        controls=OperationalControls(),
        health=HealthTracker(),
        registry=_registry(),
        incidents=stores.incidents,
        remediations=stores.remediations,
    )
    with pytest.raises(ForbiddenRemediationError):
        controller.applicator.apply(
            RemediationProposal(
                kind=RemediationKind.OPEN_CIRCUIT,
                target="authorization",
                reason="restore availability",
                classification=RemediationClass.FORBIDDEN,
            ),
            TenantScope(tenant_id="acme", user_id="ops"),
        )


def test_learning_job_is_not_promoted_without_a_passing_eval() -> None:
    stores = TenantStores()
    controller = HealController(
        controls=OperationalControls(),
        health=HealthTracker(),
        registry=_registry(),
        jobs=stores.learning_jobs,
    )
    scope = TenantScope(tenant_id="acme", user_id="ops")
    record = controller.applicator.apply(
        RemediationProposal(
            kind=RemediationKind.LEARNING_JOB,
            target="classifier",
            reason="intent drift",
            classification=RemediationClass.LEARNING,
        ),
        scope,
    )
    assert record.applied
    jobs = stores.learning_jobs.list(scope)
    assert jobs[0].status is LearningJobStatus.PROPOSED
    rejected = controller.applicator.promote_job(
        scope,
        jobs[0],
        eval_run=_eval_run(run_id="bad", accuracy=0.1),
        gates=[EvalGate(metric="accuracy", minimum=0.6, critical=True)],
    )
    assert rejected.status is LearningJobStatus.REJECTED
    passed = controller.applicator.promote_job(
        scope,
        LearningJob(tenant_id="acme", reason="retry"),
        eval_run=_eval_run(run_id="ok", accuracy=0.8),
        gates=[EvalGate(metric="accuracy", minimum=0.6, critical=True)],
    )
    assert passed.status is LearningJobStatus.PROMOTED


def test_held_open_breaker_does_not_half_open_after_cooldown() -> None:
    health = HealthTracker()
    health.force_open("cheap", reason="remediation", hold=True)
    assert health.snapshot("cheap").state is BreakerState.OPEN
    assert not health.admits("cheap")
    health.force_close("cheap")
    assert health.admits("cheap")


def test_traffic_shift_excludes_a_model_from_the_planner() -> None:
    controls = OperationalControls()
    controls.traffic.exclude("cheap")
    planner = RoutePlanner(_registry(), traffic=controls.traffic)
    plan = planner.plan(RouteRequest(requested_model="cheap", tenant_id="acme"))
    assert any(row.rule is ExclusionRule.TRAFFIC_SHIFTED for row in plan.excluded)
    assert plan.selected is None or plan.selected.id != "cheap"


def test_model_rollback_restores_a_remembered_spec() -> None:
    registry = _registry()
    controls = OperationalControls()
    original = registry.get("cheap")
    controls.models.snapshot(original)
    registry.replace(replace(original, enabled=False))
    assert not registry.get("cheap").enabled
    controller = HealController(controls=controls, health=HealthTracker(), registry=registry)
    record = controller.applicator.apply(
        RemediationProposal(
            kind=RemediationKind.ROLLBACK_MODEL,
            target="cheap",
            reason="bad enablement",
            classification=RemediationClass.OPERATIONAL,
        ),
        TenantScope(tenant_id="acme", user_id="ops"),
    )
    assert record.applied
    assert registry.get("cheap").enabled


def test_prompt_rollback_needs_a_prior_published_version() -> None:
    stores = TenantStores()
    scope = TenantScope(tenant_id="acme", user_id="ops")
    stores.prompts.publish(
        scope,
        PromptDefinition(
            tenant_id="acme",
            prompt_id="greet",
            version=1,
            owner="alice",
            purpose="greet",
            template="hello",
            status=PromptStatus.PRODUCTION,
        ),
    )
    stores.prompts.publish(
        scope,
        PromptDefinition(
            tenant_id="acme",
            prompt_id="greet",
            version=2,
            owner="alice",
            purpose="greet",
            template="HELLO",
            status=PromptStatus.PRODUCTION,
        ),
    )
    controller = HealController(
        controls=OperationalControls(),
        health=HealthTracker(),
        registry=_registry(),
        prompts=stores.prompts,
    )
    record = controller.applicator.apply(
        RemediationProposal(
            kind=RemediationKind.ROLLBACK_PROMPT,
            target="greet",
            reason="bad prompt",
            classification=RemediationClass.OPERATIONAL,
        ),
        scope,
    )
    assert record.applied
    assert stores.prompts.require(scope, "greet", 1).status is PromptStatus.PRODUCTION
    assert stores.prompts.require(scope, "greet", 2).status is PromptStatus.RETIRED


def test_classifier_rollback_refuses_without_a_prior_pin() -> None:
    controller = HealController(
        controls=OperationalControls(),
        health=HealthTracker(),
        registry=_registry(),
    )
    record = controller.applicator.apply(
        RemediationProposal(
            kind=RemediationKind.ROLLBACK_CLASSIFIER,
            target="classifier",
            reason="bad cascade",
            classification=RemediationClass.OPERATIONAL,
        ),
        TenantScope(tenant_id="acme", user_id="ops"),
    )
    assert not record.applied
    assert "no prior classifier pin" in (record.note or "")


def test_invalidate_cache_and_context_ceiling_are_real() -> None:
    cache = TenantScopedCache()
    scope = TenantScope(tenant_id="acme", user_id="ops")
    cache.put(scope, CacheNamespace.INTENT, {"k": "v"}, "cached")
    controls = OperationalControls()
    controller = HealController(
        controls=controls,
        health=HealthTracker(),
        registry=_registry(),
        cache=cache,
    )
    wiped = controller.applicator.apply(
        RemediationProposal(
            kind=RemediationKind.INVALIDATE_CACHE,
            target="intent",
            reason="stale",
            classification=RemediationClass.OPERATIONAL,
        ),
        scope,
    )
    assert wiped.applied
    assert cache.get(scope, CacheNamespace.INTENT, {"k": "v"}) is None
    controller.applicator.apply(
        RemediationProposal(
            kind=RemediationKind.REDUCE_CONTEXT,
            target="context",
            reason="pressure",
            classification=RemediationClass.OPERATIONAL,
            parameters={"tokens": 8},
        ),
        scope,
    )
    assert controls.context_ceiling_tokens == 8
    app = create_app(registry=_registry())
    app.state.controls.set_context_ceiling(8)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "cheap",
                "messages": [{"role": "user", "content": "word " * 40}],
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "context_too_large"


def test_tick_raises_an_incident_on_significant_error_drift() -> None:
    stores = TenantStores()
    baseline = [_usage(request_id=f"b{i}", error=None, created_at=float(i)) for i in range(25)]
    current = [_usage(request_id=f"c{i}", error="boom", created_at=100 + i) for i in range(25)]
    controller = HealController(
        controls=OperationalControls(),
        health=HealthTracker(),
        registry=_registry(),
        incidents=stores.incidents,
        remediations=stores.remediations,
    )
    scope = TenantScope(tenant_id="acme", user_id="ops")
    report, applied = controller.tick(baseline + current, scope, min_samples=20)
    assert any(
        s.kind is DriftKind.ERROR and s.severity is DriftSeverity.SIGNIFICANT
        for s in report.signals
    )
    assert any(row.kind is RemediationKind.RAISE_INCIDENT for row in applied)
    assert stores.incidents.list(scope)


def test_unmeasured_critical_gate_still_fails() -> None:
    run = _eval_run(run_id="x", accuracy=0.9)
    run = replace(
        run,
        results=(
            EvalResult(
                task="t",
                evaluator=EvaluatorKind.DETERMINISTIC,
                metrics={"accuracy": None},
            ),
        ),
    )
    verdicts = apply_gates(run, [EvalGate(metric="accuracy", minimum=0.5, critical=True)])
    assert critical_failures(verdicts)


def test_population_stability_index_is_zero_for_identical() -> None:
    counts = {"a": 10, "b": 10}
    assert population_stability_index(counts, counts) == pytest.approx(0.0)


def test_propose_does_not_auto_promote_a_classifier() -> None:
    baseline = capture_baseline(
        [_usage(request_id=f"b{i}", intent_id="coding", created_at=float(i)) for i in range(30)],
        tenant_id="acme",
    )
    current = [_usage(request_id=f"c{i}", intent_id="math", created_at=100 + i) for i in range(30)]
    report = analyze(
        current,
        tenant_id="acme",
        health=HealthTracker(),
        registry=_registry(),
        baseline=baseline,
        min_samples=20,
    )
    kinds = {item.kind for item in propose(report)}
    autos = [item for item in propose(report) if item.auto]
    assert RemediationKind.LEARNING_JOB in kinds
    assert all(item.kind is not RemediationKind.LEARNING_JOB or not item.auto for item in autos)
