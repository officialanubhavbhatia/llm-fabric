"""Routing evaluator.

Compares a `RoutePlan` to a labelled expected route. Regret figures that need
a measured quality, latency or cost on *both* sides are left `None` when either
side is unmeasured. Declared YAML numbers are labelled as declared, never as
observed performance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_fabric.contract.openai import ChatCompletionRequest, ChatMessage
from llm_fabric.errors import ConfigurationError
from llm_fabric.eval.schema import EvalResult, EvaluatorKind, ExampleResult
from llm_fabric.router.engine import Router
from llm_fabric.router.grades import Grade
from llm_fabric.router.plan import RoutePlan
from llm_fabric.router.registry import ModelSpec

SUPPORTED = frozenset(
    {
        "route_match",
        "policy_match",
        "fallback_rate",
        "unnecessary_escalation_rate",
        "underpowered_rate",
        "overpowered_rate",
        "declared_cost_regret",
        "declared_latency_regret",
        "declared_quality_regret",
    }
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        cases.append(json.loads(text))
    return cases


def evaluate(
    *,
    task: str,
    cases: list[dict[str, Any]],
    router: Router,
    metrics: tuple[str, ...],
) -> EvalResult:
    requested = tuple(name for name in metrics if name in SUPPORTED)
    unknown = tuple(name for name in metrics if name not in SUPPORTED)
    rows: list[ExampleResult] = []
    for index, case in enumerate(cases):
        example_id = str(case.get("id") or index)
        plan = _plan(router, case)
        scores = {name: _score(name, case, plan) for name in requested}
        rows.append(
            ExampleResult(
                example_id=example_id,
                scores=scores,
                output=plan.selected_model,
                notes=plan.policy.value,
            )
        )
    aggregates = {name: _mean(rows, name) for name in requested}
    return EvalResult(
        task=task,
        evaluator=EvaluatorKind.ROUTING,
        metrics=aggregates,
        examples=tuple(rows),
        unavailable=unknown,
        note=(
            "route_match is labelled expected model vs the planner. "
            "Regret metrics use declared registry numbers when both sides have "
            "them, and stay unavailable otherwise. This is not a routing-quality "
            "eval against production traffic."
        ),
    )


def _plan(router: Router, case: dict[str, Any]) -> RoutePlan:
    request = ChatCompletionRequest(
        model=str(case.get("requested_model") or "auto"),
        messages=[
            ChatMessage(role="user", content=str(case.get("input") or case.get("text") or ""))
        ],
    )
    return router.preview(request, tenant_id=case.get("tenant_id"))


def _score(name: str, case: dict[str, Any], plan: RoutePlan) -> float | None:
    selected = plan.selected
    expected_model = case.get("expected_model")
    if name == "route_match":
        if expected_model is None:
            return None
        return 1.0 if plan.selected_model == expected_model else 0.0
    if name == "policy_match":
        expected_policy = case.get("expected_policy")
        if expected_policy is None:
            return None
        return 1.0 if plan.policy.value == expected_policy else 0.0
    if name == "fallback_rate":
        # Preview does not execute. Only a recorded execution can score this.
        recorded = case.get("failover_count")
        if recorded is None:
            return None
        return 1.0 if int(recorded) > 0 else 0.0
    if selected is None:
        return None
    expected_grade = _grade(case.get("expected_grade"))
    if name == "unnecessary_escalation_rate":
        ceiling = _grade(case.get("max_grade")) or expected_grade
        if ceiling is None or selected.grade is None:
            return None
        return 1.0 if selected.grade.ordinal > ceiling.ordinal else 0.0
    if name == "underpowered_rate":
        floor = _grade(case.get("min_grade")) or expected_grade
        if floor is None or selected.grade is None:
            return None
        return 1.0 if selected.grade.ordinal < floor.ordinal else 0.0
    if name == "overpowered_rate":
        if expected_grade is None or selected.grade is None:
            return None
        return 1.0 if selected.grade.ordinal > expected_grade.ordinal else 0.0
    if name == "declared_cost_regret":
        return _declared_regret(case, plan, "cost")
    if name == "declared_latency_regret":
        return _declared_regret(case, plan, "latency")
    if name == "declared_quality_regret":
        return _declared_regret(case, plan, "quality")
    return None


def _declared_regret(case: dict[str, Any], plan: RoutePlan, feature: str) -> float | None:
    """Selected minus expected on a *declared* registry number.

    Returns None when either side has no declared value. Never treats absence
    as zero.
    """
    expected_id = case.get("expected_model")
    if expected_id is None or plan.selected is None:
        return None
    expected_spec = _spec_on_plan(plan, str(expected_id))
    if expected_spec is None:
        # The labelled model was not a candidate; regret against it is undefined.
        return None
    if feature == "cost":
        if not plan.selected.is_priced or not expected_spec.is_priced:
            return None
        tokens = int(case.get("prompt_tokens") or 100)
        selected_cost = plan.selected.cost_usd(tokens, tokens)
        expected_cost = expected_spec.cost_usd(tokens, tokens)
        if selected_cost is None or expected_cost is None:
            return None
        return selected_cost - expected_cost
    if feature == "latency":
        left = plan.selected.performance.p50_ttft_ms
        right = expected_spec.performance.p50_ttft_ms
        if left is None or right is None:
            return None
        return float(left) - float(right)
    if feature == "quality":
        left = plan.expected_quality
        right = expected_spec.quality.mean
        if left is None or right is None:
            return None
        # Higher quality is better: regret is expected minus selected.
        return float(right) - float(left)
    return None


def _spec_on_plan(plan: RoutePlan, model_id: str) -> ModelSpec | None:
    for candidate in plan.ranked:
        if candidate.spec.id == model_id:
            return candidate.spec
    return None


def _grade(raw: object) -> Grade | None:
    if raw is None:
        return None
    try:
        return Grade.parse(str(raw))
    except ConfigurationError:
        return None


def _mean(rows: list[ExampleResult], name: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.scores.get(name)
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else None
