"""Offline shadow comparison of a chosen route against alternatives.

Never sends extra paid inference. Scores use declared registry numbers when
both sides have them, otherwise the regret field stays None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_fabric.router.plan import RoutePlan
from llm_fabric.router.registry import ModelSpec


@dataclass(frozen=True, slots=True)
class ShadowOutcome:
    chosen: str | None
    alternatives: tuple[str, ...]
    route_regret: float | None
    quality_regret: float | None
    cost_regret: float | None
    latency_regret: float | None
    sampled: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen,
            "alternatives": list(self.alternatives),
            "route_regret": self.route_regret,
            "quality_regret": self.quality_regret,
            "cost_regret": self.cost_regret,
            "latency_regret": self.latency_regret,
            "sampled": self.sampled,
        }


def shadow_plan(
    plan: RoutePlan,
    *,
    expected_model: str | None = None,
    prompt_tokens: int = 100,
    sample: bool = True,
) -> ShadowOutcome:
    """Compare the planner's choice to labelled and remaining candidates."""
    chosen = plan.selected
    alternatives = tuple(
        candidate.spec.id
        for candidate in plan.ranked
        if chosen is None or candidate.spec.id != chosen.id
    )
    expected = _spec(plan, expected_model) if expected_model else None
    return ShadowOutcome(
        chosen=plan.selected_model,
        alternatives=alternatives,
        route_regret=(
            None
            if expected_model is None or plan.selected_model is None
            else (0.0 if plan.selected_model == expected_model else 1.0)
        ),
        quality_regret=_quality_regret(chosen, expected),
        cost_regret=_cost_regret(chosen, expected, prompt_tokens),
        latency_regret=_latency_regret(chosen, expected),
        sampled=sample,
    )


def _spec(plan: RoutePlan, model_id: str | None) -> ModelSpec | None:
    if model_id is None:
        return None
    for candidate in plan.ranked:
        if candidate.spec.id == model_id:
            return candidate.spec
    if plan.selected is not None and plan.selected.id == model_id:
        return plan.selected
    return None


def _quality_regret(chosen: ModelSpec | None, expected: ModelSpec | None) -> float | None:
    if chosen is None or expected is None:
        return None
    left = chosen.quality.mean
    right = expected.quality.mean
    if left is None or right is None:
        return None
    return right - left


def _cost_regret(chosen: ModelSpec | None, expected: ModelSpec | None, tokens: int) -> float | None:
    if chosen is None or expected is None:
        return None
    if not chosen.is_priced or not expected.is_priced:
        return None
    chosen_cost = chosen.cost_usd(tokens, tokens)
    expected_cost = expected.cost_usd(tokens, tokens)
    if chosen_cost is None or expected_cost is None:
        return None
    return chosen_cost - expected_cost


def _latency_regret(chosen: ModelSpec | None, expected: ModelSpec | None) -> float | None:
    if chosen is None or expected is None:
        return None
    left = chosen.performance.p50_ttft_ms
    right = expected.performance.p50_ttft_ms
    if left is None or right is None:
        return None
    return left - right
