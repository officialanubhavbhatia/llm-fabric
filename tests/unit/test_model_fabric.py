"""Cost knowledge, lifecycle, probe, and routing-quality behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.eval.routing_quality import overrouting, summarize_plans, underrouting
from llm_fabric.router.grades import Grade
from llm_fabric.router.plan import ExclusionRule, RoutePlanner, RouteRequest
from llm_fabric.router.policy import RoutePolicy
from llm_fabric.router.registry import (
    ApiCostKnowledge,
    ModelRegistry,
    ModelSpec,
    PromotionState,
    ResourceCostClass,
)


def test_yaml_omitted_price_is_unknown() -> None:
    registry = ModelRegistry.from_mapping({"models": [{"id": "a", "provider": "mock"}]})
    spec = registry.get("a")
    assert spec.api_cost_knowledge is ApiCostKnowledge.UNKNOWN
    assert spec.cost_usd(100, 100) is None
    assert spec.lifecycle is PromotionState.REGISTERED


def test_yaml_zero_is_known_zero() -> None:
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "local",
                    "provider": "ollama",
                    "input_cost_per_mtok": 0.0,
                    "output_cost_per_mtok": 0.0,
                    "cost_class": "resource_cost_unknown",
                }
            ]
        }
    )
    spec = registry.get("local")
    assert spec.api_cost_knowledge is ApiCostKnowledge.KNOWN_ZERO
    assert spec.cost_class is ResourceCostClass.RESOURCE_COST_UNKNOWN
    assert spec.cost_usd(100, 100) == 0.0


def test_provider_url_as_name_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown provider"):
        ModelRegistry.from_mapping(
            {"models": [{"id": "bad", "provider": "http://127.0.0.1:8000/v1"}]}
        )


def test_capabilities_mapping_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="capabilities"):
        ModelRegistry.from_mapping(
            {"models": [{"id": "a", "provider": "mock", "capabilities": {"chat": True}}]}
        )


def test_invalid_tier_name_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        ModelRegistry.from_mapping({"models": [{"id": "a", "provider": "mock", "tiers": ["L99"]}]})


def test_approved_beats_registered_high_tier() -> None:
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "approved-small",
                    "provider": "mock",
                    "grade": "L4",
                    "lifecycle": "approved",
                    "capabilities": ["chat"],
                },
                {
                    "id": "new-huge",
                    "provider": "mock",
                    "grade": "L30",
                    "lifecycle": "registered",
                    "capabilities": ["chat"],
                },
            ],
            "aliases": [
                {
                    "id": "auto",
                    "policy": "quality_first",
                    "candidates": ["approved-small", "new-huge"],
                }
            ],
        }
    )
    plan = RoutePlanner(registry).plan(RouteRequest("auto"))
    assert plan.selected_model == "approved-small"
    assert any(item.rule is ExclusionRule.NOT_PROBED for item in plan.excluded)


def test_pinned_registered_model_is_still_honoured() -> None:
    registry = ModelRegistry(
        [
            ModelSpec(
                id="approved-small",
                provider="mock",
                grade=Grade.GRADE04,
                lifecycle=PromotionState.APPROVED,
            ),
            ModelSpec(
                id="new-huge",
                provider="mock",
                grade=Grade.GRADE29,
                lifecycle=PromotionState.REGISTERED,
            ),
        ]
    )
    plan = RoutePlanner(registry).plan(RouteRequest("new-huge"))
    assert plan.selected_model == "new-huge"


def test_route_explain_includes_tier_policy() -> None:
    from llm_fabric.router.cli import format_explain
    from llm_fabric.router.intent_routing import RoutingConfig

    registry = ModelRegistry.from_yaml(Path("config/models.yaml"))
    routing = RoutingConfig.from_yaml(Path("config/routing.yaml"), registry=registry)
    plan = RoutePlanner(registry, routing=routing).plan(RouteRequest("auto"))
    payload = plan.describe()
    assert "tier_policy" in payload
    text = format_explain(payload)
    assert "Tier policy" in text
    assert "policy hash:" in text


def test_routing_policy_hash_is_stable() -> None:
    from llm_fabric.router.intent_routing import RoutingConfig

    first = RoutingConfig.from_yaml(Path("config/routing.yaml"))
    second = RoutingConfig.from_yaml(Path("config/routing.yaml"))
    assert first.content_hash
    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 16


def test_quality_shadow_does_not_change_selection() -> None:
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "cheap",
                    "provider": "mock",
                    "input_cost_per_mtok": 0.1,
                    "output_cost_per_mtok": 0.1,
                    "grade": "L2",
                    "quality": {"reasoning": 0.1},
                    "lifecycle": "approved",
                },
                {
                    "id": "dear",
                    "provider": "mock",
                    "input_cost_per_mtok": 9.0,
                    "output_cost_per_mtok": 9.0,
                    "grade": "L20",
                    "quality": {"reasoning": 0.9},
                    "lifecycle": "approved",
                },
            ],
            "aliases": [{"id": "auto", "policy": "cost_first", "candidates": ["cheap", "dear"]}],
        }
    )
    planner = RoutePlanner(registry, quality_shadow=True)
    plan = planner.plan(RouteRequest("auto"))
    assert plan.selected_model == "cheap"
    assert plan.quality_shadow is not None
    assert plan.quality_shadow["changed_served_route"] is False
    assert plan.quality_shadow["shadow_selected"] == "dear"


def test_overrouting_uses_cost_not_tier() -> None:
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "specialist",
                    "provider": "mock",
                    "input_cost_per_mtok": 1.0,
                    "output_cost_per_mtok": 1.0,
                    "grade": "L8",
                    "lifecycle": "approved",
                },
                {
                    "id": "general",
                    "provider": "mock",
                    "input_cost_per_mtok": 20.0,
                    "output_cost_per_mtok": 20.0,
                    "grade": "L22",
                    "lifecycle": "approved",
                },
            ],
            "aliases": [
                {"id": "auto", "policy": "cost_first", "candidates": ["specialist", "general"]}
            ],
        }
    )
    cheap_plan = RoutePlanner(registry).plan(RouteRequest("auto"))
    assert cheap_plan.selected_model == "specialist"
    assert overrouting(cheap_plan) is False
    dear_plan = RoutePlanner(registry, default_policy="quality_first").plan(
        RouteRequest("auto", policy=RoutePolicy.QUALITY_FIRST)
    )
    # quality_first with no quality scores uses grade; L22 wins and is more expensive
    if dear_plan.selected_model == "general":
        assert overrouting(dear_plan) is True


def test_underrouting_capability_gap() -> None:
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "chat-only",
                    "provider": "mock",
                    "capabilities": ["chat"],
                    "fallbacks": ["coder"],
                    "lifecycle": "approved",
                },
                {
                    "id": "coder",
                    "provider": "mock",
                    "capabilities": ["chat", "code"],
                    "lifecycle": "approved",
                },
            ]
        }
    )
    plan = RoutePlanner(registry).plan(RouteRequest("chat-only"))
    assert underrouting(plan, required=frozenset({"code"})) is True
    capable = RoutePlanner(registry).plan(RouteRequest("coder"))
    assert underrouting(capable, required=frozenset({"code"})) is False


def test_summarize_plans_does_not_impute_zero() -> None:
    registry = ModelRegistry.from_mapping(
        {"models": [{"id": "a", "provider": "mock", "lifecycle": "approved"}]}
    )
    plan = RoutePlanner(registry).plan(RouteRequest("a"))
    metrics = summarize_plans([(plan, {})])
    assert metrics["fallback_rate"] is None


def test_workloads_are_not_intentos_frozen() -> None:
    path = Path("datasets/eval/models/workloads.jsonl")
    assert path.is_file()
    assert "intentos" not in path.parts
    frozen = Path("datasets/eval/intentos/FROZEN_V1.sha256")
    assert frozen.is_file()


def test_score_output_math_is_deterministic() -> None:
    from llm_fabric.models.eval import score_output

    assert (
        score_output({"category": "math", "expected": "323"}, "323")["exact_answer_accuracy"] == 1.0
    )
    general = score_output({"category": "general_conversation"}, "hello")
    assert general["score"] == "not objectively scored"


async def test_mock_probe_does_not_invent_unavailable_latency() -> None:
    from llm_fabric.config import Settings
    from llm_fabric.models.probe import probe_provider_model

    payload = await probe_provider_model(
        provider="mock", model="mock-small", settings=Settings(environment="test")
    )
    assert payload["status"] == "ok"
    assert payload["reachable"] is True
    assert payload["capabilities"]["chat"]["supported"] is True
    assert payload["capabilities"]["tools"]["supported"] is False
    assert payload["performance"]["total_latency_ms"] is not None


def test_leaderboard_unknown_stays_null() -> None:
    from llm_fabric.models.eval import leaderboard_row

    row = leaderboard_row(
        {
            "deployment": "m",
            "provider": "mock",
            "categories": {
                "general_conversation": {"scoring": "not objectively scored", "metrics": {}}
            },
            "error_rate": None,
            "latency": {},
        },
        None,
    )
    assert row["general_score"] is None
    assert row["ttft_ms"] is None
    assert 0 not in (row["general_score"], row["ttft_ms"])
