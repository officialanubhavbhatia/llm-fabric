"""Intent → capability → preferred-tier policy is configuration, not IntentOS."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.router.grades import Grade
from llm_fabric.router.intent_routing import RoutingConfig
from llm_fabric.router.plan import (
    ExclusionRule,
    RoutePlanner,
    RouteRequest,
    TenantRoutingPolicies,
    TenantRoutingPolicy,
)
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.router.synthetic import synthetic_model_id, synthetic_registry
from llm_fabric.router.tiers import ServiceTier


def test_shipped_routing_config_loads_against_the_default_registry() -> None:
    registry = ModelRegistry.from_yaml(Path("config/models.yaml"))
    config = RoutingConfig.from_yaml(Path("config/routing.yaml"), registry=registry)
    assert config.version == "2026.08.24"
    assert config.policy_for("coding") is not None
    assert config.policy_for("coding.debug") is not None


def test_unknown_preferred_model_fails_fast() -> None:
    registry = synthetic_registry()
    with pytest.raises(ConfigurationError, match="unknown model"):
        RoutingConfig.from_mapping(
            {
                "routing": {
                    "version": "test",
                    "intent": {"coding": {"preferred_models": ["does-not-exist"]}},
                }
            },
            registry=registry,
        )


def test_requesting_a_tier_selects_a_model_that_serves_it() -> None:
    planner = RoutePlanner(synthetic_registry())
    plan = planner.plan(RouteRequest("L12"))
    assert plan.selected is not None
    assert plan.selected.serves_tier(ServiceTier.L12)
    assert plan.selected_tier == "L12"
    assert plan.policy.value != "declared"


def test_coding_intent_prefers_coding_capable_tiers_over_a_higher_general() -> None:
    """A coding specialist at a preferred tier beats a higher-grade general model.

    The synthetic fleet unlocks `code` at Grade05 and has a coding-quality bonus
    on ordinals % 5 == 3. Preferred tiers L8/L12/L16 keep Grade08/12/16 and drop
    Grade20 even though Grade20 is stronger overall.
    """
    routing = RoutingConfig.from_mapping(
        {
            "routing": {
                "version": "test",
                "intent": {
                    "coding": {
                        "tiers": ["L8", "L12", "L16"],
                        "capabilities": ["code"],
                    }
                },
            }
        }
    )
    planner = RoutePlanner(synthetic_registry(), routing=routing)
    plan = planner.plan(RouteRequest("synth-auto", intent_id="coding"))
    assert plan.selected is not None
    assert plan.selected.grade is not None
    assert plan.selected.grade.ordinal in {8, 12, 16}
    assert synthetic_model_id(Grade.GRADE20) in {
        item.model_id for item in plan.excluded if item.rule is ExclusionRule.TIER_NOT_PREFERRED
    }


def test_trivial_intent_stays_on_low_preferred_tiers() -> None:
    routing = RoutingConfig.from_mapping(
        {
            "routing": {
                "version": "test",
                "intent": {"general_conversation": {"tiers": ["L0", "L1", "L2"]}},
            }
        }
    )
    planner = RoutePlanner(synthetic_registry(), routing=routing)
    plan = planner.plan(RouteRequest("synth-auto", intent_id="general_conversation"))
    assert plan.selected is not None
    assert plan.selected.grade is not None
    assert plan.selected.grade.ordinal <= 2


def test_tenant_max_tier_cannot_be_raised_by_requesting_l30() -> None:
    registry = synthetic_registry()
    planner = RoutePlanner(
        registry,
        tenant_policies=TenantRoutingPolicies(
            [TenantRoutingPolicy(tenant_id="acme", maximum_grade=Grade.GRADE10)]
        ),
    )
    plan = planner.plan(
        RouteRequest("synth-auto", tenant_id="acme", maximum_grade=Grade.parse("L30"))
    )
    assert plan.selected is not None
    assert plan.selected.grade is not None
    assert plan.selected.grade.ordinal <= 10
    assert any(item.rule is ExclusionRule.GRADE_ABOVE_MAXIMUM for item in plan.excluded)


def test_preferred_tiers_are_kept_when_none_match() -> None:
    routing = RoutingConfig.from_mapping(
        {
            "routing": {
                "version": "test",
                "intent": {"coding": {"tiers": ["L28", "L29"]}},
            }
        }
    )
    # Pin to a Grade00 model: preferred high tiers cannot match, so the pin stays.
    planner = RoutePlanner(synthetic_registry(), routing=routing)
    plan = planner.plan(RouteRequest(synthetic_model_id(Grade.GRADE00), intent_id="coding"))
    assert plan.selected_model == synthetic_model_id(Grade.GRADE00)
    assert any("keeping the filtered set" in note for note in plan.notes)
