"""The route planner, exercised against the synthetic thirty-grade fleet.

Every assertion here runs offline: the fleet is fictional and its provider never
makes a call, so filtering, tenant policy and explanation are tested without a
paid API and without flakiness.
"""

from __future__ import annotations

import pytest

from llm_fabric.errors import ModelNotFoundError, NoCandidateError
from llm_fabric.intent.schema import (
    Complexity,
    ContextClass,
    CostClass,
    IntentClassification,
    LatencyClass,
    Modality,
    QualityClass,
    ReasoningLevel,
    RiskClass,
)
from llm_fabric.router.capabilities import Capability
from llm_fabric.router.fallback import FallbackBudget, FallbackReason
from llm_fabric.router.grades import Grade
from llm_fabric.router.health import BreakerPolicy, HealthTracker
from llm_fabric.router.plan import (
    ExclusionRule,
    RoutePlanner,
    RouteRequest,
    TenantRoutingPolicies,
    TenantRoutingPolicy,
)
from llm_fabric.router.policy import RoutePolicy
from llm_fabric.router.registry import Locality, ModelRegistry
from llm_fabric.router.synthetic import synthetic_model_id, synthetic_registry


@pytest.fixture
def registry() -> ModelRegistry:
    return synthetic_registry()


@pytest.fixture
def planner(registry: ModelRegistry) -> RoutePlanner:
    return RoutePlanner(registry, health=HealthTracker())


def _intent(
    intent_id: str = "coding",
    *,
    domain: str | None = None,
    quality: QualityClass = QualityClass.STANDARD,
    latency: LatencyClass = LatencyClass.INTERACTIVE,
    cost: CostClass = CostClass.LOW,
    modality: Modality = Modality.TEXT,
    capabilities: frozenset[str] = frozenset(),
    abstain: bool = False,
) -> IntentClassification:
    if abstain:
        return IntentClassification.unknown(classifier_version="test", taxonomy_version="test")
    return IntentClassification(
        intent_id=intent_id,
        domain=domain or intent_id,
        complexity=Complexity.MODERATE,
        reasoning_level=ReasoningLevel.LIGHT,
        modality=modality,
        context_class=ContextClass.SHORT,
        risk_class=RiskClass.LOW,
        latency_class=latency,
        quality_class=quality,
        cost_class=cost,
        confidence=0.9,
        classifier_version="test",
        taxonomy_version="test",
        required_capabilities=capabilities,
    )


def _excluded(plan: object, rule: ExclusionRule) -> set[str]:
    return {e.model_id for e in plan.excluded if e.rule is rule}  # type: ignore[attr-defined]


# -- policy selection ---------------------------------------------------------


def test_the_alias_policy_is_used(planner: RoutePlanner) -> None:
    assert planner.plan(RouteRequest("synth-cheap")).policy is RoutePolicy.COST_FIRST
    assert planner.plan(RouteRequest("synth-best")).policy is RoutePolicy.QUALITY_FIRST
    assert planner.plan(RouteRequest("synth-fast")).policy is RoutePolicy.LATENCY_FIRST


def test_an_explicit_request_policy_beats_the_alias(planner: RoutePlanner) -> None:
    plan = planner.plan(RouteRequest("synth-cheap", policy=RoutePolicy.QUALITY_FIRST))
    assert plan.policy is RoutePolicy.QUALITY_FIRST
    assert plan.selected_model == synthetic_model_id(Grade.GRADE29)


def test_a_pinned_model_is_not_reordered(planner: RoutePlanner) -> None:
    plan = planner.plan(RouteRequest(synthetic_model_id(Grade.GRADE20)))
    assert plan.policy is RoutePolicy.DECLARED
    assert plan.selected_model == synthetic_model_id(Grade.GRADE20)


def test_intent_can_choose_the_policy_when_nobody_else_did(registry: ModelRegistry) -> None:
    planner = RoutePlanner(registry, default_policy="balanced")

    maximum = planner.plan(RouteRequest("synth-auto", intent=_intent(quality=QualityClass.MAXIMUM)))
    # The alias pins `balanced`, so the intent must not override it there.
    assert maximum.policy is RoutePolicy.BALANCED

    # With no alias policy the intent decides.
    fleet_wide = RoutePlanner(
        registry.replace_model(registry.get(synthetic_model_id(Grade.GRADE00))),
        default_policy="balanced",
    )
    plan = fleet_wide.plan(
        RouteRequest(
            synthetic_model_id(Grade.GRADE00),
            intent=_intent(latency=LatencyClass.REALTIME),
        )
    )
    assert plan.policy is RoutePolicy.DECLARED  # pinned models still win


def test_an_abstaining_intent_uses_balanced_instead_of_cheapest_alias(
    registry: ModelRegistry,
) -> None:
    planner = RoutePlanner(registry, default_policy="cost_first")
    cheap = planner.plan(RouteRequest("synth-cheap"))
    assert cheap.policy is RoutePolicy.COST_FIRST
    abstained = planner.plan(RouteRequest("synth-cheap", intent=_intent(abstain=True)))
    assert abstained.policy is RoutePolicy.BALANCED


def test_an_abstaining_intent_does_not_override_a_pinned_model(registry: ModelRegistry) -> None:
    planner = RoutePlanner(registry, default_policy="cost_first")
    plan = planner.plan(
        RouteRequest(synthetic_model_id(Grade.GRADE00), intent=_intent(abstain=True))
    )
    assert plan.policy is RoutePolicy.DECLARED


# -- capability filtering -----------------------------------------------------


def test_a_capability_requirement_excludes_those_that_lack_it(planner: RoutePlanner) -> None:
    plan = planner.plan(
        RouteRequest("synth-auto", required_capabilities=frozenset({Capability.VISION}))
    )
    assert plan.selected is not None
    assert plan.selected.capabilities.has(Capability.VISION)
    assert synthetic_model_id(Grade.GRADE00) in _excluded(plan, ExclusionRule.MISSING_CAPABILITY)


def test_the_intent_supplies_capabilities(planner: RoutePlanner) -> None:
    plan = planner.plan(RouteRequest("synth-auto", intent=_intent(modality=Modality.IMAGE)))
    assert plan.selected is not None
    assert plan.selected.capabilities.supports_vision


def test_intent_capability_extras_do_not_empty_the_fleet(planner: RoutePlanner) -> None:
    """Classification proposes capabilities. It must not turn a serveable
    prompt into a 503 because the labelled intent named a capability no
    candidate has.
    """
    plan = planner.plan(
        RouteRequest(
            "synth-auto",
            intent=_intent(capabilities=frozenset({"time_travel"})),
        )
    )
    assert plan.selected is not None
    assert any("capability extras" in note for note in plan.notes)


def test_an_alias_requirement_is_honoured(planner: RoutePlanner) -> None:
    plan = planner.plan(RouteRequest("synth-vision"))
    assert plan.selected is not None
    assert plan.selected.capabilities.supports_vision
    # Cheapest among those that can see, not cheapest overall.
    assert plan.selected.id == synthetic_model_id(Grade.GRADE18)


def test_an_impossible_requirement_selects_nothing(planner: RoutePlanner) -> None:
    plan = planner.plan(
        RouteRequest("synth-auto", required_capabilities=frozenset({"time_travel"}))
    )
    assert plan.selected is None
    assert len(plan.excluded) == 30


def test_require_plan_refuses_an_empty_selection(planner: RoutePlanner) -> None:
    with pytest.raises(NoCandidateError):
        planner.require_plan(
            RouteRequest("synth-auto", required_capabilities=frozenset({"time_travel"}))
        )


# -- locality and privacy -----------------------------------------------------


def test_local_only_never_leaves_the_machine(planner: RoutePlanner) -> None:
    plan = planner.plan(RouteRequest("synth-local"))
    assert plan.selected is not None
    assert plan.selected.locality is Locality.LOCAL
    assert all(c.spec.locality is Locality.LOCAL for c in plan.ranked)


def test_private_only_admits_local_and_private_but_not_external(planner: RoutePlanner) -> None:
    plan = planner.plan(RouteRequest("synth-private"))
    assert {c.spec.locality for c in plan.ranked} <= {Locality.LOCAL, Locality.PRIVATE}
    assert all(c.spec.keeps_data_in_house for c in plan.ranked)
    external = _excluded(plan, ExclusionRule.LOCALITY_NOT_PERMITTED)
    assert synthetic_model_id(Grade.GRADE02) in external


def test_an_undeclared_locality_is_treated_as_external(registry: ModelRegistry) -> None:
    from llm_fabric.router.registry import ModelSpec

    unlabelled = ModelSpec(id="unlabelled", provider="mystery")
    planner = RoutePlanner(ModelRegistry([unlabelled]))
    plan = planner.plan(RouteRequest("unlabelled", policy=RoutePolicy.LOCAL_ONLY))
    assert plan.selected is None
    assert _excluded(plan, ExclusionRule.LOCALITY_NOT_PERMITTED) == {"unlabelled"}


# -- grades -------------------------------------------------------------------


def test_a_minimum_grade_excludes_the_weaker_bands(planner: RoutePlanner) -> None:
    plan = planner.plan(RouteRequest("synth-cheap", minimum_grade=Grade.GRADE25))
    assert plan.selected is not None
    assert plan.selected.grade is not None
    assert plan.selected.grade.ordinal >= 25
    assert synthetic_model_id(Grade.GRADE24) in _excluded(plan, ExclusionRule.GRADE_BELOW_MINIMUM)


def test_an_ungraded_deployment_cannot_satisfy_a_grade_floor() -> None:
    from llm_fabric.router.registry import ModelSpec

    planner = RoutePlanner(ModelRegistry([ModelSpec(id="ungraded", provider="mystery")]))
    plan = planner.plan(RouteRequest("ungraded", minimum_grade=Grade.GRADE01))
    assert plan.selected is None


# -- context ------------------------------------------------------------------


def test_a_prompt_too_large_for_a_deployment_excludes_it(planner: RoutePlanner) -> None:
    plan = planner.plan(RouteRequest("synth-cheap", prompt_tokens=60_000))
    assert plan.selected is not None
    assert plan.selected.fits_context(60_000)
    assert synthetic_model_id(Grade.GRADE00) in _excluded(plan, ExclusionRule.CONTEXT_TOO_SMALL)


def test_reserved_output_counts_against_the_window(planner: RoutePlanner) -> None:
    small = planner.plan(RouteRequest("synth-cheap", prompt_tokens=4_000))
    large = planner.plan(RouteRequest("synth-cheap", prompt_tokens=4_000, max_output_tokens=40_000))
    assert small.selected is not None and large.selected is not None
    assert large.selected.grade is not None and small.selected.grade is not None
    assert large.selected.grade.ordinal > small.selected.grade.ordinal


# -- budget and SLO -----------------------------------------------------------


def test_a_budget_excludes_deployments_known_to_breach_it(planner: RoutePlanner) -> None:
    # Prices rise with grade, so a tight budget caps how good a model the
    # request can reach even under a quality-first policy.
    plan = planner.plan(RouteRequest("synth-best", prompt_tokens=5_000, budget_usd=0.0005))
    assert _excluded(plan, ExclusionRule.OVER_BUDGET)
    assert plan.selected is not None
    assert plan.selected.cost_usd(5_000, 0) <= 0.0005


def test_an_unpriced_deployment_is_never_excluded_for_cost() -> None:
    from llm_fabric.router.registry import ModelSpec

    planner = RoutePlanner(ModelRegistry([ModelSpec(id="unpriced", provider="mystery")]))
    plan = planner.plan(RouteRequest("unpriced", prompt_tokens=1_000_000, budget_usd=0.0001))
    # No declared price is not evidence of being expensive.
    assert plan.selected_model == "unpriced"


def test_a_latency_slo_excludes_deployments_declared_slower(planner: RoutePlanner) -> None:
    plan = planner.plan(RouteRequest("synth-best", latency_slo_ms=100.0, max_output_tokens=0))
    assert _excluded(plan, ExclusionRule.LATENCY_SLO_MISSED)
    assert plan.selected is not None
    assert (plan.selected.performance.p50_ttft_ms or 0) <= 100.0


def test_an_unmeasured_deployment_is_not_excluded_by_an_slo() -> None:
    from llm_fabric.router.registry import ModelSpec

    planner = RoutePlanner(ModelRegistry([ModelSpec(id="unknown-speed", provider="mystery")]))
    plan = planner.plan(RouteRequest("unknown-speed", latency_slo_ms=1.0))
    assert plan.selected_model == "unknown-speed"


# -- health -------------------------------------------------------------------


def test_an_open_circuit_removes_a_deployment(registry: ModelRegistry) -> None:
    health = HealthTracker(policy=BreakerPolicy(consecutive_failures=1))
    planner = RoutePlanner(registry, health=health)

    cheapest = synthetic_model_id(Grade.GRADE00)
    assert planner.plan(RouteRequest("synth-cheap")).selected_model == cheapest

    health.record_failure(registry.get(cheapest).deployment_id, error="boom")
    plan = planner.plan(RouteRequest("synth-cheap"))
    assert plan.selected_model != cheapest
    assert cheapest in _excluded(plan, ExclusionRule.CIRCUIT_OPEN)


def test_all_circuits_open_selects_nothing(registry: ModelRegistry) -> None:
    health = HealthTracker(policy=BreakerPolicy(consecutive_failures=1))
    for spec in registry.all_models():
        health.record_failure(spec.deployment_id, error="boom")
    plan = RoutePlanner(registry, health=health).plan(RouteRequest("synth-auto"))
    assert plan.selected is None
    assert len(_excluded(plan, ExclusionRule.CIRCUIT_OPEN)) == 30


# -- tenant policy ------------------------------------------------------------


def _with_tenant(registry: ModelRegistry, policy: TenantRoutingPolicy) -> RoutePlanner:
    return RoutePlanner(registry, tenant_policies=TenantRoutingPolicies([policy]))


def test_a_tenant_can_pin_the_policy(registry: ModelRegistry) -> None:
    planner = _with_tenant(
        registry, TenantRoutingPolicy(tenant_id="acme", policy=RoutePolicy.QUALITY_FIRST)
    )
    plan = planner.plan(RouteRequest("synth-cheap", tenant_id="acme"))
    assert plan.policy is RoutePolicy.QUALITY_FIRST
    assert any("tenant policy" in line for line in plan.explain())


def test_a_tenant_policy_cannot_be_overridden_by_the_request(registry: ModelRegistry) -> None:
    planner = _with_tenant(
        registry, TenantRoutingPolicy(tenant_id="acme", policy=RoutePolicy.QUALITY_FIRST)
    )
    plan = planner.plan(
        RouteRequest("synth-cheap", tenant_id="acme", policy=RoutePolicy.COST_FIRST)
    )
    assert plan.policy is RoutePolicy.QUALITY_FIRST


def test_require_in_house_cannot_be_widened_by_an_alias(registry: ModelRegistry) -> None:
    planner = _with_tenant(registry, TenantRoutingPolicy(tenant_id="acme", require_in_house=True))
    # `synth-auto` is balanced and would happily use external deployments.
    plan = planner.plan(RouteRequest("synth-auto", tenant_id="acme"))
    assert plan.ranked
    assert all(c.spec.keeps_data_in_house for c in plan.ranked)


def test_locality_constraints_intersect_rather_than_replace(registry: ModelRegistry) -> None:
    planner = _with_tenant(
        registry,
        TenantRoutingPolicy(tenant_id="acme", allowed_localities=frozenset({Locality.PRIVATE})),
    )
    # The policy allows local and private; the tenant allows only private.
    plan = planner.plan(RouteRequest("synth-private", tenant_id="acme"))
    assert plan.ranked
    assert all(c.spec.locality is Locality.PRIVATE for c in plan.ranked)


def test_a_tenant_can_deny_specific_models(registry: ModelRegistry) -> None:
    cheapest = synthetic_model_id(Grade.GRADE00)
    planner = _with_tenant(
        registry, TenantRoutingPolicy(tenant_id="acme", denied_models=frozenset({cheapest}))
    )
    plan = planner.plan(RouteRequest("synth-cheap", tenant_id="acme"))
    assert plan.selected_model != cheapest
    assert cheapest in _excluded(plan, ExclusionRule.MODEL_DENIED)


def test_a_tenant_can_restrict_providers(registry: ModelRegistry) -> None:
    planner = _with_tenant(
        registry,
        TenantRoutingPolicy(tenant_id="acme", allowed_providers=frozenset({"synth_local"})),
    )
    plan = planner.plan(RouteRequest("synth-auto", tenant_id="acme"))
    assert plan.ranked
    assert all(c.spec.provider == "synth_local" for c in plan.ranked)


def test_the_strictest_grade_floor_wins(registry: ModelRegistry) -> None:
    planner = _with_tenant(
        registry, TenantRoutingPolicy(tenant_id="acme", minimum_grade=Grade.GRADE20)
    )
    plan = planner.plan(RouteRequest("synth-cheap", tenant_id="acme", minimum_grade=Grade.GRADE05))
    assert plan.selected is not None and plan.selected.grade is not None
    assert plan.selected.grade.ordinal >= 20


def test_another_tenants_policy_does_not_apply(registry: ModelRegistry) -> None:
    planner = _with_tenant(registry, TenantRoutingPolicy(tenant_id="acme", require_in_house=True))
    plan = planner.plan(RouteRequest("synth-auto", tenant_id="other"))
    assert plan.tenant_policy is None
    assert any(not c.spec.keeps_data_in_house for c in plan.ranked)


def test_an_unauthenticated_plan_gets_no_tenant_policy(registry: ModelRegistry) -> None:
    planner = _with_tenant(registry, TenantRoutingPolicy(tenant_id="acme", require_in_house=True))
    assert planner.plan(RouteRequest("synth-auto")).tenant_policy is None


# -- the decision object ------------------------------------------------------


def test_every_plan_names_its_selection_and_its_chain(planner: RoutePlanner) -> None:
    plan = planner.plan(RouteRequest("synth-cheap"))
    assert plan.selected_model == plan.chain[0]
    assert len(plan.chain) == 30
    assert plan.routing_score is not None


def test_expected_values_are_present_when_declared(planner: RoutePlanner) -> None:
    plan = planner.plan(RouteRequest("synth-best", max_output_tokens=100))
    assert plan.expected_quality is not None
    assert plan.expected_latency_ms is not None
    assert plan.expected_cost_usd(1000, 100) is not None


def test_expected_cost_is_absent_for_an_unpriced_selection() -> None:
    from llm_fabric.router.registry import ModelSpec

    planner = RoutePlanner(ModelRegistry([ModelSpec(id="unpriced", provider="mystery")]))
    plan = planner.plan(RouteRequest("unpriced"))
    assert plan.expected_cost_usd(1000, 100) is None


def test_the_explanation_names_the_selection_and_the_exclusions(planner: RoutePlanner) -> None:
    plan = planner.plan(
        RouteRequest("synth-auto", required_capabilities=frozenset({Capability.VISION}))
    )
    lines = " ".join(plan.explain())
    assert plan.selected_model is not None and plan.selected_model in lines
    assert "missing_capability" in lines


def test_all_thirteen_planner_inputs_are_reported(planner: RoutePlanner) -> None:
    plan = planner.plan(RouteRequest("synth-auto"))
    reported = {item.name for item in plan.inputs}
    assert {
        "intent",
        "tenant_policy",
        "quality_requirement",
        "latency_slo",
        "budget",
        "context_requirement",
        "model_capabilities",
        "provider_health",
        "queue_pressure",
        "historical_quality",
        "historical_latency",
        "historical_error_rate",
        "cache_probability",
    } <= reported


def test_unbuilt_inputs_are_declared_unavailable(planner: RoutePlanner) -> None:
    # Two of the constitution's inputs have no source in this build. Saying so
    # is the point: silently scoring without them would hide the gap.
    by_name = {item.name: item for item in planner.plan(RouteRequest("synth-auto")).inputs}
    assert not by_name["historical_quality"].available
    assert "not built" in by_name["historical_quality"].note
    assert not by_name["cache_probability"].available
    assert "not built" in by_name["cache_probability"].note


def test_the_plan_serialises_completely(planner: RoutePlanner) -> None:
    payload = planner.plan(RouteRequest("synth-cheap", prompt_tokens=100)).describe()
    assert {
        "requested_model",
        "policy",
        "selected",
        "routing_score",
        "expected",
        "chain",
        "scoring",
        "excluded",
        "fallback",
        "inputs",
        "explanation",
    } <= set(payload)
    assert payload["selected"]["grade"] == "Grade00"
    assert payload["fallback"]["budget"]["max_depth"] >= 0


def test_expected_values_are_labelled_as_estimates(planner: RoutePlanner) -> None:
    payload = planner.plan(RouteRequest("synth-best")).describe()
    assert "estimates" in payload["expected"]["note"]


# -- the fallback graph a plan carries ----------------------------------------


def test_a_plan_carries_a_usable_fallback_graph(planner: RoutePlanner) -> None:
    plan = planner.plan(RouteRequest("synth-cheap"))
    assert plan.selected_model is not None
    nxt = plan.graph.next_hop(plan.selected_model, FallbackReason.TIMEOUT)
    assert nxt == plan.chain[1]


def test_declared_edges_beat_ranked_order_for_their_reason(registry: ModelRegistry) -> None:
    from llm_fabric.router.fallback import ANY_REASON, FallbackEdge, FallbackGraph

    cheapest = synthetic_model_id(Grade.GRADE00)
    biggest = synthetic_model_id(Grade.GRADE29)
    graph = FallbackGraph(
        [FallbackEdge(cheapest, biggest, reasons=frozenset({FallbackReason.CONTEXT_TOO_LARGE}))]
    )
    planner = RoutePlanner(registry, graph=graph)
    plan = planner.plan(RouteRequest("synth-cheap"))

    assert plan.graph.next_hop(cheapest, FallbackReason.CONTEXT_TOO_LARGE) == biggest
    # Other reasons still fall through to ranked order.
    assert plan.graph.next_hop(cheapest, FallbackReason.TIMEOUT) == plan.chain[1]
    assert ANY_REASON  # the vocabulary is the graph's, not the planner's


def test_the_budget_can_be_set_per_tenant(registry: ModelRegistry) -> None:
    planner = RoutePlanner(
        registry,
        tenant_policies=TenantRoutingPolicies(
            [TenantRoutingPolicy(tenant_id="acme", fallback_budget=FallbackBudget(max_depth=0))]
        ),
        fallback_budget=FallbackBudget(max_depth=5),
    )
    assert planner.plan(RouteRequest("synth-auto", tenant_id="acme")).budget.max_depth == 0
    assert planner.plan(RouteRequest("synth-auto")).budget.max_depth == 5


# -- errors -------------------------------------------------------------------


def test_an_unknown_model_raises(planner: RoutePlanner) -> None:
    with pytest.raises(ModelNotFoundError):
        planner.plan(RouteRequest("no-such-model"))


def test_planning_is_deterministic(planner: RoutePlanner) -> None:
    runs = {planner.plan(RouteRequest("synth-auto")).chain for _ in range(5)}
    assert len(runs) == 1
