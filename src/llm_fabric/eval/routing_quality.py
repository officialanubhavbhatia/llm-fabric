"""Routing-quality metrics with precise overrouting / underrouting definitions.

Higher constitutional grade is not treated as strictly better. A specialist at a
lower public tier can be the correct choice.

Regret is computed only when both sides have known values. Unknown is never
coerced to zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_fabric.router.plan import ExclusionRule, RoutePlan
from llm_fabric.router.registry import ModelSpec

# Overrouting: the selected deployment has a known API cost strictly greater than
# another *eligible* deployment that met the same capability and context
# requirements. Latency-only overrouting uses declared p50 TTFT the same way,
# and only when both TTFTs are known. Tier ordinal is not used.
#
# Underrouting: the selected deployment is missing a required capability, missed
# a declared SLO, or produced a recorded failure, while another eligible
# deployment in the plan would have satisfied the same constraint.

ROUTING_QUALITY_METRICS = (
    "route_success_rate",
    "overrouting_rate",
    "underrouting_rate",
    "fallback_rate",
    "escalation_rate",
    "capability_mismatch_rate",
    "policy_rejection_rate",
    "context_rejection_rate",
    "provider_unavailable_rate",
)


def overrouting(plan: RoutePlan) -> bool | None:
    """True when a cheaper known-cost eligible alternative existed.

    Returns None when cost cannot be compared (no selected model, or no pair of
    known prices). Known-zero is a valid price. Unknown is not.
    """
    selected = plan.selected
    if selected is None:
        return None
    selected_cost = selected.blended_cost_per_mtok
    if selected_cost is None:
        return None
    cheaper = False
    comparable = False
    for candidate in plan.ranked:
        other = candidate.spec
        if other.id == selected.id:
            continue
        other_cost = other.blended_cost_per_mtok
        if other_cost is None:
            continue
        comparable = True
        if other_cost < selected_cost:
            cheaper = True
    if not comparable:
        return None
    return cheaper


def underrouting(plan: RoutePlan, *, required: frozenset[str] = frozenset()) -> bool | None:
    """True when the selection missed a requirement another eligible model met."""
    selected = plan.selected
    if selected is None:
        # Failure to select is not underrouting of a model; it is recorded
        # separately as route_success_rate.
        return None
    if required and selected.capabilities.missing(required):
        return any(not candidate.spec.capabilities.missing(required) for candidate in plan.ranked)
    return False


def cost_regret(
    selected: ModelSpec | None, alternatives: Sequence[ModelSpec], tokens: int
) -> float | None:
    """selected_cost - min(cost of alternatives that have known prices)."""
    if selected is None or not selected.is_priced:
        return None
    selected_cost = selected.cost_usd(tokens, tokens)
    if selected_cost is None:
        return None
    known = [
        value
        for spec in alternatives
        if spec.is_priced and (value := spec.cost_usd(tokens, tokens)) is not None
    ]
    if not known:
        return None
    return selected_cost - min(known)


def summarize_plans(
    plans: Sequence[tuple[RoutePlan, dict[str, Any]]],
) -> dict[str, float | None]:
    """Aggregate routing-quality rates. None when a rate has no defined cases."""

    def rate(values: list[bool | None]) -> float | None:
        defined = [1.0 if item else 0.0 for item in values if item is not None]
        if not defined:
            return None
        return sum(defined) / len(defined)

    success: list[bool | None] = []
    over: list[bool | None] = []
    under: list[bool | None] = []
    fallback: list[bool | None] = []
    escalation: list[bool | None] = []
    capability: list[bool | None] = []
    policy: list[bool | None] = []
    context: list[bool | None] = []
    unavailable: list[bool | None] = []

    for plan, extra in plans:
        success.append(plan.selected is not None)
        over.append(overrouting(plan))
        required = frozenset(extra.get("required_capabilities") or ())
        under.append(underrouting(plan, required=required))
        recorded = extra.get("failover_count")
        fallback.append(None if recorded is None else int(recorded) > 0)
        escalation.append(
            any(item.rule is ExclusionRule.TIER_NOT_PREFERRED for item in plan.excluded)
        )
        capability.append(
            any(item.rule is ExclusionRule.MISSING_CAPABILITY for item in plan.excluded)
        )
        policy.append(
            any(
                item.rule
                in {
                    ExclusionRule.MODEL_DENIED,
                    ExclusionRule.PROVIDER_NOT_PERMITTED,
                    ExclusionRule.GRADE_ABOVE_MAXIMUM,
                }
                for item in plan.excluded
            )
        )
        context.append(any(item.rule is ExclusionRule.CONTEXT_TOO_SMALL for item in plan.excluded))
        unavailable.append(any(item.rule is ExclusionRule.CIRCUIT_OPEN for item in plan.excluded))

    return {
        "route_success_rate": rate(success),
        "overrouting_rate": rate(over),
        "underrouting_rate": rate(under),
        "fallback_rate": rate(fallback),
        "escalation_rate": rate(escalation),
        "capability_mismatch_rate": rate(capability),
        "policy_rejection_rate": rate(policy),
        "context_rejection_rate": rate(context),
        "provider_unavailable_rate": rate(unavailable),
    }


def offline_routing_eval(
    planner: Any,
    *,
    cases: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Plan representative requests with no provider call. Used for artifacts."""
    from llm_fabric.errors import FabricError, ModelNotFoundError
    from llm_fabric.router.plan import RouteRequest
    from llm_fabric.router.tiers import ALL_TIERS

    default_cases: list[dict[str, Any]] = [
        {"id": "auto-hello", "model": "auto", "prompt": "Hello"},
        {
            "id": "coding-capability",
            "model": "auto",
            "prompt": "Fix this Python traceback",
            "required_capabilities": ["code"],
        },
        {"id": "reasoning-alias", "model": "auto-reasoning", "prompt": "Why?"},
        {"id": "pinned-mock-small", "model": "mock-small", "prompt": "ping"},
        {
            "id": "context-large",
            "model": "auto",
            "prompt": "x",
            "prompt_tokens": 50_000,
        },
    ]
    rows: list[dict[str, Any]] = []
    pairs: list[tuple[Any, dict[str, Any]]] = []
    for case in list(cases or default_cases):
        request = RouteRequest(
            requested_model=str(case.get("model") or "auto"),
            required_capabilities=frozenset(case.get("required_capabilities") or ()),
            prompt_tokens=int(case.get("prompt_tokens") or 0),
        )
        extra = {"required_capabilities": list(request.required_capabilities)}
        try:
            plan = planner.plan(request)
        except (ModelNotFoundError, FabricError) as exc:
            rows.append(
                {
                    "id": case.get("id"),
                    "status": "error",
                    "detail": str(exc),
                    "overrouting": None,
                    "underrouting": None,
                }
            )
            continue
        pairs.append((plan, extra))
        rows.append(
            {
                "id": case.get("id"),
                "requested_model": plan.requested_model,
                "selected": plan.selected_model,
                "selected_tier": plan.selected_tier,
                "provider": plan.selected.provider if plan.selected else None,
                "overrouting": overrouting(plan),
                "underrouting": underrouting(
                    plan, required=frozenset(request.required_capabilities)
                ),
                "excluded": [item.as_dict() for item in plan.excluded],
                "routing_policy_hash": plan.routing_policy_hash or None,
                "quality_shadow": plan.quality_shadow,
            }
        )

    by_tier: list[dict[str, Any]] = []
    for tier in ALL_TIERS:
        try:
            plan = planner.plan(RouteRequest(tier.value))
        except (ModelNotFoundError, FabricError) as exc:
            by_tier.append(
                {
                    "tier": tier.value,
                    "selected": None,
                    "detail": str(exc),
                }
            )
            continue
        by_tier.append(
            {
                "tier": tier.value,
                "selected": plan.selected_model,
                "provider": plan.selected.provider if plan.selected else None,
                "eligible": [candidate.spec.id for candidate in plan.ranked],
            }
        )

    return {
        "eval_version": "routing-eval-v1",
        "note": (
            "Offline planner metrics. No provider call. Overrouting/underrouting "
            "are defined in docs/ROUTING-QUALITY.md. Higher tier is not treated "
            "as higher quality."
        ),
        "metrics": summarize_plans(pairs),
        "cases": rows,
        "by_tier": by_tier,
    }


def write_routing_artifacts(
    *,
    registry_path: Path | None = None,
    output: Path | None = None,
    shadow_output: Path | None = None,
) -> dict[str, Path]:
    """Write routing-eval.json and routing-shadow.json. No provider call."""
    from llm_fabric.config import get_settings
    from llm_fabric.eval.provenance import current_commit
    from llm_fabric.models.artifacts import DEFAULT_DIR, write_json
    from llm_fabric.router.intent_routing import RoutingConfig
    from llm_fabric.router.plan import RoutePlanner
    from llm_fabric.router.registry import ModelRegistry

    settings = get_settings()
    registry = ModelRegistry.from_yaml(registry_path or settings.registry_path)
    routing = RoutingConfig.from_yaml(settings.routing_config_path, registry=registry)
    planner = RoutePlanner(registry, routing=routing, quality_shadow=True)
    payload = offline_routing_eval(planner)
    payload["commit"] = current_commit()
    payload["routing_policy_version"] = routing.version
    payload["routing_policy_hash"] = routing.content_hash
    eval_path = output or (DEFAULT_DIR / "routing-eval.json")
    write_json(eval_path, payload)
    shadow = {
        "eval_version": "routing-shadow-v1",
        "commit": current_commit(),
        "routing_policy_version": routing.version,
        "routing_policy_hash": routing.content_hash,
        "note": (
            "Quality-shadow records only. The served route is unchanged. "
            "LLM_FABRIC_ROUTING_QUALITY_SHADOW default remains false."
        ),
        "cases": [
            {
                "id": row.get("id"),
                "selected": row.get("selected"),
                "quality_shadow": row.get("quality_shadow"),
            }
            for row in payload["cases"]
        ],
    }
    shadow_path = shadow_output or (DEFAULT_DIR / "routing-shadow.json")
    write_json(shadow_path, shadow)
    return {"routing_eval": eval_path, "routing_shadow": shadow_path}
