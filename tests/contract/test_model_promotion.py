"""Publicly inspectable promotion schemas; mutation remains CLI-only."""

from __future__ import annotations

from llm_fabric.models.promotion import PromotionStore, status_payload
from llm_fabric.router.plan import RoutePlanner, RouteRequest
from llm_fabric.router.registry import ModelRegistry, ModelSpec, PromotionState


def test_registry_lifecycle_schema_distinguishes_declared_and_approved_tiers() -> None:
    spec = ModelSpec(
        id="candidate",
        provider="mock",
        grade=None,
        tiers=(),
        lifecycle=PromotionState.EVALUATED,
    )
    payload = spec.describe()
    assert payload["lifecycle"] == "evaluated"
    assert payload["tiers"] == []
    assert payload["approved_tiers"] == []
    assert payload["promotion_evidence_bound"] is False


def test_status_schema_says_evaluated_is_not_production_eligible() -> None:
    spec = ModelSpec(
        id="candidate",
        provider="mock",
        lifecycle=PromotionState.EVALUATED,
    )
    payload = status_payload(spec, PromotionStore())
    assert payload["state"] == "evaluated"
    assert payload["production_eligible"] is False
    assert payload["note"] == "NOT PRODUCTION ELIGIBLE"
    for field in ("probe", "evaluation", "shadow", "approval", "history"):
        assert field in payload


def test_route_explain_uses_precise_promotion_rejection_reason() -> None:
    registry = ModelRegistry(
        [
            ModelSpec(
                id="candidate",
                provider="mock",
                lifecycle=PromotionState.PROBED,
            )
        ]
    )
    payload = (
        RoutePlanner(
            registry,
            require_approved=True,
            pin_requires_approved=True,
        )
        .plan(RouteRequest("candidate"))
        .describe()
    )
    assert payload["selected"] is None
    assert payload["rejected"][-1]["rule"] == "not_evaluated"
    assert payload["rejected"][-1]["detail"] == ("state = probed; production requires approved")
