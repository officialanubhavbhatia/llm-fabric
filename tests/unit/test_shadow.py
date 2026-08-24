"""Shadow routing regret stays None when a side is unmeasured."""

from __future__ import annotations

from llm_fabric.config import Settings
from llm_fabric.contract.openai import ChatCompletionRequest, ChatMessage
from llm_fabric.eval.shadow import shadow_plan
from llm_fabric.router.engine import Router
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider
from llm_fabric.serving.factory import ProviderFactory


def test_shadow_plan_records_route_regret() -> None:
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "cheap",
                    "provider": "mock",
                    "provider_model": "cheap-v1",
                    "input_cost_per_mtok": 0.1,
                    "output_cost_per_mtok": 0.2,
                    "capabilities": ["chat"],
                },
                {
                    "id": "premium",
                    "provider": "mock",
                    "provider_model": "premium-v1",
                    "input_cost_per_mtok": 3.0,
                    "output_cost_per_mtok": 9.0,
                    "capabilities": ["chat"],
                },
            ],
            "aliases": [{"id": "auto", "policy": "cheapest", "candidates": ["premium", "cheap"]}],
        }
    )
    router = Router(
        registry,
        ProviderFactory(Settings(), overrides={"mock": MockProvider()}),
        default_policy="cost_first",
    )
    plan = router.preview(
        ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="hi")])
    )
    outcome = shadow_plan(plan, expected_model="premium", prompt_tokens=100)
    assert outcome.chosen == "cheap"
    assert outcome.route_regret == 1.0
    assert "premium" in outcome.alternatives
    assert outcome.cost_regret is not None
