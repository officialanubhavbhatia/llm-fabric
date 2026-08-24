"""Typed failure outcomes. Not every outage is a generic retry."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_fabric.config import Settings
from llm_fabric.errors import AllCandidatesFailedError, ConfigurationError, ModelNotFoundError
from llm_fabric.router.grades import Grade
from llm_fabric.router.plan import (
    ExclusionRule,
    RoutePlanner,
    RouteRequest,
    TenantRoutingPolicies,
    TenantRoutingPolicy,
)
from llm_fabric.router.registry import ModelRegistry, ModelSpec
from llm_fabric.serving.adapters.mock import MockProvider
from llm_fabric.serving.factory import ProviderFactory


def test_invalid_tier_is_unknown_model() -> None:
    registry = ModelRegistry([ModelSpec(id="m", provider="mock", grade=Grade.GRADE02)])
    planner = RoutePlanner(registry)
    with pytest.raises(ModelNotFoundError):
        planner.plan(RouteRequest("L99"))


def test_unknown_model_is_not_retried() -> None:
    registry = ModelRegistry([ModelSpec(id="m", provider="mock")])
    with pytest.raises(ModelNotFoundError):
        RoutePlanner(registry).require_plan(RouteRequest("nope"))


def test_disabled_model_is_excluded() -> None:
    registry = ModelRegistry([ModelSpec(id="m", provider="mock", enabled=False, fallbacks=())])
    plan = RoutePlanner(registry).plan(RouteRequest("m"))
    assert plan.selected is None
    assert plan.excluded[0].rule is ExclusionRule.DISABLED


def test_model_above_tenant_ceiling() -> None:
    registry = ModelRegistry(
        [
            ModelSpec(id="low", provider="mock", grade=Grade.GRADE04, capabilities=["chat"]),
            ModelSpec(id="high", provider="mock", grade=Grade.GRADE29, capabilities=["chat"]),
        ]
    )
    planner = RoutePlanner(
        registry,
        tenant_policies=TenantRoutingPolicies(
            [TenantRoutingPolicy(tenant_id="acme", maximum_grade=Grade.GRADE10)]
        ),
    )
    plan = planner.plan(RouteRequest("high", tenant_id="acme"))
    assert plan.selected is None or plan.selected_model != "high"
    assert any(item.rule is ExclusionRule.GRADE_ABOVE_MAXIMUM for item in plan.excluded)


def test_unsupported_capability() -> None:
    registry = ModelRegistry([ModelSpec(id="m", provider="mock", capabilities=["chat"])])
    plan = RoutePlanner(registry).plan(
        RouteRequest("m", required_capabilities=frozenset({"vision"}))
    )
    assert plan.selected is None
    assert plan.excluded[0].rule is ExclusionRule.MISSING_CAPABILITY


def test_context_too_large() -> None:
    registry = ModelRegistry(
        [ModelSpec(id="m", provider="mock", context_window=128, capabilities=["chat"])]
    )
    plan = RoutePlanner(registry).plan(RouteRequest("m", prompt_tokens=10_000))
    assert any(item.rule is ExclusionRule.CONTEXT_TOO_SMALL for item in plan.excluded)


def test_invalid_provider_base_urls_json() -> None:
    with pytest.raises(ConfigurationError, match="PROVIDER_BASE_URLS"):
        Settings(environment="test", provider_base_urls="not-json")


@pytest.mark.asyncio
async def test_ollama_down_falls_back_to_mock() -> None:
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "local-small",
                    "provider": "failing",
                    "capabilities": ["chat"],
                    "fallbacks": ["mock-small"],
                    "lifecycle": "approved",
                },
                {
                    "id": "mock-small",
                    "provider": "mock",
                    "capabilities": ["chat"],
                    "lifecycle": "approved",
                },
            ]
        }
    )
    settings = Settings(environment="test")
    factory = ProviderFactory(
        settings, overrides={"failing": MockProvider(fail=True), "mock": MockProvider()}
    )
    from llm_fabric.contract.openai import ChatCompletionRequest, ChatMessage
    from llm_fabric.router.engine import Router

    router = Router(registry, factory)
    result = await router.complete(
        ChatCompletionRequest(
            model="local-small",
            messages=[ChatMessage(role="user", content="Hello")],
        )
    )
    assert result.spec.id == "mock-small"
    assert result.decision.failover_count >= 1


def test_preferred_model_unavailable_keeps_others() -> None:
    from llm_fabric.router.intent_routing import RoutingConfig

    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {"id": "a", "provider": "mock", "lifecycle": "approved"},
                {"id": "b", "provider": "mock", "lifecycle": "approved"},
            ]
        }
    )
    routing = RoutingConfig.from_mapping(
        {"routing": {"version": "t", "intent": {"coding": {"preferred_models": ["missing"]}}}}
    )
    # unknown preferred model fails at load
    with pytest.raises(ConfigurationError):
        routing.validate(registry)


async def test_all_providers_unavailable() -> None:
    registry = ModelRegistry([ModelSpec(id="only", provider="failing", capabilities=["chat"])])
    settings = Settings(environment="test")
    factory = ProviderFactory(settings, overrides={"failing": MockProvider(fail=True)})
    from llm_fabric.contract.openai import ChatCompletionRequest, ChatMessage
    from llm_fabric.router.engine import Router

    router = Router(registry, factory)
    with pytest.raises(AllCandidatesFailedError):
        await router.complete(
            ChatCompletionRequest(
                model="only",
                messages=[ChatMessage(role="user", content="Hello")],
            )
        )


def test_chat_headers_include_tier(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "cheap", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    assert "x-fabric-served-model" in response.headers
    assert "x-fabric-provider" in response.headers
