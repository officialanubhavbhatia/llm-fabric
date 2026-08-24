"""The routing layer: decide which model on which backend serves a request."""

from llm_fabric.router.capabilities import Capability, CapabilityVector
from llm_fabric.router.engine import Attempt, RouteDecision, RoutedResult, Router
from llm_fabric.router.fallback import (
    FallbackBudget,
    FallbackEdge,
    FallbackGraph,
    FallbackReason,
)
from llm_fabric.router.grades import Grade
from llm_fabric.router.health import BreakerPolicy, BreakerState, HealthTracker
from llm_fabric.router.intent_routing import IntentRoutePolicy, RoutingConfig
from llm_fabric.router.plan import (
    RoutePlan,
    RoutePlanner,
    RouteRequest,
    TenantRoutingPolicies,
    TenantRoutingPolicy,
)
from llm_fabric.router.policy import RoutePolicy
from llm_fabric.router.registry import Alias, Locality, ModelRegistry, ModelSpec
from llm_fabric.router.tiers import ServiceTier

__all__ = [
    "Alias",
    "Attempt",
    "BreakerPolicy",
    "BreakerState",
    "Capability",
    "CapabilityVector",
    "FallbackBudget",
    "FallbackEdge",
    "FallbackGraph",
    "FallbackReason",
    "Grade",
    "HealthTracker",
    "IntentRoutePolicy",
    "Locality",
    "ModelRegistry",
    "ModelSpec",
    "RouteDecision",
    "RoutePlan",
    "RoutePlanner",
    "RoutePolicy",
    "RouteRequest",
    "RoutedResult",
    "Router",
    "RoutingConfig",
    "ServiceTier",
    "TenantRoutingPolicies",
    "TenantRoutingPolicy",
]
