"""Candidate ordering policies.

A policy takes the candidate models permitted for a request and puts them in the
order the router should try them. It never decides *whether* a model is eligible
— that filtering happens in the engine — so policies stay pure and testable.

Two policies exist. A latency-aware policy is deliberately absent: ordering by
latency requires per-backend latency measurement, and the fabric does not yet
collect it. Adding the policy before the measurement would mean ranking on
numbers that do not exist.
"""

from __future__ import annotations

from collections.abc import Callable

from llm_fabric.errors import ConfigurationError
from llm_fabric.router.registry import ModelSpec

Policy = Callable[[list[ModelSpec]], list[ModelSpec]]


def cheapest(candidates: list[ModelSpec]) -> list[ModelSpec]:
    """Cheapest blended registry price first; ties broken by id for determinism."""
    return sorted(candidates, key=lambda spec: (spec.blended_cost_per_mtok, spec.id))


def declared(candidates: list[ModelSpec]) -> list[ModelSpec]:
    """Registry order, unchanged. The operator's preference wins."""
    return list(candidates)


POLICIES: dict[str, Policy] = {
    "cheapest": cheapest,
    "declared": declared,
}


def get_policy(name: str) -> Policy:
    try:
        return POLICIES[name]
    except KeyError:
        raise ConfigurationError(
            f"unknown routing policy '{name}' (available: {sorted(POLICIES)})"
        ) from None
