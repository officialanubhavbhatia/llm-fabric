"""Intent → capability → preferred-tier policy. Configuration, not code.

IntentOS may supply a classification. This module never talks to a provider
and never grants authorization. When serving-path classification is off (the
default), nothing here runs on chat requests.

A preferred-tier list narrows the candidate set only when at least one eligible
deployment already occupies one of those tiers. Otherwise the planner keeps the
full filtered set rather than failing the request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from llm_fabric.errors import ConfigurationError
from llm_fabric.intent.schema import UNKNOWN_INTENT_ID, IntentClassification
from llm_fabric.router.capabilities import normalise
from llm_fabric.router.grades import Grade
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.router.tiers import ServiceTier


@dataclass(frozen=True, slots=True)
class IntentRoutePolicy:
    """What one intent asks of the capability planner."""

    intent_id: str
    preferred_tiers: tuple[ServiceTier, ...] = ()
    required_capabilities: frozenset[str] = frozenset()
    max_tier: ServiceTier | None = None
    preferred_models: tuple[str, ...] = ()

    @property
    def preferred_grades(self) -> frozenset[Grade]:
        return frozenset(tier.to_grade() for tier in self.preferred_tiers)

    @property
    def maximum_grade(self) -> Grade | None:
        return self.max_tier.to_grade() if self.max_tier is not None else None


@dataclass(frozen=True, slots=True)
class EscalationPolicy:
    """Bounds on how far a request may climb. Depth is attempts, not a jump to L30."""

    enabled: bool = True
    max_steps: int = 3


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """Operator routing policy loaded from YAML."""

    version: str = "unversioned"
    content_hash: str = ""
    default_tier: ServiceTier | None = None
    max_tier: ServiceTier | None = None
    escalation: EscalationPolicy = field(default_factory=EscalationPolicy)
    intent_policies: tuple[IntentRoutePolicy, ...] = ()

    def policy_for(self, intent_id: str) -> IntentRoutePolicy | None:
        for policy in self.intent_policies:
            if policy.intent_id == intent_id:
                return policy
        return None

    def for_classification(
        self, classification: IntentClassification | None
    ) -> IntentRoutePolicy | None:
        if classification is None or classification.abstain:
            return None
        if classification.intent_id == UNKNOWN_INTENT_ID:
            return None
        return self.policy_for(classification.intent_id) or self.policy_for(classification.domain)

    def policy_for_request(
        self,
        *,
        intent: IntentClassification | None = None,
        intent_id: str | None = None,
    ) -> IntentRoutePolicy | None:
        """Resolve a policy from a classification or an explicit intent id.

        An explicit id is how routing can be developed while serving-path
        IntentOS classification stays off. It is still not authorization.
        """
        classified = self.for_classification(intent)
        if classified is not None:
            return classified
        if intent_id:
            return self.policy_for(intent_id)
        return None

    @property
    def default_grade(self) -> Grade | None:
        return self.default_tier.to_grade() if self.default_tier is not None else None

    @property
    def maximum_grade(self) -> Grade | None:
        return self.max_tier.to_grade() if self.max_tier is not None else None

    @classmethod
    def empty(cls) -> RoutingConfig:
        return cls()

    @classmethod
    def from_mapping(
        cls, data: dict[str, Any], *, registry: ModelRegistry | None = None
    ) -> RoutingConfig:
        if not isinstance(data, dict):
            raise ConfigurationError("routing config must be a mapping")
        routing = data.get("routing")
        source: dict[str, Any] = dict(routing) if isinstance(routing, dict) else dict(data)

        default_tier = _optional_tier(source.get("default_tier"))
        max_tier = _optional_tier(source.get("max_tier"))
        raw_escalation = source.get("escalation") or {}
        if raw_escalation and not isinstance(raw_escalation, dict):
            raise ConfigurationError("routing.escalation must be a mapping")
        escalation = EscalationPolicy(
            enabled=bool(raw_escalation.get("enabled", True)),
            max_steps=int(raw_escalation.get("max_steps", 3)),
        )
        if escalation.max_steps < 1:
            raise ConfigurationError("routing.escalation.max_steps must be >= 1")

        intent_block = source.get("intent") or source.get("intent_policies") or {}
        policies = _intent_policies(intent_block)
        config = cls(
            version=str(source.get("version") or "unversioned"),
            content_hash=_policy_hash(source),
            default_tier=default_tier,
            max_tier=max_tier,
            escalation=escalation,
            intent_policies=policies,
        )
        if registry is not None:
            config.validate(registry)
        return config

    def validate(self, registry: ModelRegistry) -> None:
        """Fail fast when preferred_models name unknown deployments."""
        known = {spec.id for spec in registry.all_models()}
        for policy in self.intent_policies:
            for model_id in policy.preferred_models:
                if model_id not in known:
                    raise ConfigurationError(
                        f"intent policy '{policy.intent_id}' names unknown model '{model_id}'"
                    )

    @classmethod
    def from_yaml(cls, path: Path, *, registry: ModelRegistry | None = None) -> RoutingConfig:
        if not path.exists():
            return cls.empty()
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"routing config at {path} must be a mapping")
        return cls.from_mapping(data, registry=registry)


def _policy_hash(source: dict[str, Any]) -> str:
    """Stable identity of the operator policy. Not a secret; truncated SHA-256."""
    blob = json.dumps(source, sort_keys=True, default=str, separators=(",", ":"))
    return sha256(blob.encode("utf-8")).hexdigest()[:16]


def _optional_tier(value: object) -> ServiceTier | None:
    if value is None or value == "":
        return None
    return ServiceTier.parse(str(value))


def _intent_policies(block: object) -> tuple[IntentRoutePolicy, ...]:
    if not block:
        return ()
    if not isinstance(block, dict):
        raise ConfigurationError("routing.intent must be a mapping of intent id → policy")
    policies: list[IntentRoutePolicy] = []
    for intent_id, raw in block.items():
        if not isinstance(raw, dict):
            raise ConfigurationError(f"routing.intent.{intent_id} must be a mapping")
        tiers_raw = raw.get("tiers") or raw.get("preferred_tiers") or []
        if isinstance(tiers_raw, str):
            tiers_raw = [tiers_raw]
        preferred = tuple(ServiceTier.parse(str(item)) for item in tiers_raw)
        capabilities = normalise(raw.get("required_capabilities") or raw.get("capabilities"))
        max_tier = _optional_tier(raw.get("max_tier"))
        preferred_models = tuple(str(item) for item in (raw.get("preferred_models") or ()))
        policies.append(
            IntentRoutePolicy(
                intent_id=str(intent_id),
                preferred_tiers=preferred,
                required_capabilities=capabilities,
                max_tier=max_tier,
                preferred_models=preferred_models,
            )
        )
    return tuple(policies)
