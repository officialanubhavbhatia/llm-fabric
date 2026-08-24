"""The route planner: from a request to an auditable decision.

The constitution names thirteen inputs a route planner takes and seven outputs it
produces, and requires that no decision be hidden — every one must be explainable
from stored features. This module is that requirement made literal. Planning has
two halves:

**Filtering**, which decides who is *allowed* to serve. Every rejection is
recorded with the rule that rejected it, so "why did it not pick the cheap one"
always has an answer. Filters are hard constraints: a tenant policy or a locality
requirement is never traded off against a good score.

**Scoring**, which decides who *should*. Handled by `router.policy`, which
returns the per-feature arithmetic rather than only the winner.

Three of the constitution's inputs have no source in this build, and the planner
says so rather than quietly scoring without them. Historical quality needs the
routing evals, which are not built. Cache probability needs a response cache,
which is not built. Both appear in the inputs report as unavailable, and neither
is imputed.

Tenant policy can only ever *narrow* the candidate set. A tenant that requires
in-house inference cannot have that widened by a request, an alias, or an intent
classification, because a privacy constraint that a caller can argue its way out
of is not a constraint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from llm_fabric.errors import ConfigurationError, ModelNotFoundError, NoCandidateError
from llm_fabric.intent.schema import (
    UNKNOWN_INTENT_ID,
    CostClass,
    IntentClassification,
    LatencyClass,
    Modality,
    QualityClass,
)
from llm_fabric.router.capabilities import Capability
from llm_fabric.router.fallback import (
    ANY_REASON,
    FallbackBudget,
    FallbackEdge,
    FallbackGraph,
    FallbackReason,
)
from llm_fabric.router.grades import Grade
from llm_fabric.router.health import HealthSnapshot, HealthTracker
from llm_fabric.router.policy import (
    POLICY_WEIGHTS,
    PolicyWeights,
    RoutePolicy,
    ScoredCandidate,
    ScoringInputs,
    ScoringResult,
    parse_policy,
    permitted_localities,
    score_candidates,
)
from llm_fabric.router.registry import Locality, ModelRegistry, ModelSpec

#: Which declared quality dimension matters for which intent domain.
#: A judgement about the taxonomy, not a measurement: it says "a coding prompt
#: should be ranked on the coding score", which is obvious, and stops there.
INTENT_QUALITY_DIMENSIONS: dict[str, str] = {
    "coding": "coding",
    "agent": "agent",
    "reasoning": "reasoning",
    "math": "math",
    "rag": "rag",
    "research": "rag",
    "tool_use": "tool_use",
    "extraction": "structured_output",
    "classification": "structured_output",
    "data_analysis": "reasoning",
}

#: Which policy an intent implies when the caller expressed no preference. Also
#: a judgement: a prompt classified as needing maximum quality is routed
#: quality-first, and one classified realtime is routed latency-first.
INTENT_POLICY_HINTS: tuple[tuple[str, RoutePolicy], ...] = (
    ("quality", RoutePolicy.QUALITY_FIRST),
    ("latency", RoutePolicy.LATENCY_FIRST),
    ("cost", RoutePolicy.COST_FIRST),
)


class ExclusionRule(StrEnum):
    """Why a candidate was removed. A closed vocabulary, so exclusions are countable."""

    DISABLED = "disabled"
    MODEL_DENIED = "denied_by_tenant_policy"
    PROVIDER_NOT_PERMITTED = "provider_not_permitted"
    LOCALITY_NOT_PERMITTED = "locality_not_permitted"
    GRADE_BELOW_MINIMUM = "grade_below_minimum"
    MISSING_CAPABILITY = "missing_capability"
    CONTEXT_TOO_SMALL = "context_too_small"
    CIRCUIT_OPEN = "circuit_open"
    TRAFFIC_SHIFTED = "traffic_shifted"
    OVER_BUDGET = "over_budget"
    LATENCY_SLO_MISSED = "latency_slo_missed"


class TrafficGate(Protocol):
    """Operational overlay that can remove a model from the candidate set."""

    def excludes(self, model_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class Exclusion:
    """One candidate, and the rule that removed it."""

    model_id: str
    rule: ExclusionRule
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "rule": self.rule.value,
            "detail": self.detail or None,
        }


@dataclass(frozen=True, slots=True)
class TenantRoutingPolicy:
    """Per-tenant routing constraints. Narrowing only, never widening."""

    tenant_id: str
    policy: RoutePolicy | None = None
    allowed_localities: frozenset[Locality] | None = None
    allowed_providers: frozenset[str] | None = None
    denied_models: frozenset[str] = frozenset()
    minimum_grade: Grade | None = None
    max_cost_per_request_usd: float | None = None
    require_in_house: bool = False
    fallback_budget: FallbackBudget | None = None

    def localities(self) -> frozenset[Locality] | None:
        """The localities this tenant permits, honouring `require_in_house`."""
        in_house = frozenset({Locality.LOCAL, Locality.PRIVATE})
        if self.require_in_house:
            return in_house & self.allowed_localities if self.allowed_localities else in_house
        return self.allowed_localities

    def as_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "policy": self.policy.value if self.policy else None,
            "allowed_localities": (
                sorted(item.value for item in self.allowed_localities)
                if self.allowed_localities
                else None
            ),
            "allowed_providers": (
                sorted(self.allowed_providers) if self.allowed_providers else None
            ),
            "denied_models": sorted(self.denied_models),
            "minimum_grade": self.minimum_grade.value if self.minimum_grade else None,
            "max_cost_per_request_usd": self.max_cost_per_request_usd,
            "require_in_house": self.require_in_house,
        }


class TenantRoutingPolicies:
    """Tenant-scoped routing policy, looked up by tenant id.

    In-memory like every other store in this build, and lost on restart. A tenant
    with no policy gets `None`, which means "no tenant-level narrowing" — not a
    default policy that might accidentally be more permissive than intended.
    """

    def __init__(self, policies: Sequence[TenantRoutingPolicy] = ()) -> None:
        self._policies: dict[str, TenantRoutingPolicy] = {p.tenant_id: p for p in policies}

    def get(self, tenant_id: str | None) -> TenantRoutingPolicy | None:
        if not tenant_id:
            return None
        return self._policies.get(tenant_id)

    def set(self, policy: TenantRoutingPolicy) -> None:
        self._policies[policy.tenant_id] = policy

    def remove(self, tenant_id: str) -> None:
        self._policies.pop(tenant_id, None)

    def __len__(self) -> int:
        return len(self._policies)


@dataclass(frozen=True, slots=True)
class RouteRequest:
    """Everything the planner may consider about one request."""

    requested_model: str
    tenant_id: str | None = None
    intent: IntentClassification | None = None
    policy: RoutePolicy | None = None
    required_capabilities: frozenset[str] = frozenset()
    minimum_grade: Grade | None = None
    prompt_tokens: int = 0
    max_output_tokens: int | None = None
    latency_slo_ms: float | None = None
    budget_usd: float | None = None

    #: Probability that a response cache would serve this request. Always `None`
    #: today: no response cache exists to estimate it. Present as the seam.
    cache_probability: float | None = None

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0:
            raise ConfigurationError("prompt_tokens cannot be negative")
        if self.max_output_tokens is not None and self.max_output_tokens < 0:
            raise ConfigurationError("max_output_tokens cannot be negative")

    @property
    def context_requirement(self) -> int:
        """Tokens the deployment must be able to hold: prompt plus reserved output."""
        return self.prompt_tokens + (self.max_output_tokens or 0)


@dataclass(frozen=True, slots=True)
class InputAvailability:
    """One of the constitution's planner inputs, and whether it was available."""

    name: str
    available: bool
    value: Any = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "input": self.name,
            "available": self.available,
            "value": self.value,
            "note": self.note or None,
        }


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """The auditable decision object. Every route produces one."""

    requested_model: str
    policy: RoutePolicy
    requested_policy: str
    selected: ModelSpec | None
    ranked: tuple[ScoredCandidate, ...]
    excluded: tuple[Exclusion, ...]
    scoring: ScoringResult
    graph: FallbackGraph
    budget: FallbackBudget
    inputs: tuple[InputAvailability, ...]
    tenant_policy: TenantRoutingPolicy | None = None
    notes: tuple[str, ...] = ()

    @property
    def selected_model(self) -> str | None:
        return self.selected.id if self.selected else None

    @property
    def chain(self) -> tuple[str, ...]:
        """Candidate ids in the order the engine will try them."""
        return tuple(candidate.spec.id for candidate in self.ranked)

    @property
    def routing_score(self) -> float | None:
        return self.ranked[0].score if self.ranked and self.scoring.used_features else None

    def _selected_feature(self, name: str) -> float | None:
        if not self.ranked:
            return None
        value = self.ranked[0].feature(name)
        return value.raw if value and value.usable else None

    @property
    def expected_quality(self) -> float | None:
        """Declared quality of the selected deployment. An estimate, not a prediction."""
        return self._selected_feature("quality")

    @property
    def expected_latency_ms(self) -> float | None:
        return self._selected_feature("latency")

    def expected_cost_usd(self, prompt_tokens: int, output_tokens: int) -> float | None:
        if self.selected is None or not self.selected.is_priced:
            return None
        return self.selected.cost_usd(prompt_tokens, output_tokens)

    def explain(self) -> tuple[str, ...]:
        """The decision in sentences, for logs and for the preview API."""
        lines: list[str] = []
        if self.selected is None:
            lines.append(f"No deployment could serve '{self.requested_model}'.")
        else:
            lines.append(
                f"Selected '{self.selected.id}' on provider '{self.selected.provider}' "
                f"under policy '{self.policy.value}'."
            )
        if self.requested_policy and self.requested_policy != self.policy.value:
            lines.append(f"Policy '{self.requested_policy}' resolved to '{self.policy.value}'.")
        if self.scoring.fell_back_to_registry_order:
            lines.append(
                "No feature was usable, so candidates kept registry order and the score "
                "means nothing."
            )
        elif self.scoring.used_features:
            lines.append(f"Ranked on {', '.join(self.scoring.used_features)}.")
        for feature, why in self.scoring.dropped_features:
            lines.append(f"Dropped '{feature}': {why}.")
        for exclusion in self.excluded:
            lines.append(
                f"Excluded '{exclusion.model_id}' ({exclusion.rule})"
                + (f": {exclusion.detail}" if exclusion.detail else "")
                + "."
            )
        lines.extend(self.notes)
        return tuple(lines)

    def describe(self) -> dict[str, Any]:
        """The whole decision as JSON, for `/v1/routes/preview`."""
        return {
            "requested_model": self.requested_model,
            "requested_policy": self.requested_policy,
            "policy": self.policy.value,
            "selected": self.selected.describe() if self.selected else None,
            "routing_score": (
                round(self.routing_score, 6) if self.routing_score is not None else None
            ),
            "expected": {
                "quality": self.expected_quality,
                "latency_ms": self.expected_latency_ms,
                "note": (
                    "estimates derived from declared registry attributes and observed "
                    "health; not measurements of this request"
                ),
            },
            "chain": list(self.chain),
            "scoring": self.scoring.as_dict(),
            "excluded": [exclusion.as_dict() for exclusion in self.excluded],
            "fallback": {
                "graph": self.graph.describe(root=self.selected_model),
                "budget": self.budget.as_dict(),
            },
            "inputs": [item.as_dict() for item in self.inputs],
            "tenant_policy": self.tenant_policy.as_dict() if self.tenant_policy else None,
            "explanation": list(self.explain()),
        }


class RoutePlanner:
    """Turns a `RouteRequest` into a `RoutePlan`."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        health: HealthTracker | None = None,
        tenant_policies: TenantRoutingPolicies | None = None,
        default_policy: str = "cost_first",
        fallback_budget: FallbackBudget | None = None,
        graph: FallbackGraph | None = None,
        traffic: TrafficGate | None = None,
    ) -> None:
        self._registry = registry
        self._health = health or HealthTracker()
        self._tenants = tenant_policies or TenantRoutingPolicies()
        self._default_policy = parse_policy(default_policy)
        self._budget = fallback_budget or FallbackBudget()
        self._declared_graph = graph or self._graph_from_registry(registry)
        self._traffic = traffic

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def health(self) -> HealthTracker:
        return self._health

    @property
    def tenant_policies(self) -> TenantRoutingPolicies:
        return self._tenants

    @staticmethod
    def _graph_from_registry(registry: ModelRegistry) -> FallbackGraph:
        """Read `fallbacks:` lists as reason-agnostic edges.

        A declared list answers every failure the same way, which is exactly the
        flat behaviour the constitution warns about. It is honoured because
        operators have written it, and the planner adds reason-specific structure
        on top rather than replacing it.
        """
        return FallbackGraph(
            FallbackEdge(source=spec.id, target=target, reasons=ANY_REASON)
            for spec in registry.all_models()
            for target in spec.fallbacks
        )

    # -- policy resolution ---------------------------------------------------

    def _resolve_policy(
        self,
        request: RouteRequest,
        alias_policy: str | None,
        tenant: TenantRoutingPolicy | None,
    ) -> tuple[RoutePolicy, str, list[str]]:
        """Decide the policy, and say where it came from.

        Precedence, strongest first: the tenant's pinned policy, then an explicit
        request policy, then the alias, then what the intent implies, then the
        default. The tenant wins because a tenant policy is an administrative
        decision about the tenant's own traffic.
        """
        notes: list[str] = []
        if tenant and tenant.policy:
            notes.append(f"Policy pinned to '{tenant.policy.value}' by tenant policy.")
            return tenant.policy, tenant.policy.value, notes
        if request.policy:
            return request.policy, request.policy.value, notes
        if alias_policy:
            parsed_alias = parse_policy(alias_policy)
            if parsed_alias is RoutePolicy.DECLARED:
                return parsed_alias, alias_policy, notes
        if request.intent is not None and (
            request.intent.abstain
            or request.intent.intent_id == UNKNOWN_INTENT_ID
            or request.intent.confidence < 0.50
        ):
            notes.append(
                "IntentOS abstained or was low-confidence; "
                "using balanced as the capability floor instead of cheapest."
            )
            return RoutePolicy.BALANCED, RoutePolicy.BALANCED.value, notes
        if alias_policy:
            return parse_policy(alias_policy), alias_policy, notes
        if inferred := self._policy_from_intent(request.intent):
            notes.append(f"Policy '{inferred.value}' inferred from the intent classification.")
            return inferred, inferred.value, notes
        return self._default_policy, self._default_policy.value, notes

    @staticmethod
    def _policy_from_intent(intent: IntentClassification | None) -> RoutePolicy | None:
        """Map a classification onto a policy, when it says something decisive.

        An abstaining classification says nothing decisive by definition, so it
        never selects a policy: routing an unrecognised prompt to the most
        expensive model because "quality" seemed safe would be the wrong default.
        """
        if intent is None or intent.abstain or intent.intent_id == UNKNOWN_INTENT_ID:
            return None
        if intent.quality_class is QualityClass.MAXIMUM:
            return RoutePolicy.QUALITY_FIRST
        if intent.latency_class is LatencyClass.REALTIME:
            return RoutePolicy.LATENCY_FIRST
        if intent.cost_class is CostClass.MINIMAL:
            return RoutePolicy.COST_FIRST
        return None

    @staticmethod
    def _capabilities_for(request: RouteRequest) -> frozenset[str]:
        required = set(request.required_capabilities)
        intent = request.intent
        if intent is not None and not intent.abstain:
            required |= set(intent.required_capabilities)
            if intent.structured_output:
                required.add(Capability.STRUCTURED_OUTPUT)
            if intent.tools_required or intent.agent_required:
                required.add(Capability.TOOLS)
            if intent.modality in (Modality.IMAGE, Modality.MULTIMODAL):
                required.add(Capability.VISION)
        return frozenset(required)

    @staticmethod
    def _quality_dimension(intent: IntentClassification | None) -> str | None:
        if intent is None or intent.abstain:
            return None
        return INTENT_QUALITY_DIMENSIONS.get(intent.domain)

    # -- candidate resolution ------------------------------------------------

    def _candidates(
        self, request: RouteRequest
    ) -> tuple[list[ModelSpec], str | None, Grade | None]:
        """The starting set: an alias's candidates, or a pinned model and its fallbacks."""
        requested = request.requested_model
        if alias := self._registry.alias(requested):
            candidates = [self._registry.get(candidate) for candidate in alias.candidates]
            return candidates, alias.policy, alias.minimum_grade

        if not self._registry.known(requested):
            raise ModelNotFoundError(f"unknown model '{requested}'")

        primary = self._registry.get(requested)
        chain = [primary]
        chain.extend(self._registry.get(target) for target in primary.fallbacks)
        # A pinned model is honoured as pinned: the caller named it, so ranking
        # would override the choice they made.
        return chain, RoutePolicy.DECLARED.value, None

    def plan(self, request: RouteRequest) -> RoutePlan:
        tenant = self._tenants.get(request.tenant_id)
        candidates, alias_policy, alias_grade = self._candidates(request)
        policy, requested_policy, notes = self._resolve_policy(request, alias_policy, tenant)

        alias = self._registry.alias(request.requested_model)
        hard_required = set(request.required_capabilities)
        if alias is not None:
            hard_required |= alias.requires
        intent_required = set(self._capabilities_for(request))
        extras = intent_required - hard_required
        required = frozenset(hard_required | extras)

        minimum_grade = _strictest_grade(
            request.minimum_grade, alias_grade, tenant.minimum_grade if tenant else None
        )
        localities = _intersect_localities(
            permitted_localities(policy), tenant.localities() if tenant else None
        )

        eligible, excluded = self._filter(
            candidates,
            request=request,
            required=required,
            minimum_grade=minimum_grade,
            localities=localities,
            tenant=tenant,
        )
        if not eligible and extras:
            notes.append(
                "IntentOS capability extras left no candidate; "
                "routing with the request's hard requirements only."
            )
            required = frozenset(hard_required)
            eligible, excluded = self._filter(
                candidates,
                request=request,
                required=required,
                minimum_grade=minimum_grade,
                localities=localities,
                tenant=tenant,
            )

        health = {
            spec.deployment_id: self._health.snapshot(spec.deployment_id) for spec in eligible
        }
        scoring = score_candidates(
            eligible,
            policy=policy,
            inputs=ScoringInputs(
                health=health,
                quality_dimension=self._quality_dimension(request.intent),
                expected_output_tokens=request.max_output_tokens or 0,
            ),
            weights=self._weights_for(policy, tenant),
        )

        selected = scoring.candidates[0].spec if scoring.candidates else None
        chain = [candidate.spec.id for candidate in scoring.candidates]
        graph = self._graph_for(chain)
        budget = tenant.fallback_budget if tenant and tenant.fallback_budget else self._budget

        return RoutePlan(
            requested_model=request.requested_model,
            policy=policy,
            requested_policy=requested_policy,
            selected=selected,
            ranked=scoring.candidates,
            excluded=tuple(excluded),
            scoring=scoring,
            graph=graph,
            budget=budget,
            inputs=self._inputs_report(request, tenant, health, required, minimum_grade),
            tenant_policy=tenant,
            notes=tuple(notes),
        )

    def require_plan(self, request: RouteRequest) -> RoutePlan:
        """Plan, refusing to return one that selects nothing."""
        plan = self.plan(request)
        if plan.selected is None:
            raise NoCandidateError(
                f"no deployment can serve '{request.requested_model}': "
                + ("; ".join(f"{e.model_id} {e.rule}" for e in plan.excluded) or "no candidates")
            )
        return plan

    @staticmethod
    def _weights_for(
        policy: RoutePolicy, tenant: TenantRoutingPolicy | None
    ) -> PolicyWeights | None:
        del tenant  # Per-tenant weight overrides are not configurable yet.
        return POLICY_WEIGHTS.get(policy)

    def _graph_for(self, chain: Sequence[str]) -> FallbackGraph:
        """The fallback graph for this decision.

        Declared edges between eligible candidates come first, because an
        operator saying "when this is overloaded, use that" is more informative
        than position in a ranking. The ranked order then supplies linear edges
        for reasons the declared graph does not answer, so a fallback always
        exists even with no configuration at all.
        """
        graph = self._declared_graph.restricted_to(chain)
        declared_pairs = {(edge.source, edge.target) for edge in graph.edges}
        answered: dict[str, set[FallbackReason]] = {}
        for edge in graph.edges:
            answered.setdefault(edge.source, set()).update(edge.reasons)

        for index, source in enumerate(chain[:-1]):
            unanswered = ANY_REASON - answered.get(source, set())
            if not unanswered:
                continue
            for target in chain[index + 1 :]:
                if (source, target) in declared_pairs:
                    continue
                graph.add(
                    FallbackEdge(
                        source=source,
                        target=target,
                        reasons=frozenset(unanswered),
                        note="ranked order",
                    )
                )
                break
        return graph

    # -- filtering -----------------------------------------------------------

    def _filter(
        self,
        candidates: Sequence[ModelSpec],
        *,
        request: RouteRequest,
        required: frozenset[str],
        minimum_grade: Grade | None,
        localities: frozenset[Locality] | None,
        tenant: TenantRoutingPolicy | None,
    ) -> tuple[list[ModelSpec], list[Exclusion]]:
        eligible: list[ModelSpec] = []
        excluded: list[Exclusion] = []

        for spec in candidates:
            if not spec.enabled:
                excluded.append(Exclusion(spec.id, ExclusionRule.DISABLED))
                continue

            if self._traffic is not None and self._traffic.excludes(spec.id):
                excluded.append(
                    Exclusion(
                        spec.id,
                        ExclusionRule.TRAFFIC_SHIFTED,
                        "removed by a traffic-shift remediation",
                    )
                )
                continue

            if tenant and spec.id in tenant.denied_models:
                excluded.append(
                    Exclusion(
                        spec.id,
                        ExclusionRule.MODEL_DENIED,
                        "on this tenant's deny list",
                    )
                )
                continue

            if (
                tenant
                and tenant.allowed_providers
                and spec.provider not in tenant.allowed_providers
            ):
                excluded.append(
                    Exclusion(
                        spec.id,
                        ExclusionRule.PROVIDER_NOT_PERMITTED,
                        f"provider '{spec.provider}' is not in the tenant's allow-list",
                    )
                )
                continue

            if localities is not None and spec.locality not in localities:
                excluded.append(
                    Exclusion(
                        spec.id,
                        ExclusionRule.LOCALITY_NOT_PERMITTED,
                        f"locality '{spec.locality.value}' is not among "
                        f"{sorted(item.value for item in localities)}",
                    )
                )
                continue

            if minimum_grade is not None and (
                spec.grade is None or spec.grade.ordinal < minimum_grade.ordinal
            ):
                excluded.append(
                    Exclusion(
                        spec.id,
                        ExclusionRule.GRADE_BELOW_MINIMUM,
                        f"grade {spec.grade.value if spec.grade else 'undeclared'} "
                        f"is below {minimum_grade.value}",
                    )
                )
                continue

            if missing := spec.capabilities.missing(required):
                excluded.append(
                    Exclusion(spec.id, ExclusionRule.MISSING_CAPABILITY, f"lacks {sorted(missing)}")
                )
                continue

            requirement = request.context_requirement
            if requirement and not spec.fits_context(requirement):
                excluded.append(
                    Exclusion(
                        spec.id,
                        ExclusionRule.CONTEXT_TOO_SMALL,
                        f"needs {requirement} tokens, holds {spec.usable_context_tokens}",
                    )
                )
                continue

            if not self._health.admits(spec.deployment_id):
                snapshot = self._health.snapshot(spec.deployment_id)
                excluded.append(
                    Exclusion(
                        spec.id,
                        ExclusionRule.CIRCUIT_OPEN,
                        f"circuit is {snapshot.state.value}"
                        + (f", queue depth {snapshot.queue_depth}" if snapshot.queue_depth else ""),
                    )
                )
                continue

            if reason := self._over_budget(spec, request, tenant):
                excluded.append(Exclusion(spec.id, ExclusionRule.OVER_BUDGET, reason))
                continue

            if reason := self._misses_slo(spec, request):
                excluded.append(Exclusion(spec.id, ExclusionRule.LATENCY_SLO_MISSED, reason))
                continue

            eligible.append(spec)

        return eligible, excluded

    def _over_budget(
        self, spec: ModelSpec, request: RouteRequest, tenant: TenantRoutingPolicy | None
    ) -> str | None:
        """Exclude only when the price is known to breach the ceiling.

        An unpriced deployment is never excluded for cost, because "no declared
        price" is not evidence of being cheap or expensive. The plan records the
        absence instead.
        """
        ceiling = _tightest(request.budget_usd, tenant.max_cost_per_request_usd if tenant else None)
        if ceiling is None or not spec.is_priced:
            return None
        estimate = spec.cost_usd(request.prompt_tokens, request.max_output_tokens or 0)
        if estimate > ceiling:
            return f"estimated ${estimate:.6f} exceeds the ${ceiling:.6f} budget"
        return None

    def _misses_slo(self, spec: ModelSpec, request: RouteRequest) -> str | None:
        """Exclude only when a known latency breaches a stated SLO."""
        if request.latency_slo_ms is None:
            return None
        snapshot = self._health.snapshot(spec.deployment_id)
        observed = snapshot.ewma_latency_ms if snapshot.has_signal else None
        estimate = observed or spec.performance.estimated_total_ms(request.max_output_tokens or 0)
        if estimate is None:
            return None
        if estimate > request.latency_slo_ms:
            source = "observed" if observed is not None else "declared"
            return (
                f"{source} latency {estimate:.0f}ms exceeds the {request.latency_slo_ms:.0f}ms SLO"
            )
        return None

    # -- reporting -----------------------------------------------------------

    def _inputs_report(
        self,
        request: RouteRequest,
        tenant: TenantRoutingPolicy | None,
        health: Mapping[str, HealthSnapshot],
        required: frozenset[str],
        minimum_grade: Grade | None,
    ) -> tuple[InputAvailability, ...]:
        """Each planner input the constitution names, and whether it was available."""
        observed = [snapshot for snapshot in health.values() if snapshot.has_signal]
        intent = request.intent
        return (
            InputAvailability(
                "intent",
                intent is not None,
                intent.intent_id if intent else None,
                "" if intent else "no classification was supplied with the request",
            ),
            InputAvailability(
                "tenant_policy", tenant is not None, tenant.tenant_id if tenant else None
            ),
            InputAvailability(
                "quality_requirement",
                intent is not None and not intent.abstain,
                intent.quality_class.value if intent and not intent.abstain else None,
            ),
            InputAvailability(
                "latency_slo", request.latency_slo_ms is not None, request.latency_slo_ms
            ),
            InputAvailability("budget", request.budget_usd is not None, request.budget_usd),
            InputAvailability(
                "context_requirement",
                request.context_requirement > 0,
                request.context_requirement or None,
            ),
            InputAvailability("model_capabilities", True, sorted(required) or None),
            InputAvailability(
                "provider_health",
                bool(observed),
                len(observed) or None,
                "" if observed else "no attempt has been observed yet",
            ),
            InputAvailability(
                "queue_pressure",
                bool(observed),
                sum(snapshot.queue_depth for snapshot in observed) if observed else None,
            ),
            InputAvailability(
                "historical_quality",
                False,
                None,
                "not built: requires the routing evals, which do not exist in this build",
            ),
            InputAvailability(
                "historical_latency",
                bool(observed),
                None,
                "" if observed else "no attempt has been observed yet",
            ),
            InputAvailability(
                "historical_error_rate",
                bool(observed),
                None,
                "" if observed else "no attempt has been observed yet",
            ),
            InputAvailability(
                "cache_probability",
                False,
                None,
                "not built: requires a response cache, which does not exist in this build",
            ),
            InputAvailability(
                "minimum_grade",
                minimum_grade is not None,
                minimum_grade.value if minimum_grade else None,
            ),
        )


def _strictest_grade(*grades: Grade | None) -> Grade | None:
    """The highest floor among those given. A tighter constraint always wins."""
    present = [grade for grade in grades if grade is not None]
    return max(present, key=lambda grade: grade.ordinal) if present else None


def _tightest(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _intersect_localities(
    *sets: frozenset[Locality] | None,
) -> frozenset[Locality] | None:
    """Intersect locality constraints. Every source can only narrow."""
    present = [item for item in sets if item is not None]
    if not present:
        return None
    result = present[0]
    for item in present[1:]:
        result &= item
    return result


__all__ = [
    "Exclusion",
    "ExclusionRule",
    "InputAvailability",
    "RoutePlan",
    "RoutePlanner",
    "RouteRequest",
    "TenantRoutingPolicies",
    "TenantRoutingPolicy",
]
