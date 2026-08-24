"""Route policies: how candidates are ordered, and why.

A policy is a set of weights over four features — quality, latency, cost and
health — plus, for the locality policies, a hard filter. Ordering is the
weighted sum, and every term that went into it is returned alongside the score so
the decision can be explained rather than asserted.

Three rules keep the scoring honest.

**A feature nobody can supply is dropped, not imputed.** If any eligible
candidate lacks a declared quality or latency score, that feature is dropped for
the whole decision and the remaining weights are renormalised. Scoring the
others on it would rank a measured deployment against an unmeasured one and call
the result a comparison. The explanation names the feature and the candidate
that caused it to be dropped.

**Cost is the exception.** API prices have three states: unknown (omitted),
known-zero (`0.0`, typical for self-hosted), and known-nonzero. An unknown price
is not treated as free, and it does **not** erase cost ranking for candidates
whose prices *are* known. Cost is scored only among deployments with comparable
known API prices. Unknown-cost candidates receive no cost contribution and their
remaining weights are not renormalised to fill that slot, so they cannot win a
`cost_first` decision by looking free. A known-zero price *is* a real value and
ranks as the cheapest API cost.

**When no feature survives, ordering falls back to registry order** and says so.
That is the honest end state of a decision with nothing to decide on.

The weights below are **judgements, not fitted values**. Nothing in this
repository has measured whether 0.70 is the right weight for quality under
`quality_first`; they were chosen so each policy visibly does what its name says.
Treat them as defaults to be tuned against real traffic, and note that tuning
them requires the routing evals the constitution specifies, which are not built.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from llm_fabric.errors import ConfigurationError
from llm_fabric.router.health import HealthSnapshot
from llm_fabric.router.registry import Locality, ModelSpec

#: The four features every policy weighs.
FEATURES: tuple[str, ...] = ("quality", "latency", "cost", "health")

#: Output length assumed when estimating latency from a declared per-token cost.
#: A planning assumption, overridden by the request's `max_tokens` when it sets one.
DEFAULT_EXPECTED_OUTPUT_TOKENS = 256


class FeatureSource(StrEnum):
    """Where a feature's value came from. Part of every explanation."""

    #: An operator typed it into the registry.
    DECLARED = "declared"
    #: This process measured it from real attempts.
    OBSERVED = "observed"
    #: Nobody knows. The feature was not used.
    ABSENT = "absent"


class RoutePolicy(StrEnum):
    """The constitution's policies, plus `declared` for pinned requests."""

    QUALITY_FIRST = "quality_first"
    LATENCY_FIRST = "latency_first"
    COST_FIRST = "cost_first"
    BALANCED = "balanced"
    LOCAL_ONLY = "local_only"
    PRIVATE_ONLY = "private_only"
    CUSTOM = "custom"

    #: Not a constitutional policy. It is what a pinned model gets: the caller
    #: named a deployment, so ranking would override the choice they made.
    DECLARED = "declared"


#: Names that predate the constitution's vocabulary, kept working so existing
#: registries and clients do not break. The canonical name is what gets reported.
POLICY_ALIASES: dict[str, RoutePolicy] = {
    "cheapest": RoutePolicy.COST_FIRST,
    "cost": RoutePolicy.COST_FIRST,
    "quality": RoutePolicy.QUALITY_FIRST,
    "latency": RoutePolicy.LATENCY_FIRST,
    "fastest": RoutePolicy.LATENCY_FIRST,
    "local": RoutePolicy.LOCAL_ONLY,
    "private": RoutePolicy.PRIVATE_ONLY,
}


def parse_policy(name: str) -> RoutePolicy:
    text = str(name).strip().lower().replace("-", "_")
    if alias := POLICY_ALIASES.get(text):
        return alias
    try:
        return RoutePolicy(text)
    except ValueError:
        available = sorted({member.value for member in RoutePolicy} | set(POLICY_ALIASES))
        raise ConfigurationError(
            f"unknown routing policy '{name}' (available: {available})"
        ) from None


@dataclass(frozen=True, slots=True)
class PolicyWeights:
    quality: float = 0.0
    latency: float = 0.0
    cost: float = 0.0
    health: float = 0.0

    def __post_init__(self) -> None:
        for feature in FEATURES:
            if getattr(self, feature) < 0:
                raise ConfigurationError(f"policy weight '{feature}' cannot be negative")
        if self.total == 0:
            raise ConfigurationError("policy weights cannot all be zero")

    @property
    def total(self) -> float:
        return sum(self.get(feature) for feature in FEATURES)

    def get(self, feature: str) -> float:
        value: float = getattr(self, feature)
        return value

    def over(self, usable: Sequence[str]) -> dict[str, float]:
        """Renormalise across the features that survived, so scores stay in [0, 1]."""
        subtotal = sum(self.get(feature) for feature in usable)
        if subtotal <= 0:
            return {feature: 0.0 for feature in usable}
        return {feature: self.get(feature) / subtotal for feature in usable}

    def as_dict(self) -> dict[str, float]:
        return {feature: self.get(feature) for feature in FEATURES}


#: Judgements, not measurements. See the module docstring.
POLICY_WEIGHTS: dict[RoutePolicy, PolicyWeights] = {
    RoutePolicy.QUALITY_FIRST: PolicyWeights(quality=0.70, health=0.15, latency=0.10, cost=0.05),
    RoutePolicy.LATENCY_FIRST: PolicyWeights(latency=0.70, health=0.20, cost=0.05, quality=0.05),
    RoutePolicy.COST_FIRST: PolicyWeights(cost=0.80, health=0.15, quality=0.03, latency=0.02),
    RoutePolicy.BALANCED: PolicyWeights(quality=0.30, latency=0.25, cost=0.30, health=0.15),
    # The locality policies constrain *who may serve*, not how to rank, so they
    # rank the survivors the same way `balanced` does.
    RoutePolicy.LOCAL_ONLY: PolicyWeights(quality=0.30, latency=0.25, cost=0.30, health=0.15),
    RoutePolicy.PRIVATE_ONLY: PolicyWeights(quality=0.30, latency=0.25, cost=0.30, health=0.15),
    RoutePolicy.CUSTOM: PolicyWeights(quality=0.25, latency=0.25, cost=0.25, health=0.25),
    RoutePolicy.DECLARED: PolicyWeights(quality=1.0),
}

#: Localities each policy will accept. Absent means no locality constraint.
POLICY_LOCALITIES: dict[RoutePolicy, frozenset[Locality]] = {
    RoutePolicy.LOCAL_ONLY: frozenset({Locality.LOCAL}),
    RoutePolicy.PRIVATE_ONLY: frozenset({Locality.LOCAL, Locality.PRIVATE}),
}


@dataclass(frozen=True, slots=True)
class FeatureValue:
    """One feature of one candidate, and its effect on that candidate's score."""

    name: str
    source: FeatureSource
    raw: float | None = None
    normalised: float | None = None
    weight: float = 0.0
    contribution: float = 0.0
    basis: str = ""

    @property
    def usable(self) -> bool:
        return self.source is not FeatureSource.ABSENT and self.raw is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.name,
            "source": self.source.value,
            "basis": self.basis or None,
            "raw": round(self.raw, 6) if self.raw is not None else None,
            "normalised": round(self.normalised, 6) if self.normalised is not None else None,
            "weight": round(self.weight, 4),
            "contribution": round(self.contribution, 6),
        }


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    spec: ModelSpec
    score: float
    features: tuple[FeatureValue, ...]
    rank: int = 0

    @property
    def model_id(self) -> str:
        return self.spec.id

    def feature(self, name: str) -> FeatureValue | None:
        return next((value for value in self.features if value.name == name), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.spec.id,
            "deployment_id": self.spec.deployment_id,
            "provider": self.spec.provider,
            "grade": self.spec.grade.value if self.spec.grade else None,
            "locality": self.spec.locality.value,
            "score": round(self.score, 6),
            "rank": self.rank,
            "features": [value.as_dict() for value in self.features],
        }


@dataclass(frozen=True, slots=True)
class ScoringInputs:
    """Everything scoring may look at beyond the candidates themselves."""

    health: Mapping[str, HealthSnapshot] = field(default_factory=dict)
    quality_dimension: str | None = None
    expected_output_tokens: int = DEFAULT_EXPECTED_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        if self.expected_output_tokens < 0:
            raise ConfigurationError("expected_output_tokens cannot be negative")

    def snapshot_for(self, spec: ModelSpec) -> HealthSnapshot | None:
        snapshot = self.health.get(spec.deployment_id)
        if snapshot is not None and snapshot.has_signal:
            return snapshot
        return None


# -- raw feature extraction ---------------------------------------------------


def _quality_of(spec: ModelSpec, dimension: str | None) -> tuple[float | None, str]:
    """Declared quality, preferring the dimension the intent actually needs."""
    if dimension:
        specific = spec.quality.get(dimension)
        if specific is not None:
            return specific, f"quality.{dimension}"
    mean = spec.quality.mean
    if mean is not None:
        return mean, f"quality.mean({','.join(spec.quality.measured_dimensions)})"
    if spec.grade is not None:
        # Grades are an ordering, not a score, so this is a weaker basis and the
        # explanation says which one was used.
        return spec.grade.normalised, f"grade.{spec.grade.value}"
    return None, ""


def _latency_of(
    spec: ModelSpec, snapshot: HealthSnapshot | None, output_tokens: int
) -> tuple[float | None, FeatureSource, str]:
    """Observed latency when there is any, else the declared profile."""
    if snapshot is not None and snapshot.ewma_latency_ms is not None:
        return snapshot.ewma_latency_ms, FeatureSource.OBSERVED, "health.ewma_latency_ms"
    estimated = spec.performance.estimated_total_ms(output_tokens)
    if estimated is not None:
        return estimated, FeatureSource.DECLARED, "performance.p50_ttft_ms+p50_tpot_ms"
    if spec.performance.p50_ttft_ms is not None:
        return spec.performance.p50_ttft_ms, FeatureSource.DECLARED, "performance.p50_ttft_ms"
    return None, FeatureSource.ABSENT, ""


def _cost_of(spec: ModelSpec) -> tuple[float | None, str]:
    blended = spec.blended_cost_per_mtok
    if blended is None:
        return None, ""
    return blended, f"registry.blended_cost_per_mtok.{spec.api_cost_knowledge.value}"


def _health_of(snapshot: HealthSnapshot | None) -> tuple[float | None, str]:
    if snapshot is None or snapshot.health_score is None:
        return None, ""
    return snapshot.health_score, "health.health_score"


def _raw_features(
    spec: ModelSpec, inputs: ScoringInputs
) -> dict[str, tuple[float | None, FeatureSource, str]]:
    snapshot = inputs.snapshot_for(spec)

    quality, quality_basis = _quality_of(spec, inputs.quality_dimension)
    latency, latency_source, latency_basis = _latency_of(
        spec, snapshot, inputs.expected_output_tokens
    )
    cost, cost_basis = _cost_of(spec)
    health, health_basis = _health_of(snapshot)

    return {
        "quality": (
            quality,
            FeatureSource.DECLARED if quality is not None else FeatureSource.ABSENT,
            quality_basis,
        ),
        "latency": (latency, latency_source, latency_basis),
        "cost": (
            cost,
            FeatureSource.DECLARED if cost is not None else FeatureSource.ABSENT,
            cost_basis,
        ),
        "health": (
            health,
            FeatureSource.OBSERVED if health is not None else FeatureSource.ABSENT,
            health_basis,
        ),
    }


#: Features where a smaller number is better, and so must be inverted.
_LOWER_IS_BETTER = frozenset({"latency", "cost"})


def _normalise(values: Sequence[float], *, lower_is_better: bool) -> list[float]:
    """Min-max onto [0, 1] where 1 is always the better end.

    When every candidate has the same value the feature cannot discriminate, so
    all of them score 1.0 and the feature has no effect on the ordering.
    """
    smallest, largest = min(values), max(values)
    if largest == smallest:
        return [1.0] * len(values)
    span = largest - smallest
    if lower_is_better:
        return [(largest - value) / span for value in values]
    return [(value - smallest) / span for value in values]


@dataclass(frozen=True, slots=True)
class ScoringResult:
    """The ordered candidates, and an account of how they were ordered."""

    policy: RoutePolicy
    candidates: tuple[ScoredCandidate, ...]
    used_features: tuple[str, ...]
    dropped_features: tuple[tuple[str, str], ...] = ()
    weights: PolicyWeights = PolicyWeights(quality=1.0)
    fell_back_to_registry_order: bool = False

    @property
    def best(self) -> ScoredCandidate | None:
        return self.candidates[0] if self.candidates else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "weights": self.weights.as_dict(),
            "used_features": list(self.used_features),
            "dropped_features": [
                {"feature": feature, "why": why} for feature, why in self.dropped_features
            ],
            "fell_back_to_registry_order": self.fell_back_to_registry_order,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def score_candidates(
    candidates: Sequence[ModelSpec],
    *,
    policy: RoutePolicy,
    inputs: ScoringInputs | None = None,
    weights: PolicyWeights | None = None,
) -> ScoringResult:
    """Order `candidates` under `policy`, returning the reasoning with the result."""
    resolved_inputs = inputs or ScoringInputs()
    resolved_weights = weights or POLICY_WEIGHTS[policy]

    if not candidates:
        return ScoringResult(
            policy=policy, candidates=(), used_features=(), weights=resolved_weights
        )

    # A pinned request is not a ranking problem. Preserve the caller's order and
    # record that no feature was consulted, rather than inventing a score.
    if policy is RoutePolicy.DECLARED:
        return ScoringResult(
            policy=policy,
            candidates=tuple(
                ScoredCandidate(spec=spec, score=0.0, features=(), rank=index)
                for index, spec in enumerate(candidates)
            ),
            used_features=(),
            weights=resolved_weights,
            fell_back_to_registry_order=True,
        )

    raw = {spec.id: _raw_features(spec, resolved_inputs) for spec in candidates}

    usable: list[str] = []
    dropped: list[tuple[str, str]] = []
    for feature in FEATURES:
        if resolved_weights.get(feature) <= 0:
            continue
        missing = [spec.id for spec in candidates if raw[spec.id][feature][0] is None]
        present = [spec.id for spec in candidates if raw[spec.id][feature][0] is not None]
        if not present:
            dropped.append(
                (
                    feature,
                    f"not available for {', '.join(sorted(missing))}; "
                    "comparing the others on it would rank measured against unmeasured",
                )
            )
            continue
        if missing and feature != "cost":
            dropped.append(
                (
                    feature,
                    f"not available for {', '.join(sorted(missing))}; "
                    "comparing the others on it would rank measured against unmeasured",
                )
            )
            continue
        if missing and feature == "cost":
            dropped.append(
                (
                    "cost_partial",
                    "API price unknown for "
                    + ", ".join(sorted(missing))
                    + "; cost ranked only among deployments with known API prices "
                    "(known-zero included; unknown is not treated as free)",
                )
            )
        usable.append(feature)

    effective = resolved_weights.over(usable)

    normalised: dict[str, dict[str, float]] = {}
    for feature in usable:
        priced_or_present = [spec for spec in candidates if raw[spec.id][feature][0] is not None]
        values = [raw[spec.id][feature][0] for spec in priced_or_present]
        norms = _normalise(
            [value for value in values if value is not None],
            lower_is_better=feature in _LOWER_IS_BETTER,
        )
        normalised[feature] = {
            spec.id: norm for spec, norm in zip(priced_or_present, norms, strict=True)
        }

    scored: list[ScoredCandidate] = []
    for spec in candidates:
        features: list[FeatureValue] = []
        total = 0.0
        for feature in FEATURES:
            value, source, basis = raw[spec.id][feature]
            if feature in usable and spec.id in normalised.get(feature, {}):
                norm = normalised[feature][spec.id]
                weight = effective[feature]
                contribution = norm * weight
                total += contribution
                features.append(
                    FeatureValue(
                        name=feature,
                        source=source,
                        raw=value,
                        normalised=norm,
                        weight=weight,
                        contribution=contribution,
                        basis=basis,
                    )
                )
            else:
                features.append(
                    FeatureValue(
                        name=feature,
                        source=source if value is not None else FeatureSource.ABSENT,
                        raw=value,
                        basis=basis,
                    )
                )
        scored.append(ScoredCandidate(spec=spec, score=total, features=tuple(features)))

    if usable:
        # Ties break on registry order, then id, so the same fleet and the same
        # request always produce the same route.
        order = {spec.id: index for index, spec in enumerate(candidates)}
        scored.sort(key=lambda c: (-c.score, order[c.spec.id], c.spec.id))

    ranked = tuple(replace(candidate, rank=index) for index, candidate in enumerate(scored))
    return ScoringResult(
        policy=policy,
        candidates=ranked,
        used_features=tuple(usable),
        dropped_features=tuple(dropped),
        weights=resolved_weights,
        fell_back_to_registry_order=not usable,
    )


def permitted_localities(policy: RoutePolicy) -> frozenset[Locality] | None:
    """Localities a policy allows, or `None` when it does not constrain locality."""
    return POLICY_LOCALITIES.get(policy)


# -- legacy ordering helpers --------------------------------------------------
#
# The pre-Phase-4 policy surface was two pure ordering functions. They remain
# because the shape is genuinely useful for tests and for callers that only want
# an order, but they consult a single feature and cannot explain themselves.

Policy = Callable[[list[ModelSpec]], list[ModelSpec]]


def cheapest(candidates: list[ModelSpec]) -> list[ModelSpec]:
    """Cheapest known blended API price first; unknown prices sort last."""

    def key(spec: ModelSpec) -> tuple[int, float, str]:
        blended = spec.blended_cost_per_mtok
        if blended is None:
            return (1, 0.0, spec.id)
        return (0, blended, spec.id)

    return sorted(candidates, key=key)


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
