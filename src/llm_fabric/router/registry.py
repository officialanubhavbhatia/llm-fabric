"""The model registry: what the fabric can serve, and everything routing knows about it.

The registry is declarative configuration rather than code so that adding a
model, repricing one, regrading one, or disabling one is a config change and not
a deploy of new logic.

**Everything in this file is a declaration, not a measurement.** Prices, quality
scores and the latency profile are numbers an operator typed into YAML, sourced
from a provider's pricing page or a benchmark run that happened somewhere else.
The fabric does not verify them. This matters because the router ranks on them:
a wrong number here produces a confidently wrong route, and the route explanation
therefore labels every one of these features `declared` so nobody mistakes it for
something observed.

The constitution lists `health_score`, `error_rate` and `queue_depth` among a
graded model's attributes. They are deliberately **not** stored here. Those three
are properties of a running system, and a value typed into a config file would be
a fiction that outranks reality the moment traffic starts. They live in
`router.health`, are computed from attempts this process actually made, and are
labelled `observed` — or `absent`, when no traffic has been seen yet.

Two kinds of entry exist:

* a **model**, mapping a fabric-facing id onto one provider, that provider's own
  model name, and the attributes below, and
* an **alias**, a virtual id resolving to several models under a policy. `auto`
  is an alias.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from llm_fabric.contract.openai import ModelCard
from llm_fabric.errors import ConfigurationError, ModelNotFoundError
from llm_fabric.router.capabilities import CapabilityVector
from llm_fabric.router.grades import Grade


class Locality(StrEnum):
    """Where inference physically happens, which decides who can see the prompt.

    The distinction is a privacy boundary, not a performance one. `LOCAL` means
    the same machine or cluster the fabric runs on; `PRIVATE` means infrastructure
    the operator controls but which is reached over a network; `EXTERNAL` means a
    third party receives the prompt.
    """

    LOCAL = "local"
    PRIVATE = "private"
    EXTERNAL = "external"

    @property
    def keeps_data_in_house(self) -> bool:
        """True when no third party sees the prompt."""
        return self in (Locality.LOCAL, Locality.PRIVATE)


#: Quality dimensions the constitution names. Each is optional on a deployment:
#: absent means unmeasured, and unmeasured must never be read as zero.
QUALITY_DIMENSIONS: tuple[str, ...] = (
    "reasoning",
    "coding",
    "agent",
    "math",
    "rag",
    "tool_use",
    "structured_output",
    "safety",
)


def _check_score(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ConfigurationError(
            f"{label} must lie in [0, 1] (it is a normalised score, not a percentage), "
            f"got {value!r}"
        )
    return number


@dataclass(frozen=True, slots=True)
class QualityScores:
    """Declared quality per dimension, normalised to [0, 1].

    Every field is optional and defaults to `None`, which means *nobody has
    measured this*. That is a different statement from a score of zero, and the
    router keeps them different: a quality-first policy will not rank a
    deployment on a dimension it has no score for, it will say the score was
    absent and fall back to grade order.
    """

    reasoning: float | None = None
    coding: float | None = None
    agent: float | None = None
    math: float | None = None
    rag: float | None = None
    tool_use: float | None = None
    structured_output: float | None = None
    safety: float | None = None

    def __post_init__(self) -> None:
        for dimension in QUALITY_DIMENSIONS:
            object.__setattr__(
                self,
                dimension,
                _check_score(getattr(self, dimension), f"{dimension}_score"),
            )

    def get(self, dimension: str) -> float | None:
        if dimension not in QUALITY_DIMENSIONS:
            raise ConfigurationError(
                f"unknown quality dimension '{dimension}' (available: {list(QUALITY_DIMENSIONS)})"
            )
        value: float | None = getattr(self, dimension)
        return value

    @property
    def measured_dimensions(self) -> tuple[str, ...]:
        return tuple(d for d in QUALITY_DIMENSIONS if getattr(self, d) is not None)

    @property
    def any_measured(self) -> bool:
        return bool(self.measured_dimensions)

    @property
    def mean(self) -> float | None:
        """Mean across *declared* dimensions only. `None` when none are declared."""
        declared = [getattr(self, d) for d in self.measured_dimensions]
        return sum(declared) / len(declared) if declared else None

    def as_dict(self) -> dict[str, float | None]:
        return {f"{d}_score": getattr(self, d) for d in QUALITY_DIMENSIONS}


@dataclass(frozen=True, slots=True)
class PerformanceProfile:
    """Declared latency and throughput. Absent fields mean unmeasured."""

    p50_ttft_ms: float | None = None
    p95_ttft_ms: float | None = None
    p99_ttft_ms: float | None = None
    p50_tpot_ms: float | None = None
    prefill_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None

    @property
    def any_declared(self) -> bool:
        return any(
            value is not None
            for value in (
                self.p50_ttft_ms,
                self.p95_ttft_ms,
                self.p99_ttft_ms,
                self.p50_tpot_ms,
                self.prefill_tokens_per_second,
                self.decode_tokens_per_second,
            )
        )

    def estimated_total_ms(self, output_tokens: int) -> float | None:
        """Time to finish `output_tokens`, from declared time-to-first-token and per-token cost.

        Returns `None` rather than a guess when either input is missing. An
        estimate built from half the data is worse than an admitted absence,
        because it looks like the other estimates.
        """
        if self.p50_ttft_ms is None or self.p50_tpot_ms is None:
            return None
        return self.p50_ttft_ms + self.p50_tpot_ms * max(0, output_tokens)

    def as_dict(self) -> dict[str, float | None]:
        return {
            "p50_ttft_ms": self.p50_ttft_ms,
            "p95_ttft_ms": self.p95_ttft_ms,
            "p99_ttft_ms": self.p99_ttft_ms,
            "p50_tpot_ms": self.p50_tpot_ms,
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "decode_tokens_per_second": self.decode_tokens_per_second,
        }


@dataclass(frozen=True, slots=True)
class Placement:
    """Where a deployment runs.

    `locality` defaults to `EXTERNAL`, which is the only safe default: treating an
    unlabelled deployment as local would route a local-only request to a third
    party, and a privacy control that fails open is not a control.
    """

    region: str | None = None
    hardware: str | None = None
    locality: Locality = Locality.EXTERNAL

    def as_dict(self) -> dict[str, str | None]:
        return {
            "region": self.region,
            "hardware": self.hardware,
            "locality": self.locality.value,
        }


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One deployment, and every declared attribute the router may rank it on."""

    id: str
    provider: str
    provider_model: str = ""
    deployment_id: str = ""
    grade: Grade | None = None

    context_window: int | None = None
    recommended_context_tokens: int | None = None

    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    estimated_compute_cost_per_hour_usd: float | None = None

    capabilities: CapabilityVector = field(default_factory=CapabilityVector)
    quality: QualityScores = field(default_factory=QualityScores)
    performance: PerformanceProfile = field(default_factory=PerformanceProfile)
    placement: Placement = field(default_factory=Placement)

    enabled: bool = True
    fallbacks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.provider_model:
            object.__setattr__(self, "provider_model", self.id)
        # One physical deployment usually serves one fabric-facing id, so the id
        # is the sensible default. They diverge when the same model is deployed
        # twice — two regions, or two hardware profiles — and routing must be
        # able to tell those apart even though the model is identical.
        if not self.deployment_id:
            object.__setattr__(self, "deployment_id", self.id)
        # Constructed directly (in tests, or by a caller building a fleet in
        # code) a bare collection is the natural thing to pass, so accept it.
        if isinstance(self.capabilities, (frozenset, set, list, tuple)):
            object.__setattr__(
                self, "capabilities", CapabilityVector.from_config(self.capabilities)
            )
        if self.context_window is not None and self.context_window <= 0:
            raise ConfigurationError(f"model '{self.id}' declares a non-positive context window")
        if (
            self.recommended_context_tokens is not None
            and self.context_window is not None
            and self.recommended_context_tokens > self.context_window
        ):
            raise ConfigurationError(
                f"model '{self.id}' recommends more context "
                f"({self.recommended_context_tokens}) than it can accept ({self.context_window})"
            )

    # -- context -------------------------------------------------------------

    @property
    def max_context_tokens(self) -> int | None:
        """The constitution's name for the hard ceiling."""
        return self.context_window

    @property
    def usable_context_tokens(self) -> int | None:
        """The ceiling routing should plan against: the recommendation when given."""
        return self.recommended_context_tokens or self.context_window

    def fits_context(self, tokens: int) -> bool:
        """True when `tokens` fit. An undeclared window is not treated as a limit."""
        ceiling = self.usable_context_tokens
        return ceiling is None or tokens <= ceiling

    # -- cost ----------------------------------------------------------------

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost of a call at registry prices. Zero when the model is unpriced."""
        return (
            prompt_tokens * self.input_cost_per_mtok + completion_tokens * self.output_cost_per_mtok
        ) / 1_000_000

    @property
    def blended_cost_per_mtok(self) -> float:
        """A single comparable price, weighting output more heavily than input.

        Used to order candidates by cost. The 1:3 weighting reflects that
        generation is the priced-heavier side for most providers; it is a
        heuristic for ranking, not a prediction of spend.
        """
        return (self.input_cost_per_mtok + 3 * self.output_cost_per_mtok) / 4

    @property
    def is_priced(self) -> bool:
        return self.input_cost_per_mtok > 0 or self.output_cost_per_mtok > 0

    # -- placement -----------------------------------------------------------

    @property
    def locality(self) -> Locality:
        return self.placement.locality

    @property
    def keeps_data_in_house(self) -> bool:
        return self.placement.locality.keeps_data_in_house

    def to_card(self) -> ModelCard:
        return ModelCard(id=self.id, owned_by=self.provider, context_window=self.context_window)

    def describe(self) -> dict[str, Any]:
        """Every declared attribute, for the preview API and route explanations."""
        return {
            "model_id": self.id,
            "deployment_id": self.deployment_id,
            "provider": self.provider,
            "grade": self.grade.value if self.grade else None,
            "enabled": self.enabled,
            "context": {
                "max_context_tokens": self.max_context_tokens,
                "recommended_context_tokens": self.recommended_context_tokens,
            },
            "cost": {
                "input_cost_per_mtok": self.input_cost_per_mtok,
                "output_cost_per_mtok": self.output_cost_per_mtok,
                "estimated_compute_cost_per_hour_usd": (self.estimated_compute_cost_per_hour_usd),
            },
            "capabilities": self.capabilities.as_dict(),
            "quality": self.quality.as_dict(),
            "performance": self.performance.as_dict(),
            "placement": self.placement.as_dict(),
            "fallbacks": list(self.fallbacks),
        }


@dataclass(frozen=True, slots=True)
class Alias:
    id: str
    candidates: tuple[str, ...]
    policy: str | None = None
    requires: frozenset[str] = frozenset()
    minimum_grade: Grade | None = None


def _as_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _as_float(value: Any, label: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigurationError(f"{label} must be a number, got {value!r}") from None


def _quality_from(entry: dict[str, Any]) -> QualityScores:
    """Read quality scores, accepting either a nested block or `*_score` keys."""
    block = entry.get("quality")
    if block is not None and not isinstance(block, dict):
        raise ConfigurationError(f"model '{entry.get('id')}' has a non-mapping 'quality' block")
    nested: dict[str, Any] = dict(block or {})
    values: dict[str, float | None] = {}
    for dimension in QUALITY_DIMENSIONS:
        # Accepted spellings, most specific first: `quality: {reasoning: ...}`,
        # `quality: {reasoning_score: ...}`, then a flat `reasoning_score:`.
        raw = nested.get(dimension)
        if raw is None:
            raw = nested.get(f"{dimension}_score")
        if raw is None:
            raw = entry.get(f"{dimension}_score")
        values[dimension] = _as_float(raw, f"{dimension}_score")
    return QualityScores(**values)


def _performance_from(entry: dict[str, Any]) -> PerformanceProfile:
    block = entry.get("performance")
    if block is not None and not isinstance(block, dict):
        raise ConfigurationError(f"model '{entry.get('id')}' has a non-mapping 'performance' block")
    source: dict[str, Any] = dict(block or {})
    fields = (
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "p50_tpot_ms",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
    )
    for name in fields:
        if entry.get(name) is not None:
            source.setdefault(name, entry[name])
    return PerformanceProfile(**{name: _as_float(source.get(name), name) for name in fields})


def _placement_from(entry: dict[str, Any]) -> Placement:
    block = entry.get("placement")
    if block is not None and not isinstance(block, dict):
        raise ConfigurationError(f"model '{entry.get('id')}' has a non-mapping 'placement' block")
    source: dict[str, Any] = dict(block or {})
    for name in ("region", "hardware", "locality"):
        if entry.get(name) is not None:
            source.setdefault(name, entry[name])

    raw_locality = source.get("locality")
    if raw_locality is None:
        locality = Locality.EXTERNAL
    else:
        try:
            locality = Locality(str(raw_locality).strip().lower())
        except ValueError:
            raise ConfigurationError(
                f"model '{entry.get('id')}' declares unknown locality {raw_locality!r} "
                f"(available: {[member.value for member in Locality]})"
            ) from None

    return Placement(
        region=str(source["region"]) if source.get("region") else None,
        hardware=str(source["hardware"]) if source.get("hardware") else None,
        locality=locality,
    )


def _spec_from(entry: dict[str, Any]) -> ModelSpec:
    missing = {"id", "provider"} - entry.keys()
    if missing:
        raise ConfigurationError(f"registry model missing required field(s): {sorted(missing)}")

    context_window = entry.get("context_window", entry.get("max_context_tokens"))
    return ModelSpec(
        id=str(entry["id"]),
        provider=str(entry["provider"]),
        provider_model=str(entry.get("provider_model", entry["id"])),
        deployment_id=str(entry.get("deployment_id", "") or ""),
        grade=Grade.parse(str(entry["grade"])) if entry.get("grade") else None,
        context_window=int(context_window) if context_window else None,
        recommended_context_tokens=(
            int(entry["recommended_context_tokens"])
            if entry.get("recommended_context_tokens")
            else None
        ),
        input_cost_per_mtok=_as_float(entry.get("input_cost_per_mtok", 0.0), "input_cost_per_mtok")
        or 0.0,
        output_cost_per_mtok=_as_float(
            entry.get("output_cost_per_mtok", 0.0), "output_cost_per_mtok"
        )
        or 0.0,
        estimated_compute_cost_per_hour_usd=_as_float(
            entry.get("estimated_compute_cost_per_hour_usd"),
            "estimated_compute_cost_per_hour_usd",
        ),
        capabilities=CapabilityVector.from_config(entry.get("capabilities")),
        quality=_quality_from(entry),
        performance=_performance_from(entry),
        placement=_placement_from(entry),
        enabled=bool(entry.get("enabled", True)),
        fallbacks=_as_tuple(entry.get("fallbacks")),
    )


class ModelRegistry:
    def __init__(self, models: list[ModelSpec], aliases: list[Alias] | None = None) -> None:
        self._models: dict[str, ModelSpec] = {}
        for spec in models:
            if spec.id in self._models:
                raise ConfigurationError(f"duplicate model id in registry: {spec.id}")
            self._models[spec.id] = spec

        self._aliases: dict[str, Alias] = {}
        for alias in aliases or []:
            if alias.id in self._models:
                raise ConfigurationError(f"alias '{alias.id}' collides with a model id")
            self._aliases[alias.id] = alias

        self._validate_references()

    def _validate_references(self) -> None:
        for spec in self._models.values():
            for fallback in spec.fallbacks:
                if fallback not in self._models:
                    raise ConfigurationError(
                        f"model '{spec.id}' declares unknown fallback '{fallback}'"
                    )
                if fallback == spec.id:
                    raise ConfigurationError(f"model '{spec.id}' declares itself as a fallback")
        for alias in self._aliases.values():
            if not alias.candidates:
                raise ConfigurationError(f"alias '{alias.id}' has no candidates")
            for candidate in alias.candidates:
                if candidate not in self._models:
                    raise ConfigurationError(
                        f"alias '{alias.id}' references unknown model '{candidate}'"
                    )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ModelRegistry:
        raw_models = data.get("models") or []
        if not isinstance(raw_models, list):
            raise ConfigurationError("registry 'models' must be a list")

        models: list[ModelSpec] = []
        for entry in raw_models:
            if not isinstance(entry, dict):
                raise ConfigurationError("each registry model must be a mapping")
            models.append(_spec_from(entry))

        aliases = [
            Alias(
                id=str(entry["id"]),
                candidates=_as_tuple(entry.get("candidates")),
                policy=str(entry["policy"]) if entry.get("policy") else None,
                requires=CapabilityVector.from_config(entry.get("requires")).declared,
                minimum_grade=(
                    Grade.parse(str(entry["minimum_grade"])) if entry.get("minimum_grade") else None
                ),
            )
            for entry in data.get("aliases") or []
            if isinstance(entry, dict) and entry.get("id")
        ]

        return cls(models, aliases)

    @classmethod
    def from_yaml(cls, path: Path) -> ModelRegistry:
        if not path.exists():
            raise ConfigurationError(f"model registry not found at {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"model registry at {path} must be a mapping")
        return cls.from_mapping(data)

    # -- lookup --------------------------------------------------------------

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError:
            raise ModelNotFoundError(f"unknown model '{model_id}'") from None

    def replace(self, spec: ModelSpec) -> ModelSpec:
        """Swap one in-memory spec. Used by controlled rollback, not the request path."""
        if spec.id not in self._models:
            raise ModelNotFoundError(f"unknown model '{spec.id}'")
        previous = self._models[spec.id]
        self._models[spec.id] = spec
        try:
            self._validate_references()
        except ConfigurationError:
            self._models[spec.id] = previous
            raise
        return spec

    def alias(self, alias_id: str) -> Alias | None:
        return self._aliases.get(alias_id)

    def is_alias(self, model_id: str) -> bool:
        return model_id in self._aliases

    def known(self, model_id: str) -> bool:
        return model_id in self._models or model_id in self._aliases

    def enabled_models(self) -> list[ModelSpec]:
        return [spec for spec in self._models.values() if spec.enabled]

    def all_models(self) -> list[ModelSpec]:
        return list(self._models.values())

    def providers_in_use(self) -> set[str]:
        return {spec.provider for spec in self.enabled_models()}

    def cards(self) -> list[ModelCard]:
        cards = [spec.to_card() for spec in self.enabled_models()]
        cards.extend(
            ModelCard(id=alias.id, owned_by="llm-fabric") for alias in self._aliases.values()
        )
        return cards

    def graded(self, grade: Grade) -> list[ModelSpec]:
        return [spec for spec in self.enabled_models() if spec.grade is grade]

    def with_capabilities(self, required: frozenset[str]) -> list[ModelSpec]:
        return [spec for spec in self.enabled_models() if spec.capabilities.satisfies(required)]

    def in_house(self) -> list[ModelSpec]:
        """Deployments no third party can see. The candidate set for private-only."""
        return [spec for spec in self.enabled_models() if spec.keeps_data_in_house]

    def replace_model(self, spec: ModelSpec) -> ModelRegistry:
        """A copy with one model substituted. Used by tests and by regrading."""
        existing = list(self._models.values())
        models = [spec if current.id == spec.id else current for current in existing]
        if all(current.id != spec.id for current in existing):
            models.append(spec)
        return ModelRegistry(models, list(self._aliases.values()))

    def regrade(self, model_id: str, grade: Grade) -> ModelRegistry:
        """Move a deployment between bands, as benchmark data changes."""
        return self.replace_model(replace(self.get(model_id), grade=grade))

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self._models.values())
