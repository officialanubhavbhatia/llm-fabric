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

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from llm_fabric.contract.openai import ModelCard
from llm_fabric.errors import ConfigurationError, ModelNotFoundError
from llm_fabric.router.capabilities import CapabilityVector
from llm_fabric.router.grades import Grade
from llm_fabric.router.tiers import ServiceTier, parse_service_tier
from llm_fabric.serving.topology import RuntimeKind, TransportKind, defaults_for_provider


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


class ApiCostKnowledge(StrEnum):
    """Whether API token prices are known. `0.0` is not the same as omitted.

    `unknown` — the operator did not fill prices in. Missing features are not
    imputed, so this deployment is excluded from cost ranking rather than treated
    as free.
    `known_zero` — the operator declared a $0 API price (typical for self-hosted
    Ollama/vLLM). That is a real value and *is* used for cost ranking.
    `known_nonzero` — at least one side of the API price is positive.
    """

    UNKNOWN = "unknown"
    KNOWN_ZERO = "known_zero"
    KNOWN_NONZERO = "known_nonzero"


class ResourceCostClass(StrEnum):
    """How to interpret a deployment's cost, when the operator has said so.

    A local model can have API token price `0` and still consume GPU/CPU. This
    field records that distinction; it is never invented from the provider name.
    """

    UNKNOWN = "unknown"
    RESOURCE_COST_KNOWN = "resource_cost_known"
    RESOURCE_COST_UNKNOWN = "resource_cost_unknown"
    MARGINAL_API_PRICE = "marginal_api_price"
    ESTIMATED_GPU_COST = "estimated_gpu_cost"


class PromotionState(StrEnum):
    """Operator-controlled lifecycle. A high declared tier is not an approval."""

    REGISTERED = "registered"
    PROBED = "probed"
    EVALUATED = "evaluated"
    SHADOW = "shadow"
    APPROVED = "approved"
    DISABLED = "disabled"


#: Provider names the factory can construct. Pool ids `ollama-*` / `vllm-*` are
#: accepted in addition to this closed set.
_KNOWN_PROVIDERS = frozenset(
    {"mock", "openai", "anthropic", "ollama", "vllm", "openai-compatible", "litellm"}
)


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

    #: USD per million tokens. `None` means unknown (not filled in). `0.0` means
    #: the operator declared a known-zero API price. Both sides must be known
    #: before the deployment participates in cost ranking — a missing side is
    #: not imputed as zero.
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None
    estimated_compute_cost_per_hour_usd: float | None = None
    cost_class: ResourceCostClass | None = None

    capabilities: CapabilityVector = field(default_factory=CapabilityVector)
    quality: QualityScores = field(default_factory=QualityScores)
    performance: PerformanceProfile = field(default_factory=PerformanceProfile)
    placement: Placement = field(default_factory=Placement)

    provider_adapter: str = ""
    transport: TransportKind = TransportKind.DIRECT
    runtime: RuntimeKind = RuntimeKind.EXTERNAL
    api_base: str | None = None
    health_endpoint: str | None = None
    metrics_endpoint: str | None = None
    served_model_name: str | None = None
    supports_kv_metrics: bool = False
    supports_batch_metrics: bool = False
    supports_queue_metrics: bool = False

    enabled: bool = True
    fallbacks: tuple[str, ...] = ()

    huggingface_id: str | None = None
    revision: str | None = None
    digest: str | None = None
    license: str | None = None
    commercial_use: bool | None = None
    pool: str | None = None
    trust_remote_code: bool = False
    tiers: tuple[ServiceTier, ...] = ()
    lifecycle: PromotionState = PromotionState.REGISTERED
    approved_tiers: tuple[ServiceTier, ...] = ()
    approved_workloads: Mapping[str, tuple[ServiceTier, ...]] = field(default_factory=dict)
    promotion_identity_match: bool = True
    promotion_evidence_bound: bool = False

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
        if self.trust_remote_code:
            raise ConfigurationError(
                f"model '{self.id}' sets trust_remote_code: the fabric refuses arbitrary "
                "remote code execution. Pin a revision and serve through Ollama or vLLM "
                "without enabling untrusted remote code."
            )
        adapter, transport, runtime = defaults_for_provider(self.provider)
        if not self.provider_adapter:
            object.__setattr__(self, "provider_adapter", adapter)
        if self.transport is TransportKind.DIRECT and self.provider_adapter == "litellm":
            object.__setattr__(self, "transport", TransportKind.LITELLM)
        if self.runtime is RuntimeKind.EXTERNAL and runtime is not RuntimeKind.EXTERNAL:
            object.__setattr__(self, "runtime", runtime)
        if (
            self.transport is TransportKind.LITELLM
            and self.provider_adapter not in {"litellm", self.provider}
            and not self.provider.startswith("litellm-")
        ):
            raise ConfigurationError(
                f"model '{self.id}' sets transport=litellm but provider "
                f"'{self.provider}' is not a LiteLLM adapter; use provider: litellm "
                "and runtime: ollama|vllm|external"
            )
        if (
            self.provider == "ollama" or self.provider.startswith("ollama-")
        ) and self.transport is TransportKind.LITELLM:
            raise ConfigurationError(
                f"model '{self.id}' mixes provider '{self.provider}' with "
                "transport=litellm; use provider: litellm and runtime: ollama"
            )
        if self.input_cost_per_mtok is not None and self.input_cost_per_mtok < 0:
            raise ConfigurationError(f"model '{self.id}' has a negative input_cost_per_mtok")
        if self.output_cost_per_mtok is not None and self.output_cost_per_mtok < 0:
            raise ConfigurationError(f"model '{self.id}' has a negative output_cost_per_mtok")
        if not self.tiers and self.grade is not None:
            object.__setattr__(self, "tiers", (ServiceTier.from_grade(self.grade),))
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

    @property
    def has_known_api_price(self) -> bool:
        """True when both token prices were declared, including known-zero."""
        return self.input_cost_per_mtok is not None and self.output_cost_per_mtok is not None

    @property
    def api_cost_knowledge(self) -> ApiCostKnowledge:
        if not self.has_known_api_price:
            return ApiCostKnowledge.UNKNOWN
        if (self.input_cost_per_mtok or 0.0) == 0.0 and (self.output_cost_per_mtok or 0.0) == 0.0:
            return ApiCostKnowledge.KNOWN_ZERO
        return ApiCostKnowledge.KNOWN_NONZERO

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        """Cost of a call at registry API prices.

        Returns `None` when either price is unknown. Returns `0.0` for a
        known-zero API price. Absence is never coerced into a dollar figure.
        """
        if not self.has_known_api_price:
            return None
        assert self.input_cost_per_mtok is not None
        assert self.output_cost_per_mtok is not None
        return (
            prompt_tokens * self.input_cost_per_mtok + completion_tokens * self.output_cost_per_mtok
        ) / 1_000_000

    @property
    def blended_cost_per_mtok(self) -> float | None:
        """A single comparable API price, weighting output more heavily than input.

        Used to order candidates that have comparable known cost semantics. The
        1:3 weighting reflects that generation is the priced-heavier side for
        most providers; it is a heuristic for ranking, not a prediction of spend.
        `None` when either side is unknown.
        """
        if not self.has_known_api_price:
            return None
        assert self.input_cost_per_mtok is not None
        assert self.output_cost_per_mtok is not None
        return (self.input_cost_per_mtok + 3 * self.output_cost_per_mtok) / 4

    @property
    def is_priced(self) -> bool:
        """True when API prices are known, including a declared $0."""
        return self.has_known_api_price

    # -- placement -----------------------------------------------------------

    @property
    def locality(self) -> Locality:
        return self.placement.locality

    @property
    def keeps_data_in_house(self) -> bool:
        return self.placement.locality.keeps_data_in_house

    @property
    def public_tier(self) -> ServiceTier | None:
        """The operator-facing tier label for this deployment, if any."""
        if self.tiers:
            return max(self.tiers, key=lambda tier: tier.ordinal)
        if self.grade is not None:
            return ServiceTier.from_grade(self.grade)
        return None

    def serves_tier(self, tier: ServiceTier) -> bool:
        """True when this deployment is declared eligible for the public tier."""
        if self.tiers and tier in self.tiers:
            return True
        if self.grade is None:
            return False
        return ServiceTier.from_grade(self.grade) is tier or (
            tier is ServiceTier.L30 and self.grade.ordinal == 29
        )

    def serves_approved_tier(self, tier: ServiceTier) -> bool:
        """Measured approval, not declared eligibility."""
        if not self.approved_tiers:
            return False
        if tier is ServiceTier.L30:
            return ServiceTier.L29 in self.approved_tiers or ServiceTier.L30 in self.approved_tiers
        return tier in self.approved_tiers

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
                "api_cost_knowledge": self.api_cost_knowledge.value,
                "cost_class": self.cost_class.value if self.cost_class else None,
                "estimated_compute_cost_per_hour_usd": (self.estimated_compute_cost_per_hour_usd),
            },
            "capabilities": self.capabilities.as_dict(),
            "quality": self.quality.as_dict(),
            "performance": self.performance.as_dict(),
            "placement": self.placement.as_dict(),
            "topology": {
                "provider_adapter": self.provider_adapter,
                "transport": self.transport.value,
                "runtime": self.runtime.value,
                "api_base": self.api_base,
                "health_endpoint": self.health_endpoint,
                "metrics_endpoint": self.metrics_endpoint,
                "served_model_name": self.served_model_name,
                "supports_kv_metrics": self.supports_kv_metrics,
                "supports_batch_metrics": self.supports_batch_metrics,
                "supports_queue_metrics": self.supports_queue_metrics,
            },
            "fallbacks": list(self.fallbacks),
            "identity": {
                "huggingface_id": self.huggingface_id,
                "revision": self.revision,
                "digest": self.digest,
                "license": self.license,
                "commercial_use": self.commercial_use,
                "pool": self.pool,
            },
            "tiers": [tier.value for tier in self.tiers],
            "approved_tiers": [tier.value for tier in self.approved_tiers],
            "approved_workloads": {
                name: [tier.value for tier in tiers]
                for name, tiers in self.approved_workloads.items()
            },
            "lifecycle": self.lifecycle.value,
            "promotion_identity_match": self.promotion_identity_match,
            "promotion_evidence_bound": self.promotion_evidence_bound,
            "trust_remote_code": self.trust_remote_code,
        }


@dataclass(frozen=True, slots=True)
class Alias:
    id: str
    candidates: tuple[str, ...]
    policy: str | None = None
    requires: frozenset[str] = frozenset()
    minimum_grade: Grade | None = None


def _valid_provider(name: str) -> bool:
    """True when `provider` can name an adapter or a factory override.

    Known adapters plus `ollama-*` / `vllm-*` pools are always accepted. Other
    lowercase identifiers are accepted so tests can inject a provider (for
    example `failing`) and so a new pool name is not a code change. URLs and
    empty strings are refused here; they are not provider names.
    """
    if not name or "://" in name or "/" in name or " " in name:
        return False
    if name in _KNOWN_PROVIDERS or name.startswith(("ollama-", "vllm-", "litellm-")):
        return True
    token = name.replace("_", "").replace("-", "").replace(".", "")
    return bool(token) and token.isalnum() and name[0].isalpha()


def _as_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ConfigurationError(f"expected a boolean, got {value!r}")


def _tiers_from(entry: dict[str, Any]) -> tuple[ServiceTier, ...]:
    raw = entry.get("tiers")
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    return tuple(ServiceTier.parse(str(item)) for item in raw)


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


def _cost_class_from(entry: dict[str, Any]) -> ResourceCostClass | None:
    raw = entry.get("cost_class")
    if raw is None or raw == "":
        return None
    try:
        return ResourceCostClass(str(raw).strip().lower())
    except ValueError:
        raise ConfigurationError(
            f"model '{entry.get('id')}' declares unknown cost_class {raw!r} "
            f"(available: {[member.value for member in ResourceCostClass]})"
        ) from None


def _lifecycle_from(entry: dict[str, Any]) -> PromotionState:
    raw = entry.get("lifecycle") or entry.get("promotion_state")
    if raw is None or raw == "":
        return PromotionState.REGISTERED
    try:
        return PromotionState(str(raw).strip().lower())
    except ValueError:
        raise ConfigurationError(
            f"model '{entry.get('id')}' declares unknown lifecycle {raw!r} "
            f"(available: {[member.value for member in PromotionState]})"
        ) from None


def _capabilities_from(entry: dict[str, Any]) -> CapabilityVector:
    raw = entry.get("capabilities")
    if raw is None:
        return CapabilityVector()
    if isinstance(raw, dict):
        raise ConfigurationError(
            f"model '{entry.get('id')}' capabilities must be a list or string, not a mapping"
        )
    if isinstance(raw, (int, float, bool)):
        raise ConfigurationError(
            f"model '{entry.get('id')}' capabilities must be a list or string, got {raw!r}"
        )
    return CapabilityVector.from_config(raw)


def _transport_from(entry: dict[str, Any]) -> TransportKind:
    raw = entry.get("transport")
    if raw is None or raw == "":
        return TransportKind.DIRECT
    try:
        return TransportKind(str(raw))
    except ValueError:
        raise ConfigurationError(
            f"model '{entry.get('id')}' has unknown transport {raw!r} "
            f"(available: {[member.value for member in TransportKind]})"
        ) from None


def _runtime_from(entry: dict[str, Any]) -> RuntimeKind:
    raw = entry.get("runtime")
    if raw is None or raw == "":
        return RuntimeKind.EXTERNAL
    try:
        return RuntimeKind(str(raw))
    except ValueError:
        raise ConfigurationError(
            f"model '{entry.get('id')}' has unknown runtime {raw!r} "
            f"(available: {[member.value for member in RuntimeKind]})"
        ) from None


def _spec_from(entry: dict[str, Any]) -> ModelSpec:
    missing = {"id", "provider"} - entry.keys()
    if missing:
        raise ConfigurationError(f"registry model missing required field(s): {sorted(missing)}")

    context_window = entry.get("context_window", entry.get("max_context_tokens"))
    provider = str(entry["provider"])
    if not _valid_provider(provider):
        raise ConfigurationError(
            f"model '{entry.get('id')}' names unknown provider '{provider}' "
            f"(available: {sorted(_KNOWN_PROVIDERS)} plus ollama-* / vllm-* / litellm-* pools)"
        )
    return ModelSpec(
        id=str(entry["id"]),
        provider=provider,
        provider_model=str(entry.get("provider_model", entry["id"])),
        deployment_id=str(entry.get("deployment_id", "") or ""),
        grade=Grade.parse(str(entry["grade"])) if entry.get("grade") else None,
        context_window=int(context_window) if context_window else None,
        recommended_context_tokens=(
            int(entry["recommended_context_tokens"])
            if entry.get("recommended_context_tokens")
            else None
        ),
        input_cost_per_mtok=_as_float(entry.get("input_cost_per_mtok"), "input_cost_per_mtok"),
        output_cost_per_mtok=_as_float(entry.get("output_cost_per_mtok"), "output_cost_per_mtok"),
        estimated_compute_cost_per_hour_usd=_as_float(
            entry.get("estimated_compute_cost_per_hour_usd"),
            "estimated_compute_cost_per_hour_usd",
        ),
        cost_class=_cost_class_from(entry),
        capabilities=_capabilities_from(entry),
        quality=_quality_from(entry),
        performance=_performance_from(entry),
        placement=_placement_from(entry),
        provider_adapter=str(entry.get("provider_adapter") or ""),
        transport=_transport_from(entry),
        runtime=_runtime_from(entry),
        api_base=_optional_str(entry.get("api_base")),
        health_endpoint=_optional_str(entry.get("health_endpoint")),
        metrics_endpoint=_optional_str(entry.get("metrics_endpoint")),
        served_model_name=_optional_str(entry.get("served_model_name")),
        supports_kv_metrics=bool(entry.get("supports_kv_metrics", False)),
        supports_batch_metrics=bool(entry.get("supports_batch_metrics", False)),
        supports_queue_metrics=bool(entry.get("supports_queue_metrics", False)),
        enabled=bool(entry.get("enabled", True)),
        fallbacks=_as_tuple(entry.get("fallbacks")),
        huggingface_id=_optional_str(entry.get("huggingface_id") or entry.get("hf_id")),
        revision=_optional_str(entry.get("revision")),
        digest=_optional_str(entry.get("digest")),
        license=_optional_str(entry.get("license")),
        commercial_use=_optional_bool(entry.get("commercial_use")),
        pool=_optional_str(entry.get("pool")),
        trust_remote_code=bool(entry.get("trust_remote_code", False)),
        tiers=_tiers_from(entry),
        lifecycle=_lifecycle_from(entry),
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

    def is_model(self, model_id: str) -> bool:
        return model_id in self._models

    def known(self, model_id: str) -> bool:
        if model_id in self._models or model_id in self._aliases:
            return True
        tier = parse_service_tier(model_id)
        return tier is not None and bool(self.models_serving_tier(tier))

    def models_serving_tier(self, tier: ServiceTier) -> list[ModelSpec]:
        return [spec for spec in self.enabled_models() if spec.serves_tier(tier)]

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
