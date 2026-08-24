"""The fallback graph: where a request goes when a deployment cannot serve it.

The constitution forbids a flat fallback list, and the reason becomes obvious as
soon as failures are typed. "The context was too large" and "the provider is
down" are not the same problem and should not have the same answer: the first
needs a deployment with a bigger window, the second needs a different provider,
and a flat list conflates them into "whatever is next". So edges here are
labelled with the reasons they serve, and traversal follows only the edges that
match the failure that actually happened.

Three safeguards apply to every traversal:

**Loops are impossible, not merely unlikely.** A directed graph may legitimately
contain a cycle — A falls back to B for overload, B falls back to A for context
size — so cycles are not rejected at construction. Instead traversal carries the
set of deployments already tried and never returns to one. `detect_cycles` exists
so an operator can *see* the cycles in a preview without being stopped by them.

**Depth is bounded and recorded.** Every hop increments a depth that ends up in
the decision object.

**Cost and latency are bounded.** A fallback chain that tries eight models has
spent eight models' worth of budget and the caller's patience. The ledger tracks
both against a ceiling and refuses the hop that would breach it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from llm_fabric.errors import (
    ConfigurationError,
    ContextTooLargeError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
    RetryableError,
)


class FallbackReason(StrEnum):
    """Why a request left the deployment it was on. Exactly the constitution's list."""

    TIMEOUT = "timeout"
    OVERLOADED = "overloaded"
    RATE_LIMITED = "rate_limited"
    PROVIDER_DOWN = "provider_down"
    CONTEXT_TOO_LARGE = "context_too_large"
    MODEL_UNAVAILABLE = "model_unavailable"
    SAFETY_REQUIREMENT = "safety_requirement"
    STRUCTURED_OUTPUT_FAILURE = "structured_output_failure"
    QUALITY_FAILURE = "quality_failure"

    @property
    def is_capacity(self) -> bool:
        """Reasons that say "not now" rather than "not this model"."""
        return self in (
            FallbackReason.TIMEOUT,
            FallbackReason.OVERLOADED,
            FallbackReason.RATE_LIMITED,
            FallbackReason.PROVIDER_DOWN,
        )


#: Every reason, for edges an operator wants to apply unconditionally.
ANY_REASON: frozenset[FallbackReason] = frozenset(FallbackReason)


def reason_for_error(error: BaseException) -> FallbackReason:
    """Classify a raised error into the reason that explains the fallback.

    Deliberately conservative: anything not recognised becomes `PROVIDER_DOWN`,
    the reason with the least specific remedy, rather than being guessed into a
    category whose edges would send the request somewhere inappropriate.
    """
    if isinstance(error, ProviderTimeoutError):
        return FallbackReason.TIMEOUT
    if isinstance(error, QuotaExceededError):
        return FallbackReason.RATE_LIMITED
    if isinstance(error, ContextTooLargeError):
        return FallbackReason.CONTEXT_TOO_LARGE
    if isinstance(error, ModelNotFoundError):
        return FallbackReason.MODEL_UNAVAILABLE
    if isinstance(error, ProviderUnavailableError):
        return FallbackReason.PROVIDER_DOWN
    if isinstance(error, InvalidRequestError):
        return FallbackReason.MODEL_UNAVAILABLE
    if isinstance(error, RetryableError):
        status = getattr(error, "status_code", None)
        if status == 429:
            return FallbackReason.RATE_LIMITED
        if status == 503:
            return FallbackReason.OVERLOADED
        return FallbackReason.PROVIDER_DOWN
    return FallbackReason.PROVIDER_DOWN


@dataclass(frozen=True, slots=True)
class FallbackEdge:
    """A permitted hop, and the failures it answers."""

    source: str
    target: str
    reasons: frozenset[FallbackReason] = ANY_REASON
    note: str = ""

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ConfigurationError(
                f"fallback edge from '{self.source}' points at itself, which can never help"
            )
        if not self.reasons:
            raise ConfigurationError(
                f"fallback edge {self.source} -> {self.target} lists no reasons, "
                "so it could never be taken"
            )

    def answers(self, reason: FallbackReason) -> bool:
        return reason in self.reasons

    def as_dict(self) -> dict[str, Any]:
        return {
            "from": self.source,
            "to": self.target,
            "reasons": sorted(reason.value for reason in self.reasons),
            "note": self.note or None,
        }


@dataclass(frozen=True, slots=True)
class FallbackBudget:
    """Ceilings on what falling back is allowed to consume.

    `max_depth` counts hops after the primary, so a depth of 2 permits at most
    three deployments in total.
    """

    max_depth: int = 2
    max_cost_usd: float | None = None
    max_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ConfigurationError("fallback max_depth cannot be negative")
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ConfigurationError("fallback max_cost_usd cannot be negative")
        if self.max_latency_ms is not None and self.max_latency_ms < 0:
            raise ConfigurationError("fallback max_latency_ms cannot be negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "max_cost_usd": self.max_cost_usd,
            "max_latency_ms": self.max_latency_ms,
        }


@dataclass(slots=True)
class FallbackLedger:
    """Running total of what the fallback chain has spent."""

    budget: FallbackBudget
    depth: int = 0
    spent_usd: float = 0.0
    elapsed_ms: float = 0.0

    def charge(self, *, cost_usd: float = 0.0, latency_ms: float = 0.0) -> None:
        self.spent_usd += max(0.0, cost_usd)
        self.elapsed_ms += max(0.0, latency_ms)

    def refuse_reason(self, *, next_cost_usd: float = 0.0) -> str | None:
        """Why the next hop is not permitted, or `None` when it is.

        Latency and depth are checked against what has already been spent;
        cost also considers the estimated price of the hop being contemplated,
        because a chain that overspends and then notices has already overspent.
        """
        if self.depth >= self.budget.max_depth:
            return f"fallback depth {self.depth} reached the limit of {self.budget.max_depth}"
        if self.budget.max_latency_ms is not None and self.elapsed_ms >= self.budget.max_latency_ms:
            return (
                f"fallback latency {self.elapsed_ms:.0f}ms reached the limit of "
                f"{self.budget.max_latency_ms:.0f}ms"
            )
        if self.budget.max_cost_usd is not None:
            projected = self.spent_usd + max(0.0, next_cost_usd)
            if projected > self.budget.max_cost_usd:
                return (
                    f"fallback cost would reach ${projected:.6f}, over the limit of "
                    f"${self.budget.max_cost_usd:.6f}"
                )
        return None

    def advance(self) -> None:
        self.depth += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "spent_usd": round(self.spent_usd, 6),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "budget": self.budget.as_dict(),
        }


class FallbackGraph:
    """A directed, reason-labelled graph over deployment ids."""

    def __init__(self, edges: Iterable[FallbackEdge] = ()) -> None:
        self._edges: list[FallbackEdge] = []
        self._by_source: dict[str, list[FallbackEdge]] = {}
        for edge in edges:
            self.add(edge)

    def add(self, edge: FallbackEdge) -> None:
        self._edges.append(edge)
        self._by_source.setdefault(edge.source, []).append(edge)

    @property
    def edges(self) -> tuple[FallbackEdge, ...]:
        return tuple(self._edges)

    def successors(self, node: str, reason: FallbackReason) -> tuple[str, ...]:
        """Targets reachable from `node` for this reason, in declaration order."""
        return tuple(edge.target for edge in self._by_source.get(node, ()) if edge.answers(reason))

    def next_hop(
        self,
        node: str,
        reason: FallbackReason,
        *,
        visited: Iterable[str] = (),
        eligible: Iterable[str] | None = None,
    ) -> str | None:
        """The first untried, eligible successor, or `None` when the chain ends."""
        seen = set(visited)
        allowed = set(eligible) if eligible is not None else None
        for target in self.successors(node, reason):
            if target in seen:
                continue
            if allowed is not None and target not in allowed:
                continue
            return target
        return None

    def detect_cycles(self) -> tuple[tuple[str, ...], ...]:
        """Every cycle in the graph, ignoring reasons.

        Reported rather than rejected: traversal cannot loop regardless, and an
        operator may deliberately want a mutual fallback pair.
        """
        cycles: list[tuple[str, ...]] = []
        seen_signatures: set[frozenset[str]] = set()
        colour: dict[str, int] = {}
        stack: list[str] = []

        def visit(node: str) -> None:
            colour[node] = 1
            stack.append(node)
            for edge in self._by_source.get(node, ()):
                target = edge.target
                if colour.get(target, 0) == 0:
                    visit(target)
                elif colour.get(target) == 1 and target in stack:
                    cycle = tuple(stack[stack.index(target) :])
                    signature = frozenset(cycle)
                    if signature not in seen_signatures:
                        seen_signatures.add(signature)
                        cycles.append(cycle)
            stack.pop()
            colour[node] = 2

        for node in list(self._by_source):
            if colour.get(node, 0) == 0:
                visit(node)
        return tuple(cycles)

    def restricted_to(self, nodes: Iterable[str]) -> FallbackGraph:
        """A subgraph over `nodes`, dropping edges that leave the set."""
        allowed = set(nodes)
        return FallbackGraph(
            edge for edge in self._edges if edge.source in allowed and edge.target in allowed
        )

    def describe(self, *, root: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"edges": [edge.as_dict() for edge in self._edges]}
        if root is not None:
            payload["root"] = root
        cycles = self.detect_cycles()
        if cycles:
            payload["cycles"] = [list(cycle) for cycle in cycles]
        return payload

    def __len__(self) -> int:
        return len(self._edges)

    def __bool__(self) -> bool:
        return bool(self._edges)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_chain(
        cls, chain: Sequence[str], *, reasons: frozenset[FallbackReason] = ANY_REASON
    ) -> FallbackGraph:
        """Build the degenerate linear graph from an ordered candidate list.

        This is what a flat fallback list *means* once expressed as a graph, and
        it is how a policy ordering with no operator-declared edges is
        represented. Useful, but the constitution's point stands: a linear chain
        answers every failure the same way.
        """
        return cls(
            FallbackEdge(source=chain[index], target=chain[index + 1], reasons=reasons)
            for index in range(len(chain) - 1)
        )

    @classmethod
    def from_config(cls, entries: Iterable[Mapping[str, Any]]) -> FallbackGraph:
        edges: list[FallbackEdge] = []
        for entry in entries:
            source = entry.get("from")
            target = entry.get("to")
            if not source or not target:
                raise ConfigurationError("each fallback edge needs 'from' and 'to'")
            raw_reasons = entry.get("reasons") or entry.get("on")
            if not raw_reasons:
                reasons = ANY_REASON
            else:
                if isinstance(raw_reasons, str):
                    raw_reasons = [raw_reasons]
                reasons = frozenset(_parse_reason(value) for value in raw_reasons)
            edges.append(
                FallbackEdge(
                    source=str(source),
                    target=str(target),
                    reasons=reasons,
                    note=str(entry.get("note", "")),
                )
            )
        return cls(edges)


def _parse_reason(value: Any) -> FallbackReason:
    try:
        return FallbackReason(str(value).strip().lower())
    except ValueError:
        raise ConfigurationError(
            f"unknown fallback reason {value!r} "
            f"(available: {[member.value for member in FallbackReason]})"
        ) from None


@dataclass(frozen=True, slots=True)
class FallbackHop:
    """One recorded move along the graph, for the decision object."""

    source: str
    target: str
    reason: FallbackReason
    depth: int
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "from": self.source,
            "to": self.target,
            "reason": self.reason.value,
            "depth": self.depth,
            "error": self.error,
        }


@dataclass(slots=True)
class FallbackTrace:
    """Every hop taken while serving one request."""

    hops: list[FallbackHop] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)

    def record(self, hop: FallbackHop) -> None:
        self.hops.append(hop)

    def refuse(self, why: str) -> None:
        """Note a hop the budget or the graph would not permit."""
        self.refused.append(why)

    @property
    def depth(self) -> int:
        return len(self.hops)

    def as_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "hops": [hop.as_dict() for hop in self.hops],
            "refused": list(self.refused),
        }
