"""Capability vectors: what a deployment can do, as a comparable value.

Routing on model names is the failure this module exists to prevent. A rule that
says "use GPT-4o when the prompt needs vision" breaks the moment the fleet
changes; a rule that says "use a deployment whose vector contains `vision`"
survives it. The registry stores vectors, the planner filters on them, and no
routing code anywhere is allowed to test a model id for a capability.

The vocabulary is the union of three sources that must agree, because they meet
here: the six `supports_*` flags the constitution mandates on a graded model, the
capabilities the bootstrap intent taxonomy asks for in
`IntentProfile.required_capabilities`, and the free-form strings already present
in `config/models.yaml`. Unrecognised strings are preserved rather than rejected,
so an operator can declare a capability this code has never heard of and match on
it; what they cannot do is have it silently mean something.

Capabilities imply each other in exactly one place, and the implication is
directional: a deployment that can be held to a JSON *schema* can necessarily
produce structured output, but the converse is not true. Matching therefore
compares against the implied closure rather than the declared set.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum


class Capability(StrEnum):
    """Canonical capability names.

    Membership here is not a claim that any particular deployment has the
    capability; it is only the spelling the fabric agrees on.
    """

    CHAT = "chat"
    STREAMING = "streaming"
    REASONING = "reasoning"
    CODE = "code"
    TOOLS = "tools"
    JSON_SCHEMA = "json_schema"
    STRUCTURED_OUTPUT = "structured_output"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    PREFIX_CACHE = "prefix_cache"
    SPECULATIVE_DECODE = "speculative_decode"
    RETRIEVAL = "retrieval"
    MULTILINGUAL = "multilingual"
    AGENT = "agent"


#: Directional implications, applied transitively when matching. Kept
#: deliberately small: every entry is a claim that one capability guarantees
#: another, and a wrong entry silently routes work to a deployment that cannot
#: do it.
IMPLIES: dict[str, frozenset[str]] = {
    Capability.JSON_SCHEMA: frozenset({Capability.STRUCTURED_OUTPUT}),
}


def _closure(declared: frozenset[str]) -> frozenset[str]:
    """Expand a declared set under `IMPLIES` until it stops growing."""
    effective = set(declared)
    pending = list(declared)
    while pending:
        implied = IMPLIES.get(pending.pop(), frozenset())
        for capability in implied:
            if capability not in effective:
                effective.add(capability)
                pending.append(capability)
    return frozenset(effective)


def normalise(values: Iterable[str] | str | None) -> frozenset[str]:
    """Read capabilities from configuration, which may give one or many."""
    if not values:
        return frozenset()
    if isinstance(values, str):
        return frozenset({values.strip().lower()}) if values.strip() else frozenset()
    return frozenset(str(value).strip().lower() for value in values if str(value).strip())


@dataclass(frozen=True, slots=True)
class CapabilityVector:
    """The capabilities of one deployment, and the questions worth asking of them."""

    declared: frozenset[str] = frozenset()
    effective: frozenset[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "effective", _closure(self.declared))

    @classmethod
    def of(cls, *capabilities: str) -> CapabilityVector:
        return cls(frozenset(capabilities))

    @classmethod
    def from_config(cls, values: Iterable[str] | str | None) -> CapabilityVector:
        return cls(normalise(values))

    def has(self, capability: str) -> bool:
        return capability in self.effective

    def satisfies(self, required: Iterable[str]) -> bool:
        return set(required) <= self.effective

    def missing(self, required: Iterable[str]) -> frozenset[str]:
        """The required capabilities this vector lacks. Empty means eligible."""
        return frozenset(set(required) - self.effective)

    def union(self, other: CapabilityVector) -> CapabilityVector:
        return CapabilityVector(self.declared | other.declared)

    # The constitution names six capabilities as `supports_*` attributes on a
    # graded model. They are derived from the vector rather than stored
    # separately so there is one source of truth to disagree with.

    @property
    def supports_tools(self) -> bool:
        return self.has(Capability.TOOLS)

    @property
    def supports_json_schema(self) -> bool:
        return self.has(Capability.JSON_SCHEMA)

    @property
    def supports_vision(self) -> bool:
        return self.has(Capability.VISION)

    @property
    def supports_embeddings(self) -> bool:
        return self.has(Capability.EMBEDDINGS)

    @property
    def supports_prefix_cache(self) -> bool:
        return self.has(Capability.PREFIX_CACHE)

    @property
    def supports_speculative_decode(self) -> bool:
        return self.has(Capability.SPECULATIVE_DECODE)

    def as_dict(self) -> dict[str, object]:
        return {
            "declared": sorted(self.declared),
            "effective": sorted(self.effective),
            "supports_tools": self.supports_tools,
            "supports_json_schema": self.supports_json_schema,
            "supports_vision": self.supports_vision,
            "supports_embeddings": self.supports_embeddings,
            "supports_prefix_cache": self.supports_prefix_cache,
            "supports_speculative_decode": self.supports_speculative_decode,
        }

    def __contains__(self, capability: object) -> bool:
        return isinstance(capability, str) and self.has(capability)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self.effective))
