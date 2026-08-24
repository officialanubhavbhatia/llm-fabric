"""The versioned intent taxonomy.

Two rules shape this module.

**Historical versions are never mutated.** A stored classification records the
taxonomy version that produced it, and that record is worthless if the meaning
of an intent can change underneath it. So a taxonomy is immutable once built,
and `evolve()` returns a *new* version rather than editing the old one.

**Hierarchy is carried in the id.** An intent id is a dotted path —
`coding`, `coding.debug`, `coding.debug.stacktrace` — so `domain`, `task` and
`subtask` are derivable rather than separately maintained fields that can drift
out of agreement with the tree.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType

from llm_fabric.errors import ConfigurationError, ResourceNotFoundError
from llm_fabric.intent.schema import UNKNOWN_INTENT_ID, IntentProfile

#: Guards against a malformed tree turning traversal into an infinite loop.
MAX_TAXONOMY_DEPTH = 8


class IntentStatus(StrEnum):
    EXPERIMENTAL = "experimental"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

    @property
    def is_classifiable(self) -> bool:
        """Retired intents stay readable for old records but win no new traffic."""
        return self in (IntentStatus.EXPERIMENTAL, IntentStatus.ACTIVE, IntentStatus.DEPRECATED)


@dataclass(frozen=True, slots=True)
class IntentNode:
    """One intent in one version of the taxonomy."""

    intent_id: str
    name: str
    description: str
    taxonomy_version: str = ""
    parent_intent_id: str | None = None
    examples: tuple[str, ...] = ()
    counterexamples: tuple[str, ...] = ()
    #: Prompts that look like this intent but are not. These are the examples
    #: that actually move a classifier, and the ones a benchmark should weight.
    hard_negatives: tuple[str, ...] = ()
    required_capabilities: frozenset[str] = frozenset()
    default_route_policy: str | None = None
    status: IntentStatus = IntentStatus.ACTIVE
    profile: IntentProfile = field(default_factory=IntentProfile)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.intent_id or not self.intent_id.strip():
            raise ConfigurationError("an intent needs a non-empty id")
        if self.intent_id == UNKNOWN_INTENT_ID:
            raise ConfigurationError(
                f"'{UNKNOWN_INTENT_ID}' is reserved for abstention and cannot be "
                "declared as a taxonomy node"
            )
        if any(part == "" for part in self.intent_id.split(".")):
            raise ConfigurationError(f"malformed intent id: '{self.intent_id}'")

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(self.intent_id.split("."))

    @property
    def depth(self) -> int:
        return len(self.path)

    @property
    def domain(self) -> str:
        return self.path[0]

    @property
    def task(self) -> str | None:
        return self.path[1] if len(self.path) > 1 else None

    @property
    def subtask(self) -> str | None:
        return ".".join(self.path[2:]) if len(self.path) > 2 else None

    @property
    def implied_parent_id(self) -> str | None:
        return ".".join(self.path[:-1]) if len(self.path) > 1 else None

    def training_texts(self) -> tuple[str, ...]:
        """Positive examples only.

        Counterexamples and hard negatives belong to *other* intents, so folding
        them in here would teach a classifier the opposite of the intended
        lesson.
        """
        return self.examples

    def to_dict(self) -> dict[str, object]:
        return {
            "intent_id": self.intent_id,
            "parent_intent_id": self.parent_intent_id,
            "name": self.name,
            "description": self.description,
            "positive_examples": list(self.examples),
            "counterexamples": list(self.counterexamples),
            "hard_negatives": list(self.hard_negatives),
            "required_capabilities": sorted(self.required_capabilities),
            "default_quality_class": self.profile.quality_class.value,
            "default_latency_class": self.profile.latency_class.value,
            "default_context_class": self.profile.context_class.value,
            "default_route_policy": self.default_route_policy,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "taxonomy_version": self.taxonomy_version,
            "status": self.status.value,
        }


class IntentTaxonomy:
    """An immutable, validated set of intents at one version."""

    __slots__ = ("_version", "_nodes", "_children", "_created_at")

    def __init__(
        self,
        version: str,
        nodes: Iterable[IntentNode],
        *,
        created_at: float | None = None,
    ) -> None:
        if not version or not version.strip():
            raise ConfigurationError("a taxonomy needs a version")

        self._version = version
        self._created_at = created_at if created_at is not None else time.time()

        resolved: dict[str, IntentNode] = {}
        for node in nodes:
            if node.intent_id in resolved:
                raise ConfigurationError(f"duplicate intent id: '{node.intent_id}'")
            # Nodes are stamped with this taxonomy's version on entry, so a node
            # cannot claim membership of a version it is not in.
            resolved[node.intent_id] = replace(
                node,
                taxonomy_version=version,
                parent_intent_id=node.parent_intent_id or node.implied_parent_id,
            )

        self._nodes: Mapping[str, IntentNode] = MappingProxyType(resolved)
        self._validate()

        children: dict[str, list[str]] = {intent_id: [] for intent_id in resolved}
        for node in resolved.values():
            if node.parent_intent_id is not None:
                children[node.parent_intent_id].append(node.intent_id)
        self._children: Mapping[str, tuple[str, ...]] = MappingProxyType(
            {key: tuple(sorted(value)) for key, value in children.items()}
        )

    # -- identity ------------------------------------------------------------

    @property
    def version(self) -> str:
        return self._version

    @property
    def created_at(self) -> float:
        return self._created_at

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[IntentNode]:
        return iter(self._nodes.values())

    def __contains__(self, intent_id: object) -> bool:
        return intent_id in self._nodes

    def __repr__(self) -> str:
        return f"IntentTaxonomy(version={self._version!r}, intents={len(self._nodes)})"

    def to_dict(self) -> dict[str, object]:
        """A published snapshot. Historical versions are never rewritten."""
        return {
            "taxonomy_version": self._version,
            "created_at": self._created_at,
            "intents": [
                node.to_dict()
                for node in sorted(self._nodes.values(), key=lambda item: item.intent_id)
            ],
        }

    # -- lookup --------------------------------------------------------------

    def get(self, intent_id: str) -> IntentNode | None:
        return self._nodes.get(intent_id)

    def require(self, intent_id: str) -> IntentNode:
        node = self._nodes.get(intent_id)
        if node is None:
            raise ResourceNotFoundError(
                f"no intent '{intent_id}' in taxonomy version '{self._version}'"
            )
        return node

    def children(self, intent_id: str) -> tuple[str, ...]:
        return self._children.get(intent_id, ())

    def roots(self) -> tuple[IntentNode, ...]:
        return tuple(node for node in self._nodes.values() if node.parent_intent_id is None)

    def domains(self) -> tuple[str, ...]:
        return tuple(sorted(node.intent_id for node in self.roots()))

    def leaves(self) -> tuple[IntentNode, ...]:
        return tuple(node for node in self._nodes.values() if not self.children(node.intent_id))

    def classifiable(self) -> tuple[IntentNode, ...]:
        """Nodes a classifier may assign. Retired intents are excluded."""
        return tuple(node for node in self._nodes.values() if node.status.is_classifiable)

    def ancestors(self, intent_id: str) -> tuple[str, ...]:
        """Walk to the root, nearest parent first. Bounded by construction."""
        chain: list[str] = []
        current = self.require(intent_id).parent_intent_id
        while current is not None and len(chain) < MAX_TAXONOMY_DEPTH:
            chain.append(current)
            parent = self._nodes.get(current)
            current = parent.parent_intent_id if parent else None
        return tuple(chain)

    # -- evolution -----------------------------------------------------------

    def evolve(
        self,
        version: str,
        *,
        add: Iterable[IntentNode] = (),
        replace_nodes: Iterable[IntentNode] = (),
        retire: Iterable[str] = (),
    ) -> IntentTaxonomy:
        """Produce the next version. This taxonomy is left untouched.

        Retiring marks a node rather than deleting it, so a historical
        classification that named it can still be explained.
        """
        if version == self._version:
            raise ConfigurationError(
                f"taxonomy version '{version}' already exists; "
                "historical versions are immutable and must not be rewritten"
            )

        merged: dict[str, IntentNode] = dict(self._nodes)

        for node in add:
            if node.intent_id in merged:
                raise ConfigurationError(
                    f"intent '{node.intent_id}' already exists; use replace_nodes to change it"
                )
            merged[node.intent_id] = node

        for node in replace_nodes:
            if node.intent_id not in merged:
                raise ConfigurationError(
                    f"cannot replace unknown intent '{node.intent_id}'; use add"
                )
            merged[node.intent_id] = replace(node, updated_at=time.time())

        for intent_id in retire:
            existing = merged.get(intent_id)
            if existing is None:
                raise ConfigurationError(f"cannot retire unknown intent '{intent_id}'")
            merged[intent_id] = replace(
                existing, status=IntentStatus.RETIRED, updated_at=time.time()
            )

        return IntentTaxonomy(version, merged.values())

    # -- validation ----------------------------------------------------------

    def _validate(self) -> None:
        for node in self._nodes.values():
            parent = node.parent_intent_id
            if parent is None:
                continue
            if parent not in self._nodes:
                raise ConfigurationError(
                    f"intent '{node.intent_id}' names a parent "
                    f"'{parent}' that is not in the taxonomy"
                )
            if parent == node.intent_id:
                raise ConfigurationError(f"intent '{node.intent_id}' is its own parent")

        for node in self._nodes.values():
            self._assert_no_cycle(node)

    def _assert_no_cycle(self, node: IntentNode) -> None:
        seen = {node.intent_id}
        current = node.parent_intent_id
        depth = 0
        while current is not None:
            depth += 1
            if current in seen:
                raise ConfigurationError(f"cycle in taxonomy at intent '{current}'")
            if depth > MAX_TAXONOMY_DEPTH:
                raise ConfigurationError(
                    f"intent '{node.intent_id}' nests deeper than {MAX_TAXONOMY_DEPTH} levels"
                )
            seen.add(current)
            parent = self._nodes.get(current)
            current = parent.parent_intent_id if parent else None


class TaxonomyRegistry:
    """Holds every taxonomy version the process knows about.

    Registration is append-only. Replacing a registered version is refused,
    because a classification that recorded it would silently change meaning.
    """

    def __init__(self, taxonomies: Iterable[IntentTaxonomy] = ()) -> None:
        self._versions: dict[str, IntentTaxonomy] = {}
        self._order: list[str] = []
        for taxonomy in taxonomies:
            self.register(taxonomy)

    def register(self, taxonomy: IntentTaxonomy) -> IntentTaxonomy:
        if taxonomy.version in self._versions:
            raise ConfigurationError(
                f"taxonomy version '{taxonomy.version}' is already registered; "
                "historical versions are immutable"
            )
        self._versions[taxonomy.version] = taxonomy
        self._order.append(taxonomy.version)
        return taxonomy

    def get(self, version: str) -> IntentTaxonomy | None:
        return self._versions.get(version)

    def require(self, version: str) -> IntentTaxonomy:
        taxonomy = self._versions.get(version)
        if taxonomy is None:
            raise ResourceNotFoundError(f"unknown taxonomy version '{version}'")
        return taxonomy

    def latest(self) -> IntentTaxonomy:
        if not self._order:
            raise ConfigurationError("no taxonomy has been registered")
        return self._versions[self._order[-1]]

    @property
    def versions(self) -> tuple[str, ...]:
        return tuple(self._order)

    def __len__(self) -> int:
        return len(self._versions)
