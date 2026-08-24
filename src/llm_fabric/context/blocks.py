"""The typed unit of context.

Everything the fabric puts in front of a model is a `ContextBlock`, never a bare
string. That matters for three reasons the constitution names directly.

First, tenancy. A block knows which tenant it belongs to, so the compiler can
reject a foreign block instead of trusting whoever assembled the list.

Second, trust. A retrieved document and a system policy are not the same kind of
text, and placing them as though they were is how prompt injection succeeds. The
trust level travels with the content and decides where it may sit.

Third, honesty about size. `token_count` is `None` until something has actually
counted it. A block that has not been measured reports "not measured" rather
than zero, because a zero would silently pass a budget check it never satisfied.
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Self

from llm_fabric.errors import ConfigurationError


class ContextType(StrEnum):
    """The ten context types named by the constitution.

    The order of declaration is not the order of assembly. Placement is decided
    by `ordering.py` from trust and cacheability, not by this enum.
    """

    SYSTEM = "system"
    DEVELOPER_POLICY = "developer_policy"
    TENANT_POLICY = "tenant_policy"
    CONVERSATION = "conversation"
    LONG_TERM_MEMORY = "long_term_memory"
    RETRIEVAL = "retrieval"
    TOOL_RESULT = "tool_result"
    AGENT_STATE = "agent_state"
    USER = "user"
    OUTPUT_CONTRACT = "output_contract"


class TrustLevel(StrEnum):
    """How much authority the content in a block carries.

    Ordered from most to least trusted. The ordering is load-bearing: content at
    a lower trust level must never be placed where the model would read it as an
    instruction from a higher one.
    """

    SYSTEM = "system"
    DEVELOPER = "developer"
    TENANT = "tenant"
    USER = "user"
    TOOL = "tool"
    RETRIEVED = "retrieved"
    UNTRUSTED = "untrusted"

    @property
    def ordinal(self) -> int:
        """Position in the trust order, 0 being the most trusted."""
        return _TRUST_ORDER.index(self)

    @property
    def is_authoritative(self) -> bool:
        """True when content at this level may issue instructions.

        Everything below is data the model is being shown, not told.
        """
        return self.ordinal <= TrustLevel.TENANT.ordinal


_TRUST_ORDER: tuple[TrustLevel, ...] = (
    TrustLevel.SYSTEM,
    TrustLevel.DEVELOPER,
    TrustLevel.TENANT,
    TrustLevel.USER,
    TrustLevel.TOOL,
    TrustLevel.RETRIEVED,
    TrustLevel.UNTRUSTED,
)


class Priority(StrEnum):
    """What the compiler is allowed to drop when the budget binds.

    `REQUIRED` is a contract, not a preference: a required block is never
    dropped, never truncated and never compressed. If the required blocks alone
    exceed the budget, compilation fails loudly rather than quietly shipping a
    prompt with the safety policy cut out of it.
    """

    REQUIRED = "required"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    OPTIONAL = "optional"

    @property
    def ordinal(self) -> int:
        return _PRIORITY_ORDER.index(self)

    @property
    def weight(self) -> float:
        """Priority rescaled to [0, 1] for blending into a relevance score."""
        return 1.0 - (self.ordinal / (len(_PRIORITY_ORDER) - 1))

    @property
    def is_droppable(self) -> bool:
        return self is not Priority.REQUIRED


_PRIORITY_ORDER: tuple[Priority, ...] = (
    Priority.REQUIRED,
    Priority.HIGH,
    Priority.NORMAL,
    Priority.LOW,
    Priority.OPTIONAL,
)


class Cacheability(StrEnum):
    """How stable a block is across requests.

    This drives stable-prefix ordering. Provider prefix caches key on an exact
    leading byte sequence, so putting a volatile block early invalidates the
    cache for everything behind it.
    """

    STABLE = "stable"
    SEMI_STABLE = "semi_stable"
    VOLATILE = "volatile"
    NEVER = "never"

    @property
    def ordinal(self) -> int:
        return _CACHEABILITY_ORDER.index(self)


_CACHEABILITY_ORDER: tuple[Cacheability, ...] = (
    Cacheability.STABLE,
    Cacheability.SEMI_STABLE,
    Cacheability.VOLATILE,
    Cacheability.NEVER,
)


class Visibility(StrEnum):
    """Who may see a block's content in an explanation.

    Visibility has no effect on what reaches the model. It governs only what the
    debug endpoint will render. `INTERNAL` is the setting for hidden security
    policy: the block is compiled into the prompt and appears in the assembly
    report as a typed, counted row, but its text is never returned to a caller.
    """

    CALLER = "caller"
    TENANT = "tenant"
    INTERNAL = "internal"

    @property
    def is_disclosable(self) -> bool:
        return self is not Visibility.INTERNAL


@dataclass(frozen=True, slots=True)
class Freshness:
    """When the content was produced, and how long it stays useful.

    Optional, because plenty of context has no meaningful age. A system prompt
    is not stale; a retrieved price is.
    """

    produced_at: float
    ttl_s: float | None = None

    def __post_init__(self) -> None:
        if self.ttl_s is not None and self.ttl_s <= 0:
            raise ConfigurationError("a freshness ttl must be positive when set")

    def age_s(self, now: float | None = None) -> float:
        return max(0.0, (time.time() if now is None else now) - self.produced_at)

    def is_stale(self, now: float | None = None) -> bool:
        if self.ttl_s is None:
            return False
        return self.age_s(now) > self.ttl_s

    def decay(self, now: float | None = None) -> float:
        """A [0, 1] freshness weight for ranking, 1.0 being brand new.

        Blocks without a TTL never decay. Declaring no TTL is a claim that age
        does not matter for this content, so inventing a decay curve for it
        would be inventing a fact.
        """
        if self.ttl_s is None:
            return 1.0
        return max(0.0, 1.0 - (self.age_s(now) / self.ttl_s))


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a block came from.

    Every field is optional except `origin`, because provenance is recorded from
    whatever the producing subsystem actually knows. A retriever knows its
    document and chunk ids; a system prompt knows only that it is the system
    prompt. Filling the gaps with plausible values would defeat the point.
    """

    origin: str
    producer: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    uri: str | None = None
    retriever_score: float | None = None
    retrieved_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.origin or not self.origin.strip():
            raise ConfigurationError("provenance requires a non-empty origin")

    def as_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "producer": self.producer,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "uri": self.uri,
            "retriever_score": self.retriever_score,
            "retrieved_at": self.retrieved_at,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """One typed, owned, measurable piece of context."""

    block_id: str
    content: str
    type: ContextType
    tenant_id: str
    provenance: Provenance
    trust_level: TrustLevel = TrustLevel.RETRIEVED
    priority: Priority = Priority.NORMAL
    cacheability: Cacheability = Cacheability.VOLATILE
    visibility: Visibility = Visibility.CALLER
    freshness: Freshness | None = None
    token_count: int | None = None
    #: Set when a stage replaced the original content, so the assembly report can
    #: say a block was compressed rather than merely smaller than expected.
    derived_from: str | None = None

    def __post_init__(self) -> None:
        if not self.block_id or not self.block_id.strip():
            raise ConfigurationError("a context block requires a non-empty id")
        if not self.tenant_id or not self.tenant_id.strip():
            raise ConfigurationError("a context block requires a non-empty tenant id")
        if self.token_count is not None and self.token_count < 0:
            raise ConfigurationError("a token count cannot be negative")

    @property
    def tokens(self) -> int:
        """The measured token count.

        Raises rather than defaulting to zero. An unmeasured block that reported
        zero would pass every budget check by pretending to be free.
        """
        if self.token_count is None:
            raise ConfigurationError(
                f"block {self.block_id!r} has not been counted; "
                "the compiler counts on admission, before any budget is applied"
            )
        return self.token_count

    @property
    def is_counted(self) -> bool:
        return self.token_count is not None

    @property
    def fingerprint(self) -> str:
        """A hash of normalised content, for exact deduplication.

        Normalisation folds case, collapses whitespace and applies NFKC, so two
        blocks that differ only in formatting are recognised as the same text.
        It deliberately does not fold punctuation: "delete the row" and "delete
        the row?" are not reliably the same instruction.
        """
        return hashlib.sha256(normalise(self.content).encode("utf-8")).hexdigest()

    def with_tokens(self, count: int) -> Self:
        return replace(self, token_count=count)

    def with_content(self, content: str, *, token_count: int | None = None) -> Self:
        """Return a copy carrying replaced content.

        The copy records the id it was derived from and drops any stale token
        count, so the next stage is forced to re-measure rather than inherit a
        number that described different text.
        """
        return replace(
            self,
            content=content,
            token_count=token_count,
            derived_from=self.derived_from or self.block_id,
        )

    def describe(self, *, include_content: bool) -> dict[str, Any]:
        """A report row for this block.

        `include_content` is decided by the caller's right to see it, never by
        the block itself, so there is exactly one place where that judgement is
        made and it can be tested.
        """
        row: dict[str, Any] = {
            "block_id": self.block_id,
            "type": self.type.value,
            "trust_level": self.trust_level.value,
            "priority": self.priority.value,
            "cacheability": self.cacheability.value,
            "visibility": self.visibility.value,
            "token_count": self.token_count,
            "provenance": self.provenance.as_dict(),
            "derived_from": self.derived_from,
        }
        if self.freshness is not None:
            row["freshness"] = {
                "produced_at": self.freshness.produced_at,
                "ttl_s": self.freshness.ttl_s,
            }
        row["content"] = self.content if include_content else None
        row["content_withheld"] = not include_content
        return row


_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Fold away differences that do not change meaning.

    Shared by exact deduplication and the semantic redundancy check so both
    agree on what "the same text" means.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()
