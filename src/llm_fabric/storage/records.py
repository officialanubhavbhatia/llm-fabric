"""Records held by the tenant-scoped repositories.

Every record carries its owning `tenant_id` as a first-class field rather than
relying on where it happens to be stored. That redundancy is what lets the
storage layer re-check ownership on read instead of trusting its own indexing.

Field sets follow the constitution where it names them, so that later phases
extend behaviour rather than reshape storage. Only persistence and tenant
scoping are implemented here: the prompt promotion workflow, the intent
taxonomy lifecycle and the evaluation engine are later phases.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


class PromptStatus(StrEnum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    EVALUATED = "evaluated"
    CANARY = "canary"
    PRODUCTION = "production"
    RETIRED = "retired"


#: Once a prompt version reaches these states it has been consumed by traffic or
#: by an evaluation, so its content is frozen.
PUBLISHED_PROMPT_STATUSES = frozenset(
    {PromptStatus.CANARY, PromptStatus.PRODUCTION, PromptStatus.RETIRED}
)


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    role: str
    content: str
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class Conversation:
    tenant_id: str
    user_id: str
    conversation_id: str = field(default_factory=lambda: _new_id("conv"))
    title: str | None = None
    messages: tuple[ConversationMessage, ...] = ()
    project_id: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class TraceSpan:
    name: str
    started_at: float
    duration_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TraceRecord:
    tenant_id: str
    trace_id: str
    request_id: str
    user_id: str | None = None
    project_id: str | None = None
    spans: tuple[TraceSpan, ...] = ()
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class IntentExample:
    """A labelled example belonging to one tenant's taxonomy."""

    tenant_id: str
    text: str
    intent_id: str
    taxonomy_version: str
    example_id: str = field(default_factory=lambda: _new_id("intex"))
    is_hard_negative: bool = False
    source: str = "manual"
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class PromptDefinition:
    """One immutable version of a prompt."""

    tenant_id: str
    prompt_id: str
    version: int
    owner: str
    purpose: str
    template: str
    supported_intents: tuple[str, ...] = ()
    supported_model_families: tuple[str, ...] = ()
    variables: tuple[str, ...] = ()
    output_contract: str | None = None
    evaluation_suite: str | None = None
    status: PromptStatus = PromptStatus.DRAFT
    created_at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        return f"{self.prompt_id}@{self.version}"

    @property
    def is_published(self) -> bool:
        return self.status in PUBLISHED_PROMPT_STATUSES


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """An administrative action. Secrets must never appear in before/after."""

    tenant_id: str
    actor: str
    action: str
    target: str
    event_id: str = field(default_factory=lambda: _new_id("aud"))
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    reason: str | None = None
    request_id: str | None = None
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class EvalExample:
    input: str
    expected: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalDataset:
    tenant_id: str
    name: str
    dataset_id: str = field(default_factory=lambda: _new_id("evds"))
    description: str | None = None
    examples: tuple[EvalExample, ...] = ()
    created_at: float = field(default_factory=time.time)
