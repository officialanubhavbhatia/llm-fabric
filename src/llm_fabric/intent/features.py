"""Per-request features the cheap layers can compute without a model.

Taxonomy profiles carry the *typical* shape of an intent. A specific prompt can
be shorter, longer, more constrained, or more conversational than that typical
shape. These helpers override the profile in those cases, and they bound what
the cascade is allowed to read so a 100k-token paste cannot become a classifier
input.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from llm_fabric.intent.schema import (
    ClassificationRequest,
    Complexity,
    ContextClass,
    IntentProfile,
    ReasoningLevel,
)

#: Characters of user text any layer may see. Intent is almost always visible
#: in the opening and closing; the middle of a huge paste is noise.
MAX_CLASSIFIER_CHARS = 4_000

_CONSTRAINT = re.compile(
    r"\b(must|should|cannot|don't|do not|without|at least|no more than|"
    r"exactly|required|constraint)\b",
    re.IGNORECASE,
)
_MULTI_STEP = re.compile(
    r"\b(then|after that|and then|step \d|first .+, then)\b",
    re.IGNORECASE,
)
_CODE_FENCE = re.compile(r"```")
_JSON_HINT = re.compile(r"\b(json|schema|typed dict|pydantic|structured output)\b", re.IGNORECASE)


def bound_text(text: str, *, max_chars: int = MAX_CLASSIFIER_CHARS) -> str:
    """Keep the start and end of a long prompt; drop the middle.

    Feeding an unbounded prompt into every layer is a latency, cost and
    denial-of-service problem. Classification does not need the whole body.
    """
    if len(text) <= max_chars:
        return text
    head = (max_chars * 3) // 4
    tail = max_chars - head - 5
    if tail < 32:
        return text[:max_chars]
    return f"{text[:head]}\n…\n{text[-tail:]}"


def conversation_state_signature(messages: Sequence[object], *, bound: int = 4) -> str:
    """A stable digest of recent conversation, not the conversation itself.

    Cache keys and classifiers need to know *whether* prior turns exist and
    roughly what they were about, without storing or re-sending the full
    history. Empty input yields an empty signature so request-only
    classifications stay request-only.
    """
    if not messages:
        return ""
    parts: list[str] = []
    recent = list(messages)[-bound:]
    for message in recent:
        role = getattr(message, "role", None) or (
            message.get("role") if isinstance(message, dict) else ""
        )
        content = getattr(message, "content", None) or (
            message.get("content") if isinstance(message, dict) else ""
        )
        if not isinstance(content, str):
            continue
        parts.append(f"{role}:{bound_text(content, max_chars=240)}")
    if not parts:
        return ""
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:24]


def infer_profile_overrides(request: ClassificationRequest, base: IntentProfile) -> IntentProfile:
    """Adjust complexity and context from this prompt, not from prompt length alone."""
    text = bound_text(request.text)
    constraints = len(_CONSTRAINT.findall(text))
    multi_step = bool(_MULTI_STEP.search(text))
    fences = len(_CODE_FENCE.findall(text))
    chars = len(request.text)

    complexity = base.complexity
    if constraints >= 4 or (multi_step and fences >= 2):
        complexity = _at_least(complexity, Complexity.COMPLEX)
    elif constraints >= 2 or multi_step:
        complexity = _at_least(complexity, Complexity.MODERATE)
    elif chars < 40 and constraints == 0 and not multi_step:
        complexity = _at_most(complexity, Complexity.SIMPLE)

    reasoning = base.reasoning_level
    if multi_step and constraints >= 2:
        reasoning = _reasoning_at_least(reasoning, ReasoningLevel.MODERATE)

    context = _context_class_for(chars, fences, base.context_class)
    structured = base.structured_output or bool(_JSON_HINT.search(text))

    if (
        complexity is base.complexity
        and reasoning is base.reasoning_level
        and context is base.context_class
        and structured is base.structured_output
    ):
        return base

    return IntentProfile(
        complexity=complexity,
        reasoning_level=reasoning,
        modality=base.modality,
        context_class=context,
        risk_class=base.risk_class,
        latency_class=base.latency_class,
        quality_class=base.quality_class,
        cost_class=base.cost_class,
        agent_required=base.agent_required,
        tools_required=base.tools_required,
        structured_output=structured,
        required_capabilities=base.required_capabilities,
    )


def _context_class_for(chars: int, fences: int, fallback: ContextClass) -> ContextClass:
    if chars > 20_000 or fences >= 4:
        return ContextClass.VERY_LONG
    if chars > 8_000 or fences >= 2:
        return ContextClass.LONG
    if chars > 2_000:
        return ContextClass.MEDIUM
    if chars < 80 and fences == 0:
        return ContextClass.TINY
    return fallback


_COMPLEXITY_ORDER = (
    Complexity.TRIVIAL,
    Complexity.SIMPLE,
    Complexity.MODERATE,
    Complexity.COMPLEX,
    Complexity.VERY_COMPLEX,
)
_REASONING_ORDER = (
    ReasoningLevel.NONE,
    ReasoningLevel.LIGHT,
    ReasoningLevel.MODERATE,
    ReasoningLevel.DEEP,
    ReasoningLevel.EXTENDED,
)


def _at_least(current: Complexity, floor: Complexity) -> Complexity:
    return floor if _COMPLEXITY_ORDER.index(current) < _COMPLEXITY_ORDER.index(floor) else current


def _at_most(current: Complexity, ceiling: Complexity) -> Complexity:
    return (
        ceiling if _COMPLEXITY_ORDER.index(current) > _COMPLEXITY_ORDER.index(ceiling) else current
    )


def _reasoning_at_least(current: ReasoningLevel, floor: ReasoningLevel) -> ReasoningLevel:
    return floor if _REASONING_ORDER.index(current) < _REASONING_ORDER.index(floor) else current
