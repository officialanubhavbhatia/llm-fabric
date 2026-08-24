"""Five-stage MyVista guardrails.

Stages run in order: INPUT → RETRIEVAL → CONTEXT → EXECUTION → OUTPUT.
Each stage is a pluggable engine that returns allow / block / redact /
transform / escalate and emits a structured decision for telemetry.

The baseline engines are deterministic. Model-backed classifiers are adapters
that may be wired later; they are not on the request path unless configured.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class GuardrailStage(StrEnum):
    INPUT = "input"
    RETRIEVAL = "retrieval"
    CONTEXT = "context"
    EXECUTION = "execution"
    OUTPUT = "output"


class GuardrailAction(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    TRANSFORM = "transform"
    ESCALATE = "escalate"


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    stage: GuardrailStage
    action: GuardrailAction
    policy: str
    reason: str
    transformed: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.action is not GuardrailAction.BLOCK


class GuardrailEngine(Protocol):
    stage: GuardrailStage

    def evaluate(self, payload: Any, *, tenant_id: str) -> GuardrailDecision: ...


_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"-----BEGIN (?:RSA )?PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)
_PII_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN-shaped
    re.compile(r"\b\d{16}\b"),  # bare card-shaped
)
_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous",
    "you are now",
    "jailbreak",
    "developer mode",
)


@dataclass(frozen=True, slots=True)
class InputLimits:
    max_request_bytes: int = 1_000_000
    max_output_tokens: int = 8_192
    max_input_tokens: int = 32_000


class InputGuardrail:
    stage = GuardrailStage.INPUT

    def __init__(self, limits: InputLimits | None = None) -> None:
        self._limits = limits or InputLimits()

    def evaluate(self, payload: Any, *, tenant_id: str) -> GuardrailDecision:
        del tenant_id
        text = _as_text(payload)
        if len(text.encode("utf-8")) > self._limits.max_request_bytes:
            return _block(self.stage, "max_request_size", "request exceeds maximum size")
        from llm_fabric.serving.tokens import approximate_token_count

        if approximate_token_count(text) > self._limits.max_input_tokens:
            return _block(
                self.stage,
                "max_input_tokens",
                "prompt tokens exceed the configured ceiling",
            )
        requested = _requested_output_tokens(payload)
        if requested is not None and requested > self._limits.max_output_tokens:
            return _block(
                self.stage,
                "max_output_tokens",
                "requested output tokens exceed the configured ceiling",
            )
        if _matches(_SECRET_PATTERNS, text):
            return _block(self.stage, "secret_scan", "payload contains a secret pattern")
        if _matches(_PII_PATTERNS, text):
            return GuardrailDecision(
                stage=self.stage,
                action=GuardrailAction.REDACT,
                policy="pii",
                reason="payload matches a PII pattern",
                transformed=_redact(_PII_PATTERNS, text),
            )
        lowered = text.lower()
        if any(marker in lowered for marker in _INJECTION_MARKERS):
            return _block(self.stage, "injection", "payload matches a jailbreak marker")
        return _allow(self.stage, "input")


class RetrievalGuardrail:
    stage = GuardrailStage.RETRIEVAL

    def __init__(self, *, max_documents: int = 32, max_bytes: int = 500_000) -> None:
        self._max_documents = max_documents
        self._max_bytes = max_bytes

    def evaluate(self, payload: Any, *, tenant_id: str) -> GuardrailDecision:
        documents = payload.get("documents") if isinstance(payload, dict) else payload
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            documents = []
        if len(documents) > self._max_documents:
            return _block(self.stage, "retrieval_size", "too many retrieved documents")
        total = sum(len(str(doc).encode("utf-8")) for doc in documents)
        if total > self._max_bytes:
            return _block(self.stage, "retrieval_bytes", "retrieved corpus exceeds byte ceiling")
        for document in documents:
            owner = document.get("tenant_id") if isinstance(document, dict) else None
            if owner is not None and owner != tenant_id:
                return _block(
                    self.stage,
                    "tenant_provenance",
                    "retrieved document is owned by another tenant",
                )
            trust = document.get("trust") if isinstance(document, dict) else None
            if trust == "untrusted":
                return GuardrailDecision(
                    stage=self.stage,
                    action=GuardrailAction.TRANSFORM,
                    policy="untrusted_instruction",
                    reason="untrusted retrieval must not become system policy",
                    metadata={"documents": len(documents)},
                )
        return _allow(self.stage, "retrieval")


class ContextGuardrail:
    stage = GuardrailStage.CONTEXT

    def __init__(self, *, max_blocks: int = 64, max_tokens: int = 32_000) -> None:
        self._max_blocks = max_blocks
        self._max_tokens = max_tokens

    def evaluate(self, payload: Any, *, tenant_id: str) -> GuardrailDecision:
        blocks = payload.get("blocks") if isinstance(payload, dict) else payload
        if not isinstance(blocks, Sequence):
            return _allow(self.stage, "context")
        if len(blocks) > self._max_blocks:
            return _block(self.stage, "context_size", "too many context blocks")
        for block in blocks:
            if not isinstance(block, dict):
                continue
            owner = block.get("tenant_id")
            if owner is not None and owner != tenant_id:
                return _block(self.stage, "tenant_block", "context block belongs to another tenant")
            if block.get("trust") == "retrieved" and block.get("role") == "system":
                return _block(
                    self.stage,
                    "trust_promotion",
                    "retrieved text cannot become higher-trust system policy",
                )
        return _allow(self.stage, "context")


class ExecutionGuardrail:
    stage = GuardrailStage.EXECUTION

    def __init__(
        self,
        *,
        allowed_tools: Sequence[str] = (),
        max_iterations: int = 8,
        timeout_s: float = 30.0,
    ) -> None:
        self._allowed = frozenset(allowed_tools)
        self._max_iterations = max_iterations
        self._timeout_s = timeout_s

    def evaluate(self, payload: Any, *, tenant_id: str) -> GuardrailDecision:
        del tenant_id
        if not isinstance(payload, dict):
            return _allow(self.stage, "execution")
        tool = payload.get("tool")
        if tool and self._allowed and tool not in self._allowed:
            return _block(self.stage, "tool_allowlist", "tool is not on the allowlist")
        iterations = int(payload.get("iterations") or 0)
        if iterations > self._max_iterations:
            return _block(self.stage, "max_iterations", "agent/tool iteration ceiling reached")
        if payload.get("network") == "denied":
            return _block(self.stage, "network_policy", "network access is not permitted")
        if payload.get("filesystem") == "denied":
            return _block(self.stage, "filesystem_policy", "filesystem access is not permitted")
        timeout = payload.get("timeout_s")
        if timeout is not None and float(timeout) > self._timeout_s:
            return _block(self.stage, "timeout", "execution timeout exceeds the configured ceiling")
        args = payload.get("arguments")
        schema = payload.get("schema")
        if schema and args is not None and not isinstance(args, dict):
            return _block(self.stage, "tool_args", "tool arguments must be an object")
        return _allow(self.stage, "execution")


#: Longest prefix of a secret/PII pattern that is not yet a match. Streaming
#: emission holds this many trailing characters so a pattern split across
#: deltas cannot leak before the inspector has seen enough to decide.
STREAMING_HOLDBACK_CHARS = 80


class OutputGuardrail:
    stage = GuardrailStage.OUTPUT

    def __init__(self, *, max_output_bytes: int = 2_000_000) -> None:
        self._max_output_bytes = max_output_bytes

    def evaluate(self, payload: Any, *, tenant_id: str) -> GuardrailDecision:
        del tenant_id
        text = _as_text(payload)
        if len(text.encode("utf-8")) > self._max_output_bytes:
            return _block(self.stage, "max_output", "output exceeds the configured ceiling")
        if _matches(_SECRET_PATTERNS, text):
            return GuardrailDecision(
                stage=self.stage,
                action=GuardrailAction.REDACT,
                policy="secret_scan",
                reason="output contains a secret pattern",
                transformed=_redact(_SECRET_PATTERNS, text),
            )
        if _matches(_PII_PATTERNS, text):
            return GuardrailDecision(
                stage=self.stage,
                action=GuardrailAction.REDACT,
                policy="pii",
                reason="output matches a PII pattern",
                transformed=_redact(_PII_PATTERNS, text),
            )
        schema = payload.get("schema") if isinstance(payload, dict) else None
        output = payload.get("output") if isinstance(payload, dict) else None
        if schema and output is not None and not isinstance(output, dict):
            return _block(self.stage, "structured_output", "structured output failed validation")
        return _allow(self.stage, "output")


class StreamingOutputInspector:
    """Inspect-then-emit for SSE deltas, including patterns that span chunks.

    Per-delta evaluation of `sk-` then `aaaaaaaaaaaaaaaa` would allow the prefix
    through before the secret is complete. This inspector holds a bounded tail
    and only releases prefix that cannot complete a configured pattern.
    """

    def __init__(
        self,
        engine: OutputGuardrail | None = None,
        *,
        holdback: int = STREAMING_HOLDBACK_CHARS,
    ) -> None:
        if holdback < 1:
            raise ValueError("holdback must be at least 1")
        self._engine = engine or OutputGuardrail()
        self._holdback = holdback
        self._buffer = ""

    def push(self, delta: str, *, tenant_id: str) -> tuple[str, GuardrailDecision]:
        return self._release(self._buffer + delta, tenant_id=tenant_id, final=False)

    def flush(self, *, tenant_id: str) -> tuple[str, GuardrailDecision]:
        text, decision = self._release(self._buffer, tenant_id=tenant_id, final=True)
        self._buffer = ""
        return text, decision

    def _release(
        self, combined: str, *, tenant_id: str, final: bool
    ) -> tuple[str, GuardrailDecision]:
        decision = self._engine.evaluate(combined, tenant_id=tenant_id)
        if decision.action is GuardrailAction.BLOCK:
            self._buffer = ""
            return "", decision
        text = (
            str(decision.transformed)
            if decision.action is GuardrailAction.REDACT and decision.transformed is not None
            else combined
        )
        if final or len(text) <= self._holdback:
            self._buffer = "" if final else text
            return (text if final else ""), decision
        emit, self._buffer = text[: -self._holdback], text[-self._holdback :]
        return emit, decision


class GuardrailPipeline:
    """Runs the five stages. A BLOCK stops the chain."""

    def __init__(self, engines: Sequence[GuardrailEngine] | None = None) -> None:
        self._engines = tuple(engines or default_engines())

    def run(self, *, tenant_id: str, **stage_payloads: Any) -> list[GuardrailDecision]:
        decisions: list[GuardrailDecision] = []
        for engine in self._engines:
            payload = stage_payloads.get(engine.stage.value)
            if payload is None:
                continue
            decision = engine.evaluate(payload, tenant_id=tenant_id)
            decisions.append(decision)
            if decision.action is GuardrailAction.BLOCK:
                break
        return decisions


def default_engines() -> tuple[GuardrailEngine, ...]:
    return (
        InputGuardrail(),
        RetrievalGuardrail(),
        ContextGuardrail(),
        ExecutionGuardrail(),
        OutputGuardrail(),
    )


def _as_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict) and "text" in payload:
        return str(payload["text"])
    try:
        return json.dumps(payload, default=str)
    except TypeError:
        return str(payload)


def _requested_output_tokens(payload: Any) -> int | None:
    if isinstance(payload, dict) and payload.get("max_tokens") is not None:
        return int(payload["max_tokens"])
    return None


def _matches(patterns: Sequence[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _redact(patterns: Sequence[re.Pattern[str]], text: str) -> str:
    redacted = text
    for pattern in patterns:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _allow(stage: GuardrailStage, policy: str) -> GuardrailDecision:
    return GuardrailDecision(stage=stage, action=GuardrailAction.ALLOW, policy=policy, reason="ok")


def _block(stage: GuardrailStage, policy: str, reason: str) -> GuardrailDecision:
    return GuardrailDecision(
        stage=stage, action=GuardrailAction.BLOCK, policy=policy, reason=reason
    )
