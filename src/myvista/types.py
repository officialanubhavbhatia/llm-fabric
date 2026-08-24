"""Response shapes the SDK returns. Extra gateway fields are kept, not dropped."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class FabricProvenance:
    """MyVista extensions from `x-fabric-*` headers or a stream trailer."""

    requested_model: str | None = None
    served_model: str | None = None
    provider: str | None = None
    policy: str | None = None
    failovers: int | None = None
    fallback_depth: int | None = None
    intent: str | None = None
    intent_confidence: float | None = None

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> FabricProvenance:
        lowered = {key.lower(): value for key, value in headers.items()}

        def _get(name: str) -> str | None:
            return lowered.get(name)

        failovers = _get("x-fabric-failovers")
        depth = _get("x-fabric-fallback-depth")
        confidence = _get("x-fabric-intent-confidence")
        return cls(
            requested_model=_get("x-fabric-requested-model"),
            served_model=_get("x-fabric-served-model"),
            provider=_get("x-fabric-provider"),
            policy=_get("x-fabric-policy"),
            failovers=int(failovers) if failovers is not None else None,
            fallback_depth=int(depth) if depth is not None else None,
            intent=_get("x-fabric-intent"),
            intent_confidence=float(confidence) if confidence is not None else None,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> FabricProvenance:
        if not payload:
            return cls()
        return cls(
            requested_model=payload.get("requested_model"),
            served_model=payload.get("served_model"),
            provider=payload.get("provider"),
            policy=payload.get("policy"),
            failovers=payload.get("failovers"),
            fallback_depth=payload.get("fallback_depth"),
            intent=payload.get("intent"),
            intent_confidence=payload.get("intent_confidence"),
        )


@dataclass(slots=True)
class ChatCompletion:
    id: str
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any]
    created: int | None
    request_id: str | None
    fabric: FabricProvenance
    raw: dict[str, Any] = field(repr=False)

    @property
    def text(self) -> str:
        if not self.choices:
            return ""
        message = self.choices[0].get("message") or {}
        return str(message.get("content") or "")

    @classmethod
    def from_http(cls, payload: dict[str, Any], headers: dict[str, str]) -> ChatCompletion:
        lowered = {key.lower(): value for key, value in headers.items()}
        return cls(
            id=str(payload.get("id") or ""),
            model=str(payload.get("model") or ""),
            choices=list(payload.get("choices") or []),
            usage=dict(payload.get("usage") or {}),
            created=payload.get("created"),
            request_id=lowered.get("x-fabric-request-id"),
            fabric=FabricProvenance.from_headers(headers),
            raw=payload,
        )


@dataclass(slots=True)
class ChatChunk:
    id: str
    model: str
    delta: str
    finish_reason: str | None
    usage: dict[str, Any] | None
    fabric: FabricProvenance | None
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ChatChunk:
        choices = payload.get("choices") or []
        first = choices[0] if choices else {}
        delta = first.get("delta") or {}
        return cls(
            id=str(payload.get("id") or ""),
            model=str(payload.get("model") or ""),
            delta=str(delta.get("content") or ""),
            finish_reason=first.get("finish_reason"),
            usage=payload.get("usage"),
            fabric=FabricProvenance.from_payload(payload.get("x_fabric")),
            raw=payload,
        )


@dataclass(slots=True)
class Response:
    """`responses.create` result. `output_text` is the only field most callers need."""

    id: str
    model: str
    output_text: str
    request_id: str | None
    fabric: FabricProvenance
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_completion(cls, completion: ChatCompletion) -> Response:
        return cls(
            id=completion.id,
            model=completion.model,
            output_text=completion.text,
            request_id=completion.request_id,
            fabric=completion.fabric,
            raw=completion.raw,
        )
