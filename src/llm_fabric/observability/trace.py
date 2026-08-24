"""Trace context, carried across the request and stamped with tenant identity.

Two things happen here. The fabric honours W3C `traceparent` so a caller's trace
joins up with the fabric's, and it attaches tenant, user and project to that
context so every log line and stored span can be attributed without a lookup.

Tenant identity in telemetry is itself a privacy boundary: identifiers are
stable and opaque, and no credential or prompt content is placed in a trace
attribute by this module.

This is the trace *context*. OpenTelemetry export is not built; the ids here are
W3C-compatible so that instrumentation can adopt them rather than replace them.
"""

from __future__ import annotations

import re
import secrets
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from llm_fabric.identity.claims import Principal

_TRACEPARENT = re.compile(
    r"^(?P<version>[0-9a-f]{2})-"
    r"(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<span_id>[0-9a-f]{16})-"
    r"(?P<flags>[0-9a-f]{2})$"
)

_INVALID_TRACE_ID = "0" * 32
_INVALID_SPAN_ID = "0" * 16


@dataclass(frozen=True, slots=True)
class TraceContext:
    trace_id: str
    span_id: str
    request_id: str
    parent_span_id: str | None = None
    sampled: bool = True
    tenant_id: str | None = None
    user_id: str | None = None
    project_id: str | None = None

    @classmethod
    def start(
        cls,
        *,
        request_id: str,
        traceparent: str | None = None,
        principal: Principal | None = None,
    ) -> TraceContext:
        trace_id, parent_span_id, sampled = _parse_traceparent(traceparent)
        return cls(
            trace_id=trace_id or secrets.token_hex(16),
            span_id=secrets.token_hex(8),
            request_id=request_id,
            parent_span_id=parent_span_id,
            sampled=sampled,
            tenant_id=principal.tenant_id if principal else None,
            user_id=principal.user_id if principal else None,
            project_id=principal.project_id if principal else None,
        )

    def with_principal(self, principal: Principal) -> TraceContext:
        return TraceContext(
            trace_id=self.trace_id,
            span_id=self.span_id,
            request_id=self.request_id,
            parent_span_id=self.parent_span_id,
            sampled=self.sampled,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            project_id=principal.project_id,
        )

    def to_traceparent(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-{'01' if self.sampled else '00'}"

    def log_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "request_id": self.request_id,
        }
        if self.tenant_id:
            fields["tenant_id"] = self.tenant_id
        if self.user_id:
            fields["user_id"] = self.user_id
        if self.project_id:
            fields["project_id"] = self.project_id
        return fields


_current: ContextVar[TraceContext | None] = ContextVar("llm_fabric_trace", default=None)


def current_trace() -> TraceContext | None:
    return _current.get()


def bind_trace(context: TraceContext) -> Token[TraceContext | None]:
    return _current.set(context)


def reset_trace(token: Token[TraceContext | None]) -> None:
    _current.reset(token)


def _parse_traceparent(raw: str | None) -> tuple[str | None, str | None, bool]:
    """Parse an inbound `traceparent`, ignoring anything malformed.

    A caller-supplied header is untrusted input. A bad one is dropped and a
    fresh trace begins rather than propagating a value that would corrupt the
    trace graph.
    """
    if not raw:
        return None, None, True
    match = _TRACEPARENT.match(raw.strip())
    if match is None:
        return None, None, True
    trace_id = match.group("trace_id")
    span_id = match.group("span_id")
    if trace_id == _INVALID_TRACE_ID or span_id == _INVALID_SPAN_ID:
        return None, None, True
    sampled = bool(int(match.group("flags"), 16) & 0x01)
    return trace_id, span_id, sampled
