"""OpenTelemetry instrumentation for the request lifecycle.

The constitution names the stages a request must be able to show: HTTP,
authentication, guardrails, intent, context, retrieval, routing, the provider
call, fallbacks, tools, output validation, evaluations. This module records a
span for a stage that *ran*. A stage that is not built is not given a span
with invented timings — the Command Center lists it as unavailable instead.

Span ids stay W3C-compatible with `TraceContext` so a caller-supplied
`traceparent` continues the same tree rather than starting a second one.
"""

from __future__ import annotations

import os
import socket
import time
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Span as OtelSpan
from opentelemetry.trace import Status, StatusCode

from llm_fabric.observability.trace import current_trace

#: Names the constitution uses for the request tree. Only stages that actually
#: execute receive a span; the rest stay in this list so dashboards can say so.
LIFECYCLE_STAGES: tuple[str, ...] = (
    "request",
    "auth",
    "input_guardrails",
    "intent",
    "context",
    "retrieval",
    "route",
    "litellm",
    "llm",
    "tool",
    "output_guardrails",
    "usage",
    "eval",
)

BUILT_STAGES: frozenset[str] = frozenset(
    {
        "request",
        "auth",
        "input_guardrails",
        "intent",
        "context",
        "route",
        "litellm",
        "llm",
        "output_guardrails",
        "usage",
    }
)

DEFAULT_SPAN_BUFFER = 5_000

_TRACER_NAME = "llm_fabric"


@dataclass(frozen=True, slots=True)
class RecordedSpan:
    """A completed span, kept in-process for the Command Center."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    duration_ms: float
    status: str
    attributes: dict[str, Any]
    tenant_id: str | None = None
    user_id: str | None = None
    started_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "attributes": self.attributes,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "started_at": self.started_at,
        }


class SpanJournal:
    """Bounded in-process span log. Lost on restart, filtered by tenant."""

    def __init__(self, buffer_size: int = DEFAULT_SPAN_BUFFER) -> None:
        self._spans: deque[RecordedSpan] = deque(maxlen=buffer_size)
        self._lock = Lock()

    def record(self, span: RecordedSpan) -> None:
        with self._lock:
            self._spans.append(span)

    def recent(
        self,
        limit: int = 100,
        *,
        tenant_id: str | None = None,
        trace_id: str | None = None,
    ) -> list[RecordedSpan]:
        with self._lock:
            spans = list(self._spans)
        if tenant_id is not None:
            spans = [span for span in spans if span.tenant_id == tenant_id]
        if trace_id is not None:
            spans = [span for span in spans if span.trace_id == trace_id]
        return spans[-limit:][::-1]

    def traces(
        self,
        limit: int = 50,
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Group recent spans into trees, newest trace first."""
        spans = self.recent(limit=DEFAULT_SPAN_BUFFER, tenant_id=tenant_id)
        grouped: dict[str, list[RecordedSpan]] = {}
        order: list[str] = []
        for span in spans:
            if span.trace_id not in grouped:
                grouped[span.trace_id] = []
                order.append(span.trace_id)
            grouped[span.trace_id].append(span)
        trees: list[dict[str, Any]] = []
        for trace_id in order[:limit]:
            members = grouped[trace_id]
            root = next((s for s in members if s.name == "request"), members[0])
            trees.append(
                {
                    "trace_id": trace_id,
                    "tenant_id": root.tenant_id,
                    "user_id": root.user_id,
                    "started_at": root.started_at,
                    "duration_ms": root.duration_ms,
                    "status": root.status,
                    "spans": [span.as_dict() for span in reversed(members)],
                }
            )
        return trees


class FabricTracer:
    """Owns the OpenTelemetry provider and the in-process journal."""

    def __init__(
        self,
        *,
        journal: SpanJournal | None = None,
        exporter: SpanExporter | None = None,
        service_name: str = "llm-fabric",
    ) -> None:
        self.journal = journal or SpanJournal()
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.namespace": "myvista",
                "service.instance.id": os.environ.get("HOSTNAME") or socket.gethostname(),
            }
        )
        provider = TracerProvider(resource=resource)
        if exporter is not None:
            provider.add_span_processor(
                BatchSpanProcessor(
                    exporter,
                    max_queue_size=2_048,
                    schedule_delay_millis=5_000,
                    max_export_batch_size=512,
                    export_timeout_millis=10_000,
                )
            )
        # Always keep a no-export processor so the SDK is initialised even when
        # nothing is listening. Spans still exist for the journal.
        provider.add_span_processor(SimpleSpanProcessor(_NullExporter()))
        self._provider = provider
        self._tracer = provider.get_tracer(_TRACER_NAME)

    @property
    def otel(self) -> trace.Tracer:
        return self._tracer

    def shutdown(self) -> None:
        self._provider.shutdown()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[OtelSpan]:
        """Open a named span, copy W3C ids from the request context, journal it.

        Attribute values must be primitives. Anything else is dropped rather
        than stringified into a prompt or a secret.
        """
        context = current_trace()
        safe = _safe_attributes(attributes)
        if context is not None:
            safe.setdefault("fabric.request_id", context.request_id)
            if context.tenant_id:
                safe.setdefault("fabric.tenant_id", context.tenant_id)
            if context.user_id:
                safe.setdefault("fabric.user_id", context.user_id)
        started = time.perf_counter()
        wall = time.time()
        with self._tracer.start_as_current_span(name) as otel_span:
            for key, value in safe.items():
                otel_span.set_attribute(key, value)
            status = "ok"
            try:
                yield otel_span
            except Exception as exc:
                status = "error"
                otel_span.record_exception(exc)
                otel_span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            finally:
                duration_ms = (time.perf_counter() - started) * 1000
                ctx = otel_span.get_span_context()
                parent = getattr(otel_span, "parent", None)
                # Re-read: the request span opens before identity is bound.
                context = current_trace() or context
                # Prefer the request's W3C ids so every stage of one call
                # lands on the same tree, even when the SDK starts a span
                # before `TraceContext` is bound or in a nested task.
                self.journal.record(
                    RecordedSpan(
                        name=name,
                        trace_id=(
                            context.trace_id
                            if context is not None
                            else (f"{ctx.trace_id:032x}" if ctx.trace_id else "")
                        ),
                        span_id=f"{ctx.span_id:016x}"
                        if ctx.span_id
                        else (context.span_id if context else ""),
                        parent_span_id=(
                            f"{parent.span_id:016x}"
                            if parent is not None
                            else (context.parent_span_id if context else None)
                        ),
                        duration_ms=duration_ms,
                        status=status,
                        attributes=safe,
                        tenant_id=context.tenant_id if context else None,
                        user_id=context.user_id if context else None,
                        started_at=wall,
                    )
                )


class _NullExporter(SpanExporter):
    """Satisfies the SDK without sending anywhere."""

    def export(self, spans: Any) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        del spans
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def _safe_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only types OpenTelemetry accepts. Drop the rest rather than coerce.

    Coercing would be how a prompt or a credential ends up in a span.
    """
    allowed = (str, bool, int, float)
    out: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, allowed):
            out[str(key)] = value
    return out


def normalize_otlp_http_endpoint(endpoint: str) -> str:
    """Accept a collector base URL or a full traces path.

    The OTLP HTTP exporter does not append `/v1/traces`. A host:port value
    therefore 404s against a standard collector. Helm/Compose may still pass
    the base URL; both forms are accepted here.
    """
    trimmed = endpoint.strip().rstrip("/")
    if trimmed.endswith("/v1/traces"):
        return trimmed
    return f"{trimmed}/v1/traces"


def try_otlp_exporter(
    endpoint: str | None,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = 10.0,
    certificate_file: str | None = None,
) -> SpanExporter | None:
    """Build an OTLP exporter only when an endpoint is configured.

    Import is deferred. A missing exporter package does not prevent the
    gateway from serving: the documented policy is continue-without-telemetry.
    The exporter queue is bounded by `BatchSpanProcessor` in `FabricTracer`.
    """
    if not endpoint:
        return None
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        return None
    kwargs: dict[str, Any] = {
        "endpoint": normalize_otlp_http_endpoint(endpoint),
        "timeout": timeout_s,
    }
    if headers:
        kwargs["headers"] = headers
    if certificate_file:
        kwargs["certificate_file"] = certificate_file
    return OTLPSpanExporter(**kwargs)
