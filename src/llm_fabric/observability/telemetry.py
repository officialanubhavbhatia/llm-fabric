"""The process-wide observability hub.

One object on `app.state` so routes, the router and the Command Center all see
the same tracer, the same Prometheus registry and the same Langfuse sink.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

from llm_fabric.observability.engine import EngineMetricsHub
from llm_fabric.observability.langfuse import LangfuseSink, NullLangfuse
from llm_fabric.observability.metering import UsageRecord
from llm_fabric.observability.otel import FabricTracer
from llm_fabric.observability.prom import FabricMetrics

_current: ContextVar[Telemetry | None] = ContextVar("llm_fabric_telemetry", default=None)


def current_telemetry() -> Telemetry | None:
    return _current.get()


def bind_telemetry(telemetry: Telemetry) -> Token[Telemetry | None]:
    return _current.set(telemetry)


def reset_telemetry(token: Token[Telemetry | None]) -> None:
    _current.reset(token)


@contextmanager
def optional_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a span when a Telemetry hub is bound, otherwise do nothing.

    Lets the router record provider calls without taking a Telemetry
    constructor argument, so unit tests that construct a Router directly
    keep working.
    """
    telemetry = current_telemetry()
    if telemetry is None:
        yield None
        return
    with telemetry.span(name, **attributes) as span:
        yield span


class Telemetry:
    def __init__(
        self,
        *,
        tracer: FabricTracer | None = None,
        metrics: FabricMetrics | None = None,
        langfuse: LangfuseSink | None = None,
        engines: EngineMetricsHub | None = None,
    ) -> None:
        self.tracer = tracer or FabricTracer()
        self.metrics = metrics or FabricMetrics()
        self.langfuse = langfuse or NullLangfuse()
        self.engines = engines or EngineMetricsHub()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        with self.tracer.span(name, **attributes) as span:
            yield span

    def observe_usage(self, record: UsageRecord) -> None:
        self.metrics.observe_usage(
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            cost_usd=record.cost_usd,
            cost_is_estimated=record.cost_is_estimated,
            provider=record.provider,
            policy=record.policy,
            failover_count=record.failover_count,
            latency_s=record.latency_ms / 1000.0,
            error=record.error is not None,
            ttft_s=record.ttft_ms / 1000.0 if record.ttft_ms is not None else None,
        )
        if record.intent_layer:
            self.metrics.observe_intent(
                layer=record.intent_layer,
                cache_hit=bool(record.intent_cache_hit),
                abstained=record.intent_id == "unknown",
            )

    async def export_recent_trace(self, trace_id: str) -> None:
        if not self.langfuse.enabled:
            return
        spans = self.tracer.journal.recent(limit=200, trace_id=trace_id)
        await self.langfuse.export_trace(list(reversed(spans)))

    async def aclose(self) -> None:
        await self.langfuse.aclose()
        self.tracer.shutdown()
