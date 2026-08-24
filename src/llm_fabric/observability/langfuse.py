"""Langfuse adapter.

The constitution names Langfuse as an integration, not as the system of record.
This module talks to Langfuse through a narrow adapter so the rest of the fabric
never imports the Langfuse SDK, and so a missing or broken Langfuse never fails
a request.

When no credentials are configured the adapter is a no-op. That is the default.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from datetime import UTC
from typing import Any, Protocol
from uuid import uuid4

import httpx

from llm_fabric.observability.otel import RecordedSpan

logger = logging.getLogger("llm_fabric")


class LangfuseSink(Protocol):
    @property
    def enabled(self) -> bool: ...

    async def export_trace(self, spans: list[RecordedSpan]) -> None: ...

    async def aclose(self) -> None: ...


class NullLangfuse:
    """Configured-off sink. Export is a no-op and says so."""

    @property
    def enabled(self) -> bool:
        return False

    async def export_trace(self, spans: list[RecordedSpan]) -> None:
        del spans

    async def aclose(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class LangfuseSettings:
    host: str
    public_key: str
    secret_key: str
    timeout_s: float = 5.0

    @property
    def ingestion_url(self) -> str:
        return self.host.rstrip("/") + "/api/public/ingestion"


class HttpLangfuseAdapter:
    """Posts completed traces to Langfuse's public ingestion API.

    Failures are logged and swallowed. A dashboard outage must not become a
    serving outage.
    """

    def __init__(self, settings: LangfuseSettings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._owned_client = client is None
        self._client = client or httpx.AsyncClient(timeout=settings.timeout_s)

    @property
    def enabled(self) -> bool:
        return True

    async def export_trace(self, spans: list[RecordedSpan]) -> None:
        if not spans:
            return
        batch = _to_ingestion_batch(spans)
        token = base64.b64encode(
            f"{self._settings.public_key}:{self._settings.secret_key}".encode()
        ).decode("ascii")
        try:
            response = await self._client.post(
                self._settings.ingestion_url,
                json={"batch": batch},
                headers={
                    "authorization": f"Basic {token}",
                    "content-type": "application/json",
                },
            )
            response.raise_for_status()
        except Exception:  # noqa: BLE001 - see docstring: never fail the request
            logger.warning("langfuse export failed", exc_info=True)

    async def aclose(self) -> None:
        if self._owned_client:
            await self._client.aclose()


def build_langfuse(
    *,
    host: str | None,
    public_key: str | None,
    secret_key: str | None,
    client: httpx.AsyncClient | None = None,
) -> LangfuseSink:
    if not (host and public_key and secret_key):
        return NullLangfuse()
    return HttpLangfuseAdapter(
        LangfuseSettings(host=host, public_key=public_key, secret_key=secret_key),
        client=client,
    )


def _to_ingestion_batch(spans: list[RecordedSpan]) -> list[dict[str, Any]]:
    """Map journalled spans onto Langfuse's ingestion event shape.

    Prompts and completions are not attached. Langfuse is an operator tool, not
    a place to copy tenant content by default.
    """
    root = next((span for span in spans if span.name == "request"), spans[0])
    events: list[dict[str, Any]] = [
        {
            "id": str(uuid4()),
            "type": "trace-create",
            "timestamp": _iso(root.started_at),
            "body": {
                "id": root.trace_id,
                "name": "request",
                "userId": root.user_id,
                "metadata": {
                    "tenant_id": root.tenant_id,
                    "status": root.status,
                },
            },
        }
    ]
    for span in spans:
        events.append(
            {
                "id": str(uuid4()),
                "type": "span-create",
                "timestamp": _iso(span.started_at),
                "body": {
                    "id": span.span_id,
                    "traceId": span.trace_id,
                    "parentObservationId": span.parent_span_id,
                    "name": span.name,
                    "startTime": _iso(span.started_at),
                    "endTime": _iso(span.started_at + span.duration_ms / 1000.0),
                    "metadata": span.attributes,
                    "statusMessage": span.status,
                },
            }
        )
    return events


def _iso(epoch_s: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(epoch_s, tz=UTC).isoformat()
