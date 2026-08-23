"""Chat completions: the fabric's primary endpoint.

Both response modes route through the same engine, so a streamed request and a
buffered one make identical routing decisions and produce identical metering
records. The only difference is how bytes reach the client.

Every response carries provenance headers naming the model that actually served
the request, the provider behind it, and the policy that chose it. Callers
sending `auto` therefore always learn what they got, without a second lookup.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from llm_fabric.config import Settings
from llm_fabric.contract.openai import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChunkChoice,
    ChunkDelta,
    Usage,
    _completion_id,
)
from llm_fabric.errors import FabricError
from llm_fabric.gateway.dependencies import (
    get_client_id,
    get_meter,
    get_request_id,
    get_router,
    get_settings,
)
from llm_fabric.observability.logging import request_logger
from llm_fabric.observability.metering import AttemptRecord, InMemoryMeter, UsageRecord
from llm_fabric.router.engine import RouteDecision, Router
from llm_fabric.router.registry import ModelSpec
from llm_fabric.serving.base import StreamDelta, StreamEnd

router = APIRouter(prefix="/v1", tags=["inference"])

SSE_MEDIA_TYPE = "text/event-stream"


def _provenance(decision: RouteDecision) -> dict[str, str]:
    headers = {
        "x-fabric-requested-model": decision.requested_model,
        "x-fabric-policy": decision.policy,
        "x-fabric-failovers": str(decision.failover_count),
    }
    if decision.selected_model:
        headers["x-fabric-served-model"] = decision.selected_model
    if decision.selected_provider:
        headers["x-fabric-provider"] = decision.selected_provider
    return headers


def _attempt_records(decision: RouteDecision) -> tuple[AttemptRecord, ...]:
    return tuple(
        AttemptRecord(
            model_id=attempt.model_id,
            provider=attempt.provider,
            duration_ms=round(attempt.duration_ms, 3),
            error=attempt.error,
        )
        for attempt in decision.attempts
    )


def _meter(
    meter: InMemoryMeter,
    *,
    request_id: str,
    client_id: str | None,
    spec: ModelSpec,
    decision: RouteDecision,
    prompt_tokens: int,
    completion_tokens: int,
    usage_reported: bool,
    latency_ms: float,
    streamed: bool,
) -> UsageRecord:
    record = UsageRecord(
        request_id=request_id,
        requested_model=decision.requested_model,
        served_model=spec.id,
        provider=spec.provider,
        policy=decision.policy,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=spec.cost_usd(prompt_tokens, completion_tokens),
        cost_is_estimated=not usage_reported,
        latency_ms=round(latency_ms, 3),
        streamed=streamed,
        failover_count=decision.failover_count,
        attempts=_attempt_records(decision),
        client_id=client_id,
    )
    meter.record(record)

    request_logger().info(
        "served chat completion",
        extra={
            "request_id": record.request_id,
            "requested_model": record.requested_model,
            "served_model": record.served_model,
            "provider": record.provider,
            "policy": record.policy,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "cost_usd": round(record.cost_usd, 6),
            "cost_is_estimated": record.cost_is_estimated,
            "latency_ms": record.latency_ms,
            "streamed": record.streamed,
            "failovers": record.failover_count,
        },
    )
    return record


def _sse(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload)}\n\n"


@router.post(
    "/chat/completions",
    response_model=None,
    summary="Create a chat completion",
)
async def create_chat_completion(
    body: ChatCompletionRequest,
    response: Response,
    fabric: Router = Depends(get_router),
    meter: InMemoryMeter = Depends(get_meter),
    request_id: str = Depends(get_request_id),
    client_id: str | None = Depends(get_client_id),
    settings: Settings = Depends(get_settings),
) -> ChatCompletionResponse | StreamingResponse:
    if body.stream:
        # Resolve before streaming starts: an unknown or disabled model must come
        # back as a proper HTTP error, and once the first byte ships the status
        # code is already on the wire.
        fabric.resolve(body.model)
        return StreamingResponse(
            _stream_completion(
                body,
                fabric=fabric,
                meter=meter,
                request_id=request_id,
                client_id=client_id,
            ),
            media_type=SSE_MEDIA_TYPE,
            headers={
                "x-fabric-request-id": request_id,
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
            },
        )

    started = time.perf_counter()
    routed = await fabric.complete(body)
    latency_ms = (time.perf_counter() - started) * 1000

    _meter(
        meter,
        request_id=request_id,
        client_id=client_id,
        spec=routed.spec,
        decision=routed.decision,
        prompt_tokens=routed.result.prompt_tokens,
        completion_tokens=routed.result.completion_tokens,
        usage_reported=routed.result.usage_reported_by_provider,
        latency_ms=latency_ms,
        streamed=False,
    )

    response.headers["x-fabric-request-id"] = request_id
    for key, value in _provenance(routed.decision).items():
        response.headers[key] = value

    return ChatCompletionResponse(
        model=routed.spec.id,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=routed.result.text),
                finish_reason=routed.result.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=routed.result.prompt_tokens,
            completion_tokens=routed.result.completion_tokens,
            total_tokens=routed.result.total_tokens,
        ),
    )


async def _stream_completion(
    body: ChatCompletionRequest,
    *,
    fabric: Router,
    meter: InMemoryMeter,
    request_id: str,
    client_id: str | None,
) -> AsyncIterator[str]:
    completion_id = _completion_id()
    started = time.perf_counter()
    first_chunk = True
    spec: ModelSpec | None = None
    decision: RouteDecision | None = None

    try:
        async for event, event_spec, event_decision in fabric.stream(body):
            spec, decision = event_spec, event_decision

            if isinstance(event, StreamDelta):
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    model=event_spec.id,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=ChunkDelta(
                                role="assistant" if first_chunk else None,
                                content=event.text,
                            ),
                        )
                    ],
                )
                first_chunk = False
                yield _sse(chunk.model_dump(exclude_none=True))
                continue

            if isinstance(event, StreamEnd):
                final = ChatCompletionChunk(
                    id=completion_id,
                    model=event_spec.id,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=ChunkDelta(),
                            finish_reason=event.finish_reason,
                        )
                    ],
                )
                payload = final.model_dump(exclude_none=True)
                payload["usage"] = Usage(
                    prompt_tokens=event.prompt_tokens,
                    completion_tokens=event.completion_tokens,
                    total_tokens=event.prompt_tokens + event.completion_tokens,
                ).model_dump()
                payload["x_fabric"] = {
                    "requested_model": event_decision.requested_model,
                    "served_model": event_spec.id,
                    "provider": event_spec.provider,
                    "policy": event_decision.policy,
                    "failovers": event_decision.failover_count,
                }
                yield _sse(payload)

                _meter(
                    meter,
                    request_id=request_id,
                    client_id=client_id,
                    spec=event_spec,
                    decision=event_decision,
                    prompt_tokens=event.prompt_tokens,
                    completion_tokens=event.completion_tokens,
                    usage_reported=event.usage_reported_by_provider,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    streamed=True,
                )
    except FabricError as exc:
        # The HTTP status is already committed, so the failure has to be reported
        # inside the stream. Clients see an error frame rather than a silent stop.
        request_logger().warning(
            "stream failed",
            extra={
                "request_id": request_id,
                "requested_model": body.model,
                "error_type": exc.error_type,
                "served_model": spec.id if spec else None,
                "policy": decision.policy if decision else None,
            },
        )
        yield _sse(
            {
                "error": {
                    "message": exc.message,
                    "type": exc.error_type,
                    "request_id": request_id,
                }
            }
        )

    yield _sse("[DONE]")
