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

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

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
from llm_fabric.errors import ContextTooLargeError, FabricError, GuardrailBlockedError
from llm_fabric.gateway.dependencies import (
    get_intent_cascade,
    get_meter,
    get_quota,
    get_request_id,
    get_router,
    get_telemetry,
    get_tenant_scope,
)
from llm_fabric.guardrails import (
    GuardrailAction,
    InputGuardrail,
    InputLimits,
    OutputGuardrail,
    StreamingOutputInspector,
)
from llm_fabric.intent.cascade import IntentCascade
from llm_fabric.intent.features import bound_text, conversation_state_signature
from llm_fabric.intent.schema import ClassificationRequest, IntentClassification
from llm_fabric.observability.logging import request_logger
from llm_fabric.observability.metering import (
    AttemptRecord,
    UsageMeter,
    UsageRecord,
    events_from_decision,
)
from llm_fabric.observability.telemetry import Telemetry, optional_span
from llm_fabric.observability.trace import current_trace
from llm_fabric.observability.usage_event import token_source_for_provider
from llm_fabric.router.engine import RouteDecision, Router
from llm_fabric.router.plan import RouteRequest
from llm_fabric.router.registry import ModelSpec
from llm_fabric.serving.base import StreamDelta, StreamEnd
from llm_fabric.serving.tokens import approximate_prompt_tokens, approximate_token_count
from llm_fabric.tenancy.quota import QuotaLedger
from llm_fabric.tenancy.scope import TenantScope

router = APIRouter(prefix="/v1", tags=["inference"])

SSE_MEDIA_TYPE = "text/event-stream"


def _provenance(
    decision: RouteDecision,
    intent: IntentClassification | None = None,
    *,
    shadow: IntentClassification | None = None,
) -> dict[str, str]:
    headers = {
        "x-fabric-requested-model": decision.requested_model,
        "x-fabric-policy": decision.policy,
        "x-fabric-failovers": str(decision.failover_count),
    }
    if decision.selected_model:
        headers["x-fabric-served-model"] = decision.selected_model
    if decision.selected_provider:
        headers["x-fabric-provider"] = decision.selected_provider
    if decision.plan is not None and decision.plan.selected_tier:
        headers["x-fabric-selected-tier"] = decision.plan.selected_tier
    if decision.plan is not None and decision.plan.routing_policy_version:
        headers["x-fabric-routing-policy-version"] = decision.plan.routing_policy_version
    if decision.plan is not None and decision.plan.routing_policy_hash:
        headers["x-fabric-routing-policy-hash"] = decision.plan.routing_policy_hash
    if decision.plan is not None and decision.plan.quality_shadow:
        shadow_sel = decision.plan.quality_shadow.get("shadow_selected")
        if shadow_sel:
            headers["x-fabric-quality-shadow-model"] = str(shadow_sel)
            headers["x-fabric-quality-shadow-same"] = (
                "true" if decision.plan.quality_shadow.get("same") else "false"
            )
    if decision.fallback_depth:
        headers["x-fabric-fallback-depth"] = str(decision.fallback_depth)
    if intent is not None:
        headers["x-fabric-intent"] = intent.intent_id
        headers["x-fabric-intent-confidence"] = f"{intent.confidence:.4f}"
        headers["x-fabric-intent-layer"] = intent.layer.value
        headers["x-fabric-intent-cache"] = intent.cache_source or "none"
        headers["x-fabric-taxonomy-version"] = intent.taxonomy_version
        headers["x-fabric-classifier-version"] = intent.classifier_version
    if shadow is not None:
        headers["x-fabric-intent-shadow"] = shadow.intent_id
        headers["x-fabric-intent-shadow-confidence"] = f"{shadow.confidence:.4f}"
        headers["x-fabric-intent-shadow-layer"] = shadow.layer.value
        headers["x-fabric-intent-shadow-latency-ms"] = f"{shadow.latency_ms:.3f}"
        headers["x-fabric-taxonomy-version"] = shadow.taxonomy_version
        headers["x-fabric-classifier-version"] = shadow.classifier_version
    return headers


async def _classify(
    cascade: IntentCascade | None, scope: TenantScope, body: ChatCompletionRequest
) -> IntentClassification | None:
    """Classify the newest user turn, or return `None` when disabled.

    Classification never fails a request. A router that has no intent falls back
    to its configured policy, which is a worse route than it might have had —
    not an error the caller can do anything about.
    """
    if cascade is None:
        return None
    text = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
    if not text:
        return None
    try:
        with optional_span("intent") as span:
            decision = await cascade.classify(
                scope,
                ClassificationRequest(
                    text=bound_text(text),
                    conversation_state_signature=conversation_state_signature(body.messages),
                ),
            )
            if span is not None:
                for key, value in decision.trace_attributes().items():
                    span.set_attribute(key, value)
    except Exception:  # noqa: BLE001 - see docstring: routing degrades, never fails
        request_logger().warning("intent classification failed", extra={"model": body.model})
        return None
    return decision.classification


def _route_request(
    body: ChatCompletionRequest,
    fabric: Router,
    scope: TenantScope,
    intent: IntentClassification | None,
) -> RouteRequest:
    return fabric.build_request(body, tenant_id=scope.tenant_id, intent=intent)


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


def _served_identity(
    spec: ModelSpec | None, decision: RouteDecision
) -> tuple[str, str, ModelSpec | None]:
    if spec is not None:
        return spec.id, spec.provider, spec
    if decision.attempts:
        last = decision.attempts[-1]
        return last.model_id, last.provider, None
    return decision.requested_model, decision.selected_provider or "", None


def _meter(
    meter: UsageMeter,
    *,
    request_id: str,
    scope: TenantScope,
    quota: QuotaLedger | None,
    spec: ModelSpec | None,
    decision: RouteDecision,
    prompt_tokens: int,
    completion_tokens: int,
    usage_reported: bool,
    latency_ms: float,
    streamed: bool,
    error: str | None = None,
    intent: IntentClassification | None = None,
    ttft_ms: float | None = None,
    telemetry: Telemetry | None = None,
) -> UsageRecord:
    served_model, provider, priced = _served_identity(spec, decision)
    raw_cost = priced.cost_usd(prompt_tokens, completion_tokens) if priced else None
    cost = raw_cost if raw_cost is not None else 0.0
    trace = current_trace()
    token_source = token_source_for_provider(
        reported=usage_reported,
        tokens_known=usage_reported or prompt_tokens > 0 or completion_tokens > 0,
    )
    events = events_from_decision(
        decision,
        request_id=request_id,
        scope=scope,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        token_source=token_source,
        streamed=streamed,
        error=error,
        intent_id=intent.intent_id if intent is not None else None,
        trace_id=trace.trace_id if trace is not None else None,
        spec=priced,
    )
    if events:
        meter.record_events(events)
    record = UsageRecord(
        request_id=request_id,
        requested_model=decision.requested_model,
        served_model=served_model,
        provider=provider,
        policy=decision.policy,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
        cost_is_estimated=not usage_reported,
        latency_ms=round(latency_ms, 3),
        streamed=streamed,
        failover_count=decision.failover_count,
        attempts=_attempt_records(decision),
        tenant_id=scope.tenant_id,
        user_id=scope.user_id,
        error=error,
        intent_id=intent.intent_id if intent is not None else None,
        intent_layer=intent.layer.value if intent is not None else None,
        intent_confidence=intent.confidence if intent is not None else None,
        intent_cache_hit=intent.cache_hit if intent is not None else None,
        ttft_ms=round(ttft_ms, 3) if ttft_ms is not None else None,
        trace_id=trace.trace_id if trace is not None else None,
        invocation_count=len(events),
        ledger_prompt_tokens=sum(event.prompt_tokens for event in events),
        ledger_completion_tokens=sum(event.completion_tokens for event in events),
        selected_tier=decision.plan.selected_tier if decision.plan is not None else None,
    )
    meter.record(record)
    if telemetry is not None:
        telemetry.observe_usage(record)

    if quota is not None:
        # Charged after the fact: tokens are only known once the backend has
        # answered. The ceiling applies to the next admission, not this one.
        quota.record_usage(scope, tokens=record.total_tokens, cost_usd=cost)

    request_logger().info(
        "served chat completion",
        extra={
            "request_id": record.request_id,
            "tenant_id": record.tenant_id,
            "user_id": record.user_id,
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
    request: Request,
    fabric: Router = Depends(get_router),
    meter: UsageMeter = Depends(get_meter),
    request_id: str = Depends(get_request_id),
    scope: TenantScope = Depends(get_tenant_scope),
    quota: QuotaLedger = Depends(get_quota),
    cascade: IntentCascade | None = Depends(get_intent_cascade),
    telemetry: Telemetry = Depends(get_telemetry),
) -> ChatCompletionResponse | StreamingResponse:
    prompt_text = "\n".join(message.content for message in body.messages)
    settings = request.app.state.settings
    with telemetry.span("input_guardrails"):
        input_decision = InputGuardrail(
            InputLimits(
                max_request_bytes=settings.max_request_bytes,
                max_output_tokens=settings.effective_max_output_tokens,
                max_input_tokens=settings.effective_max_input_tokens,
            )
        ).evaluate(
            {"text": prompt_text, "max_tokens": body.max_tokens},
            tenant_id=scope.tenant_id,
        )
    if input_decision.action is GuardrailAction.BLOCK:
        raise GuardrailBlockedError(input_decision.reason)
    ceiling = fabric.controls.context_ceiling_tokens
    if ceiling is not None:
        prompt_tokens = approximate_prompt_tokens(body.messages)
        if prompt_tokens > ceiling:
            raise ContextTooLargeError(
                f"prompt is approximately {prompt_tokens} tokens; "
                f"a remediation ceiling of {ceiling} tokens is in effect"
            )
    classified = await _classify(cascade, scope, body)
    route_intent = classified if settings.intent_classification_enabled else None
    shadow = (
        classified
        if classified is not None
        and settings.intent_shadow
        and not settings.intent_classification_enabled
        else None
    )
    route = _route_request(body, fabric, scope, route_intent)

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
                scope=scope,
                quota=quota,
                route=route,
                intent=classified,
                telemetry=telemetry,
            ),
            media_type=SSE_MEDIA_TYPE,
            headers={
                "x-fabric-request-id": request_id,
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
            },
        )

    started = time.perf_counter()
    try:
        routed = await fabric.complete(body, route=route)
    except FabricError as exc:
        if isinstance(exc.decision, RouteDecision):
            _meter(
                meter,
                request_id=request_id,
                scope=scope,
                quota=quota,
                spec=None,
                decision=exc.decision,
                prompt_tokens=approximate_prompt_tokens(body.messages),
                completion_tokens=0,
                usage_reported=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                streamed=False,
                error=exc.message,
                intent=classified,
                telemetry=telemetry,
            )
        raise
    latency_ms = (time.perf_counter() - started) * 1000
    record = _meter(
        meter,
        request_id=request_id,
        scope=scope,
        quota=quota,
        spec=routed.spec,
        decision=routed.decision,
        prompt_tokens=routed.result.prompt_tokens,
        completion_tokens=routed.result.completion_tokens,
        usage_reported=routed.result.usage_reported_by_provider,
        latency_ms=latency_ms,
        streamed=False,
        intent=classified,
        telemetry=telemetry,
    )
    if record.trace_id:
        await telemetry.export_recent_trace(record.trace_id)

    with telemetry.span("output_guardrails"):
        output_decision = OutputGuardrail().evaluate(routed.result.text, tenant_id=scope.tenant_id)
    content = routed.result.text
    if output_decision.action is GuardrailAction.BLOCK:
        raise GuardrailBlockedError(output_decision.reason)
    if output_decision.action is GuardrailAction.REDACT and output_decision.transformed:
        content = str(output_decision.transformed)

    response.headers["x-fabric-request-id"] = request_id
    response.headers["x-fabric-invocations"] = str(record.invocation_count)
    for key, value in _provenance(routed.decision, route_intent, shadow=shadow).items():
        response.headers[key] = value

    return ChatCompletionResponse(
        model=routed.spec.id,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
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
    meter: UsageMeter,
    request_id: str,
    scope: TenantScope,
    quota: QuotaLedger | None = None,
    route: RouteRequest | None = None,
    intent: IntentClassification | None = None,
    telemetry: Telemetry | None = None,
) -> AsyncIterator[str]:
    completion_id = _completion_id()
    started = time.perf_counter()
    first_chunk = True
    first_byte_at: float | None = None
    spec: ModelSpec | None = None
    decision: RouteDecision | None = None
    emitted = ""
    metered = False

    async def record(
        *,
        prompt_tokens: int,
        completion_tokens: int,
        usage_reported: bool,
        error: str | None = None,
    ) -> None:
        nonlocal metered
        if metered or decision is None:
            return
        usage = _meter(
            meter,
            request_id=request_id,
            scope=scope,
            quota=quota,
            spec=spec,
            decision=decision,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage_reported=usage_reported,
            latency_ms=(time.perf_counter() - started) * 1000,
            streamed=True,
            error=error,
            intent=intent,
            ttft_ms=(first_byte_at - started) * 1000 if first_byte_at is not None else None,
            telemetry=telemetry,
        )
        metered = True
        if telemetry is not None and usage.trace_id:
            await telemetry.export_recent_trace(usage.trace_id)

    inspector = StreamingOutputInspector()
    try:
        async for event, event_spec, event_decision in fabric.stream(body, route=route):
            spec, decision = event_spec, event_decision

            if isinstance(event, StreamDelta):
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()
                text, rail = inspector.push(event.text, tenant_id=scope.tenant_id)
                if rail.action is GuardrailAction.BLOCK:
                    raise GuardrailBlockedError(rail.reason)
                if not text:
                    continue
                emitted += text
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    model=event_spec.id,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=ChunkDelta(
                                role="assistant" if first_chunk else None,
                                content=text,
                            ),
                        )
                    ],
                )
                first_chunk = False
                yield _sse(chunk.model_dump(exclude_none=True))
                continue

            if isinstance(event, StreamEnd):
                remainder, rail = inspector.flush(tenant_id=scope.tenant_id)
                if rail.action is GuardrailAction.BLOCK:
                    raise GuardrailBlockedError(rail.reason)
                if remainder:
                    emitted += remainder
                    yield _sse(
                        ChatCompletionChunk(
                            id=completion_id,
                            model=event_spec.id,
                            choices=[
                                ChunkChoice(
                                    index=0,
                                    delta=ChunkDelta(
                                        role="assistant" if first_chunk else None,
                                        content=remainder,
                                    ),
                                )
                            ],
                        ).model_dump(exclude_none=True)
                    )
                    first_chunk = False
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
                    "invocations": len(event_decision.attempts),
                }
                yield _sse(payload)
                await record(
                    prompt_tokens=event.prompt_tokens,
                    completion_tokens=event.completion_tokens,
                    usage_reported=event.usage_reported_by_provider,
                )
        if not metered and decision is not None:
            await record(
                prompt_tokens=approximate_prompt_tokens(body.messages),
                completion_tokens=approximate_token_count(emitted),
                usage_reported=False,
            )
        yield _sse("[DONE]")
    except FabricError as exc:
        # The HTTP status is already committed, so the failure has to be reported
        # inside the stream. Clients see an error frame rather than a silent stop.
        if isinstance(exc.decision, RouteDecision):
            decision = exc.decision
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
        await record(
            prompt_tokens=approximate_prompt_tokens(body.messages),
            completion_tokens=approximate_token_count(emitted),
            usage_reported=False,
            error=exc.message,
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
    except BaseException:
        await record(
            prompt_tokens=approximate_prompt_tokens(body.messages),
            completion_tokens=approximate_token_count(emitted),
            usage_reported=False,
            error="cancelled",
        )
        raise
