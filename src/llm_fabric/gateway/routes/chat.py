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
from dataclasses import replace
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from llm_fabric.context.compiler import CompiledContext, ContextCompiler, compile_chat
from llm_fabric.context.record import ContextRecord
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
from llm_fabric.intent.schema import (
    ClassificationRequest,
    IntentClassification,
    ServingClassificationState,
)
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
    intent_routing_enabled: bool = False,
    shadow: IntentClassification | None = None,
    context: ContextRecord | None = None,
) -> dict[str, str]:
    headers = {
        "x-fabric-route-id": decision.route_id,
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
    if decision.attempts:
        last = decision.attempts[-1]
        if last.deployment_id:
            headers["x-fabric-deployment-id"] = last.deployment_id
        if last.provider_adapter:
            headers["x-fabric-provider-adapter"] = last.provider_adapter
        if last.transport:
            headers["x-fabric-transport"] = last.transport
        if last.runtime:
            headers["x-fabric-runtime"] = last.runtime
        if last.grade:
            headers["x-fabric-grade"] = last.grade
        if last.litellm_model:
            headers["x-fabric-litellm-model"] = last.litellm_model
        if last.actual_served_model:
            headers["x-fabric-actual-served-model"] = last.actual_served_model
    if intent is not None:
        headers["x-fabric-intent"] = intent.intent_id
        headers["x-fabric-intent-routing"] = "on" if intent_routing_enabled else "off"
        headers["x-fabric-intent-state"] = intent.serving_state.value
        headers["x-fabric-intent-result-id"] = intent.intent_result_id
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
    if context is not None:
        headers["x-fabric-context-record-id"] = context.context_record_id
        before = context.context_tokens_before_optimization.value
        after = context.context_tokens_after_optimization.value
        if before is not None:
            headers["x-fabric-context-tokens-before"] = str(int(before))
        if after is not None:
            headers["x-fabric-context-tokens-after"] = str(int(after))
        headers["x-fabric-context-token-provenance"] = context.token_provenance.value
    return headers


def _topology_headers(spec: ModelSpec) -> dict[str, str]:
    headers = {
        "x-fabric-served-model": spec.id,
        "x-fabric-provider": spec.provider,
        "x-fabric-deployment-id": spec.deployment_id,
        "x-fabric-provider-adapter": spec.provider_adapter,
        "x-fabric-transport": spec.transport.value,
        "x-fabric-runtime": spec.runtime.value,
    }
    if spec.grade is not None:
        headers["x-fabric-grade"] = spec.grade.value
    if spec.transport.value == "litellm":
        headers["x-fabric-litellm-model"] = spec.provider_model
    return headers


async def _classify(
    cascade: IntentCascade | None, scope: TenantScope, body: ChatCompletionRequest
) -> IntentClassification:
    """Always produce a typed IntentResult for the serving path.

    Classification never fails a request. Cascade or dependency failure degrades
    to SAFE_FALLBACK / UNKNOWN. It never continues with no IntentResult.
    """
    text = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
    if cascade is None:
        fallback = IntentClassification.safe_fallback()
        _record_serving(fallback)
        return fallback
    if not text:
        unknown = IntentClassification.unknown_result(
            classifier_version=cascade.version,
            taxonomy_version=cascade.taxonomy.version,
        )
        _record_serving(unknown)
        return unknown
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
    except Exception:  # noqa: BLE001 - serving degrades to SAFE_FALLBACK, never skips
        request_logger().warning("intent classification failed", extra={"model": body.model})
        fallback = IntentClassification.safe_fallback(
            classifier_version=cascade.version,
            taxonomy_version=cascade.taxonomy.version,
        )
        cascade.metrics.record_error()
        cascade.metrics.record_serving(ServingClassificationState.SAFE_FALLBACK)
        return fallback
    return decision.classification


def _record_serving(intent: IntentClassification) -> None:
    from llm_fabric.observability.telemetry import current_telemetry

    telemetry = current_telemetry()
    if telemetry is not None:
        telemetry.metrics.observe_intent_serving(intent.serving_state.value)


def _compile_context(
    body: ChatCompletionRequest,
    scope: TenantScope,
    *,
    request_id: str,
    ceiling: int | None,
    telemetry: Telemetry,
) -> CompiledContext:
    """Always produce a ContextRecord before routing. Never bypass compilation."""
    with telemetry.span("context") as span:
        compiled = compile_chat(
            body,
            scope,
            request_id=request_id,
            context_window=ceiling,
        )
        telemetry.record_context(compiled.record)
        before = int(compiled.record.context_tokens_before_optimization.value or 0)
        after = int(compiled.record.context_tokens_after_optimization.value or 0)
        telemetry.metrics.observe_context(
            compile_s=compiled.record.compile_latency_ms / 1000.0,
            tokens_before=before,
            tokens_after=after,
        )
        if span is not None:
            span.set_attribute("context.record_id", compiled.record.context_record_id)
            span.set_attribute("context.tokens_before", before)
            span.set_attribute("context.tokens_after", after)
            span.set_attribute("context.token_provenance", compiled.record.token_provenance.value)
    return compiled


def _stream_tpot_ms(first_byte_at: float | None, completion_tokens: int) -> float | None:
    if first_byte_at is None or completion_tokens < 1:
        return None
    decode_s = time.perf_counter() - first_byte_at
    if decode_s <= 0:
        return None
    return (decode_s / max(1, completion_tokens - 1)) * 1000


def _route_request(
    body: ChatCompletionRequest,
    fabric: Router,
    scope: TenantScope,
    intent: IntentClassification,
    compiled: CompiledContext | None = None,
) -> RouteRequest:
    route = fabric.build_request(body, tenant_id=scope.tenant_id, intent=intent)
    if compiled is None:
        return route
    after = compiled.record.context_tokens_after_optimization.value
    if after is None:
        return route
    return replace(route, prompt_tokens=int(after))


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
    context: ContextRecord | None = None,
    ttft_ms: float | None = None,
    tpot_ms: float | None = None,
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
        intent_result_id=intent.intent_result_id if intent is not None else None,
        taxonomy_version=intent.taxonomy_version if intent is not None else None,
        classifier_version=intent.classifier_version if intent is not None else None,
        context_record_id=context.context_record_id if context is not None else None,
        trace_id=trace.trace_id if trace is not None else None,
        spec=priced,
    )
    if events:
        with telemetry.span("usage") if telemetry is not None else optional_span("usage"):
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
        tpot_ms=round(tpot_ms, 3) if tpot_ms is not None else None,
        trace_id=trace.trace_id if trace is not None else None,
        context_record_id=context.context_record_id if context is not None else None,
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
    if settings.intent_routing_enabled:
        route_intent = classified
        shadow = None
    else:
        route_intent = (
            classified
            if classified.serving_state is ServingClassificationState.SAFE_FALLBACK
            else IntentClassification.safe_fallback(
                classifier_version=classified.classifier_version,
                taxonomy_version=classified.taxonomy_version,
            )
        )
        shadow = classified if settings.intent_shadow else None
    compiled = _compile_context(
        body,
        scope,
        request_id=request_id,
        ceiling=ceiling,
        telemetry=telemetry,
    )
    compiled_body = compiled.request
    route = _route_request(compiled_body, fabric, scope, route_intent, compiled)

    if body.stream:
        # Resolve before streaming starts: an unknown or disabled model must come
        # back as a proper HTTP error, and once the first byte ships the status
        # code is already on the wire.
        _, candidates = fabric.resolve(body.model)
        stream_headers = {
            "x-fabric-request-id": request_id,
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
        }
        stream_headers.update(_topology_headers(candidates[0]))
        if route_intent is not None:
            stream_headers["x-fabric-intent"] = classified.intent_id
            stream_headers["x-fabric-intent-state"] = classified.serving_state.value
            stream_headers["x-fabric-intent-result-id"] = classified.intent_result_id
            stream_headers["x-fabric-taxonomy-version"] = classified.taxonomy_version
            stream_headers["x-fabric-classifier-version"] = classified.classifier_version
            stream_headers["x-fabric-intent-routing"] = (
                "on" if settings.intent_routing_enabled else "off"
            )
        if shadow is not None:
            stream_headers["x-fabric-intent-shadow"] = shadow.intent_id
        stream_headers["x-fabric-context-record-id"] = compiled.record.context_record_id
        return StreamingResponse(
            _stream_completion(
                compiled_body,
                fabric=fabric,
                meter=meter,
                request_id=request_id,
                scope=scope,
                quota=quota,
                route=route,
                intent=classified,
                context=compiled.record,
                telemetry=telemetry,
            ),
            media_type=SSE_MEDIA_TYPE,
            headers=stream_headers,
        )

    started = time.perf_counter()
    try:
        routed = await fabric.complete(compiled_body, route=route)
    except FabricError as exc:
        if isinstance(exc.decision, RouteDecision):
            _meter(
                meter,
                request_id=request_id,
                scope=scope,
                quota=quota,
                spec=None,
                decision=exc.decision,
                prompt_tokens=approximate_prompt_tokens(compiled_body.messages),
                completion_tokens=0,
                usage_reported=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                streamed=False,
                error=exc.message,
                intent=classified,
                context=compiled.record,
                telemetry=telemetry,
            )
        raise
    latency_ms = (time.perf_counter() - started) * 1000
    context_record = compiled.record
    if routed.spec.context_window:
        context_record = ContextCompiler().bind_model_window(
            compiled.record, routed.spec.context_window
        )
        telemetry.record_context(context_record)
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
        context=context_record,
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
    for key, value in _provenance(
        routed.decision,
        classified,
        intent_routing_enabled=settings.intent_routing_enabled,
        shadow=shadow,
        context=context_record,
    ).items():
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
    context: ContextRecord | None = None,
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
            context=context,
            ttft_ms=(first_byte_at - started) * 1000 if first_byte_at is not None else None,
            tpot_ms=_stream_tpot_ms(first_byte_at, completion_tokens),
            telemetry=telemetry,
        )
        metered = True
        if telemetry is not None and usage.trace_id:
            await telemetry.export_recent_trace(usage.trace_id)

    inspector = StreamingOutputInspector()
    try:
        async for event, event_spec, event_decision in fabric.stream(body, route=route):
            spec, decision = event_spec, event_decision
            if context is not None and event_spec.context_window:
                context = ContextCompiler().bind_model_window(context, event_spec.context_window)
                if telemetry is not None:
                    telemetry.record_context(context)

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
                    "route_id": event_decision.route_id,
                    "requested_model": event_decision.requested_model,
                    "served_model": event_spec.id,
                    "provider": event_spec.provider,
                    "policy": event_decision.policy,
                    "failovers": event_decision.failover_count,
                    "invocations": len(event_decision.attempts),
                    "deployment_id": event_spec.deployment_id,
                    "provider_adapter": event_spec.provider_adapter,
                    "transport": event_spec.transport.value,
                    "runtime": event_spec.runtime.value,
                    "grade": event_spec.grade.value if event_spec.grade else None,
                    "litellm_model": (
                        event_spec.provider_model
                        if event_spec.transport.value == "litellm"
                        else None
                    ),
                    "actual_served_model": next(
                        (
                            attempt.actual_served_model
                            for attempt in reversed(event_decision.attempts)
                            if attempt.actual_served_model
                        ),
                        None,
                    ),
                    "context_record_id": context.context_record_id if context else None,
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
