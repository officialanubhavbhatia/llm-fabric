"""The routing engine: execute a plan, and record what happened.

Planning decides where a request should go (`router.plan`). This module takes
that decision and carries it out, which is a different problem with its own
rules.

**Failover follows the graph, not a list.** When an attempt fails, the error is
classified into a `FallbackReason` and the next deployment is whichever one the
fallback graph nominates *for that reason*. A timeout and an oversized context
can therefore go to different places, which is the entire point of the graph.

**Only retryable failures move on.** A caller error stops immediately: retrying a
malformed request only fails again more slowly.

**Failover under streaming stops at the first byte.** Once generated text has
been handed to the client, failing over would splice two models' output together
into a response the client cannot un-see. The engine may fail over while a stream
has produced nothing, and must not once it has.

**Every attempt updates health.** Successes and failures both feed the EWMA
trackers and the circuit breakers, so the next request routes on what this one
learned.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from llm_fabric.contract.openai import ChatCompletionRequest
from llm_fabric.errors import (
    AllCandidatesFailedError,
    FabricError,
    ModelNotFoundError,
    NoCandidateError,
    RetryableError,
)
from llm_fabric.heal.controls import OperationalControls
from llm_fabric.intent.schema import IntentClassification
from llm_fabric.observability.telemetry import current_telemetry, optional_span
from llm_fabric.observability.usage_event import TokenSource, UsageOperation
from llm_fabric.router.fallback import (
    FallbackBudget,
    FallbackHop,
    FallbackLedger,
    FallbackReason,
    FallbackTrace,
    reason_for_error,
)
from llm_fabric.router.health import HealthTracker
from llm_fabric.router.plan import (
    RoutePlan,
    RoutePlanner,
    RouteRequest,
    TenantRoutingPolicies,
)
from llm_fabric.router.policy import RoutePolicy, parse_policy
from llm_fabric.router.registry import ModelRegistry, ModelSpec
from llm_fabric.serving.base import (
    InferenceRequest,
    ProviderResult,
    StreamDelta,
    StreamEnd,
    StreamEvent,
)
from llm_fabric.serving.factory import ProviderFactory
from llm_fabric.serving.tokens import approximate_prompt_tokens


@dataclass(slots=True)
class Attempt:
    """One call to one deployment."""

    model_id: str
    provider: str
    duration_ms: float
    error: str | None = None
    #: Why this attempt was reached. `None` on the first, which needed no reason.
    reason: FallbackReason | None = None
    depth: int = 0
    invocation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    deployment_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    token_source: str = TokenSource.UNAVAILABLE.value
    started_at: float = field(default_factory=time.time)
    completed_at: float = field(default_factory=time.time)
    provider_cost_usd: float | None = None
    compute_cost_estimate_usd: float | None = None
    operation: str = UsageOperation.USER_RESPONSE.value

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "duration_ms": round(self.duration_ms, 3),
            "error": self.error,
            "reason": self.reason.value if self.reason else None,
            "depth": self.depth,
            "invocation_id": self.invocation_id,
            "token_source": self.token_source,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass(slots=True)
class RouteDecision:
    """The auditable record of one routed request, planned and executed."""

    requested_model: str
    policy: str
    considered: list[str]
    attempts: list[Attempt] = field(default_factory=list)
    selected_model: str | None = None
    selected_provider: str | None = None
    plan: RoutePlan | None = None
    fallback: FallbackTrace = field(default_factory=FallbackTrace)

    @property
    def failover_count(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.error)

    @property
    def fallback_depth(self) -> int:
        return self.fallback.depth

    def explain(self) -> dict[str, Any]:
        """Everything about this decision, for tracing and the preview API."""
        payload: dict[str, Any] = {
            "requested_model": self.requested_model,
            "policy": self.policy,
            "considered": list(self.considered),
            "selected_model": self.selected_model,
            "selected_provider": self.selected_provider,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "failover_count": self.failover_count,
            "fallback": self.fallback.as_dict(),
        }
        if self.plan is not None:
            payload["plan"] = self.plan.describe()
        return payload


@dataclass(slots=True)
class RoutedResult:
    result: ProviderResult
    spec: ModelSpec
    decision: RouteDecision


def _compute_cost_usd(spec: ModelSpec, duration_ms: float) -> float | None:
    hourly = spec.estimated_compute_cost_per_hour_usd
    if hourly is None:
        return None
    return (duration_ms / 3_600_000.0) * hourly


def _provider_cost(spec: ModelSpec, prompt: int, completion: int, source: str) -> float | None:
    if source == TokenSource.UNAVAILABLE.value:
        return None
    return spec.cost_usd(prompt, completion)


def _apply_stream_usage(attempt: Attempt, event: StreamEnd, spec: ModelSpec) -> None:
    source = (
        TokenSource.PROVIDER_MEASURED.value
        if event.usage_reported_by_provider
        else TokenSource.LOCAL_TOKENIZER_ESTIMATE.value
    )
    attempt.prompt_tokens = event.prompt_tokens
    attempt.completion_tokens = event.completion_tokens
    attempt.token_source = source
    attempt.completed_at = time.time()
    attempt.provider_cost_usd = _provider_cost(
        spec, event.prompt_tokens, event.completion_tokens, source
    )


def _new_attempt(
    spec: ModelSpec,
    *,
    elapsed: float,
    reason: FallbackReason | None,
    depth: int,
    error: str | None = None,
    result: ProviderResult | None = None,
    started_at: float,
) -> Attempt:
    source = TokenSource.UNAVAILABLE.value
    prompt = 0
    completion = 0
    if result is not None:
        source = (
            TokenSource.PROVIDER_MEASURED.value
            if result.usage_reported_by_provider
            else TokenSource.LOCAL_TOKENIZER_ESTIMATE.value
        )
        prompt = result.prompt_tokens
        completion = result.completion_tokens
    completed_at = time.time()
    return Attempt(
        model_id=spec.id,
        provider=spec.provider,
        duration_ms=elapsed,
        error=error,
        reason=reason,
        depth=depth,
        deployment_id=spec.deployment_id,
        prompt_tokens=prompt,
        completion_tokens=completion,
        token_source=source,
        started_at=started_at,
        completed_at=completed_at,
        provider_cost_usd=_provider_cost(spec, prompt, completion, source),
        compute_cost_estimate_usd=_compute_cost_usd(spec, elapsed),
    )


class Router:
    def __init__(
        self,
        registry: ModelRegistry,
        providers: ProviderFactory,
        *,
        default_policy: str = "cost_first",
        max_attempts: int = 3,
        health: HealthTracker | None = None,
        tenant_policies: TenantRoutingPolicies | None = None,
        fallback_budget: FallbackBudget | None = None,
        planner: RoutePlanner | None = None,
        controls: OperationalControls | None = None,
    ) -> None:
        self._registry = registry
        self._providers = providers
        self._default_policy = parse_policy(default_policy)
        self._max_attempts = max(1, max_attempts)
        self._health = health or HealthTracker()
        self.controls = controls or OperationalControls()
        # `max_attempts` counts the primary too, so the fallback budget's depth
        # is one less. Configuring both independently would let them disagree
        # about how many deployments a request may touch.
        self._budget = fallback_budget or FallbackBudget(max_depth=self._max_attempts - 1)
        self._planner = planner or RoutePlanner(
            registry,
            health=self._health,
            tenant_policies=tenant_policies,
            default_policy=self._default_policy.value,
            fallback_budget=self._budget,
            traffic=self.controls.traffic,
        )

    @property
    def planner(self) -> RoutePlanner:
        return self._planner

    @property
    def health(self) -> HealthTracker:
        return self._health

    # -- planning ------------------------------------------------------------

    def build_request(
        self,
        request: ChatCompletionRequest,
        *,
        tenant_id: str | None = None,
        intent: IntentClassification | None = None,
        policy: RoutePolicy | None = None,
        latency_slo_ms: float | None = None,
        budget_usd: float | None = None,
    ) -> RouteRequest:
        """Translate a chat request into what the planner needs.

        Prompt tokens are approximated, not counted: the estimate feeds context
        and budget filters, both of which compare against limits with far more
        slack than the estimator's error.
        """
        return RouteRequest(
            requested_model=request.model,
            tenant_id=tenant_id,
            intent=intent,
            policy=policy,
            prompt_tokens=approximate_prompt_tokens(list(request.messages)),
            max_output_tokens=request.max_tokens,
            latency_slo_ms=latency_slo_ms,
            budget_usd=budget_usd,
        )

    def preview(
        self,
        request: ChatCompletionRequest,
        *,
        tenant_id: str | None = None,
        intent: IntentClassification | None = None,
        policy: RoutePolicy | None = None,
        latency_slo_ms: float | None = None,
        budget_usd: float | None = None,
    ) -> RoutePlan:
        """Plan without executing. Backs `/v1/routes/preview`."""
        return self._planner.plan(
            self.build_request(
                request,
                tenant_id=tenant_id,
                intent=intent,
                policy=policy,
                latency_slo_ms=latency_slo_ms,
                budget_usd=budget_usd,
            )
        )

    def resolve(self, requested_model: str) -> tuple[str, list[ModelSpec]]:
        """The policy name and ordered candidates for a bare model id."""
        plan = self._planner.plan(RouteRequest(requested_model=requested_model))
        if not plan.ranked:
            if not self._registry.known(requested_model):
                raise ModelNotFoundError(f"unknown model '{requested_model}'")
            raise NoCandidateError(
                f"no enabled deployment can serve '{requested_model}': "
                + ("; ".join(f"{e.model_id} {e.rule.value}" for e in plan.excluded) or "none")
            )
        return plan.policy.value, [candidate.spec for candidate in plan.ranked]

    def _to_inference_request(
        self, request: ChatCompletionRequest, spec: ModelSpec
    ) -> InferenceRequest:
        return InferenceRequest(
            model=spec.provider_model,
            messages=list(request.messages),
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop_sequences(),
        )

    def _start(self, request: ChatCompletionRequest, route: RouteRequest | None) -> _Walk:
        resolved = route or self.build_request(request)
        with optional_span(
            "route",
            requested_model=resolved.requested_model,
            tenant_id=resolved.tenant_id or "",
        ):
            plan = self._planner.plan(resolved)
        if plan.selected is None:
            if not self._registry.known(resolved.requested_model):
                raise ModelNotFoundError(f"unknown model '{resolved.requested_model}'")
            failure = NoCandidateError(
                f"no deployment can serve '{resolved.requested_model}': "
                + ("; ".join(f"{e.model_id} {e.rule.value}" for e in plan.excluded) or "none")
            )
            failure.decision = RouteDecision(
                requested_model=resolved.requested_model,
                policy=plan.policy.value,
                considered=[],
                plan=plan,
            )
            raise failure
        decision = RouteDecision(
            requested_model=resolved.requested_model,
            policy=plan.policy.value,
            considered=list(plan.chain),
            plan=plan,
        )
        return _Walk(plan=plan, decision=decision, ledger=FallbackLedger(budget=plan.budget))

    # -- execution -----------------------------------------------------------

    async def complete(
        self, request: ChatCompletionRequest, *, route: RouteRequest | None = None
    ) -> RoutedResult:
        walk = self._start(request, route)
        decision = walk.decision

        last_error: FabricError | None = None
        spec: ModelSpec | None = walk.plan.selected
        while spec is not None:
            started = time.perf_counter()
            started_at = time.time()
            try:
                with self._health.in_flight(spec.deployment_id):
                    provider = self._providers.get(spec.provider)
                    _mark_in_flight(spec.provider, 1)
                    try:
                        with optional_span(
                            "llm",
                            gen_ai_system=spec.provider,
                            gen_ai_request_model=spec.provider_model,
                            fabric_model_id=spec.id,
                        ):
                            result = await provider.generate(
                                self._to_inference_request(request, spec)
                            )
                    finally:
                        _mark_in_flight(spec.provider, -1)
            except FabricError as exc:
                elapsed = (time.perf_counter() - started) * 1000
                self._health.record_failure(spec.deployment_id, latency_ms=elapsed, error=str(exc))
                reason = reason_for_error(exc)
                decision.attempts.append(
                    _new_attempt(
                        spec,
                        elapsed=elapsed,
                        reason=walk.reason_for(spec.id),
                        depth=walk.ledger.depth,
                        error=str(exc),
                        started_at=started_at,
                    )
                )
                exc.decision = decision
                if not isinstance(exc, RetryableError):
                    raise
                last_error = exc
                spec = walk.advance(spec, reason, elapsed, str(exc), self._max_attempts)
                continue

            elapsed = (time.perf_counter() - started) * 1000
            self._health.record_success(spec.deployment_id, latency_ms=elapsed)
            decision.attempts.append(
                _new_attempt(
                    spec,
                    elapsed=elapsed,
                    reason=walk.reason_for(spec.id),
                    depth=walk.ledger.depth,
                    result=result,
                    started_at=started_at,
                )
            )
            decision.selected_model = spec.id
            decision.selected_provider = spec.provider
            return RoutedResult(result=result, spec=spec, decision=decision)

        failure = AllCandidatesFailedError(
            f"all {len(decision.attempts)} candidate(s) for '{decision.requested_model}' failed; "
            f"last error: {last_error}"
        )
        failure.decision = decision
        raise failure

    async def stream(
        self, request: ChatCompletionRequest, *, route: RouteRequest | None = None
    ) -> AsyncIterator[tuple[StreamEvent, ModelSpec, RouteDecision]]:
        """Stream a response, failing over only before the first byte is emitted."""
        walk = self._start(request, route)
        decision = walk.decision

        last_error: FabricError | None = None
        spec: ModelSpec | None = walk.plan.selected
        while spec is not None:
            current = spec
            started = time.perf_counter()
            started_at = time.time()
            emitted_any = False
            stream = None
            try:
                with self._health.in_flight(spec.deployment_id):
                    provider = self._providers.get(spec.provider)
                    _mark_in_flight(spec.provider, 1)
                    with optional_span(
                        "llm",
                        gen_ai_system=spec.provider,
                        gen_ai_request_model=spec.provider_model,
                        fabric_model_id=spec.id,
                        streamed=True,
                    ):
                        stream = provider.stream(self._to_inference_request(request, spec))
                        async for event in stream:
                            if not emitted_any and isinstance(event, (StreamDelta, StreamEnd)):
                                # A stream that ends without text still served the
                                # request, so the caller receives usage either way.
                                emitted_any = True
                                elapsed = (time.perf_counter() - started) * 1000
                                self._health.record_success(spec.deployment_id, latency_ms=elapsed)
                                decision.selected_model = spec.id
                                decision.selected_provider = spec.provider
                                decision.attempts.append(
                                    _new_attempt(
                                        spec,
                                        elapsed=elapsed,
                                        reason=walk.reason_for(spec.id),
                                        depth=walk.ledger.depth,
                                        started_at=started_at,
                                    )
                                )
                            if (
                                isinstance(event, StreamEnd)
                                and decision.attempts
                                and decision.attempts[-1].model_id == spec.id
                            ):
                                _apply_stream_usage(decision.attempts[-1], event, spec)
                            yield event, spec, decision
                    return
            except FabricError as exc:
                elapsed = (time.perf_counter() - started) * 1000
                exc.decision = decision
                if emitted_any:
                    # Bytes are already with the client.
                    self._health.record_failure(
                        spec.deployment_id, latency_ms=elapsed, error=str(exc)
                    )
                    if decision.attempts and decision.attempts[-1].error is None:
                        decision.attempts[-1].error = str(exc)
                        decision.attempts[-1].completed_at = time.time()
                    raise
                self._health.record_failure(spec.deployment_id, latency_ms=elapsed, error=str(exc))
                reason = reason_for_error(exc)
                decision.attempts.append(
                    _new_attempt(
                        spec,
                        elapsed=elapsed,
                        reason=walk.reason_for(spec.id),
                        depth=walk.ledger.depth,
                        error=str(exc),
                        started_at=started_at,
                    )
                )
                if not isinstance(exc, RetryableError):
                    raise
                last_error = exc
                spec = walk.advance(spec, reason, elapsed, str(exc), self._max_attempts)
                continue
            finally:
                _mark_in_flight(current.provider, -1)
                await _aclose_stream(stream)

        failure = AllCandidatesFailedError(
            f"all {len(decision.attempts)} candidate(s) for '{decision.requested_model}' failed; "
            f"last error: {last_error}"
        )
        failure.decision = decision
        raise failure


@dataclass(slots=True)
class _Walk:
    """Mutable state of one traversal of the fallback graph."""

    plan: RoutePlan
    decision: RouteDecision
    ledger: FallbackLedger
    visited: set[str] = field(default_factory=set)
    reasons: dict[str, FallbackReason] = field(default_factory=dict)

    def reason_for(self, model_id: str) -> FallbackReason | None:
        return self.reasons.get(model_id)

    def advance(
        self,
        current: ModelSpec,
        reason: FallbackReason,
        elapsed_ms: float,
        error: str,
        max_attempts: int,
    ) -> ModelSpec | None:
        """Pick the next deployment for `reason`, or stop and say why."""
        self.visited.add(current.id)
        self.ledger.charge(latency_ms=elapsed_ms)

        if len(self.decision.attempts) >= max_attempts:
            self.decision.fallback.refuse(
                f"attempt limit of {max_attempts} reached after '{current.id}'"
            )
            return None

        eligible = [candidate.spec.id for candidate in self.plan.ranked]
        target = self.plan.graph.next_hop(
            current.id, reason, visited=self.visited, eligible=eligible
        )
        if target is None:
            self.decision.fallback.refuse(
                f"no untried fallback from '{current.id}' answers '{reason.value}'"
            )
            return None

        spec = next(c.spec for c in self.plan.ranked if c.spec.id == target)
        estimate = spec.blended_cost_per_mtok / 1_000_000 if spec.is_priced else 0.0
        if refusal := self.ledger.refuse_reason(next_cost_usd=estimate):
            self.decision.fallback.refuse(refusal)
            return None

        self.ledger.advance()
        self.reasons[target] = reason
        self.decision.fallback.record(
            FallbackHop(
                source=current.id,
                target=target,
                reason=reason,
                depth=self.ledger.depth,
                error=error,
            )
        )
        return spec


def _mark_in_flight(provider: str, delta: int) -> None:
    telemetry = current_telemetry()
    if telemetry is None:
        return
    telemetry.metrics.in_flight_by_provider.labels(telemetry.metrics.provider_label(provider)).inc(
        delta
    )


async def _aclose_stream(stream: AsyncIterator[object] | None) -> None:
    """Release the provider stream so HTTP connections return to the pool."""
    if stream is None:
        return
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except (RuntimeError, StopAsyncIteration):
        return
