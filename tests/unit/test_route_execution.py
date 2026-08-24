"""Executing a plan: reasoned failover, circuit breakers and budget ceilings.

Every test drives the synthetic fleet, whose provider fails exactly when told to
and never makes a network call.
"""

from __future__ import annotations

import pytest

from llm_fabric.config import Settings
from llm_fabric.contract.openai import ChatCompletionRequest, ChatMessage
from llm_fabric.errors import (
    AllCandidatesFailedError,
    InvalidRequestError,
    NoCandidateError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from llm_fabric.router.engine import Router
from llm_fabric.router.fallback import ContextTooLargeError, FallbackBudget, FallbackReason
from llm_fabric.router.grades import Grade
from llm_fabric.router.health import BreakerPolicy, BreakerState, HealthTracker
from llm_fabric.router.plan import RouteRequest
from llm_fabric.router.synthetic import SyntheticFleet, synthetic_model_id
from llm_fabric.serving.base import StreamDelta, StreamEnd
from llm_fabric.serving.factory import ProviderFactory

G00 = synthetic_model_id(Grade.GRADE00)
G01 = synthetic_model_id(Grade.GRADE01)
G02 = synthetic_model_id(Grade.GRADE02)
G29 = synthetic_model_id(Grade.GRADE29)


@pytest.fixture
def fleet() -> SyntheticFleet:
    return SyntheticFleet()


def _router(fleet: SyntheticFleet, **kwargs: object) -> Router:
    providers = ProviderFactory(
        Settings(openai_api_key=None, anthropic_api_key=None),
        overrides=fleet.overrides(),
    )
    options: dict[str, object] = {"default_policy": "cost_first", "max_attempts": 3}
    options.update(kwargs)
    return Router(fleet.registry, providers, **options)  # type: ignore[arg-type]


def _request(model: str = "synth-cheap", text: str = "hello") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[ChatMessage(role="user", content=text)])


# -- the happy path -----------------------------------------------------------


async def test_a_successful_route_records_its_plan(fleet: SyntheticFleet) -> None:
    routed = await _router(fleet).complete(_request())

    assert routed.spec.id == G00
    decision = routed.decision
    assert decision.selected_model == G00
    assert decision.failover_count == 0
    assert decision.fallback_depth == 0
    assert decision.plan is not None
    assert decision.plan.policy.value == "cost_first"
    assert decision.attempts[0].reason is None


async def test_the_decision_explains_itself(fleet: SyntheticFleet) -> None:
    routed = await _router(fleet).complete(_request())
    payload = routed.decision.explain()

    assert payload["selected_model"] == G00
    assert payload["plan"]["policy"] == "cost_first"
    assert payload["plan"]["explanation"]
    assert payload["attempts"][0]["error"] is None


# -- failover -----------------------------------------------------------------


async def test_a_failed_attempt_moves_to_the_next_candidate(fleet: SyntheticFleet) -> None:
    fleet.always_fail(G00)
    routed = await _router(fleet).complete(_request())

    assert routed.spec.id == G01
    assert routed.decision.failover_count == 1
    assert routed.decision.fallback_depth == 1
    assert fleet.served == [G00, G01]


async def test_the_fallback_records_the_reason(fleet: SyntheticFleet) -> None:
    fleet.always_fail(G00, ProviderTimeoutError("too slow"))
    routed = await _router(fleet).complete(_request())

    hop = routed.decision.fallback.hops[0]
    assert hop.source == G00
    assert hop.target == G01
    assert hop.reason is FallbackReason.TIMEOUT
    assert hop.depth == 1
    # The attempt that served knows why it was reached.
    assert routed.decision.attempts[-1].reason is FallbackReason.TIMEOUT


async def test_different_reasons_can_take_different_edges(fleet: SyntheticFleet) -> None:
    from llm_fabric.router.fallback import FallbackEdge, FallbackGraph
    from llm_fabric.router.plan import RoutePlanner

    health = HealthTracker()
    graph = FallbackGraph(
        [FallbackEdge(G00, G29, reasons=frozenset({FallbackReason.CONTEXT_TOO_LARGE}))]
    )
    planner = RoutePlanner(fleet.registry, health=health, graph=graph, default_policy="cost_first")
    router = _router(fleet, planner=planner, health=health)

    fleet.always_fail(G00, ContextTooLargeError("prompt too big"))
    with pytest.raises(InvalidRequestError):
        # A caller error stops rather than failing over...
        await router.complete(_request())

    # ...but the graph still knows where that reason would have gone.
    plan = planner.plan(RouteRequest("synth-cheap"))
    assert plan.graph.next_hop(G00, FallbackReason.CONTEXT_TOO_LARGE) == G29
    assert plan.graph.next_hop(G00, FallbackReason.TIMEOUT) == G01


async def test_a_caller_error_stops_immediately(fleet: SyntheticFleet) -> None:
    fleet.always_fail(G00, InvalidRequestError("malformed"))
    with pytest.raises(InvalidRequestError):
        await _router(fleet).complete(_request())
    # Retrying a malformed request only fails again more slowly.
    assert fleet.served == [G00]


async def test_the_attempt_limit_is_honoured(fleet: SyntheticFleet) -> None:
    for grade in range(6):
        fleet.always_fail(synthetic_model_id(Grade.from_index(grade)))

    with pytest.raises(AllCandidatesFailedError) as caught:
        await _router(fleet, max_attempts=3).complete(_request())

    decision = caught.value.decision
    assert decision is not None
    assert len(decision.attempts) == 3  # type: ignore[attr-defined]
    assert any("attempt limit" in note for note in decision.fallback.refused)  # type: ignore[attr-defined]


async def test_a_fallback_never_returns_to_a_tried_deployment(fleet: SyntheticFleet) -> None:
    fleet.always_fail(G00)
    fleet.always_fail(G01)
    routed = await _router(fleet, max_attempts=5).complete(_request())

    assert routed.spec.id == G02
    assert fleet.served == [G00, G01, G02]
    assert len(set(fleet.served)) == len(fleet.served)


# -- budgets ------------------------------------------------------------------


async def test_a_zero_depth_budget_forbids_failing_over(fleet: SyntheticFleet) -> None:
    fleet.always_fail(G00)
    router = _router(fleet, fallback_budget=FallbackBudget(max_depth=0), max_attempts=5)

    with pytest.raises(AllCandidatesFailedError) as caught:
        await router.complete(_request())

    decision = caught.value.decision
    assert fleet.served == [G00]
    assert any("depth" in note for note in decision.fallback.refused)  # type: ignore[attr-defined]


async def test_a_cost_ceiling_stops_the_chain(fleet: SyntheticFleet) -> None:
    fleet.always_fail(G00)
    fleet.always_fail(G01)
    router = _router(
        fleet,
        max_attempts=9,
        fallback_budget=FallbackBudget(max_depth=9, max_cost_usd=0.0),
    )
    with pytest.raises(AllCandidatesFailedError) as caught:
        await router.complete(_request())

    decision = caught.value.decision
    assert any("cost" in note for note in decision.fallback.refused)  # type: ignore[attr-defined]


# -- circuit breakers ---------------------------------------------------------


async def test_repeated_failures_open_the_breaker(fleet: SyntheticFleet) -> None:
    health = HealthTracker(policy=BreakerPolicy(consecutive_failures=2))
    router = _router(fleet, health=health)
    fleet.always_fail(G00)

    deployment = fleet.registry.get(G00).deployment_id
    for _ in range(2):
        await router.complete(_request())

    assert health.snapshot(deployment).state is BreakerState.OPEN


async def test_an_open_breaker_is_skipped_before_it_is_called(fleet: SyntheticFleet) -> None:
    health = HealthTracker(policy=BreakerPolicy(consecutive_failures=1))
    router = _router(fleet, health=health)
    fleet.always_fail(G00)

    await router.complete(_request())
    fleet.reset()

    routed = await router.complete(_request())
    assert routed.spec.id == G01
    # The dead deployment was never asked a second time.
    assert G00 not in fleet.served


async def test_a_recovered_deployment_is_used_again(fleet: SyntheticFleet) -> None:
    class Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()
    health = HealthTracker(
        policy=BreakerPolicy(consecutive_failures=1, open_duration_s=10.0, half_open_successes=1),
        clock=clock,
    )
    router = _router(fleet, health=health)

    fleet.always_fail(G00)
    await router.complete(_request())
    fleet.recover()
    fleet.reset()

    clock.now = 100.0
    routed = await router.complete(_request())
    assert routed.spec.id == G00
    assert health.snapshot(fleet.registry.get(G00).deployment_id).state is BreakerState.CLOSED


async def test_every_circuit_open_gives_no_candidate(fleet: SyntheticFleet) -> None:
    health = HealthTracker(policy=BreakerPolicy(consecutive_failures=1))
    for spec in fleet.registry.all_models():
        health.record_failure(spec.deployment_id, error="boom")

    with pytest.raises(NoCandidateError):
        await _router(fleet, health=health).complete(_request())


# -- health feedback ----------------------------------------------------------


async def test_a_success_is_recorded_as_observed_health(fleet: SyntheticFleet) -> None:
    health = HealthTracker()
    await _router(fleet, health=health).complete(_request())

    snapshot = health.snapshot(fleet.registry.get(G00).deployment_id)
    assert snapshot.successes == 1
    assert snapshot.health_score == pytest.approx(1.0)
    assert snapshot.ewma_latency_ms is not None


async def test_a_failure_is_recorded_as_observed_health(fleet: SyntheticFleet) -> None:
    health = HealthTracker()
    fleet.always_fail(G00)
    await _router(fleet, health=health).complete(_request())

    snapshot = health.snapshot(fleet.registry.get(G00).deployment_id)
    assert snapshot.failures == 1
    assert snapshot.last_error is not None


async def test_queue_depth_returns_to_zero(fleet: SyntheticFleet) -> None:
    health = HealthTracker()
    await _router(fleet, health=health).complete(_request())
    assert all(snapshot.queue_depth == 0 for snapshot in health.all_snapshots().values())


# -- streaming ----------------------------------------------------------------


async def test_streaming_fails_over_before_the_first_byte(fleet: SyntheticFleet) -> None:
    fleet.always_fail(G00)
    router = _router(fleet)

    served: list[str] = []
    text = ""
    async for event, spec, _ in router.stream(_request()):
        served.append(spec.id)
        if isinstance(event, StreamDelta):
            text += event.text

    assert set(served) == {G01}
    assert text


async def test_streaming_does_not_fail_over_after_bytes_are_sent(
    fleet: SyntheticFleet,
) -> None:
    # A stream that has already emitted text cannot be spliced onto another
    # model's output, so a mid-stream failure must surface.
    router = _router(fleet)
    events = 0
    async for _event, _spec, _decision in router.stream(_request()):
        events += 1
    assert events > 1


async def test_a_stream_ends_with_usage(fleet: SyntheticFleet) -> None:
    ends = [
        event
        async for event, _spec, _decision in _router(fleet).stream(_request())
        if isinstance(event, StreamEnd)
    ]
    assert len(ends) == 1
    assert ends[0].prompt_tokens > 0


# -- routing on intent --------------------------------------------------------


async def test_a_route_request_can_be_supplied_explicitly(fleet: SyntheticFleet) -> None:
    router = _router(fleet)
    route = RouteRequest(requested_model="synth-best")
    routed = await router.complete(_request("synth-best"), route=route)
    assert routed.spec.id == G29


async def test_a_tenant_scoped_route_request_is_honoured(fleet: SyntheticFleet) -> None:
    from llm_fabric.router.plan import TenantRoutingPolicies, TenantRoutingPolicy

    policies = TenantRoutingPolicies([TenantRoutingPolicy(tenant_id="acme", require_in_house=True)])
    router = _router(fleet, tenant_policies=policies)
    route = RouteRequest(requested_model="synth-auto", tenant_id="acme")
    routed = await router.complete(_request("synth-auto"), route=route)
    assert routed.spec.keeps_data_in_house


# -- provider failures other than outages -------------------------------------


async def test_an_unavailable_provider_is_a_provider_down_fallback(
    fleet: SyntheticFleet,
) -> None:
    fleet.always_fail(G00, ProviderUnavailableError("down"))
    routed = await _router(fleet).complete(_request())
    assert routed.decision.fallback.hops[0].reason is FallbackReason.PROVIDER_DOWN


async def test_a_transient_failure_recovers_on_the_next_request(
    fleet: SyntheticFleet,
) -> None:
    router = _router(fleet)
    fleet.fail_next(G00, times=1)

    first = await router.complete(_request())
    assert first.spec.id == G01

    second = await router.complete(_request())
    assert second.spec.id == G00
