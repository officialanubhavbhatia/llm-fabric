from __future__ import annotations

import pytest

from llm_fabric.config import Settings
from llm_fabric.contract.openai import ChatCompletionRequest, ChatMessage
from llm_fabric.errors import (
    AllCandidatesFailedError,
    InvalidRequestError,
    ModelNotFoundError,
    NoCandidateError,
    ProviderUnavailableError,
)
from llm_fabric.router.engine import Router
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider
from llm_fabric.serving.base import InferenceRequest, StreamDelta, StreamEnd, StreamEvent
from llm_fabric.serving.factory import ProviderFactory


def _router(registry: ModelRegistry, **overrides: object) -> Router:
    providers = ProviderFactory(
        Settings(openai_api_key=None, anthropic_api_key=None),
        overrides={"mock": MockProvider(), "failing": MockProvider(fail=True), **overrides},
    )
    return Router(registry, providers, default_policy="cheapest", max_attempts=3)


def _request(model: str, text: str = "hello") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[ChatMessage(role="user", content=text)])


# -- resolution --------------------------------------------------------------


def test_alias_resolves_cheapest_first(registry: ModelRegistry) -> None:
    policy, candidates = _router(registry).resolve("auto")
    # The registry still says `cheapest`; the fabric reports the constitution's
    # name for it, so an explanation and the specification use one vocabulary.
    assert policy == "cost_first"
    assert [spec.id for spec in candidates] == ["cheap", "premium"]


def test_alias_capability_requirement_filters_candidates(registry: ModelRegistry) -> None:
    _, candidates = _router(registry).resolve("auto-reasoning")
    # 'cheap' lacks the reasoning capability and must be excluded.
    assert [spec.id for spec in candidates] == ["premium"]


def test_pinned_model_is_not_reordered(registry: ModelRegistry) -> None:
    policy, candidates = _router(registry).resolve("premium")
    assert policy == "declared"
    assert candidates[0].id == "premium"


def test_pinned_model_trails_its_declared_fallbacks(registry: ModelRegistry) -> None:
    _, candidates = _router(registry).resolve("cheap")
    assert [spec.id for spec in candidates] == ["cheap", "premium"]


def test_unknown_model_raises(registry: ModelRegistry) -> None:
    with pytest.raises(ModelNotFoundError):
        _router(registry).resolve("nonexistent")


def test_disabled_model_is_not_servable(registry: ModelRegistry) -> None:
    with pytest.raises(NoCandidateError, match="disabled"):
        _router(registry).resolve("retired")


def test_alias_with_no_satisfying_candidate_raises() -> None:
    registry = ModelRegistry.from_mapping(
        {
            "models": [{"id": "a", "provider": "mock", "enabled": False}],
            "aliases": [{"id": "auto", "candidates": ["a"]}],
        }
    )
    with pytest.raises(NoCandidateError):
        _router(registry).resolve("auto")


# -- completion --------------------------------------------------------------


async def test_completion_uses_provider_native_model_name(registry: ModelRegistry) -> None:
    routed = await _router(registry).complete(_request("cheap"))
    # The mock echoes the model it was actually called with.
    assert "cheap-v1" in routed.result.text
    assert routed.spec.id == "cheap"


async def test_completion_records_the_decision(registry: ModelRegistry) -> None:
    routed = await _router(registry).complete(_request("auto"))
    decision = routed.decision

    assert decision.requested_model == "auto"
    assert decision.policy == "cost_first"
    assert decision.considered == ["cheap", "premium"]
    assert decision.selected_model == "cheap"
    assert decision.failover_count == 0
    assert [a.succeeded for a in decision.attempts] == [True]


async def test_failover_moves_to_next_candidate(registry: ModelRegistry) -> None:
    routed = await _router(registry).complete(_request("broken"))

    assert routed.spec.id == "cheap"
    assert routed.decision.failover_count == 1
    assert routed.decision.attempts[0].model_id == "broken"
    assert routed.decision.attempts[0].error is not None
    assert routed.decision.attempts[1].succeeded


async def test_failover_across_an_alias_chain(registry: ModelRegistry) -> None:
    routed = await _router(registry).complete(_request("auto-failover"))
    assert routed.spec.id == "cheap"
    assert routed.decision.policy == "declared"


async def test_error_when_every_candidate_fails() -> None:
    registry = ModelRegistry.from_mapping({"models": [{"id": "only", "provider": "failing"}]})
    with pytest.raises(AllCandidatesFailedError):
        await _router(registry).complete(_request("only"))


async def test_max_attempts_caps_the_chain() -> None:
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {"id": "a", "provider": "failing", "fallbacks": ["b"]},
                {"id": "b", "provider": "failing", "fallbacks": ["c"]},
                {"id": "c", "provider": "mock"},
            ]
        }
    )
    providers = ProviderFactory(
        Settings(), overrides={"mock": MockProvider(), "failing": MockProvider(fail=True)}
    )
    router = Router(registry, providers, max_attempts=2)

    # Only two attempts are permitted, so the working third candidate is never
    # reached and the request fails.
    with pytest.raises(AllCandidatesFailedError):
        await router.complete(_request("a"))


# -- streaming ---------------------------------------------------------------


async def test_stream_emits_deltas_then_exactly_one_end(registry: ModelRegistry) -> None:
    events = [event async for event, _, _ in _router(registry).stream(_request("cheap"))]

    assert isinstance(events[-1], StreamEnd)
    assert all(isinstance(event, StreamDelta) for event in events[:-1])
    assert sum(isinstance(event, StreamEnd) for event in events) == 1


async def test_streamed_text_matches_buffered_text(registry: ModelRegistry) -> None:
    router = _router(registry)
    buffered = await router.complete(_request("cheap", "same input"))
    deltas = [
        event.text
        async for event, _, _ in router.stream(_request("cheap", "same input"))
        if isinstance(event, StreamDelta)
    ]
    assert "".join(deltas) == buffered.result.text


async def test_stream_fails_over_before_first_byte(registry: ModelRegistry) -> None:
    served: set[str] = set()
    async for event, spec, decision in _router(registry).stream(_request("broken")):
        served.add(spec.id)
        if isinstance(event, StreamEnd):
            assert decision.failover_count == 1

    assert served == {"cheap"}, "failed candidate must not appear as a serving model"


class _RecordingStream:
    def __init__(self, events: list[StreamEvent] | None = None, error: Exception | None = None):
        self.events = events or []
        self.error = error
        self.closed = False

    def __aiter__(self) -> _RecordingStream:
        self._index = 0
        return self

    async def __anext__(self) -> StreamEvent:
        if self._index < len(self.events):
            event = self.events[self._index]
            self._index += 1
            return event
        if self.error is not None:
            raise self.error
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class _DeltaThenFail:
    name = "flaky"

    def __init__(self) -> None:
        self.streams: list[_RecordingStream] = []

    async def generate(self, request: InferenceRequest):
        raise ProviderUnavailableError("should not be called")

    def stream(self, request: InferenceRequest) -> _RecordingStream:
        stream = _RecordingStream(
            events=[StreamDelta(text="partial")],
            error=ProviderUnavailableError("died after first byte"),
        )
        self.streams.append(stream)
        return stream

    async def aclose(self) -> None:
        return None


class _ClosedOnError:
    name = "closed"

    def __init__(self) -> None:
        self.stream_obj = _RecordingStream(error=ProviderUnavailableError("no bytes"))

    async def generate(self, request: InferenceRequest):
        raise ProviderUnavailableError("no")

    def stream(self, request: InferenceRequest) -> _RecordingStream:
        return self.stream_obj

    async def aclose(self) -> None:
        return None


class _CallerError:
    name = "caller"

    async def generate(self, request: InferenceRequest):
        raise InvalidRequestError("temperature out of range")

    def stream(self, request: InferenceRequest):
        raise InvalidRequestError("temperature out of range")
        yield  # make this a generator  # pragma: no cover

    async def aclose(self) -> None:
        return None


async def test_stream_does_not_failover_after_first_byte(registry: ModelRegistry) -> None:
    flaky = _DeltaThenFail()
    router = _router(registry, failing=flaky)
    served: list[str] = []

    with pytest.raises(ProviderUnavailableError, match="died after first byte"):
        async for _event, spec, _decision in router.stream(_request("broken")):
            served.append(spec.id)

    assert served == ["broken"], "must not splice in a second model after the first byte"
    assert flaky.streams[0].closed is True


async def test_stream_closes_provider_after_pre_byte_failure(registry: ModelRegistry) -> None:
    closed = _ClosedOnError()
    router = _router(registry, failing=closed)
    events = [event async for event, _, _ in router.stream(_request("broken"))]
    assert events
    assert closed.stream_obj.closed is True


async def test_caller_error_does_not_failover(registry: ModelRegistry) -> None:
    router = _router(registry, failing=_CallerError())
    with pytest.raises(InvalidRequestError, match="temperature"):
        await router.complete(_request("broken"))


async def test_missing_credentials_fail_over_to_the_next_candidate() -> None:
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {"id": "remote", "provider": "openai", "fallbacks": ["local"]},
                {"id": "local", "provider": "mock"},
            ]
        }
    )
    routed = await _router(registry).complete(_request("remote"))
    assert routed.spec.id == "local"
    assert routed.decision.failover_count == 1
    assert routed.decision.attempts[0].error is not None


async def test_all_candidates_failed_carries_the_decision() -> None:
    registry = ModelRegistry.from_mapping({"models": [{"id": "only", "provider": "failing"}]})
    with pytest.raises(AllCandidatesFailedError) as caught:
        await _router(registry).complete(_request("only"))
    assert caught.value.decision is not None
    assert len(caught.value.decision.attempts) == 1
    assert caught.value.decision.failover_count == 1
