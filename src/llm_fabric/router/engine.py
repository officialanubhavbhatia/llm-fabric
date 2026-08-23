"""The routing engine: resolve a request to a model, then serve it.

The engine owns two things worth stating plainly.

**Failover.** Candidates are tried in policy order. Only `RetryableError` moves on
to the next candidate; a caller error stops immediately, because retrying a
malformed request only fails again more slowly.

**Failover under streaming.** Once a single byte of generated text has been handed
to the client, failing over is no longer possible — the client has already
committed to a response it cannot un-see. So the engine may fail over while a
stream has produced nothing, and must not once it has. That boundary is enforced
here rather than left to each adapter.

Every decision records the candidates considered and each attempt made, so a
route can be explained after the fact instead of guessed at.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from llm_fabric.contract.openai import ChatCompletionRequest
from llm_fabric.errors import (
    AllCandidatesFailedError,
    FabricError,
    ModelNotFoundError,
    NoCandidateError,
    RetryableError,
)
from llm_fabric.router.policy import get_policy
from llm_fabric.router.registry import ModelRegistry, ModelSpec
from llm_fabric.serving.base import (
    InferenceRequest,
    ProviderResult,
    StreamDelta,
    StreamEnd,
    StreamEvent,
)
from llm_fabric.serving.factory import ProviderFactory


@dataclass(slots=True)
class Attempt:
    model_id: str
    provider: str
    duration_ms: float
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class RouteDecision:
    requested_model: str
    policy: str
    considered: list[str]
    attempts: list[Attempt] = field(default_factory=list)
    selected_model: str | None = None
    selected_provider: str | None = None

    @property
    def failover_count(self) -> int:
        return max(0, len(self.attempts) - 1)


@dataclass(slots=True)
class RoutedResult:
    result: ProviderResult
    spec: ModelSpec
    decision: RouteDecision


class Router:
    def __init__(
        self,
        registry: ModelRegistry,
        providers: ProviderFactory,
        *,
        default_policy: str = "cheapest",
        max_attempts: int = 3,
    ) -> None:
        self._registry = registry
        self._providers = providers
        self._default_policy = default_policy
        self._max_attempts = max(1, max_attempts)

    # -- resolution ----------------------------------------------------------

    def resolve(self, requested_model: str) -> tuple[str, list[ModelSpec]]:
        """Return the policy name and the ordered candidates for a request."""
        if alias := self._registry.alias(requested_model):
            policy_name = alias.policy or self._default_policy
            candidates = [self._registry.get(c) for c in alias.candidates]
            eligible = [
                spec for spec in candidates if spec.enabled and alias.requires <= spec.capabilities
            ]
            if not eligible:
                raise NoCandidateError(
                    f"alias '{requested_model}' has no enabled model satisfying "
                    f"{sorted(alias.requires) or 'its candidate list'}"
                )
            ordered = get_policy(policy_name)(eligible)
            return policy_name, ordered

        if not self._registry.known(requested_model):
            raise ModelNotFoundError(f"unknown model '{requested_model}'")

        # A pinned model is honoured as-is: the caller asked for it explicitly, so
        # reordering would override their intent. Its declared fallbacks trail it.
        primary = self._registry.get(requested_model)
        if not primary.enabled:
            raise NoCandidateError(f"model '{requested_model}' is disabled")

        chain = [primary]
        chain.extend(
            spec for fallback in primary.fallbacks if (spec := self._registry.get(fallback)).enabled
        )
        return "declared", chain

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

    # -- execution -----------------------------------------------------------

    async def complete(self, request: ChatCompletionRequest) -> RoutedResult:
        policy_name, candidates = self.resolve(request.model)
        decision = RouteDecision(
            requested_model=request.model,
            policy=policy_name,
            considered=[spec.id for spec in candidates],
        )

        last_error: FabricError | None = None
        for spec in candidates[: self._max_attempts]:
            started = time.perf_counter()
            try:
                provider = self._providers.get(spec.provider)
                result = await provider.generate(self._to_inference_request(request, spec))
            except RetryableError as exc:
                decision.attempts.append(
                    Attempt(
                        model_id=spec.id,
                        provider=spec.provider,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        error=str(exc),
                    )
                )
                last_error = exc
                continue

            decision.attempts.append(
                Attempt(
                    model_id=spec.id,
                    provider=spec.provider,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            )
            decision.selected_model = spec.id
            decision.selected_provider = spec.provider
            return RoutedResult(result=result, spec=spec, decision=decision)

        raise AllCandidatesFailedError(
            f"all {len(decision.attempts)} candidate(s) for '{request.model}' failed; "
            f"last error: {last_error}"
        )

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[tuple[StreamEvent, ModelSpec, RouteDecision]]:
        """Stream a response, failing over only before the first byte is emitted."""
        policy_name, candidates = self.resolve(request.model)
        decision = RouteDecision(
            requested_model=request.model,
            policy=policy_name,
            considered=[spec.id for spec in candidates],
        )

        last_error: FabricError | None = None
        for spec in candidates[: self._max_attempts]:
            started = time.perf_counter()
            emitted_any = False
            try:
                provider = self._providers.get(spec.provider)
                stream = provider.stream(self._to_inference_request(request, spec))
                async for event in stream:
                    if isinstance(event, StreamDelta) and not emitted_any:
                        emitted_any = True
                        decision.selected_model = spec.id
                        decision.selected_provider = spec.provider
                        decision.attempts.append(
                            Attempt(
                                model_id=spec.id,
                                provider=spec.provider,
                                duration_ms=(time.perf_counter() - started) * 1000,
                            )
                        )
                    if isinstance(event, StreamEnd) and not emitted_any:
                        # Ended without producing text; treat as a served request
                        # so the caller still receives usage.
                        emitted_any = True
                        decision.selected_model = spec.id
                        decision.selected_provider = spec.provider
                        decision.attempts.append(
                            Attempt(
                                model_id=spec.id,
                                provider=spec.provider,
                                duration_ms=(time.perf_counter() - started) * 1000,
                            )
                        )
                    yield event, spec, decision
                return
            except RetryableError as exc:
                if emitted_any:
                    # Bytes are already with the client; failing over now would
                    # splice two different responses together.
                    raise
                decision.attempts.append(
                    Attempt(
                        model_id=spec.id,
                        provider=spec.provider,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        error=str(exc),
                    )
                )
                last_error = exc
                continue

        raise AllCandidatesFailedError(
            f"all {len(decision.attempts)} candidate(s) for '{request.model}' failed; "
            f"last error: {last_error}"
        )
