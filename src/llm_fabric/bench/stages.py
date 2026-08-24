"""Isolated stage benches. Profile the slice, not the whole machine.

Each stage is one named loop. A stage whose backend is not built returns
`available=False` and no timings, rather than a mock that would look like
inference. Errors are counted; they are never dropped to inflate throughput.

These are in-process measurements. They exclude sockets and HTTP parsing, which
the HTTP harness measures separately. Do not compare a stage µs/request to an
HTTP req/s figure as if they were the same experiment.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from llm_fabric.bench.load import WORKLOADS, _percentile
from llm_fabric.bench.profile import _REGISTRY, _drive
from llm_fabric.bench.resources import elapsed_cpu, sample_process
from llm_fabric.config import Settings
from llm_fabric.contract.openai import ChatMessage
from llm_fabric.gateway.app import create_app
from llm_fabric.identity.apikey import ApiCredential, ApiKeyVerifier
from llm_fabric.intent.bootstrap import bootstrap_taxonomy
from llm_fabric.intent.cache import ExactIntentCache, IntentCacheDiscriminators, SemanticIntentCache
from llm_fabric.intent.embeddings import HashingEmbedder
from llm_fabric.intent.factory import build_offline_cascade
from llm_fabric.intent.schema import (
    ClassificationRequest,
    Complexity,
    ContextClass,
    CostClass,
    IntentClassification,
    LatencyClass,
    Modality,
    QualityClass,
    ReasoningLevel,
    RiskClass,
)
from llm_fabric.router.plan import RoutePlanner, RouteRequest
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider
from llm_fabric.serving.base import InferenceRequest
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope

DEFAULT_ITERATIONS = 2_000
DEFAULT_WARMUP = 100

STAGE_NAMES = (
    "api_gateway",
    "auth",
    "intent_exact_cache",
    "semantic_intent_cache",
    "classifier",
    "intent_rules",
    "intent_embedding",
    "intent_mixed",
    "router",
    "ollama_inference",
    "vllm_inference",
    "streaming",
    "full_system",
)


@dataclass(frozen=True, slots=True)
class StageResult:
    name: str
    available: bool
    iterations: int
    errors: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    throughput_per_s: float | None
    error_rate: float | None
    tokens_per_s: float | None
    cpu_s: float | None
    rss_bytes: int | None
    gpu: dict[str, Any] | None
    queue_depth: int | None
    note: str
    configuration: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "iterations": self.iterations,
            "errors": self.errors,
            "p50_ms": _round(self.p50_ms),
            "p95_ms": _round(self.p95_ms),
            "p99_ms": _round(self.p99_ms),
            "throughput_per_s": _round(self.throughput_per_s, 1),
            "error_rate": _round(self.error_rate, 6),
            "tokens_per_s": _round(self.tokens_per_s, 1),
            "cpu_s": _round(self.cpu_s, 4),
            "rss_bytes": self.rss_bytes,
            "gpu": self.gpu,
            "queue_depth": self.queue_depth,
            "note": self.note,
            "configuration": self.configuration,
        }


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _unavailable(name: str, note: str) -> StageResult:
    return StageResult(
        name=name,
        available=False,
        iterations=0,
        errors=0,
        p50_ms=None,
        p95_ms=None,
        p99_ms=None,
        throughput_per_s=None,
        error_rate=None,
        tokens_per_s=None,
        cpu_s=None,
        rss_bytes=None,
        gpu=None,
        queue_depth=None,
        note=note,
    )


def _from_latencies(
    name: str,
    latencies_ms: list[float],
    *,
    errors: int,
    note: str,
    tokens_per_op: float | None = None,
    before: Any = None,
    after: Any = None,
    queue_depth: int | None = None,
    configuration: dict[str, Any] | None = None,
) -> StageResult:
    ordered = sorted(latencies_ms)
    elapsed_s = sum(latencies_ms) / 1000.0
    n = len(ordered)
    total = n + errors
    cpu = elapsed_cpu(before, after) if before is not None and after is not None else None
    throughput = (n / elapsed_s) if elapsed_s > 0 else None
    return StageResult(
        name=name,
        available=True,
        iterations=total,
        errors=errors,
        p50_ms=_percentile(ordered, 0.50) if ordered else None,
        p95_ms=_percentile(ordered, 0.95) if ordered else None,
        p99_ms=_percentile(ordered, 0.99) if ordered else None,
        throughput_per_s=throughput,
        error_rate=(errors / total) if total else None,
        tokens_per_s=(throughput * tokens_per_op) if throughput and tokens_per_op else None,
        cpu_s=cpu["cpu_s"] if cpu else None,
        rss_bytes=after.rss_bytes if after is not None else None,
        gpu=after.gpu if after is not None else None,
        queue_depth=queue_depth,
        note=note,
        configuration=configuration or {},
    )


async def _loop(
    op: Callable[[], Awaitable[None]],
    *,
    iterations: int,
    warmup: int,
) -> tuple[list[float], int]:
    for _ in range(warmup):
        try:
            await op()
        except Exception:
            continue
    latencies: list[float] = []
    errors = 0
    for _ in range(iterations):
        started = time.perf_counter()
        try:
            await op()
        except Exception:
            errors += 1
            continue
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies, errors


def run_stages(
    *,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
) -> list[StageResult]:
    return asyncio.run(_run_all(iterations=iterations, warmup=warmup))


async def _run_all(*, iterations: int, warmup: int) -> list[StageResult]:
    return [
        await _api_gateway(iterations, warmup),
        await _auth(iterations, warmup),
        await _exact_cache(iterations, warmup),
        await _semantic_cache(iterations, warmup),
        await _classifier(iterations, warmup),
        await _intent_rules(iterations, warmup),
        await _intent_embedding(iterations, warmup),
        await _intent_mixed(iterations, warmup),
        await _router(iterations, warmup),
        _unavailable(
            "ollama_inference",
            "No live Ollama process is measured in this isolated bench. "
            "The OpenAI-compatible adapter exists; this stage does not start Ollama.",
        ),
        _unavailable(
            "vllm_inference",
            "No live vLLM process is measured in this isolated bench. "
            "The OpenAI-compatible adapter exists; this stage does not start vLLM.",
        ),
        await _streaming(iterations, warmup),
        await _full_system(iterations, warmup),
    ]


async def _api_gateway(iterations: int, warmup: int) -> StageResult:
    app = _app()
    workload = WORKLOADS["liveness"]

    async def op() -> None:
        await _drive(app, workload, 1)

    async with app.router.lifespan_context(app):
        return await _measure(
            "api_gateway",
            op,
            iterations,
            warmup,
            note="In-process ASGI GET /healthz. No auth, no routing, no sockets.",
        )


async def _auth(iterations: int, warmup: int) -> StageResult:
    verifier = ApiKeyVerifier(
        [ApiCredential(key="bench-key-that-is-long", tenant_id="acme", user_id="bench")]
    )

    async def op() -> None:
        await verifier.verify("bench-key-that-is-long")

    return await _measure(
        "auth",
        op,
        iterations,
        warmup,
        note="ApiKeyVerifier.verify on one configured key. OIDC JWKS verification is unmeasured.",
        configuration={"mode": "api_key", "keys": 1},
    )


async def _exact_cache(iterations: int, warmup: int) -> StageResult:
    cache = ExactIntentCache(TenantScopedCache())
    scope = TenantScope(tenant_id="acme", user_id="bench")
    disc = IntentCacheDiscriminators(
        taxonomy_version="tax",
        classifier_version="clf",
        policy_version="v1",
        language="en",
    )
    text = "debug this python traceback"
    classification = _cached_classification()
    cache.put(scope, text, disc, classification)

    async def op() -> None:
        hit = cache.get(scope, text, disc)
        if hit is None:
            raise RuntimeError("expected an exact-cache hit")

    return await _measure(
        "intent_exact_cache",
        op,
        iterations,
        warmup,
        note="L0 exact hit. Miss path is not this stage.",
    )


async def _semantic_cache(iterations: int, warmup: int) -> StageResult:
    embedder = HashingEmbedder()
    cache = SemanticIntentCache()
    scope = TenantScope(tenant_id="acme", user_id="bench")
    disc = IntentCacheDiscriminators(
        taxonomy_version="tax",
        classifier_version="clf",
        policy_version="v1",
        language="en",
    )
    text = "debug this python traceback"
    vector = embedder.embed_one(text)
    classification = _cached_classification()
    if not cache.admit(scope, text, vector, disc, classification):
        raise RuntimeError("semantic cache refused the seed entry")

    async def op() -> None:
        hit = cache.lookup(scope, vector, disc)
        if hit is None:
            raise RuntimeError("expected a semantic-cache hit")

    return await _measure(
        "semantic_intent_cache",
        op,
        iterations,
        warmup,
        note=(
            "L1 semantic hit against HashingEmbedder, which is lexical hashing "
            "and not a semantic model."
        ),
        configuration={"embedder": "HashingEmbedder"},
    )


async def _classifier(iterations: int, warmup: int) -> StageResult:
    cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())
    scope = TenantScope(tenant_id="acme", user_id="bench")
    request = ClassificationRequest(text="Write a python function that sorts a list")

    async def op() -> None:
        await cascade.classify(scope, request)

    return await _measure(
        "classifier",
        op,
        iterations,
        warmup,
        note="Offline cascade (L0–L3). L4/L5 are off. Cache is cold on the first call only.",
        configuration={"layers": "L0-L3", "embedder": "HashingEmbedder"},
    )


async def _intent_rules(iterations: int, warmup: int) -> StageResult:
    from llm_fabric.intent.cache import ExactIntentCache
    from llm_fabric.intent.cascade import IntentCascade
    from llm_fabric.intent.classifiers.rules import DeterministicClassifier

    cascade = IntentCascade(
        taxonomy=bootstrap_taxonomy(),
        exact_cache=ExactIntentCache(TenantScopedCache()),
        rules=DeterministicClassifier(),
    )
    scope = TenantScope(tenant_id="acme", user_id="bench")
    request = ClassificationRequest(text="translate this paragraph into French")

    async def op() -> None:
        await cascade.classify(scope, request)

    return await _measure(
        "intent_rules",
        op,
        iterations,
        warmup,
        note="L2 deterministic rules only. L0 misses; L3–L5 off.",
        configuration={"layers": "L2"},
    )


async def _intent_embedding(iterations: int, warmup: int) -> StageResult:
    from llm_fabric.intent.cache import ExactIntentCache
    from llm_fabric.intent.cascade import IntentCascade
    from llm_fabric.intent.classifiers.embedding import EmbeddingClassifier

    cascade = IntentCascade(
        taxonomy=bootstrap_taxonomy(),
        exact_cache=ExactIntentCache(TenantScopedCache()),
        embedding=EmbeddingClassifier(HashingEmbedder()),
    )
    scope = TenantScope(tenant_id="acme", user_id="bench")
    request = ClassificationRequest(text="why does this type checker reject my generic")

    async def op() -> None:
        await cascade.classify(scope, request)

    return await _measure(
        "intent_embedding",
        op,
        iterations,
        warmup,
        note="L3 nearest-centroid over HashingEmbedder. Lexical, not semantic.",
        configuration={"layers": "L3", "embedder": "HashingEmbedder"},
    )


async def _intent_mixed(iterations: int, warmup: int) -> StageResult:
    cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())
    scope = TenantScope(tenant_id="acme", user_id="bench")
    prompts = (
        "translate this to French",
        "summarise the attached memo",
        "fix this Python traceback",
        "what is 15% of 80",
        "asdf qwer zxcv",
    )

    index = 0

    async def op() -> None:
        nonlocal index
        await cascade.classify(scope, ClassificationRequest(text=prompts[index % len(prompts)]))
        index += 1

    return await _measure(
        "intent_mixed",
        op,
        iterations,
        warmup,
        note="Offline cascade on a rotating mix of obvious, ordinary and OOD prompts.",
        configuration={"layers": "L0-L3", "mix": "rules/embed/ood"},
    )


async def _router(iterations: int, warmup: int) -> StageResult:
    registry = ModelRegistry.from_mapping(_REGISTRY)
    planner = RoutePlanner(registry)
    request = RouteRequest(requested_model="auto", tenant_id="public", prompt_tokens=14)

    async def op() -> None:
        planner.plan(request)

    return await _measure(
        "router",
        op,
        iterations,
        warmup,
        note="RoutePlanner.plan for alias auto. No provider call.",
        queue_depth=sum(snap.queue_depth for snap in planner.health.all_snapshots().values()),
    )


async def _streaming(iterations: int, warmup: int) -> StageResult:
    provider = MockProvider()
    request = InferenceRequest(
        model="mock-small",
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=16,
    )

    async def op() -> None:
        async for _event in provider.stream(request):
            pass

    return await _measure(
        "streaming",
        op,
        iterations,
        warmup,
        note="MockProvider.stream drained in-process. The mock does not generate tokens.",
        tokens_per_op=None,
    )


async def _full_system(iterations: int, warmup: int) -> StageResult:
    app = _app()
    workload = WORKLOADS["chat-short"]

    async def op() -> None:
        await _drive(app, workload, 1)

    async with app.router.lifespan_context(app):
        return await _measure(
            "full_system",
            op,
            iterations,
            warmup,
            note=(
                "In-process ASGI POST /v1/chat/completions against the mock provider. "
                "Same path as llm-fabric-profile --workload chat-short."
            ),
            tokens_per_op=None,
        )


async def _measure(
    name: str,
    op: Callable[[], Awaitable[None]],
    iterations: int,
    warmup: int,
    *,
    note: str,
    tokens_per_op: float | None = None,
    queue_depth: int | None = None,
    configuration: dict[str, Any] | None = None,
) -> StageResult:
    before = sample_process()
    latencies, errors = await _loop(op, iterations=iterations, warmup=warmup)
    after = sample_process()
    return _from_latencies(
        name,
        latencies,
        errors=errors,
        note=note,
        tokens_per_op=tokens_per_op,
        before=before,
        after=after,
        queue_depth=queue_depth,
        configuration=configuration,
    )


def _cached_classification() -> IntentClassification:
    return IntentClassification(
        intent_id="coding",
        domain="coding",
        complexity=Complexity.MODERATE,
        reasoning_level=ReasoningLevel.LIGHT,
        modality=Modality.TEXT,
        context_class=ContextClass.SHORT,
        risk_class=RiskClass.LOW,
        latency_class=LatencyClass.INTERACTIVE,
        quality_class=QualityClass.STANDARD,
        cost_class=CostClass.LOW,
        confidence=0.95,
        classifier_version="clf",
        taxonomy_version="tax",
    )


def _app() -> Any:
    return create_app(
        settings=Settings(api_keys=[], log_level="ERROR"),
        registry=ModelRegistry.from_mapping(_REGISTRY),
        provider_overrides={"mock": MockProvider()},
    )


#: Techniques the constitution names. None are production defaults here:
#: the backends they need are unbuilt, or a measured run has not shown a gain.
OPTIMIZATION_FLAGS: tuple[dict[str, Any], ...] = (
    {
        "name": "connection_pooling",
        "enabled": False,
        "note": (
            "httpx keep-alive limits already exist on the OpenAI/Anthropic "
            "adapters as ordinary client plumbing. They have not been A/B "
            "tested against a real provider, so they are not claimed as a "
            "Phase 10 optimization."
        ),
    },
    {
        "name": "async_batching",
        "enabled": False,
        "note": "No batching engine. Not enabled.",
    },
    {
        "name": "semantic_caching",
        "enabled": False,
        "note": (
            "The L1 intent semantic cache exists and is off on the serving path "
            "because the latency it adds has not been measured there. Response "
            "semantic cache is not built."
        ),
    },
    {
        "name": "prefix_caching",
        "enabled": False,
        "note": (
            "A vLLM/Ollama engine property. Chat adapters exist; "
            "prefix-cache metrics are not scraped."
        ),
    },
    {
        "name": "quantized_kv_cache",
        "enabled": False,
        "note": "A vLLM engine property. Chat can use vLLM over HTTP; /metrics is not scraped.",
    },
    {
        "name": "continuous_batching",
        "enabled": False,
        "note": "A vLLM engine property. Chat can use vLLM over HTTP; /metrics is not scraped.",
    },
    {
        "name": "chunked_prefill",
        "enabled": False,
        "note": "A vLLM engine property. Chat can use vLLM over HTTP; /metrics is not scraped.",
    },
    {
        "name": "speculative_decoding",
        "enabled": False,
        "note": "A vLLM engine property. Chat can use vLLM over HTTP; /metrics is not scraped.",
    },
    {
        "name": "model_residency",
        "enabled": False,
        "note": "No model loader. Not enabled.",
    },
    {
        "name": "request_coalescing",
        "enabled": False,
        "note": "Not implemented. Not enabled.",
    },
)
