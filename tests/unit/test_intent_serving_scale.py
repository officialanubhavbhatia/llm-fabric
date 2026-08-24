"""In-process IntentOS scale: classifications, not generation throughput."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pytest

from llm_fabric.intent.bootstrap import bootstrap_taxonomy
from llm_fabric.intent.factory import build_offline_cascade
from llm_fabric.intent.schema import ClassificationRequest, ServingClassificationState
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope

PROMPTS = (
    "translate this sentence into French",
    "debug this python traceback",
    "summarise the following article",
    "what is 12 times 8",
    "write a polite email to a customer",
    "extract the invoice total as json",
    "hello there",
    "as an agent, use tools to book a flight",
    "explain quantum entanglement simply",
    "classify this support ticket",
)


@pytest.mark.asyncio
async def test_ten_thousand_offline_classifications_complete() -> None:
    cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())
    scope = TenantScope(tenant_id="scale")
    n = 10_000
    latencies: list[float] = []
    started = time.perf_counter()
    for index in range(n):
        text = PROMPTS[index % len(PROMPTS)]
        if index % 17 == 0:
            text = f"{text} #{index}"
        request_started = time.perf_counter()
        decision = await cascade.classify(scope, ClassificationRequest(text=text))
        latencies.append((time.perf_counter() - request_started) * 1000)
        assert decision.classification.serving_state in ServingClassificationState
        assert decision.classification.intent_result_id
    elapsed = time.perf_counter() - started
    snapshot = cascade.metrics.snapshot()
    ordered = sorted(latencies)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[int(len(ordered) * 0.95) - 1]
    p99 = ordered[int(len(ordered) * 0.99) - 1]
    assert snapshot["serving_requests"] == n
    assert snapshot["missing"] == 0
    assert elapsed > 0
    assert p50 >= 0
    report = {
        "n": n,
        "classifications_per_sec": round(n / elapsed, 2),
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "mean_ms": round(statistics.fmean(latencies), 3),
        "elapsed_s": round(elapsed, 3),
        "by_layer": snapshot["by_layer"],
        "cache_hits": snapshot["cache_hits"],
        "known": snapshot["known"],
        "unknown": snapshot["unknown"],
        "abstentions": snapshot["abstentions"],
        "safe_fallback": snapshot["safe_fallback"],
        "cost_usd": snapshot["cost_usd"],
        "kind": "intent_classification",
    }
    out = Path("artifacts/intentos")
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase2-scale.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
