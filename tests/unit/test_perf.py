"""Stage benches stay honest: missing backends stay empty, errors count."""

from __future__ import annotations

import json
from pathlib import Path

from llm_fabric.bench.perf import main as perf_main
from llm_fabric.bench.resources import elapsed_cpu, sample_process
from llm_fabric.bench.stages import OPTIMIZATION_FLAGS, STAGE_NAMES, run_stages
from llm_fabric.bench.store import write_artifact


def test_every_named_stage_is_reported() -> None:
    results = {row.name: row for row in run_stages(iterations=20, warmup=5)}
    assert tuple(results) == STAGE_NAMES
    assert results["ollama_inference"].available is False
    assert results["vllm_inference"].available is False
    assert results["ollama_inference"].throughput_per_s is None
    assert results["vllm_inference"].p50_ms is None


def test_available_stages_count_errors_instead_of_dropping_them() -> None:
    results = {row.name: row for row in run_stages(iterations=20, warmup=5)}
    for name in (
        "api_gateway",
        "auth",
        "intent_exact_cache",
        "semantic_intent_cache",
        "classifier",
        "intent_rules",
        "intent_embedding",
        "intent_mixed",
        "router",
        "streaming",
        "full_system",
    ):
        row = results[name]
        assert row.available is True
        assert row.error_rate is not None
        assert row.iterations == 20
        assert row.p50_ms is not None or row.errors == row.iterations


def test_cache_and_auth_stages_are_hits_not_failures() -> None:
    results = {row.name: row for row in run_stages(iterations=30, warmup=5)}
    assert results["auth"].errors == 0
    assert results["intent_exact_cache"].errors == 0
    assert results["semantic_intent_cache"].errors == 0
    assert results["router"].errors == 0


def test_unproven_optimizations_are_not_production_defaults() -> None:
    enabled = {item["name"]: item["enabled"] for item in OPTIMIZATION_FLAGS}
    assert enabled["async_batching"] is False
    assert enabled["prefix_caching"] is False
    assert enabled["quantized_kv_cache"] is False
    assert enabled["continuous_batching"] is False
    assert enabled["chunked_prefill"] is False
    assert enabled["speculative_decoding"] is False
    assert enabled["model_residency"] is False
    assert enabled["request_coalescing"] is False
    assert enabled["semantic_caching"] is False
    assert enabled["connection_pooling"] is False


def test_artifact_records_commit_and_metric_version(tmp_path: Path) -> None:
    path = write_artifact("stages", {"ok": True}, root=tmp_path, repo=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "stages"
    assert payload["metric_version"] == "perf-metrics-v1"
    assert payload["commit"] is None
    assert payload["payload"] == {"ok": True}
    latest = json.loads((tmp_path / "stages-latest.json").read_text(encoding="utf-8"))
    assert latest["payload"] == {"ok": True}


def test_perf_cli_writes_an_artifact(tmp_path: Path) -> None:
    output = tmp_path / "stages.json"
    assert (
        perf_main(
            [
                "stages",
                "--iterations",
                "15",
                "--warmup",
                "3",
                "--output",
                str(output),
                "--artifact-root",
                str(tmp_path / "art"),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    names = [row["name"] for row in payload["stages"]]
    assert names == list(STAGE_NAMES)
    assert list((tmp_path / "art").glob("*/stages-*.json"))


def test_resource_sample_records_cpu_and_leaves_gpu_empty_without_nvidia() -> None:
    before = sample_process()
    after = sample_process()
    elapsed = elapsed_cpu(before, after)
    assert elapsed["cpu_s"] >= 0.0
    assert after.gpu is None
    assert after.queue_depth is None
