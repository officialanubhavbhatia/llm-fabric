"""Deterministic model-workload evaluation, separate from IntentOS frozen 98."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from llm_fabric import __version__ as fabric_version
from llm_fabric.config import Settings, get_settings
from llm_fabric.contract.openai import ChatMessage
from llm_fabric.eval.provenance import current_commit
from llm_fabric.router.registry import ModelRegistry, ModelSpec
from llm_fabric.serving.base import InferenceRequest
from llm_fabric.serving.factory import ProviderFactory

EVAL_VERSION = "model-eval-v1"
DEFAULT_DATASET = Path("datasets/eval/models/workloads.jsonl")

NOT_OBJECTIVELY_SCORED = "not objectively scored"

CATEGORIES = (
    "general_conversation",
    "summarization",
    "reasoning",
    "math",
    "coding",
    "data_analysis",
    "structured_extraction",
    "long_context",
)


def load_workloads(path: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        rows.append(json.loads(text))
    return rows


def _tokens(text: str) -> set[str]:
    return {part for part in re.findall(r"[a-z0-9]+", text.lower()) if part}


def _token_overlap(output: str, reference: str) -> float | None:
    expected = _tokens(reference)
    got = _tokens(output)
    if not expected:
        return None
    return len(expected & got) / len(expected)


def _parse_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _field_f1(got: dict[str, Any], expected: dict[str, Any]) -> float:
    keys = set(expected)
    if not keys:
        return 1.0
    tp = 0
    for key in keys:
        if key in got and str(got[key]).strip() == str(expected[key]).strip():
            tp += 1
    precision = tp / max(1, len(got))
    recall = tp / len(keys)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_output(case: dict[str, Any], output: str) -> dict[str, float | None | str]:
    category = str(case.get("category") or "")
    expected = case.get("expected")
    metrics: dict[str, float | None | str] = {}
    if category in {"general_conversation", "reasoning"}:
        metrics["score"] = NOT_OBJECTIVELY_SCORED
        return metrics
    if category == "math":
        metrics["exact_answer_accuracy"] = (
            1.0
            if expected is not None and str(expected).strip() in output.replace(",", "")
            else 0.0
        )
        return metrics
    if category == "coding":
        want = str(expected or "").strip()
        metrics["exact_expected_output"] = 1.0 if want and want == output.strip() else 0.0
        metrics["tests_passed"] = metrics["exact_expected_output"]
        return metrics
    if category == "summarization":
        reference = str(expected or "")
        metrics["token_overlap"] = _token_overlap(output, reference)
        return metrics
    if category == "structured_extraction":
        schema = case.get("expected_object") or {}
        parsed = _parse_json(output)
        metrics["schema_validity"] = 1.0 if parsed is not None else 0.0
        if parsed is None or not isinstance(schema, dict):
            metrics["exact_match"] = 0.0
            metrics["field_level_f1"] = 0.0
        else:
            metrics["exact_match"] = 1.0 if parsed == schema else 0.0
            metrics["field_level_f1"] = _field_f1(parsed, schema)
        return metrics
    if category in {"data_analysis", "long_context"}:
        needle = str(expected or "")
        metrics["contains"] = 1.0 if needle and needle.lower() in output.lower() else 0.0
        return metrics
    metrics["score"] = NOT_OBJECTIVELY_SCORED
    return metrics


async def evaluate_spec(
    spec: ModelSpec,
    cases: list[dict[str, Any]],
    *,
    settings: Settings,
) -> dict[str, Any]:
    factory = ProviderFactory(settings)
    try:
        provider = factory.get(spec.provider)
    except Exception as exc:  # noqa: BLE001
        return {
            "deployment": spec.id,
            "provider": spec.provider,
            "status": "unavailable",
            "detail": str(exc),
            "categories": {},
        }
    by_category: dict[str, list[dict[str, Any]]] = {name: [] for name in CATEGORIES}
    errors = 0
    latencies: list[float] = []
    try:
        for case in cases:
            prompt = str(case.get("input") or "")
            started = time.perf_counter()
            try:
                result = await provider.generate(
                    InferenceRequest(
                        model=spec.provider_model,
                        messages=[ChatMessage(role="user", content=prompt)],
                        max_tokens=int(case.get("max_tokens") or 64),
                        temperature=0.0,
                    )
                )
                output = result.text
            except Exception as exc:  # noqa: BLE001
                errors += 1
                output = ""
                by_category.setdefault(str(case.get("category") or "unknown"), []).append(
                    {"id": case.get("id"), "error": str(exc), "metrics": {}}
                )
                continue
            elapsed = (time.perf_counter() - started) * 1000
            latencies.append(elapsed)
            metrics = score_output(case, output)
            category = str(case.get("category") or "unknown")
            by_category.setdefault(category, []).append(
                {
                    "id": case.get("id"),
                    "output": output,
                    "metrics": metrics,
                    "latency_ms": round(elapsed, 3),
                }
            )
    finally:
        await factory.aclose()

    category_scores: dict[str, Any] = {}
    for name, rows in by_category.items():
        numeric: dict[str, list[float]] = {}
        subjective = False
        for row in rows:
            for key, value in (row.get("metrics") or {}).items():
                if value == NOT_OBJECTIVELY_SCORED:
                    subjective = True
                    continue
                if isinstance(value, int | float):
                    numeric.setdefault(key, []).append(float(value))
        aggregates = {
            key: (sum(values) / len(values) if values else None) for key, values in numeric.items()
        }
        category_scores[name] = {
            "n": len(rows),
            "metrics": aggregates,
            "scoring": NOT_OBJECTIVELY_SCORED if subjective else "deterministic",
        }
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else None
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else None
    return {
        "deployment": spec.id,
        "provider": spec.provider,
        "identity": {
            "provider_model": spec.provider_model,
            "huggingface_id": spec.huggingface_id,
            "revision": spec.revision,
            "digest": spec.digest,
            "pool": spec.pool,
        },
        "status": "ok",
        "lifecycle": spec.lifecycle.value,
        "tiers": [tier.value for tier in spec.tiers],
        "error_rate": (errors / len(cases)) if cases else None,
        "latency": {"p50_ms": p50, "p95_ms": p95, "n": len(latencies) or None},
        "categories": category_scores,
    }


def leaderboard_row(result: dict[str, Any], spec: ModelSpec | None) -> dict[str, Any]:
    categories = result.get("categories") or {}

    def category_score(name: str) -> float | None:
        block = categories.get(name) or {}
        if block.get("scoring") == NOT_OBJECTIVELY_SCORED:
            return None
        metrics = block.get("metrics") or {}
        for key in (
            "exact_answer_accuracy",
            "exact_expected_output",
            "token_overlap",
            "field_level_f1",
            "contains",
        ):
            value = metrics.get(key)
            if isinstance(value, int | float):
                return float(value)
        return None

    latency = result.get("latency") or {}
    return {
        "deployment": result.get("deployment"),
        "provider": result.get("provider"),
        "tiers": result.get("tiers") or ([t.value for t in spec.tiers] if spec else []),
        "general_score": category_score("general_conversation"),
        "reasoning_score": category_score("reasoning"),
        "coding_score": category_score("coding"),
        "math_score": category_score("math"),
        "summarization_score": category_score("summarization"),
        "structured_extraction_score": category_score("structured_extraction"),
        "ttft_ms": spec.performance.p50_ttft_ms if spec else None,
        "tokens_per_sec": spec.performance.decode_tokens_per_second if spec else None,
        "p50_ms": latency.get("p50_ms"),
        "p95_ms": latency.get("p95_ms"),
        "error_rate": result.get("error_rate"),
        "context_window": spec.context_window if spec else None,
        "cost_estimate": spec.blended_cost_per_mtok if spec and spec.is_priced else None,
        "api_cost_knowledge": spec.api_cost_knowledge.value if spec else None,
    }


async def evaluate_registry(
    registry: ModelRegistry,
    *,
    dataset: Path = DEFAULT_DATASET,
    settings: Settings | None = None,
    only: list[str] | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    cases = load_workloads(dataset)
    results = []
    for spec in registry.enabled_models():
        if only and spec.id not in only:
            continue
        results.append(await evaluate_spec(spec, cases, settings=resolved))
    board = [
        leaderboard_row(row, registry.get(row["deployment"]))
        for row in results
        if row.get("deployment")
    ]
    return {
        "eval_version": EVAL_VERSION,
        "dataset": str(dataset),
        "dataset_note": "Independent of datasets/eval/intentos frozen 98.",
        "timestamp": time.time(),
        "fabric_version": fabric_version,
        "commit": current_commit(),
        "results": results,
        "leaderboard": board,
    }
