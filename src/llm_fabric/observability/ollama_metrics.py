"""Ollama telemetry: only what Ollama actually returns.

Native `/api/chat` and `/api/generate` may include eval/prompt_eval counts and
nanosecond durations. The OpenAI-compatible `/v1/chat/completions` path usually
returns token usage only.

KV occupancy, prefix-cache hits, batch utilisation and runtime queue depth are
not exposed. Those observations are UNAVAILABLE with an explicit note. They are
never synthesized to look like vLLM.
"""

from __future__ import annotations

from typing import Any

from llm_fabric.observability.metric import CountProvenance, MetricScope, Observed
from llm_fabric.observability.tps import decode_tokens_per_second, prefill_tokens_per_second

OLLAMA_DOES_NOT_EXPOSE = "UNAVAILABLE — OLLAMA DOES NOT EXPOSE THIS METRIC"

OLLAMA_UNSUPPORTED: tuple[str, ...] = (
    "kv_cache_utilization",
    "prefix_cache_hit_tokens",
    "prefix_cache_query_tokens",
    "prefix_cache_hit_ratio",
    "batch_utilization",
    "runtime_queue_depth",
    "running_requests",
    "waiting_requests",
    "preemptions",
    "kv_cache_pressure",
)


def parse_ollama_native(payload: dict[str, Any] | None) -> list[Observed]:
    """Parse a native Ollama generate/chat JSON object."""
    body = payload or {}
    observations: list[Observed] = []
    scope = MetricScope.REQUEST

    def measured(name: str, key: str, *, unit: str | None = None, scale: float = 1.0) -> Observed:
        if key not in body or body[key] is None:
            return Observed.unavailable(
                name, scope=scope, note=f"Ollama payload did not include {key}"
            )
        try:
            value = float(body[key]) * scale
        except (TypeError, ValueError):
            return Observed.unavailable(name, scope=scope, note=f"{key} was not numeric")
        return Observed(
            name=name,
            value=value,
            provenance=CountProvenance.PROVIDER_MEASURED,
            scope=scope,
            source_metric_name=key,
            unit=unit,
        )

    prompt_tokens = measured("prompt_tokens", "prompt_eval_count", unit="tokens")
    completion_tokens = measured("completion_tokens", "eval_count", unit="tokens")
    prompt_eval_s = measured(
        "prompt_evaluation_duration_s", "prompt_eval_duration", unit="seconds", scale=1e-9
    )
    eval_s = measured("generation_eval_duration_s", "eval_duration", unit="seconds", scale=1e-9)
    load_s = measured("load_duration_s", "load_duration", unit="seconds", scale=1e-9)
    total_s = measured("total_duration_s", "total_duration", unit="seconds", scale=1e-9)
    observations.extend([prompt_tokens, completion_tokens, prompt_eval_s, eval_s, load_s, total_s])
    observations.append(prefill_tokens_per_second(prompt_tokens, prompt_eval_s))
    observations.append(decode_tokens_per_second(completion_tokens, eval_s))
    for name in OLLAMA_UNSUPPORTED:
        observations.append(
            Observed.unavailable(name, scope=MetricScope.DEPLOYMENT, note=OLLAMA_DOES_NOT_EXPOSE)
        )
    return observations


def ollama_openai_usage(usage: dict[str, Any] | None) -> list[Observed]:
    """Token counts from an OpenAI-compatible Ollama response."""
    if not usage:
        return [
            Observed.unavailable(
                "prompt_tokens",
                scope=MetricScope.REQUEST,
                note="Ollama OpenAI response did not include usage",
            ),
            Observed.unavailable(
                "completion_tokens",
                scope=MetricScope.REQUEST,
                note="Ollama OpenAI response did not include usage",
            ),
        ]
    items: list[Observed] = []
    for name, key in (
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
    ):
        if key not in usage or usage[key] is None:
            items.append(
                Observed.unavailable(name, scope=MetricScope.REQUEST, note=f"usage.{key} missing")
            )
            continue
        items.append(
            Observed(
                name=name,
                value=int(usage[key]),
                provenance=CountProvenance.PROVIDER_MEASURED,
                scope=MetricScope.REQUEST,
                source_metric_name=f"usage.{key}",
                unit="tokens",
            )
        )
    for name in (
        "prompt_evaluation_duration_s",
        "generation_eval_duration_s",
        "load_duration_s",
        "total_duration_s",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
    ):
        items.append(
            Observed.unavailable(
                name,
                scope=MetricScope.REQUEST,
                note="Ollama OpenAI-compatible chat does not return native eval durations",
            )
        )
    for name in OLLAMA_UNSUPPORTED:
        items.append(
            Observed.unavailable(name, scope=MetricScope.DEPLOYMENT, note=OLLAMA_DOES_NOT_EXPOSE)
        )
    return items
