"""Runtime observability capability matrix.

A cell is supported only when this process can actually obtain the series.
Unsupported is UNAVAILABLE with a reason, never a fabricated zero.
"""

from __future__ import annotations

from typing import Any

from llm_fabric.observability.metric import MetricScope, Observed
from llm_fabric.observability.ollama_metrics import OLLAMA_DOES_NOT_EXPOSE, OLLAMA_UNSUPPORTED

CATEGORIES: tuple[str, ...] = (
    "prompt_tokens",
    "completion_tokens",
    "ttft",
    "tpot",
    "decode_tps",
    "prefill_tps",
    "kv_utilization",
    "prefix_hits",
    "queue_depth",
    "running_requests",
    "waiting_requests",
    "batch_metrics",
)

_OLLAMA: dict[str, str] = {
    "prompt_tokens": "native prompt_eval_count or OpenAI usage.prompt_tokens",
    "completion_tokens": "native eval_count or OpenAI usage.completion_tokens",
    "ttft": "gateway streaming first-byte; Ollama has no TTFT series",
    "tpot": "gateway streaming after first byte; Ollama has no TPOT series",
    "decode_tps": "DERIVED from eval_count/eval_duration when native payload is present",
    "prefill_tps": (
        "DERIVED from prompt_eval_count/prompt_eval_duration when native payload is present"
    ),
    "kv_utilization": OLLAMA_DOES_NOT_EXPOSE,
    "prefix_hits": OLLAMA_DOES_NOT_EXPOSE,
    "queue_depth": OLLAMA_DOES_NOT_EXPOSE,
    "running_requests": OLLAMA_DOES_NOT_EXPOSE,
    "waiting_requests": OLLAMA_DOES_NOT_EXPOSE,
    "batch_metrics": OLLAMA_DOES_NOT_EXPOSE,
}

_VLLM: dict[str, str] = {
    "prompt_tokens": "vllm:prompt_tokens_total (DEPLOYMENT) plus request usage",
    "completion_tokens": "vllm:generation_tokens_total (DEPLOYMENT) plus request usage",
    "ttft": (
        "vllm:time_to_first_token_seconds histogram mean (DEPLOYMENT); "
        "request TTFT from gateway stream"
    ),
    "tpot": (
        "vllm:inter_token_latency_seconds histogram mean (DEPLOYMENT); "
        "request TPOT from gateway stream"
    ),
    "decode_tps": (
        "request decode_tps only with decode duration; engine histogram is not request TPS"
    ),
    "prefill_tps": (
        "request prefill_tps only with prefill duration; engine histogram is not request TPS"
    ),
    "kv_utilization": "vllm:kv_cache_usage_perc or legacy vllm:gpu_cache_usage_perc (DEPLOYMENT)",
    "prefix_hits": "vllm:prefix_cache_hits / vllm:prefix_cache_queries (DEPLOYMENT)",
    "queue_depth": "not a single series; waiting_requests is the waiting gauge",
    "running_requests": "vllm:num_requests_running (DEPLOYMENT)",
    "waiting_requests": "vllm:num_requests_waiting (DEPLOYMENT)",
    "batch_metrics": (
        "UNAVAILABLE — vLLM /metrics does not expose batch utilization as a stable series here"
    ),
}

_LITELLM: dict[str, str] = {
    "prompt_tokens": "upstream usage when the backend reports it; otherwise ESTIMATED",
    "completion_tokens": "upstream usage when the backend reports it; otherwise ESTIMATED",
    "ttft": "gateway streaming first-byte (REQUEST); not a LiteLLM engine metric",
    "tpot": "gateway streaming (REQUEST); not a LiteLLM engine metric",
    "decode_tps": "only if the upstream runtime exposes decode duration; LiteLLM does not",
    "prefill_tps": "only if the upstream runtime exposes prefill duration; LiteLLM does not",
    "kv_utilization": "UNAVAILABLE — LiteLLM is transport; do not relabel vLLM KV as LiteLLM",
    "prefix_hits": "UNAVAILABLE — LiteLLM is transport; vLLM prefix cache stays vLLM-scoped",
    "queue_depth": (
        "UNAVAILABLE as an engine queue; LiteLLM retry/rate-limit are transport events"
    ),
    "running_requests": "UNAVAILABLE — not a LiteLLM engine gauge",
    "waiting_requests": "UNAVAILABLE — not a LiteLLM engine gauge",
    "batch_metrics": "UNAVAILABLE — LiteLLM is not the batching engine",
}


def capability_matrix() -> dict[str, Any]:
    def row(runtime: str, catalog: dict[str, str]) -> dict[str, Any]:
        cells = {}
        for category in CATEGORIES:
            note = catalog[category]
            supported = not note.startswith("UNAVAILABLE")
            cells[category] = {
                "supported": supported,
                "note": note,
                "scope": (
                    MetricScope.DEPLOYMENT.value
                    if category
                    in {
                        "kv_utilization",
                        "prefix_hits",
                        "running_requests",
                        "waiting_requests",
                        "batch_metrics",
                        "queue_depth",
                    }
                    else MetricScope.REQUEST.value
                ),
            }
        return {"runtime": runtime, "categories": cells}

    return {
        "categories": list(CATEGORIES),
        "runtimes": [
            row("ollama", _OLLAMA),
            row("vllm", _VLLM),
            row("litellm_transport", _LITELLM),
        ],
        "ollama_unsupported": list(OLLAMA_UNSUPPORTED),
        "note": (
            "LiteLLM cells are transport-only. vLLM engine series stay scoped "
            "DEPLOYMENT even when traffic transits LiteLLM."
        ),
    }


def unsupported_as_observations(runtime: str) -> list[Observed]:
    if runtime != "ollama":
        return []
    return [
        Observed.unavailable(name, scope=MetricScope.DEPLOYMENT, note=OLLAMA_DOES_NOT_EXPOSE)
        for name in OLLAMA_UNSUPPORTED
    ]
