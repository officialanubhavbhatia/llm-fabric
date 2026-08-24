"""Ollama telemetry: native fields only; unsupported stay UNAVAILABLE."""

from __future__ import annotations

from llm_fabric.observability.metric import CountProvenance, MetricHealth
from llm_fabric.observability.ollama_metrics import (
    OLLAMA_DOES_NOT_EXPOSE,
    ollama_openai_usage,
    parse_ollama_native,
)


def _by_name(items):
    return {item.name: item for item in items}


def test_native_payload_derives_tps() -> None:
    items = _by_name(
        parse_ollama_native(
            {
                "prompt_eval_count": 10,
                "eval_count": 20,
                "prompt_eval_duration": 1_000_000_000,
                "eval_duration": 2_000_000_000,
                "load_duration": 100_000_000,
                "total_duration": 3_200_000_000,
            }
        )
    )
    assert items["prompt_tokens"].value == 10
    assert items["prompt_tokens"].provenance is CountProvenance.PROVIDER_MEASURED
    assert items["prefill_tokens_per_second"].value == 10.0
    assert items["prefill_tokens_per_second"].provenance is CountProvenance.DERIVED
    assert items["decode_tokens_per_second"].value == 10.0
    assert items["kv_cache_utilization"].health is MetricHealth.UNAVAILABLE
    assert items["kv_cache_utilization"].value is None
    assert items["kv_cache_utilization"].note == OLLAMA_DOES_NOT_EXPOSE
    assert items["running_requests"].note == OLLAMA_DOES_NOT_EXPOSE


def test_openai_usage_has_no_eval_durations() -> None:
    items = _by_name(ollama_openai_usage({"prompt_tokens": 4, "completion_tokens": 2}))
    assert items["prompt_tokens"].value == 4
    assert items["decode_tokens_per_second"].health is MetricHealth.UNAVAILABLE
    assert items["prefix_cache_hit_ratio"].note == OLLAMA_DOES_NOT_EXPOSE
