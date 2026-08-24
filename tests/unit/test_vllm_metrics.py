"""vLLM /metrics parser: documented names, UNAVAILABLE vs BROKEN, DEPLOYMENT scope."""

from __future__ import annotations

from llm_fabric.observability.metric import CountProvenance, MetricHealth, MetricScope
from llm_fabric.observability.vllm_metrics import parse_vllm_metrics

V1_FIXTURE = """
# HELP vllm:kv_cache_usage_perc KV-cache used fraction
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0"} 0.42
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries 100
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits 40
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 3
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 1
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 80
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 20
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total 2
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum 1.5
vllm:time_to_first_token_seconds_count 10
# TYPE vllm:inter_token_latency_seconds histogram
vllm:inter_token_latency_seconds_sum 0.4
vllm:inter_token_latency_seconds_count 20
# TYPE vllm:request_prefill_time_seconds histogram
vllm:request_prefill_time_seconds_sum 2.0
vllm:request_prefill_time_seconds_count 10
# TYPE vllm:request_decode_time_seconds histogram
vllm:request_decode_time_seconds_sum 3.0
vllm:request_decode_time_seconds_count 10
# TYPE vllm:request_queue_time_seconds histogram
vllm:request_queue_time_seconds_sum 0.5
vllm:request_queue_time_seconds_count 10
"""


def _by_name(items):
    return {item.name: item for item in items}


def test_v1_fixture_parses_documented_names() -> None:
    items = _by_name(parse_vllm_metrics(V1_FIXTURE))
    assert items["kv_cache_utilization"].value == 0.42
    assert items["kv_cache_utilization"].scope is MetricScope.DEPLOYMENT
    assert items["kv_cache_utilization"].source_metric_name == "vllm:kv_cache_usage_perc"
    assert items["kv_cache_utilization"].runtime_version == "vllm-v1"
    assert items["prefix_cache_query_tokens"].value == 100
    assert items["prefix_cache_hit_tokens"].value == 40
    assert items["prefix_cache_hit_ratio"].provenance is CountProvenance.DERIVED
    assert items["running_requests"].value == 3
    assert items["waiting_requests"].value == 1
    assert items["ttft_seconds"].value == 0.15
    assert "not this request" in (items["ttft_seconds"].note or "")
    assert items["kv_block_lifetime"].health is MetricHealth.UNAVAILABLE
    assert items["kv_block_lifetime"].value is None


def test_unconfigured_is_unavailable_not_zero() -> None:
    items = _by_name(parse_vllm_metrics("", configured=False))
    assert items["kv_cache_utilization"].value is None
    assert items["kv_cache_utilization"].health is MetricHealth.UNAVAILABLE


def test_failed_scrape_is_pipeline_broken() -> None:
    items = parse_vllm_metrics("", configured=True, scrape_ok=False)
    assert items[0].health is MetricHealth.METRIC_PIPELINE_BROKEN
    assert items[0].value is None


def test_missing_expected_v1_series_is_broken() -> None:
    text = "vllm:kv_cache_usage_perc 0.1\nvllm:num_requests_running 1\n"
    items = _by_name(parse_vllm_metrics(text))
    assert items["waiting_requests"].health is MetricHealth.METRIC_PIPELINE_BROKEN
    assert items["waiting_requests"].value is None
