"""Load measurement for the gateway.

Separate from `llm_fabric.intent.benchmark`, which measures classification
quality. This package measures throughput and latency, and nothing here asserts
that any number it produces is good.
"""

from __future__ import annotations

from llm_fabric.bench.load import (
    WORKLOADS,
    LoadResult,
    LoadSettings,
    Workload,
    run_load,
)

__all__ = ["LoadResult", "LoadSettings", "WORKLOADS", "Workload", "run_load"]
