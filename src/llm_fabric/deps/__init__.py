"""Serving-dependency health, readiness aggregation, and admission.

This package is the P0-FIX-4 runtime: each process decides whether *it* can
safely accept new inference. Cluster-wide consensus is not required.
"""

from llm_fabric.deps.health import (
    DEPENDENCY_NAMES,
    INFERENCE_PATHS,
    DependencyClass,
    DependencyHealth,
    DependencySnapshot,
    HealthStatus,
)

__all__ = [
    "DEPENDENCY_NAMES",
    "INFERENCE_PATHS",
    "DependencyClass",
    "DependencyHealth",
    "DependencySnapshot",
    "HealthStatus",
]
