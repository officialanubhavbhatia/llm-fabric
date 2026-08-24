"""Health scoring, drift analysis and controlled remediation."""

from __future__ import annotations

from typing import Any

from llm_fabric.heal.schema import (
    ComponentHealth,
    DriftReport,
    Incident,
    LearningJob,
    RemediationKind,
    RemediationProposal,
)

__all__ = [
    "ComponentHealth",
    "DriftReport",
    "HealController",
    "Incident",
    "LearningJob",
    "RemediationKind",
    "RemediationProposal",
]


def __getattr__(name: str) -> Any:
    # Imported lazily so `heal.store` can load without `router.engine`
    # (which imports `heal.controls`) already being on the stack.
    if name == "HealController":
        from llm_fabric.heal.engine import HealController

        return HealController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
