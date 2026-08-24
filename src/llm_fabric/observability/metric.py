"""Typed observations: a number is never enough.

Every metric the fabric exposes carries provenance and scope. Zero means the
source counted zero. Unknown is `UNAVAILABLE` with no value. A configured
source that goes silent is `METRIC_PIPELINE_BROKEN`, also with no value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from llm_fabric.errors import ConfigurationError


class CountProvenance(StrEnum):
    """Where a count or duration came from."""

    PROVIDER_MEASURED = "PROVIDER_MEASURED"
    TOKENIZER_MEASURED = "TOKENIZER_MEASURED"
    DERIVED = "DERIVED"
    ESTIMATED = "ESTIMATED"
    UNAVAILABLE = "UNAVAILABLE"


class MetricScope(StrEnum):
    """The population the number describes.

    Request-level cached tokens must never be labelled as pod KV utilisation.
    """

    REQUEST = "REQUEST"
    MODEL = "MODEL"
    POD = "POD"
    DEPLOYMENT = "DEPLOYMENT"
    FLEET = "FLEET"


class MetricHealth(StrEnum):
    OK = "OK"
    UNAVAILABLE = "UNAVAILABLE"
    METRIC_PIPELINE_BROKEN = "METRIC_PIPELINE_BROKEN"


@dataclass(frozen=True, slots=True)
class Observed:
    """One named observation, or an honest absence."""

    name: str
    provenance: CountProvenance
    scope: MetricScope
    value: float | int | None = None
    source_metric_name: str | None = None
    runtime_version: str | None = None
    health: MetricHealth = MetricHealth.OK
    unit: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ConfigurationError("an observation requires a name")
        if self.provenance is CountProvenance.UNAVAILABLE:
            if self.value is not None:
                raise ConfigurationError(
                    f"{self.name}: UNAVAILABLE must not carry a value; unknown is not zero"
                )
            if self.health is MetricHealth.OK:
                object.__setattr__(self, "health", MetricHealth.UNAVAILABLE)
        if self.health is MetricHealth.METRIC_PIPELINE_BROKEN and self.value is not None:
            raise ConfigurationError(
                f"{self.name}: METRIC_PIPELINE_BROKEN must not display a numeric value"
            )
        if self.health is MetricHealth.UNAVAILABLE and self.value is not None:
            raise ConfigurationError(
                f"{self.name}: UNAVAILABLE health must not display a numeric value"
            )

    @classmethod
    def unavailable(
        cls,
        name: str,
        *,
        scope: MetricScope,
        note: str,
        source_metric_name: str | None = None,
        runtime_version: str | None = None,
        unit: str | None = None,
    ) -> Observed:
        return cls(
            name=name,
            value=None,
            provenance=CountProvenance.UNAVAILABLE,
            scope=scope,
            source_metric_name=source_metric_name,
            runtime_version=runtime_version,
            health=MetricHealth.UNAVAILABLE,
            unit=unit,
            note=note,
        )

    @classmethod
    def broken(
        cls,
        name: str,
        *,
        scope: MetricScope,
        note: str,
        source_metric_name: str | None = None,
        runtime_version: str | None = None,
    ) -> Observed:
        return cls(
            name=name,
            value=None,
            provenance=CountProvenance.UNAVAILABLE,
            scope=scope,
            source_metric_name=source_metric_name,
            runtime_version=runtime_version,
            health=MetricHealth.METRIC_PIPELINE_BROKEN,
            note=note,
        )

    @classmethod
    def zero(
        cls,
        name: str,
        *,
        provenance: CountProvenance,
        scope: MetricScope,
        note: str | None = None,
        unit: str | None = None,
    ) -> Observed:
        if provenance is CountProvenance.UNAVAILABLE:
            raise ConfigurationError("zero is a measurement; use unavailable() for unknown")
        return cls(
            name=name,
            value=0,
            provenance=provenance,
            scope=scope,
            health=MetricHealth.OK,
            unit=unit,
            note=note,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "provenance": self.provenance.value,
            "scope": self.scope.value,
            "source_metric_name": self.source_metric_name,
            "runtime_version": self.runtime_version,
            "health": self.health.value,
            "unit": self.unit,
            "note": self.note,
        }


def provenance_missing(observations: list[Observed]) -> int:
    """Count declared observations that somehow lack provenance. PASS is 0."""
    return sum(1 for item in observations if not item.provenance)
