"""Observed: unknown is not zero; broken carries no value."""

from __future__ import annotations

import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.observability.metric import (
    CountProvenance,
    MetricHealth,
    MetricScope,
    Observed,
    provenance_missing,
)


def test_unavailable_cannot_carry_a_value() -> None:
    with pytest.raises(ConfigurationError, match="UNAVAILABLE"):
        Observed(
            name="kv",
            value=0,
            provenance=CountProvenance.UNAVAILABLE,
            scope=MetricScope.DEPLOYMENT,
        )


def test_broken_cannot_carry_a_value() -> None:
    with pytest.raises(ConfigurationError, match="METRIC_PIPELINE_BROKEN"):
        Observed(
            name="kv",
            value=0,
            provenance=CountProvenance.PROVIDER_MEASURED,
            scope=MetricScope.DEPLOYMENT,
            health=MetricHealth.METRIC_PIPELINE_BROKEN,
        )
    silent = Observed.broken("kv", scope=MetricScope.DEPLOYMENT, note="silent")
    assert silent.value is None
    assert silent.health is MetricHealth.METRIC_PIPELINE_BROKEN


def test_zero_is_a_measurement() -> None:
    item = Observed.zero(
        "retrieval_tokens",
        provenance=CountProvenance.ESTIMATED,
        scope=MetricScope.REQUEST,
    )
    assert item.value == 0
    assert item.health is MetricHealth.OK
    assert provenance_missing([item]) == 0
