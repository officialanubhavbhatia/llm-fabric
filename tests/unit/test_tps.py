"""Named TPS formulas refuse untrustworthy inputs."""

from __future__ import annotations

from llm_fabric.observability.metric import CountProvenance, MetricHealth, MetricScope, Observed
from llm_fabric.observability.tps import (
    decode_tokens_per_second,
    prefill_tokens_per_second,
    request_effective_tokens_per_second,
    tpot_seconds,
)


def _tokens(
    name: str,
    value: float,
    provenance: CountProvenance = CountProvenance.PROVIDER_MEASURED,
) -> Observed:
    return Observed(
        name=name,
        value=value,
        provenance=provenance,
        scope=MetricScope.REQUEST,
        unit="tokens",
    )


def _seconds(name: str, value: float) -> Observed:
    return Observed(
        name=name,
        value=value,
        provenance=CountProvenance.PROVIDER_MEASURED,
        scope=MetricScope.REQUEST,
        unit="seconds",
    )


def test_decode_tps_uses_decode_duration() -> None:
    rate = decode_tokens_per_second(
        _tokens("completion_tokens", 20),
        _seconds("decode_duration_s", 2),
    )
    assert rate.value == 10.0
    assert rate.provenance is CountProvenance.DERIVED


def test_zero_duration_is_unavailable() -> None:
    rate = prefill_tokens_per_second(
        _tokens("prompt_tokens", 10),
        _seconds("prefill_duration_s", 0),
    )
    assert rate.health is MetricHealth.UNAVAILABLE
    assert rate.value is None


def test_e2e_is_not_used_as_decode() -> None:
    rate = request_effective_tokens_per_second(
        _tokens("total_tokens", 30), _seconds("e2e_duration_s", 3)
    )
    assert rate.name == "request_effective_tokens_per_second"
    assert rate.value == 10.0


def test_tpot_excludes_first_token() -> None:
    observed = tpot_seconds(decode_duration_s=0.9, completion_tokens=10)
    assert observed.value == 0.1
