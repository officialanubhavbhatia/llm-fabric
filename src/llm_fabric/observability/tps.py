"""Tokens-per-second definitions.

There is no single TPS number. Each formula is named, scoped, and refused when
either input is untrustworthy.

    prefill_tokens_per_second =
        prompt_tokens / prefill_duration_s

    decode_tokens_per_second =
        completion_tokens / decode_duration_s

    aggregate_generation_tokens_per_second =
        completion_tokens / generation_duration_s

    request_effective_tokens_per_second =
        (prompt_tokens + completion_tokens) / e2e_duration_s

Prefill and decode durations must come from the runtime (PROVIDER_MEASURED) or
be derived from native timing fields. Gateway end-to-end time is not decode
time and is not used as a substitute.
"""

from __future__ import annotations

from llm_fabric.observability.metric import CountProvenance, MetricScope, Observed

_TRUSTED = {
    CountProvenance.PROVIDER_MEASURED,
    CountProvenance.TOKENIZER_MEASURED,
    CountProvenance.DERIVED,
}


def _rate(
    name: str,
    tokens: Observed | None,
    duration_s: Observed | None,
    *,
    scope: MetricScope,
) -> Observed:
    if tokens is None or duration_s is None:
        return Observed.unavailable(
            name,
            scope=scope,
            note="missing tokens or duration",
        )
    if tokens.provenance is CountProvenance.UNAVAILABLE or tokens.value is None:
        return Observed.unavailable(name, scope=scope, note="token count is unavailable")
    if duration_s.provenance is CountProvenance.UNAVAILABLE or duration_s.value is None:
        return Observed.unavailable(name, scope=scope, note="duration is unavailable")
    if tokens.provenance not in _TRUSTED and tokens.provenance is not CountProvenance.ESTIMATED:
        return Observed.unavailable(name, scope=scope, note="token provenance is not usable")
    duration = float(duration_s.value)
    if duration <= 0:
        return Observed.unavailable(name, scope=scope, note="duration is not positive")
    token_count = float(tokens.value)
    if token_count < 0:
        return Observed.unavailable(name, scope=scope, note="token count is negative")
    return Observed(
        name=name,
        value=round(token_count / duration, 4),
        provenance=CountProvenance.DERIVED,
        scope=scope,
        unit="tokens_per_second",
        note=f"{tokens.name} / {duration_s.name}",
        source_metric_name=None,
    )


def prefill_tokens_per_second(prompt_tokens: Observed, prefill_duration_s: Observed) -> Observed:
    return _rate(
        "prefill_tokens_per_second",
        prompt_tokens,
        prefill_duration_s,
        scope=MetricScope.REQUEST,
    )


def decode_tokens_per_second(completion_tokens: Observed, decode_duration_s: Observed) -> Observed:
    return _rate(
        "decode_tokens_per_second",
        completion_tokens,
        decode_duration_s,
        scope=MetricScope.REQUEST,
    )


def aggregate_generation_tokens_per_second(
    completion_tokens: Observed, generation_duration_s: Observed
) -> Observed:
    return _rate(
        "aggregate_generation_tokens_per_second",
        completion_tokens,
        generation_duration_s,
        scope=MetricScope.REQUEST,
    )


def request_effective_tokens_per_second(
    total_tokens: Observed, e2e_duration_s: Observed
) -> Observed:
    return _rate(
        "request_effective_tokens_per_second",
        total_tokens,
        e2e_duration_s,
        scope=MetricScope.REQUEST,
    )


def tpot_seconds(
    *,
    decode_duration_s: float | None,
    completion_tokens: int | None,
    first_token_counted: bool = True,
) -> Observed:
    """Time per output token after the first token.

    TPOT = decode_duration / max(1, completion_tokens - 1) when the first token
    is excluded. Requires streaming timestamps or a runtime decode duration.
    """
    if decode_duration_s is None or completion_tokens is None:
        return Observed.unavailable(
            "tpot_seconds",
            scope=MetricScope.REQUEST,
            note="need decode duration and completion tokens",
        )
    if decode_duration_s <= 0 or completion_tokens < 0:
        return Observed.unavailable(
            "tpot_seconds",
            scope=MetricScope.REQUEST,
            note="decode duration or completion tokens are not usable",
        )
    denom = max(1, completion_tokens - 1) if first_token_counted else max(1, completion_tokens)
    return Observed(
        name="tpot_seconds",
        value=round(decode_duration_s / denom, 6),
        provenance=CountProvenance.DERIVED,
        scope=MetricScope.REQUEST,
        unit="seconds",
        note="decode_duration / output_tokens_after_first" if first_token_counted else None,
    )
