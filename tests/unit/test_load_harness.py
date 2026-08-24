"""The load harness's own correctness.

A benchmark tool is worth exactly as much as its arithmetic. These tests pin the
parts that would silently produce a wrong number rather than an error: request
encoding, percentile selection, throughput division, and the arrival schedule
that a previous version of this module got wrong badly enough to inflate
measured latency roughly twentyfold.
"""

from __future__ import annotations

import json

import pytest

from llm_fabric.bench.load import (
    WORKLOADS,
    LoadResult,
    LoadSettings,
    Workload,
    _percentile,
    summarise,
)


def _settings(**overrides: object) -> LoadSettings:
    base: dict[str, object] = {"workload": WORKLOADS["chat-short"]}
    base.update(overrides)
    return LoadSettings(**base)  # type: ignore[arg-type]


def _result(**overrides: object) -> LoadResult:
    base: dict[str, object] = {
        "settings": _settings(),
        "requests": 1000,
        "duration_s": 10.0,
        "achieved_rps": 100.0,
        "p50_ms": 1.0,
        "p95_ms": 2.0,
        "p99_ms": 3.0,
        "max_ms": 9.0,
        "statuses": {200: 1000},
        "errors": {},
        "schedule_delay_p99_ms": None,
        "environment": {},
    }
    base.update(overrides)
    return LoadResult(**base)  # type: ignore[arg-type]


# -- request encoding --------------------------------------------------------


def test_a_get_carries_no_body_or_content_length() -> None:
    raw = WORKLOADS["liveness"].encode("localhost:8000", token=None)
    head, _, body = raw.partition(b"\r\n\r\n")
    assert body == b""
    assert b"content-length" not in head.lower()
    assert raw.startswith(b"GET /healthz HTTP/1.1")


def test_a_post_declares_the_exact_body_length() -> None:
    raw = WORKLOADS["chat-short"].encode("localhost:8000", token=None)
    head, _, body = raw.partition(b"\r\n\r\n")
    declared = next(
        int(line.split(b":")[1])
        for line in head.split(b"\r\n")
        if line.lower().startswith(b"content-length")
    )
    # A mismatch here would hang the server waiting for bytes that never come.
    assert declared == len(body)
    assert json.loads(body)["model"] == "auto"


def test_the_token_is_sent_only_when_the_workload_needs_auth() -> None:
    authed = WORKLOADS["chat-short"].encode("h", token="secret-token")
    public = WORKLOADS["liveness"].encode("h", token="secret-token")
    assert b"authorization: Bearer secret-token" in authed
    assert b"authorization" not in public.lower()


def test_connections_are_reused() -> None:
    # Opening a socket per request would measure the kernel, not the gateway.
    raw = WORKLOADS["chat-short"].encode("h", token=None)
    assert b"connection: keep-alive" in raw
    assert b"connection: close" not in raw


def test_every_workload_names_what_it_exercises() -> None:
    # The constitution forbids publishing an RPS number without its workload.
    for name, workload in WORKLOADS.items():
        assert workload.name == name
        assert len(workload.description) > 40


# -- percentiles -------------------------------------------------------------


def test_percentiles_index_into_the_ordered_sample() -> None:
    ordered = [float(n) for n in range(1, 101)]
    assert _percentile(ordered, 0.50) == pytest.approx(50.5, abs=1.0)
    assert _percentile(ordered, 0.99) == pytest.approx(100.0, abs=1.0)
    assert _percentile(ordered, 0.0) == 1.0


def test_an_empty_sample_reports_zero_rather_than_raising() -> None:
    assert _percentile([], 0.99) == 0.0


def test_a_single_sample_is_every_percentile() -> None:
    assert _percentile([7.0], 0.50) == 7.0
    assert _percentile([7.0], 0.99) == 7.0


# -- derived figures ---------------------------------------------------------


def test_error_rate_counts_failures_that_never_produced_a_status() -> None:
    # A connection reset returns no status. Counting only statuses would report
    # a perfect run for a server that refused half the load.
    result = _result(requests=100, statuses={200: 100}, errors={"ConnectionResetError": 100})
    assert result.successes == 100
    assert result.error_rate == pytest.approx(0.5)


def test_non_2xx_responses_are_not_successes() -> None:
    result = _result(requests=100, statuses={200: 90, 500: 10})
    assert result.successes == 90
    assert result.error_rate == pytest.approx(0.1)


def test_a_429_is_a_failure_of_the_run_even_though_it_is_correct_behaviour() -> None:
    # Quota rejection is the gateway working, but a load result that counted it
    # as served would report throughput the fabric never delivered.
    result = _result(requests=100, statuses={200: 50, 429: 50})
    assert result.error_rate == pytest.approx(0.5)


def test_an_empty_run_reports_no_errors_rather_than_dividing_by_zero() -> None:
    assert _result(requests=0, statuses={}, errors={}).error_rate == 0.0


# -- open loop validity ------------------------------------------------------


def test_a_closed_loop_run_makes_no_claim_about_offered_load() -> None:
    assert _result(settings=_settings(rate=None)).offered_load_was_met is None


def test_an_open_loop_run_that_kept_its_schedule_is_valid() -> None:
    result = _result(settings=_settings(rate=1000.0), schedule_delay_p99_ms=2.0)
    assert result.offered_load_was_met is True


def test_an_open_loop_run_that_fell_behind_is_marked_invalid() -> None:
    # The generator, not the server, failed to keep up. Reporting the achieved
    # rate as if the server had been offered the target would be a lie.
    result = _result(settings=_settings(rate=1000.0), schedule_delay_p99_ms=800.0)
    assert result.offered_load_was_met is False
    assert "never offered" in summarise(result)


# -- reporting ---------------------------------------------------------------


def test_the_summary_names_the_workload_before_the_numbers() -> None:
    text = summarise(_result())
    assert text.index("Workload") < text.index("Achieved")
    assert "chat-short" in text


def test_the_summary_refuses_to_imply_a_comparison() -> None:
    assert "not a comparison against any other system" in summarise(_result())


def test_the_serialised_result_carries_the_workload_that_produced_it() -> None:
    payload = _result().as_dict()
    assert payload["workload"]["name"] == "chat-short"
    assert payload["workload"]["path"] == "/v1/chat/completions"
    assert payload["load"]["mode"] == "closed-loop"
    assert payload["environment"] == {}


def test_an_open_loop_result_records_its_target() -> None:
    payload = _result(settings=_settings(rate=1000.0)).as_dict()
    assert payload["load"]["mode"] == "open-loop"
    assert payload["load"]["target_rps"] == 1000.0


# -- arrival schedule --------------------------------------------------------


def test_arrivals_are_spread_within_the_interval_not_by_whole_intervals() -> None:
    """The bug that made a smooth rate arrive as periodic bursts.

    Each of `connections` connections fires every `interval` seconds. Staggering
    by whole intervals leaves every connection firing at the same instants, so
    the server receives `connections` requests at once and then nothing, and the
    queueing that causes is charged to the server. The stagger must divide one
    interval.
    """
    connections, rate = 16, 250.0
    interval = connections / rate
    offsets = [offset * (interval / connections) for offset in range(connections)]

    assert offsets[0] == 0.0
    assert max(offsets) < interval
    gaps = [b - a for a, b in zip(offsets, offsets[1:], strict=False)]
    assert all(gap == pytest.approx(1.0 / rate) for gap in gaps)


def test_a_workload_is_immutable_so_a_run_cannot_alter_it() -> None:
    with pytest.raises(AttributeError):
        WORKLOADS["chat-short"].name = "something-else"  # type: ignore[misc]


def test_settings_are_immutable() -> None:
    with pytest.raises(AttributeError):
        _settings().connections = 9999  # type: ignore[misc]


def test_a_custom_workload_round_trips_through_encoding() -> None:
    workload = Workload(
        name="custom",
        description="A workload defined by a caller rather than shipped with the harness.",
        method="POST",
        path="/v1/thing",
        body={"a": 1},
    )
    raw = workload.encode("h:1", token=None)
    _, _, body = raw.partition(b"\r\n\r\n")
    assert json.loads(body) == {"a": 1}
