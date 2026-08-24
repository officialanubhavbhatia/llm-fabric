"""Trace context propagation and tenant stamping."""

from __future__ import annotations

from llm_fabric.identity.claims import Principal
from llm_fabric.observability.trace import TraceContext, bind_trace, current_trace, reset_trace

VALID_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
VALID_SPAN_ID = "00f067aa0ba902b7"


def _principal() -> Principal:
    return Principal(
        tenant_id="acme",
        user_id="alice",
        subject="sub-1",
        issuer="https://issuer.example",
        project_id="proj-1",
    )


def test_a_fresh_trace_is_generated_without_a_header() -> None:
    trace = TraceContext.start(request_id="req-1")

    assert len(trace.trace_id) == 32
    assert len(trace.span_id) == 16
    assert trace.parent_span_id is None


def test_a_valid_traceparent_is_continued() -> None:
    trace = TraceContext.start(
        request_id="req-1",
        traceparent=f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01",
    )

    assert trace.trace_id == VALID_TRACE_ID
    assert trace.parent_span_id == VALID_SPAN_ID
    assert trace.sampled is True
    assert trace.span_id != VALID_SPAN_ID, "a new span must be started"


def test_the_sampled_flag_is_honoured() -> None:
    trace = TraceContext.start(
        request_id="req-1", traceparent=f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-00"
    )

    assert trace.sampled is False
    assert trace.to_traceparent().endswith("-00")


def test_a_malformed_traceparent_starts_a_new_trace() -> None:
    """Untrusted input. A bad header is dropped, not propagated."""
    for bad in ("", "garbage", "00-short-00f067aa0ba902b7-01", "00-" + "z" * 32 + "-x-01"):
        trace = TraceContext.start(request_id="req-1", traceparent=bad)
        assert len(trace.trace_id) == 32
        assert trace.parent_span_id is None


def test_an_all_zero_trace_id_is_rejected() -> None:
    """W3C reserves all-zero ids as invalid."""
    trace = TraceContext.start(request_id="req-1", traceparent=f"00-{'0' * 32}-{VALID_SPAN_ID}-01")

    assert trace.trace_id != "0" * 32


def test_the_principal_stamps_tenant_metadata() -> None:
    trace = TraceContext.start(request_id="req-1", principal=_principal())

    fields = trace.log_fields()
    assert fields["tenant_id"] == "acme"
    assert fields["user_id"] == "alice"
    assert fields["project_id"] == "proj-1"


def test_log_fields_omit_absent_identity() -> None:
    fields = TraceContext.start(request_id="req-1").log_fields()

    assert "tenant_id" not in fields
    assert set(fields) == {"trace_id", "span_id", "request_id"}


def test_the_wire_format_carries_no_tenant() -> None:
    trace = TraceContext.start(request_id="req-1", principal=_principal())

    assert "acme" not in trace.to_traceparent()


def test_the_context_variable_is_set_and_reset() -> None:
    assert current_trace() is None

    trace = TraceContext.start(request_id="req-1")
    token = bind_trace(trace)
    assert current_trace() is trace

    reset_trace(token)
    assert current_trace() is None
