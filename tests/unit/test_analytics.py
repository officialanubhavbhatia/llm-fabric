"""Analytics sink is fail-soft and bounded."""

from __future__ import annotations

from llm_fabric.observability.analytics import AnalyticsEvent, BufferedAnalyticsSink


def test_the_buffer_drops_oldest_events_rather_than_growing() -> None:
    sink = BufferedAnalyticsSink(max_events=3)
    for index in range(5):
        sink.emit(AnalyticsEvent(kind="token", tenant_id="acme", payload={"n": index}))

    assert sink.buffered == 3
    assert sink.dropped == 2
    drained = sink.drain(10)
    assert [event.payload["n"] for event in drained] == [2, 3, 4]
