"""Command Center contract: every named view exists, unbuilt backends stay empty."""

from __future__ import annotations

from fastapi.testclient import TestClient

from llm_fabric.observability.dashboards import VIEWS


def test_command_center_html_is_served(client: TestClient) -> None:
    response = client.get("/command-center")
    assert response.status_code == 200
    assert "MyVista Command Center" in response.text


def test_metrics_scrape_is_prometheus_text(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "fabric_requests_total" in response.text or response.text.startswith("#")


def test_every_dashboard_is_reachable(client: TestClient) -> None:
    listed = client.get("/v1/observability/dashboards").json()["views"]
    assert listed == list(VIEWS)
    for view in VIEWS:
        payload = client.get(f"/v1/observability/dashboards/{view}").json()
        assert payload["view"] == view
        assert "available" in payload
        assert "unavailable_fields" in payload
        if view == "traces":
            assert "local-pod diagnostic only" in (payload.get("note") or "")
        if not payload["available"]:
            assert payload["data"] is None
            assert payload["note"]


def test_chat_writes_a_request_trace(client: TestClient) -> None:
    client.post(
        "/v1/chat/completions",
        json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]},
    )
    traces = client.get("/v1/observability/traces").json()
    assert "local-pod diagnostic only" in traces["scope"]
    chat = next(
        (tree for tree in traces["traces"] if any(span["name"] == "llm" for span in tree["spans"])),
        None,
    )
    assert chat is not None
    names = {span["name"] for span in chat["spans"]}
    assert "request" in names
    assert "route" in names
    assert "llm" in names
    assert "context" not in names
    assert "eval" not in names


def test_unknown_dashboard_is_a_client_error(client: TestClient) -> None:
    assert client.get("/v1/observability/dashboards/made_up").status_code == 400
