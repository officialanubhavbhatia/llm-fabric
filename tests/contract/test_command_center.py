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


def test_intents_view_leads_with_frozen_safety_gates(client: TestClient) -> None:
    payload = client.get("/v1/observability/dashboards/intents").json()
    assert payload["available"] is True
    gates = payload["data"]["safety_gates"]
    assert gates["hard_negative_accuracy"] == 0.50
    assert gates["required"] == 0.58
    assert gates["routing"] == "OFF"
    assert gates["serving_path_classification"] is False


def test_tiers_view_has_l0_to_l30_histogram(client: TestClient) -> None:
    payload = client.get("/v1/observability/dashboards/tiers").json()
    assert payload["available"] is True
    histogram = payload["data"]["histogram"]
    assert len(histogram) == 31
    assert histogram[0]["tier"] == "L0"
    assert histogram[-1]["tier"] == "L30"


def test_promotion_view_does_not_equate_evaluated_with_approved(
    client: TestClient,
) -> None:
    payload = client.get("/v1/observability/dashboards/promotion").json()
    assert payload["available"] is True
    assert "Evaluated is not approved" in payload["note"]
    for row in payload["data"]["models"]:
        assert "state" in row
        assert "production_eligible" in row
        assert "declared_tiers" in row
        assert "approved_tiers" in row


def test_unknown_dashboard_is_a_client_error(client: TestClient) -> None:
    assert client.get("/v1/observability/dashboards/made_up").status_code == 400
