"""The route preview API.

Preview is the constitution's "no hidden routing decision" made reachable: the
same planner that serves traffic answers here, and returns the same decision
object without spending anything on inference.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _preview(client: TestClient, **body: object) -> dict:
    payload = {"model": "auto", "messages": [{"role": "user", "content": "hello"}]}
    payload.update(body)
    response = client.post("/v1/routes/preview", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_preview_names_the_deployment_that_would_serve(client: TestClient) -> None:
    body = _preview(client)
    assert body["selected"]["model_id"] == "cheap"
    assert body["policy"] == "cost_first"
    assert body["executed"] is False


def test_preview_does_not_perform_inference(client: TestClient, meter) -> None:
    _preview(client)
    # Nothing was served, so nothing may be metered or billed.
    assert meter.recent() == []
    assert meter.totals().requests == 0


def test_preview_returns_the_ranked_chain(client: TestClient) -> None:
    body = _preview(client)
    assert body["chain"] == ["cheap", "premium"]


def test_preview_shows_the_scoring_arithmetic(client: TestClient) -> None:
    scoring = _preview(client)["scoring"]
    assert scoring["policy"] == "cost_first"
    assert "cost" in scoring["used_features"]

    candidate = scoring["candidates"][0]
    features = {feature["feature"]: feature for feature in candidate["features"]}
    assert set(features) == {"quality", "latency", "cost", "health"}
    assert features["cost"]["source"] == "declared"
    assert features["cost"]["raw"] is not None


def test_preview_explains_exclusions(client: TestClient) -> None:
    body = _preview(client, model="auto-reasoning")
    rules = {item["model_id"]: item["rule"] for item in body["excluded"]}
    assert rules["cheap"] == "missing_capability"
    assert any("missing_capability" in line for line in body["explanation"])


def test_preview_reports_the_fallback_graph(client: TestClient) -> None:
    fallback = _preview(client)["fallback"]
    assert fallback["graph"]["root"] == "cheap"
    assert fallback["graph"]["edges"]
    assert fallback["budget"]["max_depth"] >= 0


def test_preview_reports_which_planner_inputs_were_available(client: TestClient) -> None:
    inputs = {item["input"]: item for item in _preview(client)["inputs"]}
    assert inputs["model_capabilities"]["available"] is True
    # The two inputs with no source in this build must say so.
    assert inputs["historical_quality"]["available"] is False
    assert "not built" in inputs["historical_quality"]["note"]
    assert inputs["cache_probability"]["available"] is False


def test_expected_values_are_labelled_as_estimates(client: TestClient) -> None:
    expected = _preview(client)["expected"]
    assert "estimates" in expected["note"]
    assert "not measurements" in expected["note"]


def test_a_policy_override_changes_the_answer(client: TestClient) -> None:
    # `declared` keeps the alias order, which puts premium first.
    assert _preview(client, policy="cost_first")["selected"]["model_id"] == "cheap"
    assert _preview(client, policy="declared")["selected"]["model_id"] == "premium"


def test_quality_first_admits_when_it_cannot_rank_on_quality(client: TestClient) -> None:
    # Neither test model declares a quality score or a grade. The honest answer
    # is to drop the feature and say so, not to invent a ranking.
    body = _preview(client, policy="quality_first")
    scoring = body["scoring"]

    assert "quality" not in scoring["used_features"]
    dropped = {item["feature"]: item["why"] for item in scoring["dropped_features"]}
    assert "quality" in dropped
    assert "not available" in dropped["quality"]
    assert any("quality" in line for line in body["explanation"])


def test_a_legacy_policy_name_still_works(client: TestClient) -> None:
    body = _preview(client, policy="cheapest")
    assert body["policy"] == "cost_first"
    assert body["requested_policy"] == "cost_first"


def test_a_capability_requirement_narrows_the_field(client: TestClient) -> None:
    body = _preview(client, required_capabilities=["reasoning"])
    assert body["selected"]["model_id"] == "premium"


def test_a_context_requirement_narrows_the_field(client: TestClient) -> None:
    body = _preview(client, prompt_tokens=20_000)
    assert body["selected"]["model_id"] == "premium"
    assert body["prompt_tokens"] == 20_000
    assert body["prompt_tokens_are_estimated"] is False


def test_prompt_tokens_are_flagged_when_estimated(client: TestClient) -> None:
    body = _preview(client)
    assert body["prompt_tokens"] > 0
    assert body["prompt_tokens_are_estimated"] is True


def test_a_minimum_grade_is_honoured(client: TestClient) -> None:
    # Neither test model is graded, so a grade floor can select nothing.
    body = _preview(client, minimum_grade="Grade10")
    assert body["selected"] is None
    assert all(item["rule"] == "grade_below_minimum" for item in body["excluded"])


def test_a_request_that_can_be_served_by_nobody_is_reported_not_raised(
    client: TestClient,
) -> None:
    body = _preview(client, required_capabilities=["telepathy"])
    assert body["selected"] is None
    assert body["routing_score"] is None
    assert "No deployment could serve" in " ".join(body["explanation"])


def test_an_unknown_model_is_a_client_error(client: TestClient) -> None:
    response = client.post("/v1/routes/preview", json={"model": "no-such-model"})
    # Same status and shape the chat endpoint uses for the same mistake.
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "model_not_found"


def test_a_nonsense_policy_is_a_client_error(client: TestClient) -> None:
    response = client.post("/v1/routes/preview", json={"model": "auto", "policy": "vibes"})
    assert response.status_code == 400


def test_a_nonsense_grade_is_a_client_error(client: TestClient) -> None:
    response = client.post("/v1/routes/preview", json={"model": "auto", "minimum_grade": "Grade99"})
    assert response.status_code == 400


def test_negative_token_counts_are_rejected(client: TestClient) -> None:
    response = client.post("/v1/routes/preview", json={"model": "auto", "prompt_tokens": -1})
    assert response.status_code == 422


# -- fleet health -------------------------------------------------------------


def test_fleet_health_is_empty_before_any_traffic(client: TestClient) -> None:
    body = client.get("/v1/routes/health").json()
    assert body["observed"] == 0
    assert body["deployments"] == []


def test_fleet_health_reports_after_a_request(client: TestClient) -> None:
    client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    body = client.get("/v1/routes/health").json()

    assert body["observed"] >= 1
    served = next(d for d in body["deployments"] if d["deployment_id"] == "cheap")
    assert served["successes"] == 1
    assert served["circuit_state"] == "closed"
    assert served["health_score"] == 1.0
    assert served["queue_depth"] == 0


def test_fleet_health_reports_absence_rather_than_a_default(client: TestClient) -> None:
    # A deployment that failed to construct is still recorded, but a deployment
    # nobody touched must simply not appear rather than claim to be healthy.
    body = client.get("/v1/routes/health").json()
    assert "null rather than a default" in body["note"]
