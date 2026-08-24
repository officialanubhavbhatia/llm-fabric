"""Policy scoring: the ordering, and the arithmetic that explains it.

The rules about *absent* data get the most attention here, because they are the
ones that decide whether a route is honest. A feature nobody can supply must be
dropped rather than imputed, and an unpriced deployment must not be mistaken for
a free one.
"""

from __future__ import annotations

import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.router.grades import Grade
from llm_fabric.router.health import HealthTracker
from llm_fabric.router.policy import (
    POLICY_WEIGHTS,
    FeatureSource,
    PolicyWeights,
    RoutePolicy,
    ScoringInputs,
    parse_policy,
    permitted_localities,
    score_candidates,
)
from llm_fabric.router.registry import (
    Locality,
    ModelSpec,
    PerformanceProfile,
    Placement,
    QualityScores,
)


def _spec(
    model_id: str,
    *,
    grade: Grade | None = None,
    quality: QualityScores | None = None,
    ttft: float | None = None,
    tpot: float | None = None,
    in_cost: float | None = None,
    out_cost: float | None = None,
    locality: Locality = Locality.EXTERNAL,
) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        provider="synthetic",
        provider_model=model_id,
        grade=grade,
        quality=quality or QualityScores(),
        performance=PerformanceProfile(p50_ttft_ms=ttft, p50_tpot_ms=tpot),
        input_cost_per_mtok=in_cost,
        output_cost_per_mtok=out_cost,
        placement=Placement(locality=locality),
    )


# -- the policy vocabulary ----------------------------------------------------


def test_every_constitutional_policy_exists() -> None:
    assert {
        "quality_first",
        "latency_first",
        "cost_first",
        "balanced",
        "local_only",
        "private_only",
        "custom",
    } <= {member.value for member in RoutePolicy}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cheapest", RoutePolicy.COST_FIRST),
        ("cost_first", RoutePolicy.COST_FIRST),
        ("cost-first", RoutePolicy.COST_FIRST),
        ("QUALITY_FIRST", RoutePolicy.QUALITY_FIRST),
        ("fastest", RoutePolicy.LATENCY_FIRST),
        ("local", RoutePolicy.LOCAL_ONLY),
        ("private", RoutePolicy.PRIVATE_ONLY),
    ],
)
def test_legacy_and_canonical_names_both_parse(text: str, expected: RoutePolicy) -> None:
    assert parse_policy(text) is expected


def test_an_unknown_policy_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="unknown routing policy"):
        parse_policy("vibes")


def test_only_the_locality_policies_constrain_locality() -> None:
    assert permitted_localities(RoutePolicy.LOCAL_ONLY) == {Locality.LOCAL}
    assert permitted_localities(RoutePolicy.PRIVATE_ONLY) == {Locality.LOCAL, Locality.PRIVATE}
    assert permitted_localities(RoutePolicy.BALANCED) is None
    assert permitted_localities(RoutePolicy.COST_FIRST) is None


def test_weights_must_not_be_negative_or_all_zero() -> None:
    with pytest.raises(ConfigurationError):
        PolicyWeights(quality=-1.0)
    with pytest.raises(ConfigurationError):
        PolicyWeights()


# -- ordering -----------------------------------------------------------------


def test_cost_first_picks_the_cheapest() -> None:
    cheap = _spec("cheap", in_cost=0.1, out_cost=0.2)
    dear = _spec("dear", in_cost=10.0, out_cost=30.0)
    result = score_candidates([dear, cheap], policy=RoutePolicy.COST_FIRST)
    assert [c.model_id for c in result.candidates] == ["cheap", "dear"]


def test_quality_first_picks_the_best_declared_score() -> None:
    weak = _spec("weak", quality=QualityScores(reasoning=0.2, coding=0.2))
    strong = _spec("strong", quality=QualityScores(reasoning=0.9, coding=0.9))
    result = score_candidates([weak, strong], policy=RoutePolicy.QUALITY_FIRST)
    assert result.candidates[0].model_id == "strong"


def test_latency_first_picks_the_fastest() -> None:
    slow = _spec("slow", ttft=900.0, tpot=9.0)
    quick = _spec("quick", ttft=20.0, tpot=1.0)
    result = score_candidates([slow, quick], policy=RoutePolicy.LATENCY_FIRST)
    assert result.candidates[0].model_id == "quick"


def test_the_quality_dimension_the_intent_needs_is_the_one_used() -> None:
    coder = _spec("coder", quality=QualityScores(coding=0.95, reasoning=0.10))
    thinker = _spec("thinker", quality=QualityScores(coding=0.10, reasoning=0.95))

    on_coding = score_candidates(
        [thinker, coder],
        policy=RoutePolicy.QUALITY_FIRST,
        inputs=ScoringInputs(quality_dimension="coding"),
    )
    assert on_coding.candidates[0].model_id == "coder"
    assert on_coding.candidates[0].feature("quality").basis == "quality.coding"  # type: ignore[union-attr]

    on_reasoning = score_candidates(
        [thinker, coder],
        policy=RoutePolicy.QUALITY_FIRST,
        inputs=ScoringInputs(quality_dimension="reasoning"),
    )
    assert on_reasoning.candidates[0].model_id == "thinker"


def test_grade_stands_in_when_no_score_was_measured() -> None:
    low = _spec("low", grade=Grade.GRADE02)
    high = _spec("high", grade=Grade.GRADE25)
    result = score_candidates([low, high], policy=RoutePolicy.QUALITY_FIRST)

    assert result.candidates[0].model_id == "high"
    quality = result.candidates[0].feature("quality")
    assert quality is not None
    # The explanation must say the ranking rested on a grade, not on a score.
    assert quality.basis == "grade.Grade25"


def test_a_measured_score_is_preferred_over_a_grade() -> None:
    spec = _spec("m", grade=Grade.GRADE00, quality=QualityScores(coding=0.9))
    other = _spec("n", grade=Grade.GRADE29, quality=QualityScores(coding=0.1))
    result = score_candidates(
        [spec, other],
        policy=RoutePolicy.QUALITY_FIRST,
        inputs=ScoringInputs(quality_dimension="coding"),
    )
    assert result.candidates[0].model_id == "m"


# -- absent data --------------------------------------------------------------


def test_an_unpriced_model_is_not_treated_as_free() -> None:
    priced = _spec("priced", in_cost=5.0, out_cost=15.0)
    unpriced = _spec("unpriced")

    result = score_candidates([priced, unpriced], policy=RoutePolicy.COST_FIRST)

    by_id = {candidate.model_id: candidate for candidate in result.candidates}
    assert by_id["unpriced"].feature("cost").source is FeatureSource.ABSENT  # type: ignore[union-attr]
    assert by_id["priced"].feature("cost").source is FeatureSource.DECLARED  # type: ignore[union-attr]

    # Cost is still used for the priced deployment. The unknown price does not
    # drop ranking for the whole fleet, and the unpriced model does not win by
    # looking free.
    assert "cost" in result.used_features
    assert "cost_partial" in dict(result.dropped_features)
    assert result.candidates[0].model_id == "priced"


def test_known_zero_is_not_unknown() -> None:
    free = _spec("local", in_cost=0.0, out_cost=0.0)
    dear = _spec("api", in_cost=10.0, out_cost=30.0)
    result = score_candidates([dear, free], policy=RoutePolicy.COST_FIRST)
    assert result.candidates[0].model_id == "local"
    assert "cost" in result.used_features
    cost = result.candidates[0].feature("cost")
    assert cost is not None
    assert cost.raw == 0.0


def test_one_unknown_price_does_not_erase_known_prices() -> None:
    cheap = _spec("cheap", in_cost=0.1, out_cost=0.2)
    dear = _spec("dear", in_cost=10.0, out_cost=30.0)
    unknown = _spec("unknown")
    result = score_candidates([dear, unknown, cheap], policy=RoutePolicy.COST_FIRST)
    assert [c.model_id for c in result.candidates] == ["cheap", "dear", "unknown"]


def test_a_feature_missing_for_one_candidate_is_dropped_for_all() -> None:
    measured = _spec("measured", quality=QualityScores(reasoning=0.9), in_cost=1.0, out_cost=1.0)
    unmeasured = _spec("unmeasured", in_cost=2.0, out_cost=2.0)

    result = score_candidates([measured, unmeasured], policy=RoutePolicy.QUALITY_FIRST)
    assert "quality" not in result.used_features
    assert "quality" in dict(result.dropped_features)


def test_dropping_a_feature_renormalises_the_rest() -> None:
    # With quality unavailable, the surviving weights must still sum to one, so
    # scores stay on a comparable scale.
    a = _spec("a", in_cost=1.0, out_cost=1.0, ttft=10.0, tpot=1.0)
    b = _spec("b", in_cost=2.0, out_cost=2.0, ttft=20.0, tpot=2.0)
    result = score_candidates([a, b], policy=RoutePolicy.BALANCED)

    assert "quality" not in result.used_features
    best = result.candidates[0]
    assert best.score == pytest.approx(1.0)
    weights = sum(f.weight for f in best.features if f.usable)
    assert weights == pytest.approx(1.0)


def test_with_nothing_usable_registry_order_is_kept_and_admitted() -> None:
    first = _spec("first")
    second = _spec("second")
    result = score_candidates([first, second], policy=RoutePolicy.BALANCED)

    assert result.fell_back_to_registry_order
    assert result.used_features == ()
    assert [c.model_id for c in result.candidates] == ["first", "second"]
    assert all(c.score == 0.0 for c in result.candidates)


def test_health_is_absent_until_traffic_is_observed() -> None:
    tracker = HealthTracker()
    spec = _spec("m", in_cost=1.0, out_cost=1.0)
    inputs = ScoringInputs(health={spec.deployment_id: tracker.snapshot(spec.deployment_id)})

    result = score_candidates([spec], policy=RoutePolicy.BALANCED, inputs=inputs)
    health = result.candidates[0].feature("health")
    assert health is not None
    assert health.source is FeatureSource.ABSENT
    assert health.raw is None


def test_observed_latency_outranks_the_declared_profile() -> None:
    tracker = HealthTracker()
    # Declared fast, observed slow. What actually happened must win.
    optimistic = _spec("optimistic", ttft=10.0, tpot=1.0)
    honest = _spec("honest", ttft=500.0, tpot=5.0)
    for _ in range(3):
        tracker.record_success(optimistic.deployment_id, latency_ms=5000.0)
        tracker.record_success(honest.deployment_id, latency_ms=50.0)

    inputs = ScoringInputs(
        health={
            spec.deployment_id: tracker.snapshot(spec.deployment_id)
            for spec in (optimistic, honest)
        }
    )
    result = score_candidates([optimistic, honest], policy=RoutePolicy.LATENCY_FIRST, inputs=inputs)
    assert result.candidates[0].model_id == "honest"
    latency = result.candidates[0].feature("latency")
    assert latency is not None
    assert latency.source is FeatureSource.OBSERVED


def test_an_unhealthy_deployment_loses_to_a_healthy_one() -> None:
    tracker = HealthTracker()
    good = _spec("good", in_cost=1.0, out_cost=1.0, ttft=10.0, tpot=1.0)
    bad = _spec("bad", in_cost=1.0, out_cost=1.0, ttft=10.0, tpot=1.0)
    for _ in range(6):
        tracker.record_success(good.deployment_id, latency_ms=10.0)
        tracker.record_failure(bad.deployment_id, latency_ms=10.0, error="boom")

    inputs = ScoringInputs(
        health={spec.deployment_id: tracker.snapshot(spec.deployment_id) for spec in (good, bad)}
    )
    result = score_candidates([good, bad], policy=RoutePolicy.BALANCED, inputs=inputs)
    assert result.candidates[0].model_id == "good"


# -- explanation --------------------------------------------------------------


def test_every_feature_is_reported_even_when_unused() -> None:
    result = score_candidates([_spec("m", in_cost=1.0, out_cost=1.0)], policy=RoutePolicy.BALANCED)
    names = {feature.name for feature in result.candidates[0].features}
    assert names == {"quality", "latency", "cost", "health"}


def test_contributions_sum_to_the_score() -> None:
    a = _spec("a", in_cost=1.0, out_cost=1.0, ttft=10.0, tpot=1.0)
    b = _spec("b", in_cost=5.0, out_cost=5.0, ttft=90.0, tpot=9.0)
    result = score_candidates([a, b], policy=RoutePolicy.BALANCED)

    for candidate in result.candidates:
        total = sum(feature.contribution for feature in candidate.features)
        assert total == pytest.approx(candidate.score)


def test_the_result_serialises_for_the_preview_api() -> None:
    result = score_candidates(
        [_spec("m", in_cost=1.0, out_cost=1.0)], policy=RoutePolicy.COST_FIRST
    )
    payload = result.as_dict()
    assert payload["policy"] == "cost_first"
    assert payload["candidates"][0]["model_id"] == "m"
    assert {"feature", "source", "weight", "contribution"} <= set(
        payload["candidates"][0]["features"][0]
    )


# -- determinism --------------------------------------------------------------


def test_identical_candidates_keep_registry_order() -> None:
    a = _spec("a", in_cost=1.0, out_cost=1.0)
    b = _spec("b", in_cost=1.0, out_cost=1.0)
    scored = score_candidates([b, a], policy=RoutePolicy.COST_FIRST)
    assert [candidate.model_id for candidate in scored.candidates] == ["b", "a"]


def test_scoring_is_repeatable() -> None:
    specs = [
        _spec("a", in_cost=1.0, out_cost=3.0, ttft=10.0, tpot=1.0, grade=Grade.GRADE05),
        _spec("b", in_cost=2.0, out_cost=1.0, ttft=30.0, tpot=2.0, grade=Grade.GRADE20),
        _spec("c", in_cost=0.5, out_cost=9.0, ttft=5.0, tpot=4.0, grade=Grade.GRADE12),
    ]
    runs = [
        [c.model_id for c in score_candidates(specs, policy=RoutePolicy.BALANCED).candidates]
        for _ in range(5)
    ]
    assert len(set(map(tuple, runs))) == 1


def test_an_empty_candidate_set_scores_nothing() -> None:
    result = score_candidates([], policy=RoutePolicy.BALANCED)
    assert result.candidates == ()
    assert result.best is None


def test_declared_policy_does_not_rank() -> None:
    a = _spec("expensive", in_cost=100.0, out_cost=100.0)
    b = _spec("cheap", in_cost=0.1, out_cost=0.1)
    result = score_candidates([a, b], policy=RoutePolicy.DECLARED)
    assert [c.model_id for c in result.candidates] == ["expensive", "cheap"]
    assert result.fell_back_to_registry_order


def test_each_policy_has_weights() -> None:
    for policy in RoutePolicy:
        assert policy in POLICY_WEIGHTS
        assert POLICY_WEIGHTS[policy].total > 0
