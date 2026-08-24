"""The synthetic fleet's invariants.

Other tests assert routing behaviour *through* this fleet, so if its shape drifts
those tests start proving something other than what they claim. These assertions
pin the shape, and pin the fact that the numbers are fixtures rather than
measurements of anything real.
"""

from __future__ import annotations

import pytest

from llm_fabric.contract.openai import ChatMessage
from llm_fabric.errors import ProviderUnavailableError
from llm_fabric.router.capabilities import Capability
from llm_fabric.router.grades import ALL_GRADES, Grade
from llm_fabric.router.registry import Locality
from llm_fabric.router.synthetic import (
    SyntheticFleet,
    SyntheticProvider,
    synthetic_model_id,
    synthetic_models,
    synthetic_registry,
)
from llm_fabric.serving.base import InferenceRequest, StreamDelta, StreamEnd


def test_there_is_one_deployment_per_grade() -> None:
    models = synthetic_models()
    assert len(models) == 30
    assert [spec.grade for spec in models] == list(ALL_GRADES)
    assert len({spec.id for spec in models}) == 30
    assert len({spec.deployment_id for spec in models}) == 30


@pytest.mark.parametrize(
    "attribute",
    ["input_cost_per_mtok", "output_cost_per_mtok", "context_window"],
)
def test_these_attributes_rise_with_grade(attribute: str) -> None:
    values = [getattr(spec, attribute) for spec in synthetic_models()]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_declared_latency_rises_with_grade() -> None:
    ttft = [spec.performance.p50_ttft_ms for spec in synthetic_models()]
    assert ttft == sorted(ttft)  # type: ignore[type-var]


def test_quality_rises_with_grade_apart_from_the_specialists() -> None:
    for spec in synthetic_models():
        assert spec.grade is not None
        assert spec.quality.safety is not None
        # `safety` carries no specialist bonus, so it is the clean baseline.
        assert spec.quality.safety == pytest.approx((spec.grade.ordinal + 1) / 32, abs=1e-4)


def test_the_baseline_leaves_headroom_for_a_specialist_bonus() -> None:
    # Otherwise the top band's bonus would be clamped away and any test relying
    # on it would pass for the wrong reason.
    top = synthetic_models()[-1]
    assert (top.quality.safety or 0) < 0.95


def test_the_specialist_bands_score_above_their_own_baseline() -> None:
    for spec in synthetic_models():
        assert spec.grade is not None
        baseline = spec.quality.safety
        assert baseline is not None
        if spec.grade.ordinal % 5 == 3:
            assert (spec.quality.coding or 0) > baseline
        if spec.grade.ordinal % 5 == 4:
            assert (spec.quality.reasoning or 0) > baseline


def test_localities_cycle_evenly() -> None:
    counts = {locality: 0 for locality in Locality}
    for spec in synthetic_models():
        counts[spec.locality] += 1
    assert counts == {Locality.LOCAL: 10, Locality.PRIVATE: 10, Locality.EXTERNAL: 10}


def test_capabilities_unlock_as_grade_rises() -> None:
    registry = synthetic_registry()
    weakest = registry.get(synthetic_model_id(Grade.GRADE00))
    strongest = registry.get(synthetic_model_id(Grade.GRADE29))

    assert weakest.capabilities.has(Capability.CHAT)
    assert not weakest.capabilities.supports_vision
    assert not weakest.capabilities.supports_tools

    assert strongest.capabilities.supports_vision
    assert strongest.capabilities.supports_tools
    assert strongest.capabilities.supports_json_schema
    # And the implication still holds in the fleet.
    assert strongest.capabilities.has(Capability.STRUCTURED_OUTPUT)


def test_embeddings_are_not_merely_a_function_of_grade() -> None:
    with_embeddings = {
        spec.id for spec in synthetic_models() if spec.capabilities.supports_embeddings
    }
    assert with_embeddings
    assert synthetic_model_id(Grade.GRADE29) not in with_embeddings


def test_every_alias_resolves() -> None:
    registry = synthetic_registry()
    for alias_id in ("synth-auto", "synth-cheap", "synth-best", "synth-fast"):
        alias = registry.alias(alias_id)
        assert alias is not None
        assert len(alias.candidates) == 30


def test_the_fleet_is_reproducible() -> None:
    first = [spec.describe() for spec in synthetic_models()]
    second = [spec.describe() for spec in synthetic_models()]
    assert first == second


# -- the provider -------------------------------------------------------------


def _inference(model: str = "synthetic/grade00") -> InferenceRequest:
    return InferenceRequest(model=model, messages=[ChatMessage(role="user", content="hello")])


async def test_the_provider_returns_deterministic_text() -> None:
    provider = SyntheticProvider()
    first = await provider.generate(_inference())
    second = await provider.generate(_inference())
    assert first.text == second.text
    assert first.prompt_tokens == second.prompt_tokens


async def test_queued_failures_are_consumed_one_per_call() -> None:
    provider = SyntheticProvider()
    provider.fail_next("m", times=2)

    for _ in range(2):
        with pytest.raises(ProviderUnavailableError):
            await provider.generate(_inference("m"))
    # Third call succeeds: this is the transient-then-recovered shape.
    assert await provider.generate(_inference("m"))


async def test_permanent_failures_never_recover_until_told() -> None:
    provider = SyntheticProvider()
    provider.always_fail("m")
    for _ in range(3):
        with pytest.raises(ProviderUnavailableError):
            await provider.generate(_inference("m"))

    provider.recover("m")
    assert await provider.generate(_inference("m"))


async def test_streaming_yields_deltas_then_exactly_one_end() -> None:
    provider = SyntheticProvider()
    events = [event async for event in provider.stream(_inference())]
    assert isinstance(events[-1], StreamEnd)
    assert all(isinstance(event, StreamDelta) for event in events[:-1])
    assert len(events) > 1


async def test_calls_are_recorded_across_providers_in_order() -> None:
    fleet = SyntheticFleet()
    for grade in (Grade.GRADE00, Grade.GRADE01, Grade.GRADE02):
        spec = fleet.registry.get(synthetic_model_id(grade))
        await fleet.providers[spec.provider].generate(_inference(spec.provider_model))

    # Those three sit on three different providers; the shared sequence is what
    # keeps the order readable.
    assert fleet.served == [
        synthetic_model_id(Grade.GRADE00),
        synthetic_model_id(Grade.GRADE01),
        synthetic_model_id(Grade.GRADE02),
    ]


def test_the_fleet_exposes_one_provider_per_locality() -> None:
    fleet = SyntheticFleet()
    assert set(fleet.overrides()) == {"synth_local", "synth_private", "synth_cloud"}
    assert fleet.describe()["models"] == 30


def test_reset_clears_failures_and_calls() -> None:
    fleet = SyntheticFleet()
    fleet.always_fail(synthetic_model_id(Grade.GRADE00))
    fleet.reset()
    assert fleet.served == []
