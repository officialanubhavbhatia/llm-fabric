from __future__ import annotations

import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.router.policy import cheapest, declared, get_policy
from llm_fabric.router.registry import ModelSpec


def _spec(model_id: str, in_cost: float, out_cost: float) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        provider="mock",
        provider_model=model_id,
        input_cost_per_mtok=in_cost,
        output_cost_per_mtok=out_cost,
    )


def test_cheapest_orders_by_blended_price() -> None:
    expensive = _spec("expensive", 10.0, 30.0)
    mid = _spec("mid", 1.0, 3.0)
    cheap = _spec("cheap", 0.1, 0.3)

    assert [s.id for s in cheapest([expensive, mid, cheap])] == ["cheap", "mid", "expensive"]


def test_cheapest_weights_output_more_than_input() -> None:
    # Same total, different split: heavier output should rank as more expensive.
    output_heavy = _spec("output-heavy", 1.0, 9.0)
    input_heavy = _spec("input-heavy", 9.0, 1.0)

    assert [s.id for s in cheapest([output_heavy, input_heavy])] == [
        "input-heavy",
        "output-heavy",
    ]


def test_cheapest_breaks_ties_deterministically() -> None:
    first = _spec("b", 1.0, 1.0)
    second = _spec("a", 1.0, 1.0)
    assert [s.id for s in cheapest([first, second])] == ["a", "b"]


def test_declared_preserves_registry_order() -> None:
    specs = [_spec("z", 9.0, 9.0), _spec("a", 0.1, 0.1)]
    assert [s.id for s in declared(specs)] == ["z", "a"]


def test_policies_do_not_mutate_input() -> None:
    specs = [_spec("z", 9.0, 9.0), _spec("a", 0.1, 0.1)]
    cheapest(specs)
    declared(specs)
    assert [s.id for s in specs] == ["z", "a"]


def test_unknown_policy_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown routing policy"):
        get_policy("lowest-latency")
