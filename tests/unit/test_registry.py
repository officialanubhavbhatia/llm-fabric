from __future__ import annotations

import pytest

from llm_fabric.errors import ConfigurationError, ModelNotFoundError
from llm_fabric.router.registry import ModelRegistry, ModelSpec


def test_cost_is_computed_from_registry_prices() -> None:
    spec = ModelSpec(
        id="m",
        provider="mock",
        provider_model="m",
        input_cost_per_mtok=1.0,
        output_cost_per_mtok=2.0,
    )
    # 1000 input at $1/Mtok + 500 output at $2/Mtok
    assert spec.cost_usd(1000, 500) == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1_000_000)


def test_unpriced_model_costs_nothing() -> None:
    spec = ModelSpec(id="m", provider="mock", provider_model="m")
    assert spec.cost_usd(10_000, 10_000) == 0.0


def test_provider_model_defaults_to_id() -> None:
    registry = ModelRegistry.from_mapping({"models": [{"id": "abc", "provider": "mock"}]})
    assert registry.get("abc").provider_model == "abc"


def test_unknown_model_raises(registry: ModelRegistry) -> None:
    with pytest.raises(ModelNotFoundError):
        registry.get("does-not-exist")


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ConfigurationError, match="duplicate model id"):
        ModelRegistry.from_mapping(
            {"models": [{"id": "a", "provider": "mock"}, {"id": "a", "provider": "mock"}]}
        )


def test_unknown_fallback_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown fallback"):
        ModelRegistry.from_mapping(
            {"models": [{"id": "a", "provider": "mock", "fallbacks": ["ghost"]}]}
        )


def test_alias_referencing_unknown_model_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown model"):
        ModelRegistry.from_mapping(
            {
                "models": [{"id": "a", "provider": "mock"}],
                "aliases": [{"id": "auto", "candidates": ["ghost"]}],
            }
        )


def test_alias_colliding_with_model_id_rejected() -> None:
    with pytest.raises(ConfigurationError, match="collides"):
        ModelRegistry.from_mapping(
            {
                "models": [{"id": "a", "provider": "mock"}],
                "aliases": [{"id": "a", "candidates": ["a"]}],
            }
        )


def test_missing_required_field_rejected() -> None:
    with pytest.raises(ConfigurationError, match="missing required field"):
        ModelRegistry.from_mapping({"models": [{"id": "a"}]})


def test_disabled_models_excluded_from_listing(registry: ModelRegistry) -> None:
    listed = {spec.id for spec in registry.enabled_models()}
    assert "retired" not in listed
    assert "cheap" in listed


def test_cards_include_aliases(registry: ModelRegistry) -> None:
    ids = {card.id for card in registry.cards()}
    assert {"auto", "auto-reasoning"} <= ids


def test_shipped_registry_is_valid() -> None:
    """The registry shipped in config/ must actually load."""
    from pathlib import Path

    registry = ModelRegistry.from_yaml(Path("config/models.yaml"))
    assert registry.enabled_models(), "shipped registry should enable at least one model"
    assert registry.is_alias("auto")
