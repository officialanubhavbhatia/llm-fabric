from __future__ import annotations

import pytest

from llm_fabric.errors import ConfigurationError, ModelNotFoundError
from llm_fabric.router.registry import ModelRegistry, ModelSpec
from llm_fabric.router.tiers import ServiceTier


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


def test_unpriced_model_cost_is_unknown() -> None:
    spec = ModelSpec(id="m", provider="mock", provider_model="m")
    assert spec.cost_usd(10_000, 10_000) is None
    assert spec.api_cost_knowledge.value == "unknown"


def test_known_zero_api_price_costs_zero() -> None:
    spec = ModelSpec(
        id="m",
        provider="mock",
        provider_model="m",
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
    )
    assert spec.cost_usd(10_000, 10_000) == 0.0
    assert spec.is_priced
    assert spec.api_cost_knowledge.value == "known_zero"


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


def test_local_registry_is_valid() -> None:
    from pathlib import Path

    registry = ModelRegistry.from_yaml(Path("config/models.local.yaml"))
    assert {spec.id for spec in registry.enabled_models()} >= {
        "mock-small",
        "local-small",
        "local-reasoning",
    }
    assert registry.get("local-small").provider == "ollama"
    assert registry.get("local-small").provider_model == "llama3.2"


def test_ollama_grades_registry_covers_grade00_to_grade29() -> None:
    from pathlib import Path

    from llm_fabric.router.grades import GRADE_COUNT, Grade

    registry = ModelRegistry.from_yaml(Path("config/models.ollama-grades.yaml"))
    tags = [
        line.split("#", 1)[0].strip()
        for line in Path("config/ollama-grade-tags.txt").read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]
    ollama = [spec for spec in registry.enabled_models() if spec.provider == "ollama"]
    assert len(ollama) == GRADE_COUNT == 30
    assert len(tags) == GRADE_COUNT
    assert {spec.provider_model for spec in ollama} == set(tags)
    grades = {spec.grade for spec in ollama}
    assert grades == {Grade.from_index(index) for index in range(GRADE_COUNT)}
    assert registry.is_alias("auto")
    assert registry.is_alias("auto-coding")
    assert registry.is_alias("auto-reasoning")
    assert registry.is_alias("auto-agent")
    for spec in ollama:
        assert spec.quality.reasoning is None
        assert spec.quality.coding is None
        assert spec.quality.agent is None
        assert spec.lifecycle.value == "approved"


def test_identity_metadata_round_trips() -> None:
    registry = ModelRegistry.from_mapping(
        {
            "models": [
                {
                    "id": "qwen",
                    "provider": "vllm",
                    "huggingface_id": "Qwen/Qwen2.5-7B-Instruct",
                    "revision": "abc123",
                    "digest": "sha256:deadbeef",
                    "license": "apache-2.0",
                    "commercial_use": True,
                    "pool": "general",
                    "grade": "L12",
                    "tiers": ["L12", "L13"],
                }
            ]
        }
    )
    spec = registry.get("qwen")
    assert spec.huggingface_id == "Qwen/Qwen2.5-7B-Instruct"
    assert spec.revision == "abc123"
    assert spec.license == "apache-2.0"
    assert spec.commercial_use is True
    assert spec.pool == "general"
    assert spec.serves_tier(ServiceTier.L13)
    assert spec.public_tier is ServiceTier.L13


def test_trust_remote_code_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="trust_remote_code"):
        ModelSpec(id="unsafe", provider="vllm", trust_remote_code=True)
