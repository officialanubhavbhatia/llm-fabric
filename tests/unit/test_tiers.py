"""Public L0–L30 tiers are a spelling of Grade00–Grade29, plus L30 → Grade29."""

from __future__ import annotations

import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.router.grades import GRADE_COUNT, Grade
from llm_fabric.router.tiers import ALL_TIERS, TIER_COUNT, ServiceTier, parse_service_tier


def test_there_are_thirty_one_public_tiers() -> None:
    assert len(ALL_TIERS) == TIER_COUNT == 31
    assert ALL_TIERS[0] is ServiceTier.L0
    assert ALL_TIERS[-1] is ServiceTier.L30


def test_l_n_maps_onto_grade_n_until_l30() -> None:
    for index in range(GRADE_COUNT):
        assert ServiceTier.from_ordinal(index).to_grade() is Grade.from_index(index)
    assert ServiceTier.L30.to_grade() is Grade.GRADE29
    assert ServiceTier.L29.to_grade() is Grade.GRADE29


def test_parse_accepts_l_spellings() -> None:
    assert ServiceTier.parse("L7") is ServiceTier.L7
    assert ServiceTier.parse("l12") is ServiceTier.L12
    assert ServiceTier.parse("Grade03") is ServiceTier.L3
    assert ServiceTier.parse("L30") is ServiceTier.L30


def test_parse_service_tier_ignores_model_ids() -> None:
    assert parse_service_tier("auto") is None
    assert parse_service_tier("llama3.2") is None
    assert parse_service_tier("L12") is ServiceTier.L12
    assert parse_service_tier("L99") is None


def test_out_of_range_is_refused() -> None:
    with pytest.raises(ConfigurationError):
        ServiceTier.from_ordinal(31)
    with pytest.raises(ConfigurationError):
        ServiceTier.parse("L31")
