"""Grades are an ordering the router depends on, so the ordering is pinned here."""

from __future__ import annotations

import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.router.grades import ALL_GRADES, GRADE_COUNT, Grade, at_least


def test_there_are_exactly_thirty_grades() -> None:
    assert len(ALL_GRADES) == GRADE_COUNT == 30
    assert ALL_GRADES[0] is Grade.GRADE00
    assert ALL_GRADES[-1] is Grade.GRADE29


def test_index_matches_the_name() -> None:
    for index, grade in enumerate(ALL_GRADES):
        assert grade.ordinal == index
        assert grade.value == f"Grade{index:02d}"


def test_zero_padding_makes_string_order_match_numeric_order() -> None:
    # Sorting grades as plain strings must not put Grade10 before Grade09.
    assert sorted(grade.value for grade in ALL_GRADES) == [g.value for g in ALL_GRADES]


def test_normalised_spans_the_unit_interval() -> None:
    assert Grade.GRADE00.normalised == 0.0
    assert Grade.GRADE29.normalised == 1.0
    assert Grade.GRADE00.normalised < Grade.GRADE15.normalised < Grade.GRADE29.normalised


@pytest.mark.parametrize("text", ["Grade07", "grade07", "GRADE07", "7", "07", " Grade07 "])
def test_parse_accepts_the_spellings_people_actually_write(text: str) -> None:
    assert Grade.parse(text) is Grade.GRADE07


@pytest.mark.parametrize("text", ["", "Grade30", "Grade-1", "G7", "best", "Grade07x"])
def test_parse_refuses_anything_else(text: str) -> None:
    with pytest.raises(ConfigurationError):
        Grade.parse(text)


def test_from_index_rejects_out_of_range() -> None:
    assert Grade.from_index(0) is Grade.GRADE00
    assert Grade.from_index(29) is Grade.GRADE29
    for bad in (-1, 30, 100):
        with pytest.raises(ConfigurationError):
            Grade.from_index(bad)


def test_at_least_is_inclusive_of_its_floor() -> None:
    permitted = at_least(Grade.GRADE27)
    assert permitted == {Grade.GRADE27, Grade.GRADE28, Grade.GRADE29}
    assert Grade.GRADE26 not in permitted
