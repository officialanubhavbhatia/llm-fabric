"""Logical model grades, `Grade00` through `Grade29`.

A grade is a *capability class*, not a model. `Grade22` does not mean "GPT-4o";
it means "whatever this deployment currently believes sits in the twenty-third
capability band". The point of the indirection is that benchmark data moves: when
a model is re-measured, or a provider ships a better checkpoint under the same
name, the operator changes that deployment's grade and every routing rule
expressed in grades keeps working untouched.

Two conventions are chosen here because the constitution names the grades without
fixing their semantics, and routing needs both:

**Ascending capability.** `Grade00` is the least capable band and `Grade29` the
most. Nothing forces this reading — the constitution says only that grades are
logical classes — so it is a decision of this implementation, and every
comparison in the router depends on it.

**Zero-padded names.** Because every name is `Grade` plus exactly two digits,
lexicographic order and numeric order are the same, so sorting grades as plain
strings is correct. `index` is still the honest way to ask for the number.

A grade carries no quality score of its own. Grades order deployments when
measured scores are absent; they do not substitute for measurement, and the
router says which of the two it used.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from llm_fabric.errors import ConfigurationError

#: How many grades exist. Fixed by the constitution at `Grade00` ... `Grade29`.
GRADE_COUNT = 30


class Grade(StrEnum):
    """A logical capability band. Higher is more capable; see the module docstring."""

    GRADE00 = "Grade00"
    GRADE01 = "Grade01"
    GRADE02 = "Grade02"
    GRADE03 = "Grade03"
    GRADE04 = "Grade04"
    GRADE05 = "Grade05"
    GRADE06 = "Grade06"
    GRADE07 = "Grade07"
    GRADE08 = "Grade08"
    GRADE09 = "Grade09"
    GRADE10 = "Grade10"
    GRADE11 = "Grade11"
    GRADE12 = "Grade12"
    GRADE13 = "Grade13"
    GRADE14 = "Grade14"
    GRADE15 = "Grade15"
    GRADE16 = "Grade16"
    GRADE17 = "Grade17"
    GRADE18 = "Grade18"
    GRADE19 = "Grade19"
    GRADE20 = "Grade20"
    GRADE21 = "Grade21"
    GRADE22 = "Grade22"
    GRADE23 = "Grade23"
    GRADE24 = "Grade24"
    GRADE25 = "Grade25"
    GRADE26 = "Grade26"
    GRADE27 = "Grade27"
    GRADE28 = "Grade28"
    GRADE29 = "Grade29"

    @property
    def ordinal(self) -> int:
        """Position in the band, 0 through 29.

        Named `ordinal` rather than `index` because `Grade` is a `StrEnum`, and
        a property called `index` would shadow `str.index` for every caller that
        treats a grade as the string it also is.
        """
        return int(self.value[5:])

    @property
    def normalised(self) -> float:
        """Position rescaled to [0, 1], for blending into a routing score."""
        return self.ordinal / (GRADE_COUNT - 1)

    @classmethod
    def from_index(cls, index: int) -> Self:
        if not 0 <= index < GRADE_COUNT:
            raise ConfigurationError(f"grade index must lie in [0, {GRADE_COUNT - 1}], got {index}")
        return cls(f"Grade{index:02d}")

    @classmethod
    def parse(cls, value: str) -> Self:
        """Accept `Grade07`, `grade07`, `7`, `07`, or the public tier spelling `L7`.

        `L30` maps onto `Grade29`. Tiers are documented in `router.tiers`; the
        constitution still has thirty grades.
        """
        text = str(value).strip()
        if not text:
            raise ConfigurationError("a grade cannot be empty")
        if text.isdigit():
            return cls.from_index(int(text))
        if len(text) >= 2 and text[0] in "Ll" and text[1:].isdigit():
            ordinal = int(text[1:])
            if ordinal == GRADE_COUNT:
                return cls.from_index(GRADE_COUNT - 1)
            return cls.from_index(ordinal)
        normalised = text[:5].capitalize() + text[5:]
        try:
            return cls(normalised)
        except ValueError:
            raise ConfigurationError(
                f"unknown grade '{value}'; expected Grade00 through Grade{GRADE_COUNT - 1:02d} "
                f"or L0 through L{GRADE_COUNT}"
            ) from None


#: Every grade, weakest first. Materialised once because policies iterate it.
ALL_GRADES: tuple[Grade, ...] = tuple(Grade)


def at_least(minimum: Grade) -> frozenset[Grade]:
    """The grades that clear a floor, for `minimum_grade` style requirements."""
    return frozenset(grade for grade in ALL_GRADES if grade.ordinal >= minimum.ordinal)
