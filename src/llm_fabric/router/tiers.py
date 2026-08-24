"""Operator-facing service tiers `L0` through `L30`.

The constitution names thirty capability bands, `Grade00` … `Grade29`. Tiers
are a public spelling of those bands plus one extra label, `L30`, for
exceptional escalation. `L30` maps onto `Grade29`: it is not a thirty-first
constitutional grade.

A tier is not a model name. Routing still selects deployments from the registry.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from llm_fabric.errors import ConfigurationError
from llm_fabric.router.grades import GRADE_COUNT, Grade

TIER_COUNT = 31
_LAST_GRADE_ORDINAL = GRADE_COUNT - 1


class ServiceTier(StrEnum):
    """Public service-level label. Higher is a stronger capability class."""

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"
    L6 = "L6"
    L7 = "L7"
    L8 = "L8"
    L9 = "L9"
    L10 = "L10"
    L11 = "L11"
    L12 = "L12"
    L13 = "L13"
    L14 = "L14"
    L15 = "L15"
    L16 = "L16"
    L17 = "L17"
    L18 = "L18"
    L19 = "L19"
    L20 = "L20"
    L21 = "L21"
    L22 = "L22"
    L23 = "L23"
    L24 = "L24"
    L25 = "L25"
    L26 = "L26"
    L27 = "L27"
    L28 = "L28"
    L29 = "L29"
    L30 = "L30"

    @property
    def ordinal(self) -> int:
        return int(self.value[1:])

    def to_grade(self) -> Grade:
        """The constitutional grade this tier occupies."""
        return Grade.from_index(min(self.ordinal, _LAST_GRADE_ORDINAL))

    @classmethod
    def from_ordinal(cls, index: int) -> Self:
        if not 0 <= index < TIER_COUNT:
            raise ConfigurationError(
                f"service tier index must lie in [0, {TIER_COUNT - 1}], got {index}"
            )
        return cls(f"L{index}")

    @classmethod
    def from_grade(cls, grade: Grade) -> Self:
        return cls.from_ordinal(grade.ordinal)

    @classmethod
    def parse(cls, value: str) -> Self:
        text = str(value).strip()
        if not text:
            raise ConfigurationError("a service tier cannot be empty")
        upper = text.upper()
        if upper.startswith("L") and upper[1:].isdigit():
            return cls.from_ordinal(int(upper[1:]))
        grade = Grade.parse(text)
        return cls.from_grade(grade)


def parse_service_tier(value: str) -> ServiceTier | None:
    """Parse `L12` / `l12`. Returns `None` when the string is not a tier name.

    Model ids that happen to look like other things must not be coerced.
    """
    text = str(value).strip()
    if len(text) < 2 or text[0] not in "Ll":
        return None
    if not text[1:].isdigit():
        return None
    try:
        return ServiceTier.from_ordinal(int(text[1:]))
    except ConfigurationError:
        return None


ALL_TIERS: tuple[ServiceTier, ...] = tuple(ServiceTier)
