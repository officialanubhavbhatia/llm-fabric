"""How many tokens the context is allowed to occupy.

The window is not the budget. A deployment that accepts 8,192 tokens must still
leave room for the completion, and the fabric counts tokens with a heuristic
rather than the model's own tokenizer, so the plan also carries a margin sized
to the counter's inaccuracy.

That margin is the honest part of this module. An approximate counter that
undercounts by a few percent will happily certify a prompt that the provider
then rejects, and the failure surfaces as a confusing upstream error rather than
a budget the fabric got wrong. Reserving headroom converts an unpredictable
provider rejection into a predictable local decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from llm_fabric.errors import ConfigurationError
from llm_fabric.serving.tokens import approximate_token_count

#: Fraction of the window held back when the counter is approximate. The value
#: is a judgement, not a measurement: it is roughly twice the error the
#: character-per-token heuristic showed on this repository's own fixtures, and
#: it is configurable because a different corpus will behave differently.
DEFAULT_SAFETY_FRACTION = 0.02

#: Floor for the same margin, so a small window still keeps usable headroom.
DEFAULT_SAFETY_FLOOR_TOKENS = 32


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens in text.

    `is_exact` is what the budget planner keys its safety margin on, so an
    implementation that wraps a real tokenizer can claim the headroom back.
    """

    @property
    def name(self) -> str: ...

    @property
    def is_exact(self) -> bool: ...

    def count(self, text: str) -> int: ...


class ApproximateTokenCounter:
    """The dependency-free default: characters divided by a constant.

    Declares itself inexact, which is the whole reason it is safe to use as a
    default. Everything downstream that reports a token figure derived from it
    is labelled an estimate.
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        return "approximate-chars-v1"

    @property
    def is_exact(self) -> bool:
        return False

    def count(self, text: str) -> int:
        return approximate_token_count(text)


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """The token allowance for one compilation, and where it went.

    Every subtraction from the window is named. A caller who wants to know why
    only 3,000 of an 8,192-token window were usable can read it off this object
    rather than reconstructing the arithmetic.
    """

    context_window: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    framing_overhead_tokens: int
    counter_name: str
    counter_is_exact: bool

    def __post_init__(self) -> None:
        if self.context_window <= 0:
            raise ConfigurationError("a context budget requires a positive context window")
        for label, value in (
            ("reserved output", self.reserved_output_tokens),
            ("safety margin", self.safety_margin_tokens),
            ("framing overhead", self.framing_overhead_tokens),
        ):
            if value < 0:
                raise ConfigurationError(f"the {label} allowance cannot be negative")

    @property
    def available_tokens(self) -> int:
        """Tokens the context blocks may actually occupy.

        Clamped at zero: a window smaller than its own reservations is a
        configuration problem, and the compiler reports it as required blocks
        not fitting rather than as a negative allowance.
        """
        spent = self.reserved_output_tokens + self.safety_margin_tokens
        return max(0, self.context_window - spent - self.framing_overhead_tokens)

    @property
    def is_estimated(self) -> bool:
        """True when every token figure under this budget is an estimate."""
        return not self.counter_is_exact

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_window": self.context_window,
            "reserved_output_tokens": self.reserved_output_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "framing_overhead_tokens": self.framing_overhead_tokens,
            "available_tokens": self.available_tokens,
            "counter": self.counter_name,
            "counts_are_estimates": self.is_estimated,
        }


def plan_budget(
    *,
    context_window: int,
    reserved_output_tokens: int,
    counter: TokenCounter,
    block_count: int = 0,
    framing_overhead_per_block: int = 4,
    safety_fraction: float = DEFAULT_SAFETY_FRACTION,
    safety_floor_tokens: int = DEFAULT_SAFETY_FLOOR_TOKENS,
) -> ContextBudget:
    """Build the budget for one compilation.

    The safety margin collapses to zero for an exact counter. There is no reason
    to hold back headroom against a tokenizer that is right, and doing so would
    quietly shrink every window by two percent for no benefit.
    """
    if context_window <= 0:
        raise ConfigurationError("a context budget requires a positive context window")
    if reserved_output_tokens < 0:
        raise ConfigurationError("reserved output tokens cannot be negative")
    if not 0.0 <= safety_fraction < 1.0:
        raise ConfigurationError("the safety fraction must lie in [0, 1)")

    if counter.is_exact:
        margin = 0
    else:
        margin = max(safety_floor_tokens, math.ceil(context_window * safety_fraction))

    return ContextBudget(
        context_window=context_window,
        reserved_output_tokens=reserved_output_tokens,
        safety_margin_tokens=margin,
        framing_overhead_tokens=max(0, block_count) * max(0, framing_overhead_per_block),
        counter_name=counter.name,
        counter_is_exact=counter.is_exact,
    )
