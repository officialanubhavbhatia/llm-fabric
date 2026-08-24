"""Confidence calibration for IntentOS.

Layer confidences are heuristic scores, not probabilities. This module fits a
one-parameter temperature on a **validation** split and applies it at serving
time. It must not be fitted on the frozen test set.

A temperature of 1.0 is a no-op. Fitting is refused when the val set is too
small to say anything honest.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from llm_fabric.intent.benchmark import CaseOutcome

MIN_FIT_CASES = 20


@dataclass(frozen=True, slots=True)
class TemperatureScaler:
    """Maps a raw score in [0, 1] onto a calibrated score in [0, 1]."""

    temperature: float = 1.0
    version: str = "identity"

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

    def calibrate(self, confidence: float) -> float:
        if self.temperature == 1.0:
            return max(0.0, min(1.0, confidence))
        # Treat the score as a two-class logit around 0.5, then re-sigmoid.
        clipped = min(1.0 - 1e-6, max(1e-6, confidence))
        logit = math.log(clipped / (1.0 - clipped))
        scaled = logit / self.temperature
        return 1.0 / (1.0 + math.exp(-scaled))


def fit_temperature(outcomes: Sequence[CaseOutcome]) -> TemperatureScaler:
    """Grid-search a temperature that minimises ECE on `outcomes`.

    Returns identity when there is not enough labelled data. Never claims a
    calibration it did not measure.
    """
    if len(outcomes) < MIN_FIT_CASES:
        return TemperatureScaler()

    best = TemperatureScaler()
    best_error = _ece(outcomes, best)
    for step in range(5, 41):
        candidate = TemperatureScaler(temperature=step / 10.0, version=f"temp-{step / 10.0:.1f}")
        error = _ece(outcomes, candidate)
        if error < best_error:
            best, best_error = candidate, error
    return best


def _ece(outcomes: Sequence[CaseOutcome], scaler: TemperatureScaler, bins: int = 10) -> float:
    buckets: list[list[float]] = [[] for _ in range(bins)]
    correct: list[list[int]] = [[] for _ in range(bins)]
    for outcome in outcomes:
        confidence = scaler.calibrate(outcome.confidence)
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append(confidence)
        correct[index].append(1 if outcome.correct else 0)
    error = 0.0
    n = len(outcomes)
    for confs, hits in zip(buckets, correct, strict=True):
        if not confs:
            continue
        error += (len(confs) / n) * abs(sum(confs) / len(confs) - sum(hits) / len(hits))
    return error
