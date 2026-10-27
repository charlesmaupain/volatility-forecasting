"""Temporal alignment on an audited consecutive-session variance series.

Callers must validate trading calendars and publication timestamps upstream.
"""
from dataclasses import dataclass
from math import isfinite, log
from typing import Sequence


@dataclass(frozen=True)
class Example:
    origin: int
    target_end: int
    features: tuple[float, float, float]
    target: float


def make_examples(variance: Sequence[float], horizon: int = 1) -> list[Example]:
    """Trailing 1/5/22-session features; sum of next h session variances."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    values = [float(value) for value in variance]
    if any(not isfinite(value) or value < 0 for value in values):
        raise ValueError("variance must be finite and nonnegative")
    return [Example(t, t + horizon,
                    (values[t], sum(values[t-4:t+1]) / 5,
                     sum(values[t-21:t+1]) / 22),
                    sum(values[t+1:t+horizon+1]))
            for t in range(21, len(values) - horizon)]


def available_training(examples: Sequence[Example], fit_origin: int) -> list[Example]:
    """Labels ending at t are available after session t; later ones are not."""
    return [row for row in examples
            if row.origin < fit_origin and row.target_end <= fit_origin]


def qlike(observed: float, forecast: float) -> float:
    """QLIKE up to an observation-only term; allows zero observed variance."""
    if not isfinite(observed) or observed < 0:
        raise ValueError("observed variance must be finite and nonnegative")
    if not isfinite(forecast) or forecast <= 0:
        raise ValueError("forecast variance must be finite and positive")
    return log(forecast) + observed / forecast
