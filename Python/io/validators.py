"""IO validation policies for ingestion safeguards.

Nyquist Injection Frequency:
    The minimum injection frequency is enforced via besov_nyquist_interval
    parameter. This ensures that the signal sampling rate respects the Nyquist
    theorem and prevents aliasing artifacts in downstream processing.

    Reference: IO.tex §3.1.2 - Minimum Injection Frequency (Nyquist Soft Limit)
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OutlierRejectedEvent:
    timestamp_ns: int
    value: float
    sigma_bound: float
    sigma_val: float


@dataclass(frozen=True)
class FrozenSignalAlarmEvent:
    timestamp_ns: int
    value: float
    window: int


@dataclass(frozen=True)
class StaleSignalEvent:
    timestamp_ns: int
    staleness_ns: int
    max_ttl_ns: int


def detect_catastrophic_outlier(
    value: float,
    sigma_bound: float,
    sigma_val: float,
) -> bool:
    magnitude = float(np.asarray(value))
    return abs(magnitude) > (sigma_bound * sigma_val)


def detect_frozen_signal(
    values: Iterable[float],
    min_steps: int,
) -> bool:
    buffer = np.asarray(list(values), dtype=np.float64)
    if buffer.size < min_steps:
        return False
    window = buffer[-min_steps:]
    return bool(np.all(window == window[0]))


def compute_staleness_ns(timestamp_ns: int, now_ns: int) -> int:
    return int(now_ns - timestamp_ns)


def is_stale(staleness_ns: int, max_ttl_ns: int) -> bool:
    return staleness_ns > max_ttl_ns


def detect_frozen_recovery(
    variance_history: Iterable[float],
    historical_variance: float,
    ratio_threshold: float,
    consecutive_steps: int,
) -> bool:
    history = np.asarray(list(variance_history), dtype=np.float64)
    if history.size < consecutive_steps or historical_variance <= 0.0:
        return False
    recent = history[-consecutive_steps:]
    return bool(np.all(recent > (ratio_threshold * historical_variance)))


def validate_besov_nyquist_interval(
    injection_frequency_hz: float,
    besov_nyquist_interval: float,
) -> bool:
    """Enforce minimum Nyquist sampling per IO.tex §3.1.2.

    Validates that signal injection frequency meets Besov-space Nyquist
    requirement for preventing aliasing in stochastic observations.
    """
    return injection_frequency_hz >= (1.0 / besov_nyquist_interval)
