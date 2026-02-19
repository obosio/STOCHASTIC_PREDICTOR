"""IO validation policies for ingestion safeguards."""

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
