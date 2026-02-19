"""Non-blocking telemetry buffering and parity hash utilities."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional
import hashlib
import threading
import numpy as np


@dataclass(frozen=True)
class TelemetryRecord:
    step: int
    payload: dict


class TelemetryBuffer:
    """Thread-safe telemetry ring buffer with explicit capacity injection.
    
    Design: Capacity must be injected from PredictorConfig to comply with
    the zero-heuristics policy (no implicit buffer defaults).
    """
    def __init__(self, capacity: int) -> None:
        self._buffer: Deque[TelemetryRecord] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def enqueue(self, record: TelemetryRecord) -> bool:
        with self._lock:
            before = len(self._buffer)
            self._buffer.append(record)
            return len(self._buffer) >= before

    def drain(self) -> list[TelemetryRecord]:
        with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
            return items

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _to_float64_bytes(values: Iterable[float]) -> bytes:
    array = np.asarray(list(values), dtype=np.float64)
    return array.tobytes(order="C")


def parity_hashes(rho: Iterable[float], ot_cost: float) -> dict:
    rho_bytes = _to_float64_bytes(rho)
    cost_bytes = _to_float64_bytes([ot_cost])
    return {
        "rho_sha256": _hash_bytes(rho_bytes),
        "ot_cost_sha256": _hash_bytes(cost_bytes),
    }


def should_emit_hash(step: int, interval_steps: int) -> bool:
    return interval_steps > 0 and (step % interval_steps == 0)
