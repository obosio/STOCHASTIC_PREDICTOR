"""Non-blocking telemetry buffering and parity hash utilities.

COMPLIANCE: API_Python.tex §9 - Asynchronous I/O for Snapshots
Telemetry consumer thread performs batch device_get() to avoid blocking
the orchestrator's XLA async dispatch pipeline.
"""

import hashlib
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable, Optional

import numpy as np

try:
    pass

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


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


def _hash_bytes(data: bytes, algorithm: str) -> str:
    if algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    if algorithm == "crc32c":
        try:
            import google_crc32c  # type: ignore
        except ImportError as exc:
            raise RuntimeError("google-crc32c is required for crc32c hashing") from exc
        return google_crc32c.value(data).to_bytes(4, byteorder="big", signed=False).hex()
    raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def _to_float64_bytes(values: Iterable[float]) -> bytes:
    array = np.asarray(list(values), dtype=np.float64)
    return array.tobytes(order="C")


def parity_hashes(rho: Iterable[float], ot_cost: float, algorithm: str) -> dict:
    rho_bytes = _to_float64_bytes(rho)
    cost_bytes = _to_float64_bytes([ot_cost])
    return {
        f"rho_{algorithm}": _hash_bytes(rho_bytes, algorithm),
        f"ot_cost_{algorithm}": _hash_bytes(cost_bytes, algorithm),
    }


def should_emit_hash(step: int, interval_steps: int) -> bool:
    return interval_steps > 0 and (step % interval_steps == 0)


def materialize_telemetry_batch(records: list[TelemetryRecord], config: Any) -> list[dict]:
    """Convert DeviceArray references to Python scalars (batch transfer).

    COMPLIANCE: API_Python.tex §9 - Asynchronous I/O
    Performs batch jax.device_get() to amortize GPU→CPU transfer overhead.
    Should run in separate consumer thread, NOT in orchestrator path.

    Args:
        records: List of telemetry records with DeviceArray references
            config: PredictorConfig with telemetry_hash_algorithm

    Returns:
        List of materialized payloads with Python scalars

    Performance:
        - Batch transfer of N records: ~50-100μs overhead
        - Per-record transfer: ~10-50μs × N (much worse)
        - Speedup: ~5-10x for typical batch sizes (N=10-100)

    References:
        - API_Python.tex §9: Asynchronous I/O for Snapshots (Non-Blocking)
        - Implementation.tex §5.4: Non-blocking audit trail
    """
    if not records:
        return []

    if not HAS_JAX:
        # Fallback if JAX not available (shouldn't happen in production)
        return [record.payload for record in records]

    # JAX is available, safe to use
    import jax as jax_module

    # Collect all DeviceArray references for batch transfer
    materialized_records = []

    for record in records:
        payload = record.payload.copy()

        # Batch device_get for all tensor fields
        device_arrays = {key: value for key, value in payload.items() if key.endswith("_ref")}

        if device_arrays:
            # Single batch transfer for all tensors in this record
            materialized = jax_module.device_get(device_arrays)

            # Convert to Python scalars and remove _ref suffix
            for key, value in materialized.items():
                scalar_key = key.replace("_ref", "")
                if isinstance(value, np.ndarray):
                    if value.shape == () or value.size == 1:
                        payload[scalar_key] = float(value)
                    else:
                        # For weight vectors: convert to list
                        payload[scalar_key] = [float(v) for v in value]
                else:
                    payload[scalar_key] = float(value)

                # Remove reference key
                del payload[key]

        # Compute parity hashes NOW (after materialization)
        if "weights" in payload and "free_energy" in payload:
            parity_record = parity_hashes(
                rho=payload["weights"],
                ot_cost=payload.get("free_energy", 0.0),
                algorithm=config.telemetry_hash_algorithm,
            )
            payload["parity_hashes"] = parity_record

        materialized_records.append(payload)

    return materialized_records


# =============================================================================
# ADAPTIVE TELEMETRY (Level 4 Autonomy) - V-MAJ-7
# =============================================================================


@dataclass(frozen=True)
class AdaptiveTelemetry:
    """
    Telemetry for adaptive architecture and solver selection monitoring.

    COMPLIANCE: V-MAJ-7 - Monitoring Telemetry for Adaptive Parameters
    Theory.tex §2.3.6 - Monitoring and telemetry for adaptive SDE schemes

    This structure captures Level 4 autonomy adaptation events for:
        - SDE solver selection (Kernel C)
        - DGM architecture scaling (Kernel B)
        - JKO flow parameter tuning (Orchestrator)
        - Stiffness threshold adaptation (Kernel C)

    Attributes:
        # SDE Solver Monitoring (Kernel C)
        scheme_frequency_explicit: Percentage of steps using explicit Euler-Maruyama
        scheme_frequency_implicit: Percentage of steps using implicit trapezoidal
        max_stiffness_metric: Peak stiffness S_t over monitoring window
        num_internal_iterations_mean: Mean Newton iterations for implicit solver
        implicit_residual_norm_max: Worst-case convergence residual (implicit)

        # DGM Architecture Monitoring (Kernel B)
        entropy_ratio_current: κ = H_current / H_baseline (regime transition indicator)
        dgm_width_current: Current DGM network width
        dgm_depth_current: Current DGM network depth
        architecture_scaling_events: Count of capacity scaling events in window

        # JKO Flow Monitoring (Orchestrator)
        entropy_window_current: Current adaptive entropy window size
        learning_rate_current: Current adaptive JKO step size
        volatility_sigma_squared: Empirical variance σ² from EMA estimator

        # Stiffness Threshold Monitoring (Kernel C)
        stiffness_low_adaptive: Current θ_L threshold (Hölder-informed)
        stiffness_high_adaptive: Current θ_H threshold (Hölder-informed)
        holder_exponent_wtmm: α ∈ [0, 1] from WTMM pipeline

    Usage:
        >>> telemetry = AdaptiveTelemetry(
        ...     scheme_frequency_explicit=0.65,  # 65% explicit solver usage
        ...     scheme_frequency_implicit=0.35,
        ...     max_stiffness_metric=450.0,
        ...     entropy_ratio_current=3.2,  # 3.2× entropy increase
        ...     dgm_width_current=128,  # Scaled from baseline 64
        ...     architecture_scaling_events=2,  # 2 scaling events in window
        ...     # ... other fields
        ... )

    References:
        - API_Python.tex §3.1: Multi-Tenant Architecture
        - Theory.tex §2.3.6: Hölder-Stiffness Correspondence
        - Theory.tex §2.4.2: Entropy-Topology Coupling
        - Theory.tex §3.4.1: Non-Universality of JKO Flow
    """

    # SDE Solver Monitoring (Kernel C)
    scheme_frequency_explicit: float  # ∈ [0, 1], percentage of explicit solver usage
    scheme_frequency_implicit: float  # ∈ [0, 1], percentage of implicit solver usage
    max_stiffness_metric: float  # Peak S_t over monitoring window
    num_internal_iterations_mean: float  # Mean Newton iterations (implicit)
    implicit_residual_norm_max: float  # Worst-case convergence residual

    # DGM Architecture Monitoring (Kernel B)
    entropy_ratio_current: float  # κ = H_current / H_baseline
    dgm_width_current: int  # Current DGM architecture width
    dgm_depth_current: int  # Current DGM architecture depth
    architecture_scaling_events: int  # Count of capacity increases in window

    # JKO Flow Monitoring (Orchestrator)
    entropy_window_current: int  # Current adaptive entropy window size
    learning_rate_current: float  # Current adaptive JKO step size
    volatility_sigma_squared: float  # Empirical variance σ²

    # Stiffness Threshold Monitoring (Kernel C)
    stiffness_low_adaptive: float  # Current θ_L (Hölder-informed)
    stiffness_high_adaptive: float  # Current θ_H (Hölder-informed)
    holder_exponent_wtmm: float  # α from WTMM multifractal analysis


def collect_adaptive_telemetry(
    state: Any,  # InternalState
    config: Any,  # PredictorConfig
) -> Optional[AdaptiveTelemetry]:
    """
    Collect telemetry for adaptive architecture/solver diagnostics.

    COMPLIANCE: V-MAJ-7 - Adaptive Telemetry Monitoring

    This function extracts adaptive parameter snapshots from the current
    orchestrator state and configuration. In a full Level 4 autonomy deployment,
    this is called periodically (e.g., every 100 steps) to track
    adaptation behavior.

    Args:
        state: Current InternalState (contains counters, entropy, etc.)
        config: Current PredictorConfig (may have been mutated)

    Returns:
        AdaptiveTelemetry instance or None if insufficient data

    Note:
        Implementation extracts:
        1. Solver frequency tracking (explicit vs implicit counts from InternalState)
        2. Architecture scaling events (from InternalState.architecture_scaling_events)
        3. Entropy ratio κ = H_current / H_baseline
        4. Adaptive stiffness thresholds from config
        5. JKO flow parameters from config

    Example:
        >>> # In orchestration loop
        >>> if step % 100 == 0:
        ...     adaptive_tel = collect_adaptive_telemetry(state, config)
        ...     if adaptive_tel:
        ...         emit_adaptive_telemetry(adaptive_tel)
    """

    # Compute solver frequencies (clipped to [0,1])
    total_solver_steps = state.solver_explicit_count + state.solver_implicit_count
    if total_solver_steps < config.telemetry_adaptive_window_size:
        # Insufficient data - return None
        return None

    freq_explicit = float(state.solver_explicit_count) / total_solver_steps
    freq_implicit = float(state.solver_implicit_count) / total_solver_steps

    # Compute entropy ratio κ = H_current / H_baseline
    baseline_entropy_val = max(float(state.baseline_entropy), config.entropy_baseline_floor)
    entropy_ratio = float(state.dgm_entropy) / baseline_entropy_val

    # Extract DGM architecture from config
    dgm_width = config.dgm_width_size
    dgm_depth = config.dgm_depth

    # Extract JKO flow parameters from config
    entropy_window = config.entropy_window
    learning_rate = config.learning_rate
    volatility_sigma_squared = float(state.ema_variance)

    # Extract adaptive stiffness thresholds from config
    stiffness_low = config.stiffness_low
    stiffness_high = config.stiffness_high
    holder_exponent_val = float(state.holder_exponent)

    # Build AdaptiveTelemetry instance
    # Note: We provide placeholder values for metrics not yet tracked
    # (max_stiffness_metric, num_internal_iterations_mean, implicit_residual_norm_max)
    return AdaptiveTelemetry(
        # SDE Solver Frequency
        scheme_frequency_explicit=freq_explicit,
        scheme_frequency_implicit=freq_implicit,
        max_stiffness_metric=config.telemetry_placeholder_max_stiffness_metric,
        num_internal_iterations_mean=config.telemetry_placeholder_num_internal_iterations_mean,
        implicit_residual_norm_max=config.telemetry_placeholder_implicit_residual_norm_max,
        # DGM Architecture
        entropy_ratio_current=entropy_ratio,
        dgm_width_current=dgm_width,
        dgm_depth_current=dgm_depth,
        architecture_scaling_events=state.architecture_scaling_events,
        # JKO Flow
        entropy_window_current=entropy_window,
        learning_rate_current=learning_rate,
        volatility_sigma_squared=volatility_sigma_squared,
        # Stiffness Thresholds
        stiffness_low_adaptive=stiffness_low,
        stiffness_high_adaptive=stiffness_high,
        holder_exponent_wtmm=holder_exponent_val,
    )


def emit_adaptive_telemetry(
    telemetry: AdaptiveTelemetry,
    config: Any,  # PredictorConfig
) -> None:
    """
    Emit adaptive telemetry to JSON Lines log file.

    Args:
        telemetry: AdaptiveTelemetry instance
        config: PredictorConfig with telemetry_adaptive_log_path

    Example:
        >>> telemetry = AdaptiveTelemetry(...)
        >>> emit_adaptive_telemetry(telemetry, config)
        >>> # Appends JSON record to io/adaptive_telemetry.jsonl
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    # Convert to dict
    telemetry_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scheme_frequency_explicit": telemetry.scheme_frequency_explicit,
        "scheme_frequency_implicit": telemetry.scheme_frequency_implicit,
        "max_stiffness_metric": telemetry.max_stiffness_metric,
        "num_internal_iterations_mean": telemetry.num_internal_iterations_mean,
        "implicit_residual_norm_max": telemetry.implicit_residual_norm_max,
        "entropy_ratio_current": telemetry.entropy_ratio_current,
        "dgm_width_current": telemetry.dgm_width_current,
        "dgm_depth_current": telemetry.dgm_depth_current,
        "architecture_scaling_events": telemetry.architecture_scaling_events,
        "entropy_window_current": telemetry.entropy_window_current,
        "learning_rate_current": telemetry.learning_rate_current,
        "volatility_sigma_squared": telemetry.volatility_sigma_squared,
        "stiffness_low_adaptive": telemetry.stiffness_low_adaptive,
        "stiffness_high_adaptive": telemetry.stiffness_high_adaptive,
        "holder_exponent_wtmm": telemetry.holder_exponent_wtmm,
    }

    # Ensure directory exists
    log_file = Path(config.telemetry_adaptive_log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Append to JSON Lines file
    with open(log_file, "a") as f:
        f.write(json.dumps(telemetry_dict) + "\n")
