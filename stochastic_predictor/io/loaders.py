"""Ingestion pipeline helpers for IO policies."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from stochastic_predictor.api.types import InternalState, PredictorConfig, ProcessState
from stochastic_predictor.io.validators import (
    OutlierRejectedEvent,
    FrozenSignalAlarmEvent,
    StaleSignalEvent,
    compute_staleness_ns,
    detect_catastrophic_outlier,
    detect_frozen_signal,
    detect_frozen_recovery,  # V-MAJ-6: Import recovery detector
    is_stale,
)


@dataclass(frozen=True)
class IngestionDecision:
    accept_observation: bool
    suspend_jko_update: bool
    degraded_mode: bool
    freeze_kernel_d: bool
    staleness_ns: int
    events: List[object]


def evaluate_ingestion(
    state: InternalState,
    observation: ProcessState,
    now_ns: int,
    config: PredictorConfig,
) -> IngestionDecision:
    events: List[object] = []

    magnitude = float(np.asarray(observation.magnitude).reshape(-1)[0])
    signal_history = np.asarray(state.signal_history, dtype=np.float64)

    staleness_ns = compute_staleness_ns(observation.timestamp_ns, now_ns)
    stale = is_stale(staleness_ns, config.staleness_ttl_ns)
    if stale:
        events.append(
            StaleSignalEvent(
                timestamp_ns=observation.timestamp_ns,
                staleness_ns=staleness_ns,
                max_ttl_ns=config.staleness_ttl_ns,
            )
        )

    outlier = detect_catastrophic_outlier(
        magnitude,
        config.sigma_bound,
        config.sigma_val,
    )
    if outlier:
        events.append(
            OutlierRejectedEvent(
                timestamp_ns=observation.timestamp_ns,
                value=magnitude,
                sigma_bound=config.sigma_bound,
                sigma_val=config.sigma_val,
            )
        )

    history_with_new = np.concatenate([signal_history, np.array([magnitude], dtype=np.float64)])
    frozen = detect_frozen_signal(history_with_new, config.frozen_signal_min_steps)
    
    # V-MAJ-6: Check for frozen signal recovery
    # If frozen, but variance has recovered above threshold, lift the frozen flag
    in_recovery = False
    if frozen:
        # Use EMA variance from state as variance history proxy
        ema_variance_scalar = float(state.ema_variance)
        # Build variance history from residual buffer std
        residual_buffer = np.asarray(state.residual_buffer, dtype=np.float64)
        residual_variance = float(np.var(residual_buffer)) if residual_buffer.size > 0 else 1e-10
        
        # Check if variance has recovered: recent_var > ratio_threshold * baseline_var
        in_recovery = detect_frozen_recovery(
            variance_history=[residual_variance],  # Single recent measurement
            historical_variance=float(np.maximum(residual_variance, 1e-10)),
            ratio_threshold=config.frozen_signal_recovery_ratio,
            consecutive_steps=config.frozen_signal_recovery_steps,
        )
        
        # If in recovery, lift the frozen flag
        if in_recovery:
            frozen = False
    
    if frozen:
        events.append(
            FrozenSignalAlarmEvent(
                timestamp_ns=observation.timestamp_ns,
                value=magnitude,
                window=config.frozen_signal_min_steps,
            )
        )

    if outlier:
        return IngestionDecision(
            accept_observation=False,
            suspend_jko_update=True,
            degraded_mode=True,
            freeze_kernel_d=False,
            staleness_ns=staleness_ns,
            events=events,
        )

    return IngestionDecision(
        accept_observation=True,
        suspend_jko_update=bool(stale or frozen),
        degraded_mode=bool(stale or frozen),
        freeze_kernel_d=bool(frozen),
        staleness_ns=staleness_ns,
        events=events,
    )
