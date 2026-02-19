"""Core orchestration pipeline.

Executes kernel ensemble, applies JKO/Sinkhorn fusion, updates InternalState,
and emits PredictionResult with configuration-driven validation.
"""

from dataclasses import dataclass, replace
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from stochastic_predictor.api.state_buffer import atomic_state_update, reset_cusum_statistics
from stochastic_predictor.api.types import InternalState, KernelType, OperatingMode, PredictionResult, PredictorConfig, ProcessState
from stochastic_predictor.api.validation import validate_simplex
from stochastic_predictor.api.prng import RNG_SPLIT_COUNT
from stochastic_predictor.core.fusion import FusionResult, fuse_kernel_outputs
from stochastic_predictor.io.loaders import evaluate_ingestion
from stochastic_predictor.kernels import kernel_a_predict, kernel_b_predict, kernel_c_predict, kernel_d_predict
from stochastic_predictor.kernels.base import KernelOutput, validate_kernel_input


@dataclass(frozen=True)
class OrchestrationResult:
    """Outputs of a single orchestration step."""
    prediction: PredictionResult
    state: InternalState
    kernel_outputs: tuple[KernelOutput, KernelOutput, KernelOutput, KernelOutput]
    fusion: Optional[FusionResult]


def initialize_state(
    signal: Float[Array, "n"],
    timestamp_ns: int,
    rng_key: Array,
    config: PredictorConfig
) -> InternalState:
    """Initialize InternalState buffers from an initial signal."""
    min_length = config.base_min_signal_length
    is_valid, msg = validate_kernel_input(signal, min_length)
    if not is_valid:
        raise ValueError(msg)

    signal_history = signal[-min_length:]
    residual_buffer = jnp.zeros_like(signal_history)
    residual_window = jnp.zeros(config.residual_window_size)  # For kurtosis tracking
    rho = jnp.full((KernelType.N_KERNELS,), 1.0 / KernelType.N_KERNELS)

    return InternalState(
        signal_history=signal_history,
        residual_buffer=residual_buffer,
        residual_window=residual_window,
        rho=rho,
        cusum_g_plus=jnp.array(0.0),
        cusum_g_minus=jnp.array(0.0),
        grace_counter=0,
        adaptive_h_t=jnp.array(config.cusum_h),  # Initialize with static value
        ema_variance=jnp.array(0.0),
        kurtosis=jnp.array(0.0),
        holder_exponent=jnp.array(0.0),
        dgm_entropy=jnp.array(0.0),
        mode_collapse_consecutive_steps=0,  # V-MAJ-5: Initialize counter
        degraded_mode=False,
        emergency_mode=False,
        regime_changed=False,
        last_update_ns=timestamp_ns,
        rng_key=rng_key,
    )


def _run_kernels(
    signal: Float[Array, "n"],
    rng_key: Array,
    config: PredictorConfig,
    freeze_kernel_d: bool = False,
    ema_variance: Optional[Float[Array, ""]] = None,  # V-MAJ-1: Optional parameter for adaptive entropy threshold
) -> tuple[KernelOutput, KernelOutput, KernelOutput, KernelOutput]:
    """Execute kernels A-D with independent PRNG keys.
    
    Args:
        signal: Time series signal history.
        rng_key: JAX PRNG key.
        config: System configuration.
        freeze_kernel_d: If True, mark kernel D output as frozen (no weight update).
        ema_variance: Optional EWMA variance for V-MAJ-1 adaptive entropy threshold
    
    Returns:
        Tuple of KernelOutput from kernels A, B, C, D.
        
    Note: freeze_kernel_d does not skip computation; it marks the output
          so downstream fusion logic can handle it appropriately.
    """
    key_a, key_b, key_c, key_d = jax.random.split(rng_key, KernelType.N_KERNELS)
    output_a = kernel_a_predict(signal, key_a, config)
    output_b = kernel_b_predict(signal, key_b, config, ema_variance=ema_variance)  # V-MAJ-1: Pass ema_variance
    output_c = kernel_c_predict(signal, key_c, config)
    output_d = kernel_d_predict(signal, key_d, config)
    
    # Mark kernel D if frozen (no state update to its weight)
    if freeze_kernel_d:
        output_d = KernelOutput(
            prediction=output_d.prediction,
            confidence=output_d.confidence,
            metadata={**output_d.metadata, "frozen": True},
        )
    
    return output_a, output_b, output_c, output_d


def _compute_mode(
    degraded: bool,
    emergency: bool
) -> str:
    if emergency:
        return OperatingMode.EMERGENCY
    if degraded:
        return OperatingMode.ROBUST
    return OperatingMode.STANDARD


def orchestrate_step(
    signal: Float[Array, "n"],
    timestamp_ns: int,
    state: InternalState,
    config: PredictorConfig,
    observation: ProcessState,
    now_ns: int,
) -> OrchestrationResult:
    """Run a single orchestration step with IO ingestion validation."""
    min_length = config.base_min_signal_length
    is_valid, msg = validate_kernel_input(signal, min_length)
    if not is_valid:
        raise ValueError(msg)

    # Evaluate ingestion decision (outlier, frozen signal, staleness checks)
    ingestion_decision = evaluate_ingestion(
        state=state,
        observation=observation,
        now_ns=now_ns,
        config=config,
    )

    delta_ns = timestamp_ns - state.last_update_ns
    staleness_degraded = bool(delta_ns > config.staleness_ttl_ns)
    ingestion_degraded = ingestion_decision.degraded_mode

    # Use ingestion decision flags to override or augment degraded mode
    reject_observation = not ingestion_decision.accept_observation
    degraded_mode = bool(staleness_degraded or ingestion_degraded or reject_observation)

    kernel_outputs = _run_kernels(
        signal, 
        state.rng_key, 
        config, 
        freeze_kernel_d=ingestion_decision.freeze_kernel_d,
        ema_variance=state.ema_variance  # V-MAJ-1: Pass for adaptive entropy threshold
    )

    if degraded_mode or ingestion_decision.suspend_jko_update:
        predictions = jnp.array([ko.prediction for ko in kernel_outputs]).reshape(-1)
        updated_weights = state.rho
        fused_prediction = jnp.sum(updated_weights * predictions)
        fusion = None
        sinkhorn_converged = jnp.array(False)
        free_energy = jnp.array(0.0)
        sinkhorn_epsilon = jnp.array(0.0)
    else:
        fusion = fuse_kernel_outputs(
            kernel_outputs=kernel_outputs,
            current_weights=state.rho,
            ema_variance=state.ema_variance,
            config=config,
        )
        updated_weights = fusion.updated_weights
        fused_prediction = fusion.fused_prediction
        sinkhorn_converged = jnp.asarray(fusion.sinkhorn_converged)
        free_energy = jnp.asarray(fusion.free_energy)
        sinkhorn_epsilon = jnp.asarray(fusion.sinkhorn_epsilon)

    PredictionResult.validate_simplex(updated_weights, config.validation_simplex_atol)
    is_simplex, msg = validate_simplex(updated_weights, config.validation_simplex_atol, "weights")
    if not is_simplex:
        raise ValueError(msg)

    current_value = signal[-1]
    residual = jnp.abs(current_value - fused_prediction)

    # If observation is rejected, skip state update entirely (return unchanged state)
    if reject_observation:
        updated_state = state
        regime_change_detected = False
    else:
        updated_state, regime_change_detected = atomic_state_update(
            state=state,
            new_signal=current_value,
            new_residual=residual,
            config=config,
        )

    # CAPA 1: Entropy reset on regime change (CUSUM alarm)
    # Mandato: ρ → Softmax(0) = uniform simplex [0.25, 0.25, 0.25, 0.25]
    # References: MIGRATION_AUTOTUNING_v1.0.md §2.1, Theory.tex §3.4
    uniform_simplex = jnp.full((KernelType.N_KERNELS,), 1.0 / KernelType.N_KERNELS)
    
    # Apply entropy reset if regime changed AND not in grace period already
    # (grace_counter starts at 0, gets set to grace_period_steps on alarm)
    entropy_reset_triggered = regime_change_detected and (state.grace_counter == 0)
    
    # During grace period: freeze weights (no JKO update)
    # After grace period or normal operation: use fused weights
    in_grace_period = updated_state.grace_counter > 0
    
    if reject_observation:
        final_rho = state.rho
    elif entropy_reset_triggered:
        final_rho = uniform_simplex  # Max entropy reset
    elif in_grace_period:
        final_rho = state.rho  # Freeze during grace period
    else:
        final_rho = updated_weights  # Normal JKO update

    updated_state = replace(
        updated_state,
        rho=final_rho,
        holder_exponent=jnp.asarray(kernel_outputs[KernelType.KERNEL_A].metadata.get("holder_exponent", 0.0)),  # V-MAJ-2: State update
        dgm_entropy=jnp.asarray(kernel_outputs[KernelType.KERNEL_B].metadata.get("entropy_dgm", 0.0)),
        last_update_ns=timestamp_ns if not reject_observation else state.last_update_ns,
        rng_key=jax.random.split(state.rng_key, RNG_SPLIT_COUNT)[1],
    )

    # Grace period decay during normal operations (CUSUM reset is done in update_cusum_statistics)
    # Note: grace_counter is already set in update_cusum_statistics when alarm triggers
    # Here we just handle the decay (no need to touch rho again, already handled above)
    grace_counter = updated_state.grace_counter
    if grace_counter > 0:
        grace_counter -= 1
        updated_state = replace(updated_state, grace_counter=grace_counter)

    # V-MAJ-5: Mode Collapse Detection (consecutive low-entropy steps)
    # If dgm_entropy < threshold → increment counter, else reset
    dgm_entropy_threshold = config.entropy_threshold
    low_entropy = float(updated_state.dgm_entropy) < dgm_entropy_threshold
    mode_collapse_counter = updated_state.mode_collapse_consecutive_steps
    
    if low_entropy:
        mode_collapse_counter = mode_collapse_counter + 1
    else:
        mode_collapse_counter = 0
    
    # Warning threshold: emit if counter exceeds config.entropy_window steps (reuse as window)
    mode_collapse_warning_threshold = max(10, config.entropy_window // 10)  # Default 10 or 1/10 of entropy_window
    mode_collapse_warning = bool(mode_collapse_counter >= mode_collapse_warning_threshold)
    
    updated_state = replace(
        updated_state,
        mode_collapse_consecutive_steps=mode_collapse_counter
    )

    emergency_mode = bool(updated_state.holder_exponent < config.holder_threshold)
    
    # Override emergency mode if observation was rejected
    if reject_observation:
        emergency_mode = True

    # Use adaptive threshold h_t (not static cusum_h) for telemetry
    prediction = PredictionResult(
        predicted_next=jnp.atleast_1d(fused_prediction),
        holder_exponent=updated_state.holder_exponent,
        cusum_drift=updated_state.cusum_g_plus,
        distance_to_collapse=jnp.atleast_1d(updated_state.adaptive_h_t - updated_state.cusum_g_plus),
        free_energy=jnp.atleast_1d(free_energy),
        kurtosis=updated_state.kurtosis,
        dgm_entropy=updated_state.dgm_entropy,
        adaptive_threshold=updated_state.adaptive_h_t,
        weights=final_rho,  # Use final_rho (already computed above)
        sinkhorn_converged=jnp.atleast_1d(sinkhorn_converged),
        degraded_inference_mode=degraded_mode,
        emergency_mode=emergency_mode,
        regime_change_detected=regime_change_detected,
        mode_collapse_warning=mode_collapse_warning,  # V-MAJ-5: Use calculated warning
        mode=_compute_mode(degraded_mode, emergency_mode),
    )

    updated_state = replace(
        updated_state,
        degraded_mode=degraded_mode,
        emergency_mode=emergency_mode,
        regime_changed=regime_change_detected,
    )

    return OrchestrationResult(
        prediction=prediction,
        state=updated_state,
        kernel_outputs=kernel_outputs,
        fusion=fusion,
    )
