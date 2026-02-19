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
from stochastic_predictor.api.types import InternalState, KernelType, OperatingMode, PredictionResult, PredictorConfig
from stochastic_predictor.api.validation import validate_simplex
from stochastic_predictor.core.fusion import FusionResult, fuse_kernel_outputs
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
    rho = jnp.full((KernelType.N_KERNELS,), 1.0 / KernelType.N_KERNELS)

    return InternalState(
        signal_history=signal_history,
        residual_buffer=residual_buffer,
        rho=rho,
        cusum_g_plus=jnp.array(0.0),
        cusum_g_minus=jnp.array(0.0),
        grace_counter=0,
        ema_variance=jnp.array(0.0),
        kurtosis=jnp.array(0.0),
        holder_exponent=jnp.array(0.0),
        dgm_entropy=jnp.array(0.0),
        degraded_mode=False,
        emergency_mode=False,
        regime_changed=False,
        last_update_ns=timestamp_ns,
        rng_key=rng_key,
    )


def _run_kernels(
    signal: Float[Array, "n"],
    rng_key: Array,
    config: PredictorConfig
) -> tuple[KernelOutput, KernelOutput, KernelOutput, KernelOutput]:
    """Execute kernels A-D with independent PRNG keys."""
    key_a, key_b, key_c, key_d = jax.random.split(rng_key, KernelType.N_KERNELS)
    output_a = kernel_a_predict(signal, key_a, config)
    output_b = kernel_b_predict(signal, key_b, config)
    output_c = kernel_c_predict(signal, key_c, config)
    output_d = kernel_d_predict(signal, key_d, config)
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
    config: PredictorConfig
) -> OrchestrationResult:
    """Run a single orchestration step and update InternalState."""
    min_length = config.base_min_signal_length
    is_valid, msg = validate_kernel_input(signal, min_length)
    if not is_valid:
        raise ValueError(msg)

    delta_ns = timestamp_ns - state.last_update_ns
    degraded_mode = bool(delta_ns > config.staleness_ttl_ns)

    kernel_outputs = _run_kernels(signal, state.rng_key, config)

    if degraded_mode:
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

    updated_state = atomic_state_update(
        state=state,
        new_signal=current_value,
        new_residual=residual,
        cusum_k=config.cusum_k,
        volatility_alpha=config.volatility_alpha,
    )

    updated_state = replace(
        updated_state,
        rho=updated_weights,
        dgm_entropy=jnp.asarray(kernel_outputs[KernelType.KERNEL_B].metadata.get("entropy_dgm", 0.0)),
        last_update_ns=timestamp_ns,
        rng_key=jax.random.split(state.rng_key, 2)[1],
    )

    regime_change_detected = bool(updated_state.cusum_g_plus > config.cusum_h)
    grace_counter = updated_state.grace_counter

    if regime_change_detected:
        updated_state = reset_cusum_statistics(updated_state)
        grace_counter = config.grace_period_steps

    if grace_counter > 0:
        grace_counter -= 1
        updated_state = replace(updated_state, rho=state.rho)

    updated_state = replace(updated_state, grace_counter=grace_counter)

    emergency_mode = bool(updated_state.holder_exponent < config.holder_threshold)

    prediction = PredictionResult(
        predicted_next=jnp.atleast_1d(fused_prediction),
        holder_exponent=updated_state.holder_exponent,
        cusum_drift=updated_state.cusum_g_plus,
        distance_to_collapse=jnp.atleast_1d(config.cusum_h - updated_state.cusum_g_plus),
        free_energy=jnp.atleast_1d(free_energy),
        kurtosis=updated_state.kurtosis,
        dgm_entropy=updated_state.dgm_entropy,
        adaptive_threshold=jnp.atleast_1d(config.cusum_h),
        weights=updated_weights,
        sinkhorn_converged=jnp.atleast_1d(sinkhorn_converged),
        degraded_inference_mode=degraded_mode,
        emergency_mode=emergency_mode,
        regime_change_detected=regime_change_detected,
        mode_collapse_warning=bool(kernel_outputs[KernelType.KERNEL_B].metadata.get("mode_collapse", False)),
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
