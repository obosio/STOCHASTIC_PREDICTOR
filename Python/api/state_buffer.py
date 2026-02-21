"""Zero-Copy State Buffer Management.

This module provides utilities for efficient, atomic updates to InternalState
buffers using JAX's dynamic_update_slice for zero-copy operations.

Design Rationale:
    - Avoid full array copies when updating rolling windows (signal_history, residual_buffer)
    - Use jax.lax.dynamic_update_slice for in-place updates (Zero-Copy)
    - Maintain functional purity (returns new InternalState, old unchanged)
    - GPU-friendly (no host-device transfers)

References:
    - API_Python.tex §3: State Management
    - Python.tex §6.2: Zero-Copy Buffer Updates
    - JAX docs: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dynamic_update_slice.html

Usage:
    >>> from Python.api.state_buffer import update_signal_history
    >>> state = InternalState(...)
    >>> new_state = update_signal_history(state, new_value=3.14)
"""

from dataclasses import replace

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from Python.api.types import InternalState, PredictorConfig


def update_signal_history(state: InternalState, new_value: Float[Array, ""]) -> InternalState:
    """
    Update signal_history buffer with new value (Zero-Copy rolling window).

    Implements efficient rolling window update:
    1. Shift existing values left by 1 position (drop oldest)
    2. Insert new value at rightmost position
    3. Use dynamic_update_slice to avoid full array copy

    Args:
        state: Current internal state
        new_value: New signal magnitude to append

    Returns:
        New InternalState with updated signal_history (functional update)

    Example:
        >>> state = InternalState(signal_history=jnp.array([1.0, 2.0, 3.0]), ...)
        >>> new_state = update_signal_history(state, jnp.array(4.0))
        >>> new_state.signal_history  # [2.0, 3.0, 4.0]

    References:
        - API_Python.tex §3.1: Rolling Window Management
        - Python.tex §6.2.1: dynamic_update_slice Pattern
    """
    history = lax.stop_gradient(state.signal_history)
    new_value = lax.stop_gradient(new_value)
    N = history.shape[0]

    # Shift left: [x[1], x[2], ..., x[N-1], 0.0]
    shifted = lax.dynamic_slice(history, start_indices=(1,), slice_sizes=(N - 1,))

    # Prepare new value as array
    new_val_array = jnp.atleast_1d(new_value)

    # Concatenate: [x[1], x[2], ..., x[N-1], new_value]
    updated_history = jnp.concatenate([shifted, new_val_array])

    # Return new state (functional update via dataclass replace)
    return replace(state, signal_history=updated_history)


def update_residual_buffer(state: InternalState, new_residual: Float[Array, ""]) -> InternalState:
    """
    Update residual_buffer with new prediction error (Zero-Copy).

    Same pattern as update_signal_history but for residual tracking.

    Args:
        state: Current internal state
        new_residual: New prediction error (|y_true - y_pred|)

    Returns:
        New InternalState with updated residual_buffer

    References:
        - API_Python.tex §3.2: Residual Tracking
    """
    buffer = lax.stop_gradient(state.residual_buffer)
    new_residual = lax.stop_gradient(new_residual)
    N = buffer.shape[0]

    shifted = lax.dynamic_slice(buffer, start_indices=(1,), slice_sizes=(N - 1,))
    new_res_array = jnp.atleast_1d(new_residual)
    updated_buffer = jnp.concatenate([shifted, new_res_array])

    return replace(state, residual_buffer=updated_buffer)


def batch_update_signal_history(state: InternalState, new_values: Float[Array, "M"]) -> InternalState:
    """
    Update signal_history with multiple values (batch Zero-Copy).

    Useful for initialization or recovery scenarios where M > 1 new values
    need to be appended simultaneously.

    Args:
        state: Current internal state
        new_values: Array of M new values to append

    Returns:
        New InternalState with updated signal_history

    Example:
        >>> state = InternalState(signal_history=jnp.zeros(100), ...)
        >>> new_state = batch_update_signal_history(state, jnp.array([1.0, 2.0, 3.0]))

    References:
        - API_Python.tex §3.3: Batch Updates
    """
    history = lax.stop_gradient(state.signal_history)
    new_values = lax.stop_gradient(new_values)
    N = history.shape[0]
    M = new_values.shape[0]

    if M >= N:
        # If M >= N, just take the last N values
        updated_history = new_values[-N:]
    else:
        # Shift left by M positions, append M new values
        shifted = lax.dynamic_slice(history, start_indices=(M,), slice_sizes=(N - M,))
        updated_history = jnp.concatenate([shifted, new_values])

    return replace(state, signal_history=updated_history)


@jax.jit
def compute_rolling_kurtosis(
    residual_window: Float[Array, "W"],
    config: PredictorConfig,
) -> Float[Array, ""]:
    """
    Compute empirical kurtosis of rolling residuals.

    κ_t = (1/n) Σ(e_i - μ)^4 / σ^4

    Args:
        residual_window: Rolling window of standardized residuals

    Returns:
        Scalar kurtosis value bounded [1.0, 100.0]

    References:
        - Implementation.tex §2.3: Algorithm 2.2 (CUSUM with Kurtosis)
    """
    residual_window.shape[0]
    mean_res = jnp.mean(residual_window)
    var_res = jnp.var(residual_window)
    std_res = jnp.sqrt(jnp.maximum(var_res, config.numerical_epsilon))

    # Fourth central moment
    fourth_moment = jnp.mean((residual_window - mean_res) ** 4)

    # Kurtosis: μ4 / σ^4
    kurtosis = fourth_moment / (std_res**4 + config.numerical_epsilon)

    # Bound to avoid numerical explosion
    return jnp.clip(kurtosis, config.kurtosis_min, config.kurtosis_max)


@jax.jit
def update_residual_window(state: InternalState, new_residual: Float[Array, ""]) -> InternalState:
    """
    Update residual_window with new residual (Zero-Copy rolling window).

    Shifts window left and appends new value at the end.

    Args:
        state: Current internal state
        new_residual: New residual to append

    Returns:
        New InternalState with updated residual_window

    References:
        - API_Python.tex §3.2: Residual Window Management
    """
    window = lax.stop_gradient(state.residual_window)
    new_residual = lax.stop_gradient(new_residual)
    W = window.shape[0]

    # Shift left: drop oldest, append new
    shifted = lax.dynamic_slice(window, start_indices=(1,), slice_sizes=(W - 1,))
    new_res_array = jnp.atleast_1d(new_residual)
    updated_window = jnp.concatenate([shifted, new_res_array])

    return replace(state, residual_window=updated_window)


def update_cusum_statistics(
    residual: Float[Array, ""], state: InternalState, config
) -> tuple[InternalState, bool, float]:
    """
    Update CUSUM statistics with kurtosis-adaptive threshold [V-CRIT-1 FIX].

    NEW: h_t = k · σ_t · (1 + ln(κ_t / 3))

    With grace period logic to suppress false positives post-alarm.

    CRITICAL GRADIENT ISOLATION:
    All statistical accumulators (CUSUM, kurtosis, variance) are wrapped with
    stop_gradient() to prevent gradient leakage during meta-optimization loops
    (jax.lax.scan, jax.vmap). This ensures VRAM efficiency and prevents
    catastrophic memory exhaustion when tuning hyperparameters.

    Args:
        residual: Current prediction residual / standardized error
        state: Current internal state
        config: PredictorConfig with cusum_k, grace_period_steps, etc.

    Returns:
        Tuple: (updated_state, should_alarm, h_t)
        - updated_state: State with CUSUM, kurtosis, grace counter updated
        - should_alarm: True if alarm AND NOT in grace period
        - h_t: Adaptive threshold value

    References:
        - Implementation.tex §2.3, Algorithm 2.2: CUSUM with Kurtosis
        - Implementation.tex §2.5: Grace Period Logic (V-CRIT-3)
        - Implementation.tex §2.0.0: FP64 precision policy for immutability
    """
    # Get current state components with gradient isolation (VRAM protection)
    # CRITICAL: stop_gradient ensures these accumulators do not participate in
    # autodiff when orchestrator is nested in optimization loops (jax.lax.scan)
    residual = lax.stop_gradient(residual)
    cusum_g_plus = lax.stop_gradient(state.cusum_g_plus)
    cusum_g_minus = lax.stop_gradient(state.cusum_g_minus)
    grace_counter = lax.stop_gradient(jnp.array(state.grace_counter, dtype=jnp.int32))
    sigma_t = jnp.sqrt(jnp.maximum(state.ema_variance, config.numerical_epsilon))

    # 1. Update rolling residual window
    new_state = update_residual_window(state, residual)

    # 2. Compute kurtosis from updated window (stop_gradient applied)
    kurtosis = compute_rolling_kurtosis(new_state.residual_window, config)

    # 3. Compute adaptive threshold with kurtosis adjustment
    # h_t = k · σ_t · (1 + ln(κ_t / 3))
    # CRITICAL: stop_gradient prevents backprop through threshold computation,
    # ensuring CUSUM statistics remain diagnostic-only (no gradient contamination)
    h_t = jax.lax.stop_gradient(
        config.cusum_k
        * sigma_t
        * (1.0 + jnp.log(jnp.maximum(kurtosis, config.kurtosis_reference) / config.kurtosis_reference))
    )

    # 4. CUSUM update equations (standard CUSUM recursion)
    g_plus_new = jnp.maximum(0.0, cusum_g_plus + residual - config.cusum_k)
    g_minus_new = jnp.maximum(0.0, cusum_g_minus - residual - config.cusum_k)

    # 5. Alarm detection
    alarm = (g_plus_new > h_t) | (g_minus_new > h_t)
    in_grace_period = grace_counter > 0
    should_alarm = alarm & ~in_grace_period

    # 6. CUSUM reset if alarm (stop_gradient applied to prevent gradient flow)
    final_g_plus = lax.stop_gradient(jnp.where(should_alarm, 0.0, g_plus_new))
    final_g_minus = lax.stop_gradient(jnp.where(should_alarm, 0.0, g_minus_new))

    # 7. Update grace counter (diagnostic, no gradients)
    new_grace_counter = jnp.where(should_alarm, config.grace_period_steps, jnp.maximum(0, grace_counter - 1))

    # 8. Return updated state with all components (including adaptive_h_t)
    # All stateful components are gradient-isolated for VRAM efficiency
    final_state = replace(
        new_state,
        cusum_g_plus=final_g_plus,
        cusum_g_minus=final_g_minus,
        grace_counter=int(jnp.asarray(new_grace_counter)),
        adaptive_h_t=h_t,  # Persist adaptive threshold
        kurtosis=kurtosis,
    )

    return final_state, bool(should_alarm), float(h_t)


def update_ema_variance(state: InternalState, new_value: Float[Array, ""], alpha: float) -> InternalState:
    """
    Update EWMA (Exponential Weighted Moving Average) variance.

    EWMA update: σ²_t = α * (x_t - μ)² + (1 - α) * σ²_{t-1}
    Assumes zero mean (μ = 0) for residuals.

    Args:
        state: Current internal state
        new_value: New observation (typically residual)
        alpha: Smoothing factor (0 < α < 1, typically 0.1-0.3)

    Returns:
        New InternalState with updated EMA variance

    References:
        - Teoria.tex §3.3: Volatility Monitoring
        - Python.tex §5.2: EWMA Estimation
    """
    ema_variance = lax.stop_gradient(state.ema_variance)
    new_value = lax.stop_gradient(new_value)

    variance_new = alpha * (new_value**2) + (1.0 - alpha) * ema_variance

    return replace(state, ema_variance=variance_new)


@jax.jit
def atomic_state_update(
    state: InternalState,
    new_signal: Float[Array, ""],
    new_residual: Float[Array, ""],
    config: PredictorConfig,
) -> tuple[InternalState, bool]:
    """
    Perform atomic update of all state buffers simultaneously.

    Combines multiple Zero-Copy updates into a single functional operation:
    - Signal history rolling window
    - Residual buffer rolling window
    - CUSUM statistics (G+, G-) with kurtosis adaptation
    - EWMA variance
    - Grace period management

    Args:
        state: Current internal state
        new_signal: New signal magnitude
        new_residual: New prediction error (absolute deviation)
        config: System configuration (PredictorConfig)

    Returns:
        Tuple of (updated_state, should_alarm: bool)
        - updated_state: New InternalState with all buffers updated atomically
        - should_alarm: True if CUSUM alarm triggered, False otherwise

    Example:
        >>> from Python.api.config import get_config
        >>> config = get_config()
        >>> new_state, alarm = atomic_state_update(
        ...     state, new_signal=3.14, new_residual=0.05, config=config
        ... )

    References:
        - API_Python.tex §3.4: Atomic State Updates
        - Implementation_v2.0.1_API.tex §2.3: CUSUM Kurtosis Adjustment (V-CRIT-1)
        - Stochastic_Predictor_Theory.tex §3.4.2: CUSUM Grace Period Logic
    """
    # Chain functional updates (JAX will optimize this)
    state = update_signal_history(state, new_signal)
    state = update_residual_buffer(state, new_residual)
    state, should_alarm, h_t = update_cusum_statistics(new_residual, state, config)
    state = update_ema_variance(state, new_residual, config.volatility_alpha)

    return state, should_alarm


def reset_cusum_statistics(state: InternalState) -> InternalState:
    """
    Reset CUSUM accumulators to zero (after alarm trigger).

    Called after CUSUM alarm is triggered or during grace period exit.

    Args:
        state: Current internal state

    Returns:
        New InternalState with CUSUM statistics reset

    References:
        - Teoria.tex §3.4.2: CUSUM Reset Logic
    """
    return replace(
        state,
        cusum_g_plus=jnp.array(0.0),
        cusum_g_minus=jnp.array(0.0),
        grace_counter=0,
    )


# Export public API
__all__ = [
    "update_signal_history",
    "update_residual_buffer",
    "batch_update_signal_history",
    "compute_rolling_kurtosis",
    "update_residual_window",
    "update_cusum_statistics",
    "update_ema_variance",
    "atomic_state_update",
    "reset_cusum_statistics",
]
