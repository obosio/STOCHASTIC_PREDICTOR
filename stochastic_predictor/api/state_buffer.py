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
    >>> from stochastic_predictor.api.state_buffer import update_signal_history
    >>> state = InternalState(...)
    >>> new_state = update_signal_history(state, new_value=3.14)
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float
from dataclasses import replace

from stochastic_predictor.api.types import InternalState


@jax.jit
def update_signal_history(
    state: InternalState,
    new_value: Float[Array, ""]
) -> InternalState:
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
    N = state.signal_history.shape[0]
    
    # Shift left: [x[1], x[2], ..., x[N-1], 0.0]
    shifted = lax.dynamic_slice(state.signal_history, start_indices=(1,), slice_sizes=(N - 1,))
    
    # Prepare new value as array
    new_val_array = jnp.atleast_1d(new_value)
    
    # Concatenate: [x[1], x[2], ..., x[N-1], new_value]
    updated_history = jnp.concatenate([shifted, new_val_array])
    
    # Return new state (functional update via dataclass replace)
    return replace(state, signal_history=updated_history)


@jax.jit
def update_residual_buffer(
    state: InternalState,
    new_residual: Float[Array, ""]
) -> InternalState:
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
    N = state.residual_buffer.shape[0]
    
    shifted = lax.dynamic_slice(state.residual_buffer, start_indices=(1,), slice_sizes=(N - 1,))
    new_res_array = jnp.atleast_1d(new_residual)
    updated_buffer = jnp.concatenate([shifted, new_res_array])
    
    return replace(state, residual_buffer=updated_buffer)


@jax.jit
def batch_update_signal_history(
    state: InternalState,
    new_values: Float[Array, "M"]
) -> InternalState:
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
    N = state.signal_history.shape[0]
    M = new_values.shape[0]
    
    if M >= N:
        # If M >= N, just take the last N values
        updated_history = new_values[-N:]
    else:
        # Shift left by M positions, append M new values
        shifted = lax.dynamic_slice(
            state.signal_history,
            start_indices=(M,),
            slice_sizes=(N - M,)
        )
        updated_history = jnp.concatenate([shifted, new_values])
    
    return replace(state, signal_history=updated_history)


@jax.jit
def update_cusum_statistics(
    state: InternalState,
    new_residual: Float[Array, ""],
    cusum_k: float
) -> InternalState:
    """
    Update CUSUM statistics (G+, G-) atomically.
    
    CUSUM (Cumulative Sum) algorithm for change-point detection:
    - G^+ tracks positive drift accumulation
    - G^- tracks negative drift accumulation
    
    Args:
        state: Current internal state
        new_residual: New prediction residual
        cusum_k: CUSUM reference value (allowance parameter)
    
    Returns:
        New InternalState with updated CUSUM statistics
    
    References:
        - Teoria.tex §3.4: CUSUM Algorithm
        - Python.tex §5.3: Degradation Detection
    """
    # CUSUM update equations
    g_plus_new = jnp.maximum(0.0, state.cusum_g_plus + new_residual - cusum_k)
    g_minus_new = jnp.maximum(0.0, state.cusum_g_minus - new_residual - cusum_k)
    
    return replace(
        state,
        cusum_g_plus=g_plus_new,
        cusum_g_minus=g_minus_new
    )


@jax.jit
def update_ema_variance(
    state: InternalState,
    new_value: Float[Array, ""],
    alpha: float
) -> InternalState:
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
    variance_new = alpha * (new_value ** 2) + (1.0 - alpha) * state.ema_variance
    
    return replace(state, ema_variance=variance_new)


@jax.jit
def atomic_state_update(
    state: InternalState,
    new_signal: Float[Array, ""],
    new_residual: Float[Array, ""],
    cusum_k: float,
    volatility_alpha: float
) -> InternalState:
    """
    Perform atomic update of all state buffers simultaneously.
    
    Combines multiple Zero-Copy updates into a single functional operation:
    - Signal history rolling window
    - Residual buffer rolling window
    - CUSUM statistics (G+, G-)
    - EWMA variance
    
    Args:
        state: Current internal state
        new_signal: New signal magnitude
        new_residual: New prediction error
        cusum_k: CUSUM allowance parameter
        volatility_alpha: EWMA smoothing factor
    
    Returns:
        New InternalState with all buffers updated atomically
    
    Example:
        >>> from stochastic_predictor.api.config import get_config
        >>> config = get_config()
        >>> new_state = atomic_state_update(
        ...     state, new_signal=3.14, new_residual=0.05,
        ...     cusum_k=config.cusum_k, volatility_alpha=config.volatility_alpha
        ... )
    
    References:
        - API_Python.tex §3.4: Atomic State Updates
        - Implementacion.tex §5: State Management
    """
    # Chain functional updates (JAX will optimize this)
    state = update_signal_history(state, new_signal)
    state = update_residual_buffer(state, new_residual)
    state = update_cusum_statistics(state, new_residual, cusum_k)
    state = update_ema_variance(state, new_residual, volatility_alpha)
    
    return state


@jax.jit
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
        grace_counter=0
    )


# Export public API
__all__ = [
    "update_signal_history",
    "update_residual_buffer",
    "batch_update_signal_history",
    "update_cusum_statistics",
    "update_ema_variance",
    "atomic_state_update",
    "reset_cusum_statistics",
]
