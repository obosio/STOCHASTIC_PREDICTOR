"""Base classes and utilities for prediction kernels.

This module provides the abstract base class and common utilities for all
prediction kernels (A, B, C, D). All kernels must be stateless JAX functions
to enable JIT compilation and vmap vectorization.

References:
    - Python.tex §2: Prediction Kernels Architecture
    - Teoria.tex §2: Mathematical Foundations for Four Branches
    - Implementacion.tex §1.3: Stateless Kernel Design Pattern

Design Principles:
    - Pure functions (no side effects)
    - JIT-compilable (all operations traceable by XLA)
    - vmap-compatible (vectorization over batch dimension)
    - stop_gradient for diagnostics (VRAM optimization)
"""

from abc import ABC, abstractmethod
from typing import Protocol, NamedTuple
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class KernelOutput(NamedTuple):
    """
    Standardized output from all prediction kernels.
    
    Fields:
        prediction: Predicted next value or trajectory
        confidence: Uncertainty estimate (standard deviation)
        entropy: Diagnostic entropy estimate
        probability_density: Discrete PDF over a standardized grid
        kernel_id: Kernel identifier (A|B|C|D)
        computation_time_us: Execution time in microseconds
        numerics_flags: Diagnostic numerical flags
        metadata: Kernel-specific diagnostic information
    
    References:
        - API_Python.tex §2.1: KernelOutput schema
    """
    prediction: Float[Array, "..."]
    confidence: Float[Array, "..."]
    entropy: Float[Array, "..."]
    probability_density: Float[Array, "n_targets"]
    kernel_id: str
    computation_time_us: Float[Array, ""]
    numerics_flags: dict
    metadata: dict


def build_pdf_grid(
    mean: Float[Array, ""],
    sigma: Float[Array, ""],
    config,
) -> tuple[Float[Array, "n_targets"], Float[Array, ""]]:
    """Construct a standardized grid for probability density outputs."""
    sigma_safe = jnp.maximum(sigma, config.pdf_min_sigma)
    z_grid = jnp.linspace(
        config.pdf_grid_min_z,
        config.pdf_grid_max_z,
        config.pdf_grid_num_points,
    )
    grid = mean + sigma_safe * z_grid
    dx = grid[1] - grid[0]
    return grid, dx


def compute_normal_pdf(
    grid: Float[Array, "n_targets"],
    mean: Float[Array, ""],
    sigma: Float[Array, ""],
    config,
) -> Float[Array, "n_targets"]:
    """Compute a normal PDF over a provided grid."""
    sigma_safe = jnp.maximum(sigma, config.pdf_min_sigma)
    norm = 1.0 / (sigma_safe * jnp.sqrt(2.0 * jnp.pi))
    exponent = -0.5 * ((grid - mean) / sigma_safe) ** 2
    return norm * jnp.exp(exponent)


def compute_density_entropy(
    density: Float[Array, "n_targets"],
    dx: Float[Array, ""],
    config,
) -> Float[Array, ""]:
    """Compute discrete entropy from a density and grid spacing."""
    density_safe = jnp.maximum(density, config.numerical_epsilon)
    return -jnp.sum(density * jnp.log(density_safe)) * dx


class PredictionKernel(Protocol):
    """
    Protocol defining the interface for all prediction kernels.
    
    All kernels must implement:
        - __call__: Pure prediction function
        - Stateless operation (no internal state modifications)
        - JAX-compatible (JIT/vmap/grad)
    
    References:
        - Python.tex §2.2: Kernel Interface Contract
        - Implementacion.tex §4.1: Stateless Design
    """
    
    def __call__(
        self,
        signal: Float[Array, "n"],
        key: Array,
        *args,
        **kwargs
    ) -> KernelOutput:
        """
        Compute prediction from input signal.
        
        Args:
            signal: Input time series (length n)
            key: JAX PRNG key for stochastic operations
            *args: Kernel-specific positional arguments
            **kwargs: Kernel-specific keyword arguments
        
        Returns:
            KernelOutput with prediction, confidence, and metadata
        
        Note:
            Must be a pure function (same inputs -> same outputs).
            All randomness must be derived from `key` parameter.
        """
        ...


@jax.jit
def apply_stop_gradient_to_diagnostics(
    prediction: Float[Array, "..."],
    diagnostics: dict
) -> tuple[Float[Array, "..."], dict]:
    """
    Apply stop_gradient to diagnostic computations to save VRAM.
    
    This function ensures diagnostic calculations (entropy, WTMM, CUSUM)
    do not contribute to gradient backpropagation, protecting VRAM budget
    during large-scale training or inference.
    
    Args:
        prediction: Primary prediction tensor (gradients flow)
        diagnostics: Dictionary of diagnostic values (gradients stopped)
    
    Returns:
        Tuple of (prediction, diagnostics_stopped) where diagnostics_stopped
        has stop_gradient applied to all values
    
    References:
        - Python.tex §3.1: VRAM Optimization with stop_gradient
        - Implementacion.tex §2.2: Diagnostic Detachment
    
    Example:
        >>> pred = jnp.array([1.0, 2.0, 3.0])
        >>> diag = {"entropy": jnp.array(0.5), "cusum": jnp.array(2.1)}
        >>> pred_out, diag_out = apply_stop_gradient_to_diagnostics(pred, diag)
        >>> # pred_out: gradients flow normally
        >>> # diag_out: gradients stopped (no backprop)
    """
    # Apply stop_gradient to all diagnostic values
    diagnostics_stopped = jax.tree_map(jax.lax.stop_gradient, diagnostics)
    
    # Prediction passes through unchanged (gradients flow)
    return prediction, diagnostics_stopped


@jax.jit(static_argnames=["min_length"])
def validate_kernel_input(
    signal: Float[Array, "n"],
    min_length: int
) -> tuple[bool, str]:
    """
    Validate input signal for kernel processing.
    
    Args:
        signal: Input time series
        min_length: Minimum required signal length (from config.base_min_signal_length - REQUIRED)
    
    Returns:
        Tuple of (is_valid, error_message)
    
    References:
        - Python.tex §2.3: Input Validation Protocol
    """
    # Check minimum length
    if signal.shape[0] < min_length:
        return False, f"Signal too short: {signal.shape[0]} < {min_length}"
    
    # Check for NaN or Inf
    if not jnp.all(jnp.isfinite(signal)):
        return False, "Signal contains NaN or Inf"
    
    # Check for all zeros (degenerate signal)
    if jnp.all(signal == 0.0):
        return False, "Signal is all zeros (degenerate)"
    
    return True, ""


@jax.jit
def compute_signal_statistics(
    signal: Float[Array, "n"]
) -> dict:
    """
    Compute basic statistics of input signal.
    
    These statistics are used across multiple kernels for normalization
    and adaptive parameter selection.
    
    Args:
        signal: Input time series
    
    Returns:
        Dictionary with keys: mean, std, min, max, range
    
    References:
        - Python.tex §2.4: Signal Preprocessing
    """
    return {
        "mean": jnp.mean(signal),
        "std": jnp.std(signal),
        "min": jnp.min(signal),
        "max": jnp.max(signal),
        "range": jnp.max(signal) - jnp.min(signal)
    }


@jax.jit(static_argnames=["method"])
def normalize_signal(
    signal: Float[Array, "n"],
    method: str,
    epsilon: float = 1e-10
) -> Float[Array, "n"]:
    """
    Normalize signal to zero mean and unit variance.
    
    Args:
        signal: Input time series
        method: Normalization method (from config.signal_normalization_method - REQUIRED)
               Options: 'zscore' or 'minmax'
        epsilon: Small constant to prevent division by zero (from config.numerical_epsilon)
    
    Returns:
        Normalized signal
    
    References:
        - Implementacion.tex §1.5: Signal Normalization
    """
    if method == "zscore":
        mean = jnp.mean(signal)
        std = jnp.std(signal)
        # Avoid division by zero
        std_safe = jnp.where(std < epsilon, 1.0, std)
        return (signal - mean) / std_safe
    
    elif method == "minmax":
        min_val = jnp.min(signal)
        max_val = jnp.max(signal)
        range_val = max_val - min_val
        # Avoid division by zero
        range_safe = jnp.where(range_val < epsilon, 1.0, range_val)
        return (signal - min_val) / range_safe
    
    else:
        # Return unchanged if method unknown
        return signal
