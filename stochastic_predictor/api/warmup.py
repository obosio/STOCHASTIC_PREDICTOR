"""JIT Warm-up for Production Deployment.

This module provides utilities to pre-compile all kernels via dummy executions,
eliminating first-inference JIT compilation latency in production.

Design Rationale:
    - JAX JIT compilation occurs on first call, causing ~100-500ms latency
    - Production systems require predictable sub-10ms latency from start
    - Warm-up pass pre-compiles all critical paths with representative shapes

References:
    - API_Python.tex §5.2: JIT Compilation Strategy
    - Python.tex §4: Performance Optimization
    - Implementacion.tex §6: Production Deployment

Usage:
    >>> from stochastic_predictor.api.warmup import warmup_all_kernels
    >>> from stochastic_predictor.api.config import get_config
    >>> config = get_config()
    >>> warmup_all_kernels(config)  # Pre-compile all kernels
    >>> # Now first real inference will have no JIT overhead
"""

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import time
from typing import Optional

from stochastic_predictor.api.types import PredictorConfig
from stochastic_predictor.api.prng import initialize_jax_prng, split_key


def warmup_kernel_a(config: PredictorConfig, key: PRNGKeyArray) -> float:
    """
    Warm-up Kernel A (RKHS Ridge Regression).
    
    Executes a dummy prediction to trigger JIT compilation of:
    - Gaussian kernel matrix computation
    - Ridge regression solver
    - WTMM wavelet transform
    
    Args:
        config: Configuration object
        key: PRNG key for reproducibility
    
    Returns:
        Warm-up execution time (ms)
    
    References:
        - Python.tex §2.2.1: Kernel A Implementation
    """
    from stochastic_predictor.kernels.kernel_a import kernel_a_predict
    
    # Create dummy signal (minimum length from config)
    signal_length = max(config.base_min_signal_length, 100)
    dummy_signal = jnp.linspace(0.0, 1.0, signal_length)
    
    # Execute once to compile
    start = time.perf_counter()
    _ = kernel_a_predict(dummy_signal, key, config)
    jax.block_until_ready(_)  # Wait for asynchronous dispatch
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
    
    return elapsed


def warmup_kernel_b(config: PredictorConfig, key: PRNGKeyArray) -> float:
    """
    Warm-up Kernel B (Deep Galerkin Method PDE Solver).
    
    Executes a dummy prediction to trigger JIT compilation of:
    - DGM neural network forward pass
    - HJB PDE residual computation
    - Entropy estimation
    
    Args:
        config: Configuration object
        key: PRNG key for reproducibility
    
    Returns:
        Warm-up execution time (ms)
    
    References:
        - Python.tex §2.2.2: Kernel B Implementation
    """
    from stochastic_predictor.kernels.kernel_b import kernel_b_predict
    
    signal_length = max(config.base_min_signal_length, 100)
    dummy_signal = jnp.linspace(0.0, 1.0, signal_length)
    
    start = time.perf_counter()
    _ = kernel_b_predict(dummy_signal, key, config)
    jax.block_until_ready(_)
    elapsed = (time.perf_counter() - start) * 1000
    
    return elapsed


def warmup_kernel_c(config: PredictorConfig, key: PRNGKeyArray) -> float:
    """
    Warm-up Kernel C (SDE Integration with Diffrax).
    
    Executes a dummy prediction to trigger JIT compilation of:
    - Stiffness estimation
    - Dynamic solver selection
    - Diffrax SDE integration
    - VirtualBrownianTree construction
    
    Args:
        config: Configuration object
        key: PRNG key for reproducibility
    
    Returns:
        Warm-up execution time (ms)
    
    References:
        - Python.tex §2.2.3: Kernel C Implementation
        - Teoria.tex §2.3.3: Stiffness-Adaptive SDE Solvers
    """
    from stochastic_predictor.kernels.kernel_c import kernel_c_predict
    
    signal_length = max(config.base_min_signal_length, 100)
    dummy_signal = jnp.linspace(0.0, 1.0, signal_length)
    
    start = time.perf_counter()
    _ = kernel_c_predict(dummy_signal, key, config)
    jax.block_until_ready(_)
    elapsed = (time.perf_counter() - start) * 1000
    
    return elapsed


def warmup_kernel_d(config: PredictorConfig, key: PRNGKeyArray) -> float:
    """
    Warm-up Kernel D (Path Signatures with Signax).
    
    Executes a dummy prediction to trigger JIT compilation of:
    - Path signature computation
    - Log-signature truncation
    - Ridge regression on signature features
    
    Args:
        config: Configuration object
        key: PRNG key for reproducibility
    
    Returns:
        Warm-up execution time (ms)
    
    References:
        - Python.tex §2.2.4: Kernel D Implementation
    """
    from stochastic_predictor.kernels.kernel_d import kernel_d_predict
    
    signal_length = max(config.base_min_signal_length, 100)
    dummy_signal = jnp.linspace(0.0, 1.0, signal_length)
    
    start = time.perf_counter()
    _ = kernel_d_predict(dummy_signal, key, config)
    jax.block_until_ready(_)
    elapsed = (time.perf_counter() - start) * 1000
    
    return elapsed


def warmup_all_kernels(
    config: PredictorConfig,
    key: Optional[PRNGKeyArray] = None,
    verbose: bool = True
) -> dict[str, float]:
    """
    Execute warm-up pass for all kernels.
    
    Pre-compiles all critical execution paths to eliminate first-inference
    JIT latency. Should be called during service initialization.
    
    Args:
        config: Configuration object with all kernel parameters
        key: Optional PRNG key (creates default if None)
        verbose: Print warm-up progress
    
    Returns:
        Dictionary mapping kernel names to compilation times (ms)
    
    Example:
        >>> from stochastic_predictor.api.config import get_config
        >>> from stochastic_predictor.api.warmup import warmup_all_kernels
        >>> config = get_config()
        >>> timings = warmup_all_kernels(config)
        >>> print(f"Total warm-up: {sum(timings.values()):.1f} ms")
    
    References:
        - Implementacion.tex §6.2: Production Deployment Checklist
    """
    if key is None:
        key = initialize_jax_prng(seed=42)
    
    timings = {}
    
    # Split key for each kernel (deterministic warm-up)
    keys = split_key(key, num=4)
    
    if verbose:
        print("🔥 JIT Warm-up: Pre-compiling kernels...")
    
    # Kernel A: RKHS
    if verbose:
        print("  ⏳ Kernel A (RKHS Ridge)...", end=" ", flush=True)
    timings["kernel_a"] = warmup_kernel_a(config, keys[0])
    if verbose:
        print(f"✓ {timings['kernel_a']:.1f} ms")
    
    # Kernel B: DGM
    if verbose:
        print("  ⏳ Kernel B (DGM PDE)...", end=" ", flush=True)
    timings["kernel_b"] = warmup_kernel_b(config, keys[1])
    if verbose:
        print(f"✓ {timings['kernel_b']:.1f} ms")
    
    # Kernel C: SDE
    if verbose:
        print("  ⏳ Kernel C (SDE Integration)...", end=" ", flush=True)
    timings["kernel_c"] = warmup_kernel_c(config, keys[2])
    if verbose:
        print(f"✓ {timings['kernel_c']:.1f} ms")
    
    # Kernel D: Signatures
    if verbose:
        print("  ⏳ Kernel D (Path Signatures)...", end=" ", flush=True)
    timings["kernel_d"] = warmup_kernel_d(config, keys[3])
    if verbose:
        print(f"✓ {timings['kernel_d']:.1f} ms")
    
    total_time = sum(timings.values())
    if verbose:
        print(f"✅ Warm-up complete: {total_time:.1f} ms total")
    
    return timings


def warmup_with_retry(
    config: PredictorConfig,
    max_retries: int = 3,
    verbose: bool = True
) -> dict[str, float]:
    """
    Warm-up with automatic retry on failure.
    
    Useful for production environments where initial compilation may fail
    due to transient GPU memory issues or XLA initialization delays.
    
    Args:
        config: Configuration object
        max_retries: Maximum retry attempts per kernel
        verbose: Print progress
    
    Returns:
        Warm-up timings dictionary
    
    Raises:
        RuntimeError: If warm-up fails after max_retries
    """
    for attempt in range(max_retries):
        try:
            timings = warmup_all_kernels(config, verbose=verbose)
            return timings
        except Exception as e:
            if attempt < max_retries - 1:
                if verbose:
                    print(f"⚠️  Warm-up attempt {attempt + 1} failed: {e}")
                    print(f"   Retrying ({attempt + 2}/{max_retries})...")
                continue
            else:
                raise RuntimeError(
                    f"Warm-up failed after {max_retries} attempts: {e}"
                ) from e
    
    # Should never reach here
    raise RuntimeError("Warm-up loop exit anomaly")
