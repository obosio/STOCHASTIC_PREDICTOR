"""JIT Warm-up for Production Deployment.

This module provides utilities to pre-compile all kernels via dummy executions,
eliminating first-inference JIT compilation latency in production.

Design Rationale:
    - JAX JIT compilation occurs on first call, causing ~100-500ms latency
    - Production systems require predictable sub-10ms latency from start
    - Warm-up pass pre-compiles all critical paths with representative shapes

References:
    - Stochastic_Predictor_API_Python.tex §5.2: JIT Compilation Strategy
    - Stochastic_Predictor_Python.tex §4: Performance Optimization
    - Stochastic_Predictor_Implementation.tex §6: Production Deployment

Usage:
    >>> from stochastic_predictor.api.warmup import warmup_all_kernels
    >>> from stochastic_predictor.api.config import get_config
    >>> config = get_config()
    >>> warmup_all_kernels(config)  # Pre-compile all kernels
    >>> # Now first real inference will have no JIT overhead
"""

import time
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

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
        - Stochastic_Predictor_Python.tex §2.2.1: Kernel A Implementation
    """
    from stochastic_predictor.kernels.kernel_a import kernel_a_predict
    
    # Create dummy signal (representative length from config)
    signal_length = config.warmup_signal_length
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
        - Stochastic_Predictor_Python.tex §2.2.2: Kernel B Implementation
    """
    from stochastic_predictor.kernels.kernel_b import kernel_b_predict
    
    signal_length = config.warmup_signal_length
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
        - Stochastic_Predictor_Python.tex §2.2.3: Kernel C Implementation
        - Stochastic_Predictor_Theory.tex §2.3.3: Stiffness-Adaptive SDE Solvers
    """
    from stochastic_predictor.kernels.kernel_c import kernel_c_predict
    
    signal_length = config.warmup_signal_length
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
        - Stochastic_Predictor_Python.tex §2.2.4: Kernel D Implementation
    """
    from stochastic_predictor.kernels.kernel_d import kernel_d_predict
    
    signal_length = config.warmup_signal_length
    dummy_signal = jnp.linspace(0.0, 1.0, signal_length)
    
    start = time.perf_counter()
    _ = kernel_d_predict(dummy_signal, key, config)
    jax.block_until_ready(_)
    elapsed = (time.perf_counter() - start) * 1000
    
    return elapsed


def warmup_kernel_d_load_shedding(
    config: PredictorConfig,
    key: PRNGKeyArray,
    verbose: bool = False
) -> dict[int, float]:
    """
    Pre-compile Kernel D for multiple signature depths (Load Shedding).
    
    COMPLIANCE: API_Python.tex §14 - Load Shedding (Adaptive Topological Pruning)
    "Precompile multiple JIT graphs for M ∈ {2,3,5} and switch by thresholds
    to prevent backlog."
    
    During high-latency regimes, the system must dynamically reduce signature
    depth M to maintain sub-10ms inference. Without pre-compilation, switching
    to M=2 would trigger a JIT cache miss (~300-500ms compilation stall),
    exacerbating the latency problem.
    
    This defensive warmup pre-compiles all emergency topologies to enable
    zero-overhead switching during crisis scenarios.
    
    Args:
        config: Configuration object
        key: PRNG key for reproducibility
        verbose: Print compilation times per depth
    
    Returns:
        Dictionary mapping depth → compilation time (ms)
        e.g., {2: 234.5, 3: 456.7, 5: 789.1}
    
    Design Rationale:
        - M=2: Emergency mode (latency >>threshold, ~50% feature reduction)
        - M=3: Normal mode (baseline, typical config value)
        - M=5: Rich mode (low latency, maximum topological memory)
    
    Performance Impact:
        - One-time startup cost: +500-1500ms (acceptable)
        - Runtime benefit: Eliminates 300-500ms JIT miss during degradation
        - Memory footprint: +15-40MB XLA cache (negligible)
    
    References:
        - API_Python.tex §14: Load Shedding (Adaptive Topological Pruning)
        - Theory.tex §3.6: Signature Depth Truncation Trade-offs
        - Implementation.tex §5.4: Dynamic signature depth adjustment
    
    Example:
        >>> from stochastic_predictor.api.warmup import warmup_kernel_d_load_shedding
        >>> from stochastic_predictor.api.config import get_config
        >>> from stochastic_predictor.api.prng import initialize_jax_prng
        >>> config = get_config()
        >>> key = initialize_jax_prng(seed=42)
        >>> timings = warmup_kernel_d_load_shedding(config, key, verbose=True)
        🔥 Load Shedding Warmup: Pre-compiling Kernel D topologies...
          • M=2 (emergency): 234.5 ms ✓
          • M=3 (normal): 456.7 ms ✓
          • M=5 (rich): 789.1 ms ✓
        ✅ Load shedding ready: 1480.3 ms total
    """
    from stochastic_predictor.kernels.kernel_d import kernel_d_predict
    from dataclasses import replace
    
    signal_length = config.warmup_signal_length
    dummy_signal = jnp.linspace(0.0, 1.0, signal_length)
    
    # Emergency topologies as mandated by specification
    emergency_depths = [2, 3, 5]
    compilation_times = {}
    
    if verbose:
        print("🔥 Load Shedding Warmup: Pre-compiling Kernel D topologies...")
    
    for depth in emergency_depths:
        # Create temporary config with modified depth
        temp_config = replace(config, kernel_d_depth=depth)
        
        # Trigger JIT compilation
        start = time.perf_counter()
        _ = kernel_d_predict(dummy_signal, key, temp_config)
        jax.block_until_ready(_)  # Wait for GPU kernel compilation
        elapsed = (time.perf_counter() - start) * 1000
        
        compilation_times[depth] = elapsed
        
        if verbose:
            mode_label = {
                2: "emergency",
                3: "normal",
                5: "rich"
            }.get(depth, "unknown")
            print(f"  • M={depth} ({mode_label}): {elapsed:.1f} ms ✓")
    
    total_time = sum(compilation_times.values())
    if verbose:
        print(f"✅ Load shedding ready: {total_time:.1f} ms total")
    
    return compilation_times


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
        - Stochastic_Predictor_Implementation.tex §6.2: Production Deployment Checklist
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
    
    # Kernel D: Signatures (baseline depth)
    if verbose:
        print("  ⏳ Kernel D (Path Signatures - baseline)...", end=" ", flush=True)
    timings["kernel_d"] = warmup_kernel_d(config, keys[3])
    if verbose:
        print(f"✓ {timings['kernel_d']:.1f} ms")
    
    # Kernel D: Load Shedding (emergency topologies)
    # COMPLIANCE: API_Python.tex §14 mandates pre-compilation of M ∈ {2,3,5}
    if verbose:
        print("  ⏳ Kernel D (Load Shedding topologies)...")
    load_shedding_timings = warmup_kernel_d_load_shedding(config, keys[3], verbose=verbose)
    timings["kernel_d_load_shedding"] = load_shedding_timings
    
    total_time = sum(timings.values()) if not isinstance(timings.get("kernel_d_load_shedding"), dict) else \
                 sum(v for k, v in timings.items() if k != "kernel_d_load_shedding") + \
                 sum(timings["kernel_d_load_shedding"].values())
    
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


def profile_warmup_and_recommend_timeout(
    config: PredictorConfig,
    verbose: bool = True
) -> dict[str, float | int]:
    """
    Profile warm-up times and recommend data_feed_timeout adjustment.
    
    Executes warm-up pass and provides recommendations for timeout configuration
    based on actual JIT compilation latency. Useful for deployment on different
    GPU hardware (e.g., T4 vs A100 vs CPU-only).
    
    Recommendation Logic:
        - If max(kernel_times) > 500ms: Recommend timeout ≥ 60s (slow GPU/CPU)
        - If max(kernel_times) > 300ms: Recommend timeout ≥ 45s (mid-tier GPU)
        - If max(kernel_times) ≤ 300ms: Keep default timeout (30s, fast GPU)
    
    Args:
        config: Configuration object
        verbose: Print profiling results and recommendations
    
    Returns:
        Dictionary with profiling results:
        {
            "kernel_a": <ms>,
            "kernel_b": <ms>,
            "kernel_c": <ms>,
            "kernel_d": <ms>,
            "total": <ms>,
            "max_kernel": <ms>,
            "recommended_timeout": <seconds>
        }
    
    Example:
        >>> from stochastic_predictor.api.warmup import profile_warmup_and_recommend_timeout
        >>> from stochastic_predictor.api.config import get_config
        >>> config = get_config()
        >>> profile = profile_warmup_and_recommend_timeout(config)
        >>> # Update config.toml [io] section:
        >>> # data_feed_timeout = <profile["recommended_timeout"]>
    
    References:
        - Stochastic_Predictor_Implementation.tex §6.3: Deployment on Heterogeneous Hardware
    """
    if verbose:
        print("🔍 Profiling JIT Compilation Times...")
        print()
    
    # Execute warm-up and measure
    timings = warmup_all_kernels(config, verbose=verbose)
    
    # Compute statistics
    total_time = sum(timings.values())
    max_kernel_time = max(timings.values())
    
    # Determine slowest kernel (for diagnostic purposes)
    slowest_kernel = "kernel_a"
    for kernel_name, kernel_time in timings.items():
        if kernel_time == max_kernel_time:
            slowest_kernel = kernel_name
            break
    
    # Recommendation logic based on max kernel compilation time
    if max_kernel_time > 500.0:
        recommended_timeout = 60  # Slow GPU/CPU - conservative timeout
        hardware_tier = "SLOW (CPU or low-end GPU)"
    elif max_kernel_time > 300.0:
        recommended_timeout = 45  # Mid-tier GPU
        hardware_tier = "MEDIUM (mid-tier GPU)"
    else:
        recommended_timeout = 30  # Fast GPU - default timeout
        hardware_tier = "FAST (high-end GPU)"
    
    if verbose:
        print()
        print("📊 Profiling Summary:")
        print(f"  • Total warm-up time: {total_time:.1f} ms")
        print(f"  • Max kernel time: {max_kernel_time:.1f} ms ({slowest_kernel})")
        print(f"  • Hardware tier: {hardware_tier}")
        print()
        print("💡 Recommendation:")
        print(f"  • Set data_feed_timeout ≥ {recommended_timeout} seconds in config.toml")
        print(f"  • Rationale: JIT compilation latency suggests {hardware_tier} hardware")
        if max_kernel_time > 500.0:
            print("  ⚠️  WARNING: High compilation times detected!")
            print("     Consider enabling XLA_FLAGS for persistent cache or upgrading GPU")
    
    # Return full profiling data
    return {
        **timings,
        "total": total_time,
        "max_kernel": max_kernel_time,
        "recommended_timeout": recommended_timeout,
    }
