"""
PRNG Key Management for Deterministic Reproducibility.

This module provides utilities to handle JAX's PRNG state in a thread-safe
and deterministic manner, critical for bit-exact reproducibility required in
portability tests (CPU/GPU/FPGA).

References:
    - Stochastic_Predictor_Python.tex §1: JAX PRNG (threefry2x32)
    - Stochastic_Predictor_API_Python.tex §5: Floating-Point Determinism
    - Stochastic_Predictor_Tests_Python.tex §1.2: Shared Fixtures (rng_key)
"""

from typing import Sequence, Any
import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import PRNGKeyArray
import os


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL PRNG CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

def initialize_jax_prng(seed: int) -> PRNGKeyArray:
    """
    Initialize JAX PRNG generator with deterministic configuration.
    
    Prerequisites:
        - Environment variables must be set BEFORE importing JAX:
          * JAX_DEFAULT_PRNG_IMPL='threefry2x32'
          * JAX_DETERMINISTIC_REDUCTIONS='1'
    
    Args:
        seed: Seed for reproducibility
        
    Returns:
        PRNGKeyArray: Root key for deriving subkeys
        
    References:
        - Stochastic_Predictor_API_Python.tex §5.1: Deterministic XLA and PRNG Configuration
        - Stochastic_Predictor_Python.tex §1.3: Global Numerical Precision Management
        
    Example:
        >>> import os
        >>> os.environ['JAX_DEFAULT_PRNG_IMPL'] = 'threefry2x32'
        >>> import jax
        >>> from Python.api.prng import initialize_jax_prng
        >>> key = initialize_jax_prng(seed=42)
        >>> print(key.shape)  # (2,) for threefry2x32
    """
    # Verify deterministic configuration is active
    prng_impl = os.getenv("JAX_DEFAULT_PRNG_IMPL", "default")
    if prng_impl != "threefry2x32":
        import warnings
        warnings.warn(
            f"PRNG implementation is '{prng_impl}', expected 'threefry2x32'. "
            "Set JAX_DEFAULT_PRNG_IMPL='threefry2x32' before importing JAX "
            "for bit-exact reproducibility across backends.",
            RuntimeWarning
        )
    
    # Generate root key
    key = random.PRNGKey(seed)
    
    return key


# ═══════════════════════════════════════════════════════════════════════════
# SUBKEY MANAGEMENT (Splitting)
# ═══════════════════════════════════════════════════════════════════════════

def split_key(key: PRNGKeyArray, num: int) -> tuple[PRNGKeyArray, ...]:
    """
    Split a PRNG key into multiple independent subkeys.
    
    Design: Wrapper over jax.random.split for API consistency
    and clarity in key splitting.
    
    Args:
        key: PRNG key to split
        num: Number of subkeys to generate
        
    Returns:
        Tuple of PRNGKeyArray subkeys
        
    References:
        - Python.tex §2: Vectorized Generators
        - API_Python.tex: Split usage in kernels
        
    Example:
        >>> key = initialize_jax_prng(42)
        >>> k1, k2, k3 = split_key(key, num=3)
        >>> # k1, k2, k3 are statistically independent
    """
    return tuple(random.split(key, num=num))


def split_key_like(
    key: PRNGKeyArray, 
    target_shape: Sequence[int]
) -> tuple[PRNGKeyArray, PRNGKeyArray]:
    """
    Split a key and generate an array of subkeys with specific shape.
    
    Useful for vectorized operations that require independent PRNG keys
    for each element of a batch.
    
    Args:
        key: Root PRNG key
        target_shape: Desired array shape of subkeys
        
    Returns:
        Tuple (new_root_key, array_of_subkeys)
        
    Example:
        >>> key = initialize_jax_prng(42)
        >>> new_key, batch_keys = split_key_like(key, (100,))
        >>> # batch_keys.shape == (100, 2) for threefry2x32
        >>> # new_key can be reused for next operation
    """
    # Split root key and generate batch of subkeys
    new_key, subkey = random.split(key)
    batch_keys = random.split(subkey, num=int(jnp.prod(jnp.array(target_shape))))
    
    # Reshape to target shape
    batch_keys = batch_keys.reshape(*target_shape, -1)
    
    return new_key, batch_keys


# ═══════════════════════════════════════════════════════════════════════════
# BASIC STOCHASTIC GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def uniform_samples(
    key: PRNGKeyArray,
    shape: Sequence[int],
    minval: float,
    maxval: float,
    dtype: jnp.dtype
) -> jnp.ndarray:
    """
    Generate uniform samples in [minval, maxval).
    
    Args:
        key: PRNG key
        shape: Output array shape
        minval: Lower bound (inclusive)
        maxval: Upper bound (exclusive)
        dtype: Data type (float32 or float64)
        
    Returns:
        JAX array with uniform samples
        
    References:
        - Python.tex §2.1: CMS Algorithm (uniform variables)
    """
    return random.uniform(
        key, 
        shape=shape, 
        minval=minval, 
        maxval=maxval,
        dtype=dtype
    )


def normal_samples(
    key: PRNGKeyArray,
    shape: Sequence[int],
    mean: float,
    std: float,
    dtype: jnp.dtype
) -> jnp.ndarray:
    """
    Generate normal (Gaussian) distribution samples.
    
    Args:
        key: PRNG key
        shape: Output array shape
        mean: Distribution mean
        std: Standard deviation
        dtype: Data type
        
    Returns:
        JAX array with normal samples
        
    References:
        - Tests_Python.tex §1.2: synthetic_brownian fixture
        - Python.tex §3: Brownian Processes
    """
    samples = random.normal(key, shape=shape, dtype=dtype)
    return mean + std * samples


def exponential_samples(
    key: PRNGKeyArray,
    shape: Sequence[int],
    rate: float,
    dtype: jnp.dtype
) -> jnp.ndarray:
    """
    Generate exponential distribution samples.
    
    Args:
        key: PRNG key
        shape: Output array shape
        rate: Rate parameter (lambda)
        dtype: Data type
        
    Returns:
        JAX array with exponential samples
        
    References:
        - Python.tex §2.1: CMS Algorithm (exponential variables)
    """
    samples = random.exponential(key, shape=shape, dtype=dtype)
    return samples / rate


# ═══════════════════════════════════════════════════════════════════════════
# VERIFICATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def check_prng_state(key: PRNGKeyArray) -> dict[str, Any]:
    """
    Verify the state of a PRNG key.
    
    Args:
        key: PRNG key to verify
        
    Returns:
        dict: Information about the key (shape, dtype, impl)
        
    Useful for debugging and configuration verification.
    """
    return {
        "shape": key.shape,
        "dtype": str(key.dtype),
        "impl": os.getenv("JAX_DEFAULT_PRNG_IMPL", "default"),
        "is_valid": key.shape == (2,),  # threefry2x32 uses shape (2,)
        "sample_value": int(key[0]) if len(key) > 0 else None
    }


def verify_determinism(
    seed: int,
    n_trials: int = 3
) -> bool:
    """
    Verify that PRNG is deterministic (same seed -> same results).
    
    Args:
        seed: Seed for test
        n_trials: Number of repetitions
        
    Returns:
        bool: True if all repetitions yield identical results
        
    References:
        - Tests_Python.tex §4: Determinism Test
        - API_Python.tex §5: Bit-Exact Reproducibility
        
    Example:
        >>> from Python.api.prng import verify_determinism
        >>> assert verify_determinism(seed=42, n_trials=5)
    """
    results = []
    
    for _ in range(n_trials):
        key = initialize_jax_prng(seed)
        sample = random.normal(key, shape=(10,))
        results.append(sample)
    
    # Verify all results are identical
    reference = results[0]
    all_equal = all(jnp.allclose(r, reference, atol=1e-10) for r in results)
    
    return bool(all_equal)


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Initialization
    "initialize_jax_prng",
    # Splitting
    "split_key",
    "split_key_like",
    # Generators
    "uniform_samples",
    "normal_samples",
    "exponential_samples",
    # Verification
    "check_prng_state",
    "verify_determinism",
]
