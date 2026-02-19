"""
Global pytest configuration and shared fixtures.

This file defines reusable fixtures for all project tests,
ensuring deterministic reproducibility through fixed seeds.

References:
    - Predictor_Estocastico_Tests_Python.tex §1.2: Shared Fixtures
    - Predictor_Estocastico_API_Python.tex §5: Bit-Exact Determinism
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from typing import Tuple

# Import utilities from project
from stochastic_predictor.api.prng import initialize_jax_prng


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL PYTEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """
    Hook for pytest global configuration before test execution.
    
    Configures JAX for bit-exact determinism per specification.
    
    References:
        - API_Python.tex §5.1: XLA Environment Variables
    """
    import os
    
    # CRITICAL: Configure BEFORE any JAX operation
    os.environ['JAX_DEFAULT_PRNG_IMPL'] = 'threefry2x32'
    os.environ['JAX_DETERMINISTIC_REDUCTIONS'] = '1'
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
    
    # Configure tensor precision
    jax.config.update('jax_enable_x64', True)
    jax.config.update("jax_default_matmul_precision", "highest")


# ═══════════════════════════════════════════════════════════════════════════
# PRNG FIXTURES (Reproducibility)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def rng_key():
    """
    Fixture: Deterministic PRNG key for reproducibility.
    
    Returns:
        PRNGKeyArray: JAX key with seed=42 (threefry2x32)
        
    References:
        - Tests_Python.tex §1.2: rng_key fixture
        - Python.tex §1: JAX PRNG
        
    Example:
        >>> def test_random_generation(rng_key):
        ...     samples = random.normal(rng_key, shape=(10,))
        ...     assert samples.shape == (10,)
    """
    return initialize_jax_prng(seed=42)


@pytest.fixture
def rng_key_alt():
    """
    Fixture: Alternative PRNG key (different seed).
    
    Useful for tests requiring multiple independent streams.
    
    Returns:
        PRNGKeyArray: JAX key with seed=123
    """
    return initialize_jax_prng(seed=123)


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_brownian() -> Tuple[np.ndarray, float]:
    """
    Fixture: Synthetic Brownian motion trajectory for tests.
    
    Generates a standard Brownian process W(t) with independent Gaussian
    increments, used to validate kernels and SDEs.
    
    Returns:
        Tuple (X: np.ndarray, dt: float):
            - X: Brownian motion trajectory [N] (N=1000)
            - dt: Time step (0.001)
            
    References:
        - Tests_Python.tex §1.2: synthetic_brownian fixture
        - Python.tex §3: Brownian Processes
        
    Example:
        >>> def test_brownian_properties(synthetic_brownian):
        ...     X, dt = synthetic_brownian
        ...     assert X.shape == (1000,)
        ...     # E[W(T)] ≈ 0
        ...     assert abs(X[-1]) < 5.0 * np.sqrt(1.0)  # 5-sigma check
    """
    np.random.seed(123)
    T = 1.0
    N = 1000
    dt = T / N
    
    # Generate increments dW ~ N(0, sqrt(dt))
    dW = np.random.randn(N) * np.sqrt(dt)
    
    # Integrate: W(t) = ∫ dW
    X = np.cumsum(dW)
    
    return X, dt


@pytest.fixture
def synthetic_levy_stable() -> Tuple[np.ndarray, float]:
    """
    Fixture: Stable Lévy process trajectory.
    
    Generates samples from an alpha-stable distribution with parameters:
        - alpha = 1.5 (heavy tails)
        - beta = 0.0 (symmetric)
    
    Returns:
        Tuple (samples: np.ndarray, alpha: float):
            - samples: Array of 1000 samples
            - alpha: Stability parameter used
            
    References:
        - Tests_Python.tex §1.2: synthetic_levy_stable fixture
        - Teoria.tex §2.2: Alpha-Stable Distributions
        
    Example:
        >>> def test_levy_generation(synthetic_levy_stable):
        ...     samples, alpha = synthetic_levy_stable
        ...     assert samples.shape == (1000,)
        ...     assert alpha == 1.5
    """
    from scipy.stats import levy_stable
    
    np.random.seed(456)
    alpha = 1.5  # Stability index
    beta = 0.0   # Symmetry
    
    samples = np.asarray(levy_stable.rvs(alpha, beta, size=1000))
    
    return samples, alpha


@pytest.fixture
def mock_market_data() -> np.ndarray:
    """
    Fixture: Synthetic market data with regime change.
    
    Generates time series with two regimes:
        - Regime 1 (t=0..499): Low volatility (σ=0.01, μ=100)
        - Regime 2 (t=500..999): High volatility (σ=0.05, μ=105)
    
    Design: Simulates market shock (flash crash, volatility spike)
    to validate CUSUM and regime change detection.
    
    Returns:
        np.ndarray: Time series [1000] with regime change at t=500
        
    References:
        - Tests_Python.tex §1.2: mock_market_data fixture
        - API_Python.tex §3.2: CUSUM Drift Detection
        
    Example:
        >>> def test_regime_detection(mock_market_data):
        ...     data = mock_market_data
        ...     assert data.shape == (1000,)
        ...     # Verify level change
        ...     assert np.mean(data[500:]) > np.mean(data[:500])
    """
    np.random.seed(789)
    
    # Regime 1: low volatility
    regime1 = np.random.randn(500) * 0.01 + 100
    
    # Regime 2: high volatility (abrupt change)
    regime2 = np.random.randn(500) * 0.05 + 105
    
    data = np.concatenate([regime1, regime2])
    
    return data


@pytest.fixture
def dgm_reference_solution():
    """
    Fixture: Black-Scholes reference solution to validate DGM.
    
    Returns function computing analytic European call price
    under Black-Scholes model, used as ground truth to validate
    Deep Galerkin Method convergence (Kernel B).
    
    Returns:
        Callable: Function bs_call(S, K, T, r, sigma) -> price
        
    References:
        - Tests_Python.tex §1.2: dgm_reference_solution fixture
        - Python.tex §4: Kernel B (DGM for HJB)
        
    Example:
        >>> def test_dgm_convergence(dgm_reference_solution):
        ...     bs_call = dgm_reference_solution
        ...     price = bs_call(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        ...     assert 0 < price < 100  # Sanity check
    """
    from scipy.stats import norm
    
    def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Black-Scholes European Call Option Price.
        
        Args:
            S: Spot price of underlying
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            
        Returns:
            float: European call price
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        return call_price
    
    return bs_call


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def default_config():
    """
    Fixture: Default predictor configuration.
    
    Returns:
        PredictorConfig: Configuration with default values
        
    References:
        - API_Python.tex §1.1: PredictorConfig
        - config.toml: Reference values
    """
    from stochastic_predictor.api.types import PredictorConfig
    
    return PredictorConfig()


@pytest.fixture
def test_config_low_threshold():
    """
    Fixture: Configuration with low thresholds (edge case testing).
    
    Useful for tests requiring degraded or emergency mode triggers.
    
    Returns:
        PredictorConfig: Configuration with holder_threshold=0.1
    """
    from stochastic_predictor.api.types import PredictorConfig
    
    return PredictorConfig(
        holder_threshold=0.1,       # Highly sensitive circuit breaker
        cusum_h=2.0,                # Sensitive CUSUM
        grace_period_steps=5,       # Short refractory period
    )


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def jax_device():
    """
    Fixture: Current JAX device information.
    
    Returns:
        jax.Device: Default device (CPU or GPU)
        
    Useful for portability tests (CPU vs GPU).
    
    References:
        - Tests_Python.tex §4: Portability Test
        - API_Python.tex §5.2: CPU/GPU Parity
    """
    return jax.devices()[0]


@pytest.fixture
def assert_jax_deterministic():
    """
    Fixture: Helper function to verify JAX operation determinism.
    
    Returns:
        Callable: Function assert_deterministic(fn, *args, n_trials=3)
        
    Example:
        >>> def test_levy_determinism(assert_jax_deterministic, rng_key):
        ...     def generate_levy(key):
        ...         return levy_stable_cms(key, alpha=1.5, beta=0.0)
        ...     
        ...     assert_jax_deterministic(generate_levy, rng_key, n_trials=5)
    """
    def assert_deterministic(fn, *args, n_trials: int = 3, atol: float = 1e-10):
        """
        Verify that fn(*args) produces identical results over multiple trials.
        
        Args:
            fn: Function to test
            *args: Function arguments
            n_trials: Number of repetitions
            atol: Absolute tolerance for equality
            
        Raises:
            AssertionError: If results differ between trials
        """
        results = [fn(*args) for _ in range(n_trials)]
        
        reference = results[0]
        for i, result in enumerate(results[1:], start=1):
            assert jnp.allclose(result, reference, atol=atol), \
                f"Trial {i} differs from reference (max_diff={jnp.max(jnp.abs(result - reference))})"
    
    return assert_deterministic
