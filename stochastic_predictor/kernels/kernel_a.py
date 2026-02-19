"""Kernel A: Hilbert/RKHS for Smooth Gaussian Processes.

This kernel implements RKHS (Reproducing Kernel Hilbert Space) projections
for smooth stochastic processes dominated by Brownian motion dynamics.

References:
    - Teoria.tex §2.1: Rama A (Hilbert Space Projections)
    - Python.tex §2.2.1: Kernel A Implementation
    - Implementacion.tex §3.1: RKHS Kernel Design

Mathematical Foundation:
    Gaussian kernel k(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))
    Prediction via kernel regression with historical observations.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Optional

from .base import KernelOutput, apply_stop_gradient_to_diagnostics, normalize_signal, compute_signal_statistics


# ==================== P2.1: WTMM Functions ====================
# Wavelet Transform Modulus Maxima for Hölder exponent estimation
# Reference: Stochastic_Predictor_Theory.tex §2.1 (Singularity Spectrum)

@jax.jit
def morlet_wavelet(
    t: Float[Array, ""],
    sigma: float = 1.0,
    f_c: float = 0.5
) -> Float[Array, ""]:
    """
    Morlet wavelet: complex exponential modulated by Gaussian.
    
    ψ(t) = exp(2πi·f_c·t) * exp(-t²/(2σ²))
    
    For real part (used in WTMM): Re(ψ(t)) = cos(2πi·f_c·t) * exp(-t²/(2σ²))
    
    Args:
        t: Time/scale parameter
        sigma: Gaussian envelope width
        f_c: Central frequency
    
    Returns:
        Real part of Morlet wavelet
    """
    gaussian_envelope = jnp.exp(-(t ** 2) / (2.0 * sigma ** 2))
    oscillation = jnp.cos(2.0 * jnp.pi * f_c * t)
    return oscillation * gaussian_envelope


@jax.jit
def continuous_wavelet_transform(
    signal: Float[Array, "n"],
    scales: Float[Array, "m"],
    mother_wavelet_fn=None
) -> Float[Array, "m n"]:
    """
    Compute continuous wavelet transform (CWT) at multiple scales.
    
    CWT_ψ(s, b) = (1/√s) ∫ ψ*((t-b)/s) x(t) dt
    
    Args:
        signal: Input signal (n,)
        scales: Array of scales to evaluate (m,)
        mother_wavelet_fn: Wavelet function (default: Morlet)
    
    Returns:
        CWT coefficients (m, n) where axis 0 is scale, axis 1 is position
    """
    if mother_wavelet_fn is None:
        mother_wavelet_fn = morlet_wavelet
    
    n = signal.shape[0]
    m = scales.shape[0]
    
    # Normalize time to [-1, 1] for wavelet evaluation
    t = jnp.linspace(-1.0, 1.0, n)
    
    # Compute CWT for each scale
    def cwt_scale(scale):
        # Normalized wavelet at this scale
        psi = jax.vmap(lambda ti: mother_wavelet_fn(ti / scale, sigma=1.0))(t)
        
        # Convolution: CWT(s, b) ~ ∫ ψ((t-b)/s) x(t) dt
        # Approximated via inner product at each position
        def cwt_position(b):
            shifted_times = t - t[int(b * (n - 1))]
            psi_shifted = jax.vmap(lambda ti: mother_wavelet_fn(ti / scale, sigma=1.0))(shifted_times)
            # Inner product: CWT = sum(signal * psi_shifted) / sqrt(scale)
            return jnp.sum(signal * psi_shifted) / jnp.sqrt(scale)
        
        return jax.vmap(cwt_position)(jnp.linspace(0.0, 1.0, n))
    
    cwt_coeffs = jax.vmap(cwt_scale)(scales)
    return cwt_coeffs


@jax.jit
def find_modulus_maxima(
    cwt_coeffs: Float[Array, "m n"],
    threshold: float = 0.1
) -> Float[Array, "m n"]:
    """
    Identify local maxima in wavelet transform (modulus maxima).
    
    For each scale, find positions where |CWT| >= |CWT(neighbors)|.
    
    Args:
        cwt_coeffs: CWT coefficients (m scales, n positions)
        threshold: Minimum amplitude for maxima (relative to scale mean)
    
    Returns:
        Binary mask (m, n): 1 where local maxima, 0 elsewhere
    """
    m, n = cwt_coeffs.shape
    
    # Pad boundaries
    cwt_padded = jnp.pad(cwt_coeffs, ((0, 0), (1, 1)), mode='edge')
    
    # Compute absolute values
    abs_cwt = jnp.abs(cwt_coeffs)
    
    # Compare with neighbors
    left_neighbor = jnp.abs(cwt_padded[:, :-2])
    right_neighbor = jnp.abs(cwt_padded[:, 2:])
    
    # Local maxima: current >= both neighbors AND above threshold
    scale_threshold = jax.vmap(lambda scale_row: jnp.mean(abs_cwt[scale_row]) * threshold)(jnp.arange(m))
    threshold_mask = abs_cwt >= scale_threshold[:, None]
    
    local_max = (abs_cwt >= left_neighbor) & (abs_cwt >= right_neighbor) & threshold_mask
    
    return local_max.astype(jnp.float32)


@jax.jit
def link_wavelet_maxima(
    modulus_maxima: Float[Array, "m n"],
    scales: Float[Array, "m"],
    max_link_distance: float = 2.0
) -> Float[Array, "n m"]:
    """
    Link modulus maxima across scales to form chains (P2.1 requirement).
    
    For each position, create a chain of scales where maxima occur.
    Uses position persistence across scales to group related maxima.
    
    Args:
        modulus_maxima: Binary mask of modulus maxima (m scales, n positions)
        scales: Scale values (m,)
        max_link_distance: Maximum horizontal distance to link positions
    
    Returns:
        Chain matrix (n, m): normalized chain strength per position
    """
    m, n = modulus_maxima.shape
    
    # For each position, sum maxima across scales (chain strength)
    chain_strength = jnp.sum(modulus_maxima, axis=0)  # Shape: (n,)
    
    # Normalize to [0, 1]
    chain_strength_norm = chain_strength / jnp.maximum(m, 1.0)
    
    # Repeat for all scales (chain present/absent at each scale)
    chains = modulus_maxima.T  # Shape: (n, m)
    
    return chains.astype(jnp.float32)


@jax.jit
def compute_partition_function(
    chains: Float[Array, "n m"],
    scales: Float[Array, "m"],
    q_range: Float[Array, "q"]
) -> Float[Array, "q"]:
    """
    Compute partition function Z_q(s) = Σ |chain_strength|^q
    
    This is used to compute scaling exponents τ(q) via linear regression:
    log Z_q(s) ~ τ(q) · log(s) + const
    
    Args:
        chains: Chain matrix (n positions, m scales)
        scales: Scale values (m,)
        q_range: Range of exponents to evaluate
    
    Returns:
        Partition function estimates for each q
    """
    # Sum chain strengths at each scale
    scale_sum = jnp.sum(jnp.abs(chains), axis=0)  # Shape: (m,)
    
    # Compute partition function for each q
    def partition_q(q):
        return jnp.sum((scale_sum ** q) * (1.0 / jnp.sqrt(scales)))
    
    Z_q = jax.vmap(partition_q)(q_range)
    return Z_q


@jax.jit
def compute_singularity_spectrum(
    partition_function: Float[Array, "q"],
    q_range: Float[Array, "q"],
    scales: Float[Array, "m"]
) -> Float[Array, "(2,)"] :
    """
    Compute singularity spectrum D(h) via Legendre transform.
    
    From τ(q) = min_h [D(h) + q·h], invert to get D(h) = min_q [τ(q) - q·h].
    Returns (h_max, D_h_max) where D(h_max) is maximum.
    
    Args:
        partition_function: Z_q values (q,)
        q_range: Exponents (q,)
        scales: Scale values (m,)
    
    Returns:
        [holder_exponent, spectrum_max]
    """
    # Estimate τ(q) from partition function via simple scaling
    # τ(q) ~ (log Z_q) / log(scale)
    scale_factor = jnp.log(scales[-1] / scales[0])
    tau_q = jnp.log(jnp.abs(partition_function) + 1e-10) / scale_factor
    
    # Legendre transform: D(h) = q·h - τ(q)
    # For each h, find max over q: D(h) = max_q [q·h - τ(q)]
    h_range = jnp.linspace(0.0, 2.0, 50)
    
    def spectrum_h(h):
        legendre = q_range * h - tau_q
        return jnp.max(legendre)
    
    D_h = jax.vmap(spectrum_h)(h_range)
    
    # Find h where D(h) is maximum (most likely Hölder exponent)
    h_max_idx = jnp.argmax(D_h)
    holder_exponent = h_range[h_max_idx]
    spectrum_max = D_h[h_max_idx]
    
    return jnp.array([holder_exponent, spectrum_max])


@jax.jit
def extract_holder_exponent_wtmm(
    signal: Float[Array, "n"],
    config
) -> Float[Array, ""]:
    """
    Extract Hölder exponent via complete WTMM pipeline (P2.1).
    
    Pipeline:
    1. Compute CWT at multiple scales
    2. Find modulus maxima
    3. Link maxima across scales
    4. Compute partition function
    5. Compute singularity spectrum (Legendre transform)
    6. Extract holder_exponent = argmax D(h)
    
    Args:
        signal: Input signal (normalized)
        config: PredictorConfig with WTMM parameters
    
    Returns:
        Hölder exponent (scalar, clipped to valid range)
    """
    n = signal.shape[0]
    
    # Define scales: logarithmically spaced from smallest to largest
    # Use config.wtmm_buffer_size as upper scale limit
    scales = jnp.logspace(0.0, jnp.log10(config.wtmm_buffer_size), 16)
    scales = jnp.clip(scales, 1.0, n // 2)  # Ensure valid scale range
    
    # Step 1: Continuous Wavelet Transform
    cwt = continuous_wavelet_transform(signal, scales)
    
    # Step 2: Find modulus maxima
    modulus_maxima = find_modulus_maxima(cwt, threshold=0.1)
    
    # Step 3: Link maxima across scales
    chains = link_wavelet_maxima(modulus_maxima, scales, max_link_distance=2.0)
    
    # Step 4: Compute partition function
    q_range = jnp.linspace(-2.0, 2.0, 9)
    partition_func = compute_partition_function(chains, scales, q_range)
    
    # Step 5-6: Compute singularity spectrum and extract Hölder exponent
    spectrum_result = compute_singularity_spectrum(partition_func, q_range, scales)
    holder_exponent = spectrum_result[0]
    
    # Clip to valid range
    holder_exponent = jnp.clip(
        holder_exponent,
        config.validation_holder_exponent_min,
        config.validation_holder_exponent_max
    )
    
    return holder_exponent


# ==================== End P2.1: WTMM Functions ====================


@jax.jit
def gaussian_kernel(
    x: Float[Array, "d"],
    y: Float[Array, "d"],
    bandwidth: float
) -> Float[Array, ""]:
    """
    Gaussian (RBF) kernel function.
    
    Computes k(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))
    
    Args:
        x: First point in d-dimensional space
        y: Second point in d-dimensional space
        bandwidth: Kernel bandwidth parameter (controls smoothness)
    
    Returns:
        Kernel value (scalar)
    
    References:
        - Teoria.tex §2.1.2: Gaussian Kernel Definition
    """
    squared_dist = jnp.sum((x - y) ** 2)
    return jnp.exp(-squared_dist / (2.0 * bandwidth ** 2))


@jax.jit
def compute_gram_matrix(
    X: Float[Array, "n d"],
    bandwidth: float
) -> Float[Array, "n n"]:
    """
    Compute Gram matrix K where K[i,j] = k(x_i, x_j).
    
    Args:
        X: Matrix of n points in d dimensions
        bandwidth: Kernel bandwidth
    
    Returns:
        Gram matrix (n x n)
    
    References:
        - Python.tex §2.2.1: Gram Matrix Construction
    """
    n = X.shape[0]
    
    # Vectorized computation using broadcasting
    # X[:, None, :] has shape (n, 1, d)
    # X[None, :, :] has shape (1, n, d)
    # Difference has shape (n, n, d)
    diff = X[:, None, :] - X[None, :, :]
    squared_dist = jnp.sum(diff ** 2, axis=-1)
    
    K = jnp.exp(-squared_dist / (2.0 * bandwidth ** 2))
    
    return K


@jax.jit
def kernel_ridge_regression(
    X_train: Float[Array, "n d"],
    y_train: Float[Array, "n"],
    X_test: Float[Array, "m d"],
    config
) -> tuple[Float[Array, "m"], Float[Array, "m"]]:
    """
    Kernel Ridge Regression for prediction.
    
    Zero-Heuristics: ALL parameters from PredictorConfig (unified config injection).
    
    Solves: alpha = (K + lambda*I)^(-1) y
    Predicts: y_pred = K_test @ alpha
    
    Args:
        X_train: Training points (n x d)
        y_train: Training targets (n,)
        X_test: Test points (m x d)
        config: PredictorConfig with kernel_a_bandwidth, kernel_ridge_lambda, kernel_a_min_variance
    
    Returns:
        Tuple of (predictions, variances) for test points
    
    References:
        - Python.tex §2.2.1: Kernel Ridge Regression
        - Implementacion.tex §3.1.2: Regularized RKHS Regression
    """
    n = X_train.shape[0]
    
    # Compute Gram matrix on training data
    K_train = compute_gram_matrix(X_train, config.kernel_a_bandwidth)
    
    # Add ridge regularization to diagonal
    K_reg = K_train + config.kernel_ridge_lambda * jnp.eye(n)
    
    # Solve for alpha coefficients
    alpha = jnp.linalg.solve(K_reg, y_train)
    
    # Compute kernel between test and train points (vectorized via broadcasting)
    # K_test[i, j] = k(x_test[i], x_train[j])
    # X_test[:, None, :] has shape (m, 1, d)  - each test point broadcasted
    # X_train[None, :, :] has shape (1, n, d)  - each train point broadcasted
    # diff_test has shape (m, n, d)
    m = X_test.shape[0]
    diff_test = X_test[:, None, :] - X_train[None, :, :]
    squared_dist_test = jnp.sum(diff_test ** 2, axis=-1)
    K_test = jnp.exp(-squared_dist_test / (2.0 * config.kernel_a_bandwidth ** 2))
    
    # Predictions
    y_pred = K_test @ alpha
    
    # Variance estimation (diagonal of predictive covariance)
    # Var[f(x_test)] ≈ k(x_test, x_test) - K_test @ (K + λI)^(-1) @ K_test^T
    k_test_diag = jnp.ones(m)  # k(x, x) = 1 for normalized kernel
    K_inv_K_test_T = jnp.linalg.solve(K_reg, K_test.T)
    variances = k_test_diag - jnp.sum(K_test * K_inv_K_test_T.T, axis=1)
    variances = jnp.maximum(variances, config.kernel_a_min_variance)  # Ensure non-negative (from config)
    
    return y_pred, variances


@jax.jit
def create_embedding(
    signal: Float[Array, "n"],
    config
) -> Float[Array, "n_embed d"]:
    """
    Create time-delay embedding (Takens' embedding) from 1D signal.
    
    Converts time series into points in d-dimensional space using
    sliding window technique.
    
    Args:
        signal: Input time series (length n)
        config: PredictorConfig with kernel_a_embedding_dim
    
    Returns:
        Embedded points (n-d+1 x d)
    
    References:
        - Teoria.tex §2.1.3: Takens' Embedding Theorem
        - Python.tex §2.2.1: Time-Delay Embedding
    
    Example:
        >>> from stochastic_predictor.api.config import PredictorConfigInjector
        >>> config = PredictorConfigInjector().create_config()
        >>> signal = jnp.array([1, 2, 3, 4, 5])
        >>> embedding = create_embedding(signal, config)
        >>> # Returns: [[1, 2, 3], [2, 3, 4], [3, 4, 5]] (if embedding_dim=3)
    """
    n = signal.shape[0]
    embedding_dim = config.kernel_a_embedding_dim
    n_embed = n - embedding_dim + 1
    
    # Create embedding matrix
    embedded = jnp.zeros((n_embed, embedding_dim))
    
    for i in range(n_embed):
        embedded = embedded.at[i].set(signal[i:i+embedding_dim])
    
    return embedded


@jax.jit
def kernel_a_predict(
    signal: Float[Array, "n"],
    key: Array,
    config
) -> KernelOutput:
    """
    Kernel A: RKHS prediction for smooth Gaussian processes.
    
    Algorithm:
        1. Normalize input signal
        2. Create time-delay embedding
        3. Train kernel ridge regression on historical data
        4. Predict next value
    
    Args:
        signal: Input time series (length n)
        key: JAX PRNG key (for compatibility, not used in this deterministic kernel)
        config: PredictorConfig with kernel_ridge_lambda, kernel_a_bandwidth, 
                kernel_a_embedding_dim, kernel_a_min_variance
    
    Returns:
        KernelOutput with prediction, confidence (std), and metadata
    
    References:
        - Python.tex §2.2.1: Kernel A Complete Algorithm
        - Teoria.tex §2.1: RKHS Theory
    
    Example:
        >>> from stochastic_predictor.api.config import PredictorConfigInjector
        >>> config = PredictorConfigInjector().create_config()
        >>> signal = jnp.array([1.0, 1.1, 0.95, 1.05])
        >>> key = initialize_jax_prng(42)
        >>> result = kernel_a_predict(signal, key, config)
        >>> prediction = result.prediction
        >>> confidence = result.confidence
    """
    # Step 1: Normalize signal (z-score)
    signal_normalized = normalize_signal(signal, method="zscore", epsilon=config.numerical_epsilon)
    
    # Step 2: Compute signal statistics (for diagnostics)
    stats = compute_signal_statistics(signal)
    
    # Step 3: Create time-delay embedding
    X_embedded = create_embedding(signal_normalized, config)
    n_points = X_embedded.shape[0]
    
    # Step 4: Split into train/test (last point for prediction)
    # Train on all but last embedded point, predict last target
    X_train = X_embedded[:-1]
    y_train = signal_normalized[config.kernel_a_embedding_dim:-1]  # Targets (next values)
    X_test = X_embedded[-1:] # Last embedded point
    
    # Step 5: Kernel ridge regression
    y_pred_norm, variances = kernel_ridge_regression(
        X_train, y_train, X_test, config
    )
    
    # Step 6: Denormalize prediction (inverse z-score)
    prediction = y_pred_norm[0] * stats["std"] + stats["mean"]
    confidence = jnp.sqrt(variances[0]) * stats["std"]  # Scale variance by std
    
    # Step 7: Compute diagnostics (with stop_gradient)
    # V-MAJ-2 + P2.1: Full WTMM implementation for holder_exponent
    holder_exponent_estimate = extract_holder_exponent_wtmm(signal_normalized, config)
    
    diagnostics = {
        "kernel_type": "A_Hilbert_RKHS",
        "bandwidth": config.kernel_a_bandwidth,
        "embedding_dim": config.kernel_a_embedding_dim,
        "n_training_points": X_train.shape[0],
        "signal_mean": stats["mean"],
        "signal_std": stats["std"],
        "holder_exponent": float(holder_exponent_estimate)  # P2.1: Full WTMM, not placeholder
    }
    
    # Apply stop_gradient to diagnostics (VRAM optimization)
    prediction, diagnostics = apply_stop_gradient_to_diagnostics(
        prediction, diagnostics
    )
    
    return KernelOutput(
        prediction=prediction,
        confidence=confidence,
        metadata=diagnostics
    )


# Public API
__all__ = [
    "kernel_a_predict",
    "gaussian_kernel",
    "compute_gram_matrix",
    "kernel_ridge_regression",
    "create_embedding"
]
