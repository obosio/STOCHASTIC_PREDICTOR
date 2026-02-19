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
    diagnostics = {
        "kernel_type": "A_Hilbert_RKHS",
        "bandwidth": config.kernel_a_bandwidth,
        "embedding_dim": config.kernel_a_embedding_dim,
        "n_training_points": X_train.shape[0],
        "signal_mean": stats["mean"],
        "signal_std": stats["std"]
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
