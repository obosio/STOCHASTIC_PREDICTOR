"""Kernel D: Signature-based Prediction for Rough Paths.

This kernel uses signature methods (via Signax library) for prediction on
rough paths with low Hölder regularity (H < 0.5).

References:
    - Teoria.tex §5: Rama D (Rough Paths + Signatures)
    - Python.tex §2.2.4: Kernel D (Log-Signatures)
    - Implementacion.tex §3.4: Signature Truncation and Memory

Mathematical Foundation:
    Signature S(X) of path X is the sequence of iterated integrals:
    S(X)_0,t = (1, ∫dX, ∫∫dX⊗dX, ...)
    
    Log-signature provides a more compact representation via BCH formula.
    Truncation depth M controls approximation quality vs. memory.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import signax
from typing import Optional

from .base import KernelOutput, apply_stop_gradient_to_diagnostics


@jax.jit
def compute_log_signature(
    path: Float[Array, "n d"],
    config
) -> Float[Array, "signature_dim"]:
    """
    Compute log-signature of a path using Signax.
    
    The log-signature is a more compact representation than the full
    signature, using the Baker-Campbell-Hausdorff formula.
    
    Args:
        path: Discrete path (n time steps, d dimensions)
        config: PredictorConfig with kernel_d_depth
    
    Returns:
        Log-signature vector (dimension depends on depth and d)
    
    References:
        - Teoria.tex §5.1: Signature Transform
        - Python.tex §2.2.4: Signax Integration
    
    Example:
        >>> from stochastic_predictor.api.config import PredictorConfigInjector
        >>> config = PredictorConfigInjector().create_config()
        >>> path = jnp.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        >>> logsig = compute_log_signature(path, config)
    """
    # Signax expects shape (batch, length, channels) but path is (length, channels)
    # Add batch dimension
    path_batched = path[None, :, :]  # Shape: (1, n, d)
    
    # Compute log-signature
    logsig = signax.logsignature(path_batched, depth=config.kernel_d_depth)
    
    # Remove batch dimension
    logsig_unbatched = logsig[0]  # Shape: (signature_dim,)
    
    return logsig_unbatched


@jax.jit
def create_path_augmentation(
    signal: Float[Array, "n"]
) -> Float[Array, "n 2"]:
    """
    Create 2D path from 1D signal via time augmentation.
    
    Converts signal [y_1, y_2, ..., y_n] into path [(0, y_1), (1, y_2), ..., (n-1, y_n)]
    where first coordinate is time.
    
    Args:
        signal: 1D time series (length n)
    
    Returns:
        2D path (n x 2) with time in first column, values in second
    
    References:
        - Teoria.tex §5.2: Time Augmentation
        - Python.tex §2.2.4: Path Construction
    """
    n = signal.shape[0]
    time_coords = jnp.arange(n, dtype=jnp.float64)
    path = jnp.stack([time_coords, signal.astype(jnp.float64)], axis=1)
    return path


@jax.jit
def reparameterize_path(
    path: Float[Array, "n d"]
) -> Float[Array, "n d"]:
    """
    Apply monotone time reparameterization for invariance check.

    Uses quadratic time warp t -> t^2 and resamples path.

    Args:
        path: Input path (n x d)

    Returns:
        Reparameterized path (n x d)

    References:
        - Theory.tex §5.2: Reparametrization invariance
    """
    n = path.shape[0]
    t = jnp.linspace(0.0, 1.0, n)
    warped = t ** 2
    indices = jnp.clip((warped * (n - 1)).astype(jnp.int32), 0, n - 1)
    return path[indices]


@jax.jit
def predict_from_signature(
    logsig: Float[Array, "signature_dim"],
    last_value: float,
    config
) -> tuple[float, float]:
    """
    Generate prediction from log-signature features.
    
    This is a simple linear extrapolation model. In production,
    this would be replaced with a trained model (e.g., signature kernel
    or neural network on signature features).
    
    Args:
        logsig: Log-signature vector
        last_value: Last observed value (for baseline prediction)
        config: PredictorConfig with kernel_d_alpha, kernel_d_confidence_scale
    
    Returns:
        Tuple of (prediction, confidence)
    
    References:
        - Python.tex §2.2.4: Signature Regression
        - Teoria.tex §5.3: Signature-based Forecasting
    
    Note:
        This simplified version uses the L2 norm of signature as a
        trend indicator. Production models should use signature kernels
        or neural networks trained on historical data.
    """
    # Compute signature magnitude (proxy for path activity)
    sig_norm = jnp.linalg.norm(logsig)
    
    # Simple heuristic: prediction = last_value + alpha * sign(first_sig_component)
    # where alpha is scaled by signature magnitude (from config, NOT hardcoded)
    if logsig.shape[0] > 1:
        direction = jnp.sign(logsig[1])  # First non-trivial component
    else:
        direction = 0.0
    
    # Prediction: slight extrapolation based on signature trend
    prediction = last_value + config.kernel_d_alpha * direction * sig_norm
    
    # Confidence: Scale from config (Zero-Heuristics: all factors from config)
    # More activity (larger sig_norm) = less certainty
    confidence = config.kernel_d_confidence_scale * (config.kernel_d_confidence_base + sig_norm)
    
    return prediction, confidence


@jax.jit
def kernel_d_predict(
    signal: Float[Array, "n"],
    key: Array,
    config
) -> KernelOutput:
    """
    Kernel D: Signature-based prediction for rough paths.
    
    Algorithm:
        1. Create 2D path from 1D signal (time augmentation)
        2. Compute log-signature up to truncation depth
        3. Extract prediction from signature features
        4. Return with confidence estimate
    
    Zero-Heuristics Policy: All hyperparameters MUST be injected from config.
    No hardcoded constants remain in kernel implementation.
    
    Args:
        signal: Input time series (historical trajectory)
        key: JAX PRNG key (for compatibility, not used in deterministic signature)
        config: PredictorConfig with kernel_d_depth and kernel_d_alpha parameters
    
    Returns:
        KernelOutput with prediction, confidence, and diagnostics
    
    References:
        - Python.tex §2.2.4: Kernel D Complete Algorithm
        - Teoria.tex §5: Rough Paths Theory
        - Implementacion.tex §3.4: Signature Memory Optimization
    
    Example:
        >>> from stochastic_predictor.api.config import PredictorConfigInjector
        >>> signal = jnp.array([1.0, 1.2, 1.1, 1.3, 1.2])
        >>> key = initialize_jax_prng(42)
        >>> config = PredictorConfigInjector().create_config()
        >>> result = kernel_d_predict(signal, key, config)
        >>> prediction = result.prediction
    
    Note:
        Activated when Hölder exponent H < holder_threshold (typically 0.4)
        indicating rough/irregular dynamics where classical methods fail.
    """
    # Step 1: Create 2D path (time augmentation)
    path = create_path_augmentation(signal)
    
    # Step 2: Compute log-signature
    logsig = compute_log_signature(path, config)

    # Reparametrization invariance check (diagnostic)
    path_warped = reparameterize_path(path)
    logsig_warped = compute_log_signature(path_warped, config)
    reparam_invariance_error = jnp.linalg.norm(logsig_warped - logsig)
    
    # Step 3: Predict from signature (config injection pattern)
    last_value = signal[-1]
    prediction, confidence = predict_from_signature(
        logsig, 
        last_value, 
        config
    )
    
    # Diagnostics
    diagnostics = {
        "kernel_type": "D_Signature_Rough_Paths",
        "signature_depth": config.kernel_d_depth,
        "signature_dim": logsig.shape[0],
        "signature_norm": jnp.linalg.norm(logsig),
        "path_length": path.shape[0],
        "last_value": last_value,
        "reparam_invariance_error": reparam_invariance_error,
    }
    
    # Apply stop_gradient to diagnostics
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
    "kernel_d_predict",
    "compute_log_signature",
    "create_path_augmentation",
    "predict_from_signature"
]
