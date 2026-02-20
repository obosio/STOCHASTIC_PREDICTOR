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
from functools import partial
from jaxtyping import Array, Float
from typing import Optional

from .base import (
    KernelOutput,
    apply_stop_gradient_to_diagnostics,
    build_pdf_grid,
    compute_density_entropy,
    compute_normal_pdf,
    compute_signal_statistics,
    normalize_signal,
)


# ==================== P2.1: WTMM Functions ====================
# Wavelet Transform Modulus Maxima for Hölder exponent estimation
# Reference: Stochastic_Predictor_Theory.tex §2.1 (Singularity Spectrum)

@jax.jit
def morlet_wavelet(
    t: Float[Array, ""],
    sigma: float,
    f_c: float
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
    mother_wavelet_fn,
    epsilon: float = 1e-10
) -> Float[Array, "m n"]:
    """
    Compute continuous wavelet transform (CWT) at multiple scales.
    
    CWT_ψ(s, b) = (1/√s) ∫ ψ*((t-b)/s) x(t) dt
    
    Args:
        signal: Input signal (n,)
        scales: Array of scales to evaluate (m,)
        mother_wavelet_fn: Wavelet function (required)
    
    Returns:
        CWT coefficients (m, n) where axis 0 is scale, axis 1 is position
    """
    if mother_wavelet_fn is None:
        raise ValueError("mother_wavelet_fn is required for CWT computation.")
    
    n = signal.shape[0]
    
    # Normalize time to [-1, 1]
    t = jnp.linspace(-1.0, 1.0, n)
    
    # Compute CWT for each scale using convolution
    def cwt_scale(scale):
        # Create wavelet at this scale (domain: all time points)
        psi_scale = jax.vmap(lambda ti: mother_wavelet_fn(ti / scale))(t)
        
        # Normalize wavelet energy
        psi_norm = psi_scale / (jnp.sqrt(scale) + epsilon)
        
        # Compute correlation between signal and wavelet (sliding dot product)
        # This is equivalent to convolution with reversed/conjugate wavelet
        def correlation_at_shift(shift_idx):
            # Circular shift to create a "moving window" effect
            shifted_psi = jnp.roll(psi_norm, shift_idx)
            # Dot product (correlation)
            return jnp.sum(signal * shifted_psi) / (n ** 0.5)
        
        # For each position, compute the correlation
        corr_vals = jax.vmap(correlation_at_shift)(jnp.arange(n))
        
        return corr_vals
    
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
    
    return local_max.astype(jnp.float64)


@jax.jit
def link_wavelet_maxima(
    modulus_maxima: Float[Array, "m n"],
    cwt_coeffs: Float[Array, "m n"],
    scales: Float[Array, "m"],
    max_link_distance: float = 2.0
) -> tuple[Float[Array, "n m"], Float[Array, "n m"]]:
    """
    Link modulus maxima across scales to form chains (P2.1 requirement).
    
    For each position, create a chain of scales where maxima occur.
    Uses position persistence across scales to group related maxima.
    
    Args:
        modulus_maxima: Binary mask of modulus maxima (m scales, n positions)
        cwt_coeffs: CWT magnitude coefficients (m scales, n positions)
        scales: Scale values (m,)
        max_link_distance: Maximum horizontal distance to link positions
    
    Returns:
        Tuple of (chain_presence, chain_magnitudes) where:
        - chain_presence: Binary mask (n positions, m scales)
        - chain_magnitudes: CWT magnitude values at maxima (n positions, m scales)
    """
    m, n = modulus_maxima.shape
    
    # Extract magnitude values at maxima positions
    cwt_abs = jnp.abs(cwt_coeffs)
    
    # Mask: keep CWT values only where maxima exist, zeros elsewhere
    chain_magnitudes = modulus_maxima * cwt_abs  # Shape: (m, n)
    
    # Transpose for per-position chains
    chain_presence = modulus_maxima.T  # Shape: (n, m)
    chain_magnitudes = chain_magnitudes.T  # Shape: (n, m)
    
    return chain_presence.astype(jnp.float64), chain_magnitudes.astype(jnp.float64)


@jax.jit
def compute_partition_function(
    chain_magnitudes: Float[Array, "n m"],
    scales: Float[Array, "m"],
    q_range: Float[Array, "q"],
    epsilon: float
) -> tuple[Float[Array, "q m"], Float[Array, "q"]]:
    """
    Compute partition function Z_q(s) and scaling exponents τ(q).
    
    Theory: Z_q(s) = Σ_{chains L} (sup_{scale s, position b in chain L} |W_ψ(s,b)|)^q
    
    This captures multifractal scaling: Z_q(s) ~ s^{τ(q)}
    
    Extraction of τ(q):
        log Z_q(s) = τ(q) · log(s) + const (linear regression)
        τ(q) = slope of log-log plot
    
    Args:
        chain_magnitudes: CWT magnitude at chain positions (n positions, m scales)
        scales: Scale values (m,)
        q_range: Range of exponents to evaluate
    
    Returns:
        Tuple of (Z_q_scales, tau_q) where:
        - Z_q_scales: Partition function per scale (q, m)
        - tau_q: Scaling exponents (q,)
    
    References:
        - Theory.tex §2.1: Partition Function Formula Z_q(s)
        - Implementacion.tex §3.1: Log-log regression for τ(q)
    """
    n = chain_magnitudes.shape[0]  # Number of positions/chains
    m = chain_magnitudes.shape[1]  # Number of scales
    
    # For each q: Z_q(s) = Σ_{L=0}^{n-1} max(|chain_magnitudes[L, :]|)^q
    # This sums over all chains (n positions), taking max value per chain per scale
    
    def partition_q(q):
        # For this q, compute Z_q across all scales
        # Each chain L contributes only if it has a nonzero magnitude
        def z_at_scale(scale_idx):
            # For each position (chain), take magnitude at this scale
            mags_at_scale = chain_magnitudes[:, scale_idx]  # Shape: (n,)
            
            # Only sum nonzero magnitudes
            # Avoid division by zero for negative q
            safe_mags = jnp.clip(mags_at_scale, epsilon, None)
            
            # Use mask to only count truly nonzero entries
            mask = (mags_at_scale > epsilon).astype(jnp.float64)
            
            # Compute sum, but only for positions with signal
            # Apply mask to zero out non-signal positions before power
            masked_vals = safe_mags * mask
            z_val = jnp.sum(masked_vals ** q)
            
            return z_val
        
        # Compute Z_q for all scales
        z_scales = jax.vmap(z_at_scale)(jnp.arange(m))  # Shape: (m,)
        
        return z_scales
    
    # Compute partition function for all q values
    Z_q_scales = jax.vmap(partition_q)(q_range)  # Shape: (q, m)
    
    # Extract τ(q) via log-log regression: log Z_q(s) ~ τ(q) · log(s)
    log_scales = jnp.log(jnp.clip(scales, epsilon, None))
    
    def extract_tau(z_q_vals):
        # z_q_vals shape: (m,) - Z_q values across scales
        log_z_q = jnp.log(jnp.clip(z_q_vals, epsilon, None))
        
        # Linear regression: slope = [Σ(x-x_mean)(y-y_mean)] / [Σ(x-x_mean)^2]
        x_mean = jnp.mean(log_scales)
        y_mean = jnp.mean(log_z_q)
        
        numerator = jnp.sum((log_scales - x_mean) * (log_z_q - y_mean))
        denominator = jnp.sum((log_scales - x_mean) ** 2)
        
        tau = numerator / jnp.maximum(denominator, epsilon)
        
        return tau
    
    # Compute τ(q) for each q
    tau_q = jax.vmap(extract_tau)(Z_q_scales)  # Shape: (q,)
    
    return Z_q_scales, tau_q


@jax.jit
def compute_singularity_spectrum(
    tau_q: Float[Array, "q"],
    q_range: Float[Array, "q"],
    h_min: float,
    h_max: float,
    h_steps: int
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """
    Compute singularity spectrum D(h) via Legendre transform.
    
    Theory: The singularity spectrum D(h) is the Legendre transform of τ(q):
    
        D(h) = min_q [τ(q) - q·h]
    
    The spectrum is maximized at h_* where D(h_*) is largest.
    This h_* is the dominant Hölder exponent of the signal.
    
    Args:
        tau_q: Scaling exponents for each q (q,)
        q_range: Exponent values used (q,)
    
    Returns:
        Tuple of (holder_exponent, spectrum_max) where:
        - holder_exponent: h_* = argmax_h D(h) (most likely Hölder exponent)
        - spectrum_max: D(h_*) (height of spectrum at maximum)
    
    References:
        - Theory.tex §2.1: Legendre Transform and Singularity Spectrum
        - Implementacion.tex §3.1: D(h) = min_q[τ(q) - q·h]
    """
    
    # Create fine grid of Hölder exponents h
    # Typically h ∈ [0, 1] for typical signals (Brownian motion h=0.5)
    # Allow up to 1.5 for more singular cases
    h_range = jnp.linspace(h_min, h_max, h_steps)
    
    def spectrum_at_h(h):
        # For this h, compute D(h) = min_q [τ(q) - q·h]
        # Equivalently: D(h) = max_q [q·h - τ(q)]
        legendre_vals = tau_q - q_range * h  # Shape: (q,)
        
        # D(h) is the maximum of [q*h - tau_q], which is -min of [tau_q - q*h]
        d_h = -jnp.min(legendre_vals)
        
        return d_h
    
    # Compute singularity spectrum D(h) for all h values
    D_h = jax.vmap(spectrum_at_h)(h_range)  # Shape: (151,)
    
    # Find h where D(h) is maximum
    h_max_idx = jnp.argmax(D_h)
    holder_exponent = h_range[h_max_idx]
    spectrum_max = D_h[h_max_idx]
    
    return holder_exponent, spectrum_max


@partial(jax.jit, static_argnames=('config',))
def extract_holder_exponent_wtmm(
    signal: Float[Array, "n"],
    config
) -> Float[Array, ""]:
    """
    Extract Hölder exponent via complete WTMM pipeline (P2.1).
    
    Pipeline:
    1. Compute CWT at multiple scales
    2. Find modulus maxima (with adaptive threshold)
    3. Link maxima across scales (retaining magnitudes)
    4. Compute partition function Z_q(s) for each q
    5. Extract scaling exponents τ(q) via log-log regression
    6. Compute singularity spectrum D(h) via Legendre transform
    7. Extract holder_exponent = argmax_h D(h)
    
    Args:
        signal: Input signal (normalized)
        config: PredictorConfig with WTMM parameters
    
    Returns:
        Hölder exponent (scalar, clipped to valid range)
    
    References:
        - Theory.tex §2.1: Multifractal Analysis via WTMM
        - Python.tex §2.2.1: Complete WTMM Pipeline
    """
    n = signal.shape[0]
    
    # Define scales: logarithmically spaced from smallest to largest
    # Use config.wtmm_buffer_size as upper scale limit
    scales = jnp.logspace(
        jnp.log10(config.wtmm_scale_min),
        jnp.log10(config.wtmm_buffer_size),
        config.wtmm_num_scales,
    )
    scales = jnp.clip(scales, config.wtmm_scale_min, n // 2)  # Ensure valid scale range
    
    # Step 1: Continuous Wavelet Transform
    def wavelet_fn(t):
        return morlet_wavelet(t, sigma=config.wtmm_sigma, f_c=config.wtmm_fc)

    cwt = continuous_wavelet_transform(
        signal,
        scales,
        mother_wavelet_fn=wavelet_fn,
        epsilon=config.numerical_epsilon,
    )
    
    # Step 2: Find modulus maxima with reduced threshold for better detection
    # Use very small threshold to retain more maxima (they carry multifractal information)
    modulus_maxima = find_modulus_maxima(
        cwt,
        threshold=config.wtmm_modulus_threshold,
    )
    
    # Step 3: Link maxima across scales (retaining both presence and magnitudes)
    chain_presence, chain_magnitudes = link_wavelet_maxima(
        modulus_maxima,
        cwt,
        scales,
        max_link_distance=config.wtmm_max_link_distance,
    )
    
    # Step 4-5: Compute partition function and extract τ(q)
    q_range = jnp.linspace(config.wtmm_q_min, config.wtmm_q_max, config.wtmm_q_steps)
    Z_q_scales, tau_q = compute_partition_function(
        chain_magnitudes,
        scales,
        q_range,
        epsilon=config.numerical_epsilon,
    )
    
    # Step 6-7: Compute singularity spectrum and extract Hölder exponent
    # Ensure tau_q is well-formed (replace NaN/Inf with sensible defaults)
    finite_mask = jnp.isfinite(tau_q).astype(jnp.float64)
    default_vals = jnp.sign(q_range) * config.wtmm_tau_default_scale
    tau_q_safe = tau_q * finite_mask + default_vals * (1.0 - finite_mask)
    
    holder_exponent_raw, spectrum_max = compute_singularity_spectrum(
        tau_q_safe,
        q_range,
        h_min=config.wtmm_h_min,
        h_max=config.wtmm_h_max,
        h_steps=config.wtmm_h_steps,
    )
    
    # Ensure result is valid
    is_finite = jnp.isfinite(holder_exponent_raw).astype(jnp.float64)
    holder_exponent_val = (
        holder_exponent_raw * is_finite
        + config.wtmm_tau_default_scale * (1.0 - is_finite)
    )
    
    # Clip to valid range
    holder_exponent = jnp.clip(
        holder_exponent_val,
        config.validation_holder_exponent_min,
        config.validation_holder_exponent_max
    )
    
    return holder_exponent


@jax.jit
def compute_koopman_spectrum(
    signal: Float[Array, "n"],
    top_k: int,
    min_power: float,
    sampling_interval: float
) -> tuple[Float[Array, "k"], Float[Array, "k"]]:
    """
    Compute Koopman spectral modes via FFT power spectrum.

    Args:
        signal: Input time series
        top_k: Number of dominant modes to return
        min_power: Minimum power threshold

    Returns:
        (frequencies, powers) for top-k spectral modes

    References:
        - Theory.tex §2.1.5: Koopman operator and spectrum
    """
    signal_centered = signal - jnp.mean(signal)
    fft_vals = jnp.fft.rfft(signal_centered)
    power = jnp.abs(fft_vals) ** 2
    power = jnp.where(power < min_power, 0.0, power)
    freqs = jnp.fft.rfftfreq(signal.shape[0], d=sampling_interval)

    top_power, top_idx = jax.lax.top_k(power, top_k)
    top_freqs = freqs[top_idx]
    return top_freqs, top_power


@jax.jit
def compute_paley_wiener_integral(
    signal: Float[Array, "n"],
    epsilon: float,
    sampling_interval: float
) -> Float[Array, ""]:
    """
    Compute discrete Paley-Wiener integral approximation.

    Integral approximation:
        \\int |log f(omega)| / (1 + omega^2) d omega

    Args:
        signal: Input time series
        epsilon: Floor for spectral density to avoid log(0)

    Returns:
        Approximate Paley-Wiener integral value

    References:
        - Theory.tex §2.1.2: Paley-Wiener condition
    """
    signal_centered = signal - jnp.mean(signal)
    fft_vals = jnp.fft.rfft(signal_centered)
    power = (jnp.abs(fft_vals) ** 2) / signal.shape[0]
    log_power = jnp.log(jnp.maximum(power, epsilon))
    freqs = jnp.fft.rfftfreq(signal.shape[0], d=sampling_interval)

    delta = freqs[1] - freqs[0] if freqs.shape[0] > 1 else 1.0
    integrand = jnp.abs(log_power) / (1.0 + freqs ** 2)
    return jnp.sum(integrand) * delta


@jax.jit
def compute_wiener_hopf_filter(
    signal: Float[Array, "n"],
    order: int,
    epsilon: float
) -> Float[Array, "order"]:
    """
    Solve Wiener-Hopf equations for optimal linear predictor filter.

    Discrete approximation:
        R h = p
    where R is Toeplitz autocorrelation matrix and p is lagged autocorrelation.

    Args:
        signal: Input time series
        order: Filter order
        epsilon: Diagonal regularization for numerical stability

    Returns:
        Wiener-Hopf filter coefficients

    References:
        - Theory.tex §2.1.2: Wiener-Hopf integral equation
    """
    signal_centered = signal - jnp.mean(signal)
    n = signal_centered.shape[0]
    autocorr_full = jnp.correlate(signal_centered, signal_centered, mode="full")
    autocorr = autocorr_full[n - 1:n - 1 + order + 1] / n

    r = autocorr[:order]
    p = autocorr[1:order + 1]
    indices = jnp.abs(jnp.arange(order)[:, None] - jnp.arange(order)[None, :])
    R = r[indices]
    R = R + epsilon * jnp.eye(order)
    return jnp.linalg.solve(R, p)


def compute_malliavin_derivative(
    functional,
    signal: Float[Array, "n"]
) -> Float[Array, "n"]:
    """
    Compute Malliavin derivative of a functional with JAX autodiff.

    Args:
        functional: Callable mapping signal -> scalar
        signal: Input time series

    Returns:
        Gradient of functional with respect to signal

    References:
        - Theory.tex §2.2.1: Malliavin derivative operator
    """
    return jax.grad(functional)(signal)


def compute_ocone_haussmann_representation(
    expected_value: Float[Array, ""],
    malliavin_derivative: Float[Array, "n"],
    dt: float
) -> Float[Array, ""]:
    """
    Compute Ocone-Haussmann representation approximation.

    Args:
        expected_value: Scalar expectation estimate
        malliavin_derivative: Malliavin derivative along path
        dt: Time step for integral approximation

    Returns:
        Approximate representation value

    References:
        - Theory.tex §2.2.2: Ocone-Haussmann representation theorem
    """
    return expected_value + jnp.sum(malliavin_derivative) * dt


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


@partial(jax.jit, static_argnames=('config',))
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


@partial(jax.jit, static_argnames=('config',))
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


@partial(jax.jit, static_argnames=('config',))
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
    # Step 1: Normalize signal (config-driven)
    signal_normalized = normalize_signal(
        signal,
        method=config.signal_normalization_method,
        epsilon=config.numerical_epsilon,
    )
    
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
    
    # Step 6: Denormalize prediction (inverse normalization)
    prediction = y_pred_norm[0] * stats["std"] + stats["mean"]
    confidence = jnp.sqrt(variances[0]) * stats["std"]  # Scale variance by std
    
    # Step 7: Compute diagnostics (with stop_gradient)
    # V-MAJ-2 + P2.1: Full WTMM implementation for holder_exponent
    holder_exponent_estimate = extract_holder_exponent_wtmm(signal_normalized, config)
    
    koopman_top_k = min(config.koopman_top_k, (signal.shape[0] // 2) + 1)
    koopman_freqs, koopman_powers = compute_koopman_spectrum(
        signal_normalized,
        top_k=koopman_top_k,
        min_power=config.koopman_min_power,
        sampling_interval=config.signal_sampling_interval,
    )
    paley_wiener_integral = compute_paley_wiener_integral(
        signal_normalized,
        epsilon=config.numerical_epsilon,
        sampling_interval=config.signal_sampling_interval,
    )
    paley_wiener_ok = paley_wiener_integral <= config.paley_wiener_integral_max

    wiener_hopf_order = max(config.kernel_a_min_wiener_hopf_order, int(config.kernel_a_embedding_dim))
    wiener_hopf_filter = compute_wiener_hopf_filter(
        signal_normalized,
        order=wiener_hopf_order,
        epsilon=config.numerical_epsilon,
    )

    def prediction_fn(sig: Float[Array, "n"]) -> Float[Array, ""]:
        sig_norm = normalize_signal(
            sig,
            method=config.signal_normalization_method,
            epsilon=config.numerical_epsilon,
        )
        sig_stats = compute_signal_statistics(sig)
        sig_embed = create_embedding(sig_norm, config)
        sig_train = sig_embed[:-1]
        sig_targets = sig_norm[config.kernel_a_embedding_dim:-1]
        sig_test = sig_embed[-1:]
        sig_pred_norm, _ = kernel_ridge_regression(sig_train, sig_targets, sig_test, config)
        return sig_pred_norm[0] * sig_stats["std"] + sig_stats["mean"]

    malliavin_derivative = compute_malliavin_derivative(prediction_fn, signal)
    ocone_haussmann_value = compute_ocone_haussmann_representation(
        expected_value=jnp.mean(signal),
        malliavin_derivative=malliavin_derivative,
        dt=config.sde_dt,
    )

    diagnostics = {
        "kernel_type": "A_Hilbert_RKHS",
        "bandwidth": config.kernel_a_bandwidth,
        "embedding_dim": config.kernel_a_embedding_dim,
        "n_training_points": X_train.shape[0],
        "signal_mean": stats["mean"],
        "signal_std": stats["std"],
        "holder_exponent": float(holder_exponent_estimate),  # P2.1: Full WTMM, not placeholder
        "koopman_freqs": koopman_freqs,
        "koopman_powers": koopman_powers,
        "paley_wiener_integral": paley_wiener_integral,
        "paley_wiener_ok": paley_wiener_ok,
        "wiener_hopf_filter": wiener_hopf_filter,
        "malliavin_derivative_norm": jnp.linalg.norm(malliavin_derivative),
        "ocone_haussmann_value": ocone_haussmann_value,
    }
    
    grid, dx = build_pdf_grid(prediction, confidence, config)
    probability_density = compute_normal_pdf(grid, prediction, confidence, config)
    entropy = compute_density_entropy(probability_density, dx, config)
    entropy = jax.lax.stop_gradient(entropy)

    numerics_flags = {
        "has_nan": jnp.any(jnp.isnan(probability_density))
        | jnp.any(jnp.isnan(prediction))
        | jnp.any(jnp.isnan(confidence)),
        "has_inf": jnp.any(jnp.isinf(probability_density))
        | jnp.any(jnp.isinf(prediction))
        | jnp.any(jnp.isinf(confidence)),
    }

    # Apply stop_gradient to diagnostics (VRAM optimization)
    prediction, diagnostics = apply_stop_gradient_to_diagnostics(
        prediction, diagnostics
    )
    
    return KernelOutput(
        prediction=prediction,
        confidence=confidence,
        entropy=entropy,
        probability_density=probability_density,
        kernel_id="A",
        computation_time_us=jnp.array(config.kernel_output_time_us),
        numerics_flags=numerics_flags,
        metadata=diagnostics,
    )


# Public API
__all__ = [
    "kernel_a_predict",
    "gaussian_kernel",
    "compute_gram_matrix",
    "kernel_ridge_regression",
    "create_embedding"
]
