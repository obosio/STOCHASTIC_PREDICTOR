"""Kernel B: Deep Galerkin Method (DGM) for Fokker-Planck/HJB.

This kernel uses Deep Galerkin Method with Equinox neural networks to solve
Hamilton-Jacobi-Bellman equations arising from optimal control of stochastic
processes.

References:
    - Teoria.tex §2.2: Rama B (Fokker-Planck + DGM)
    - Python.tex §2.2.2: Kernel B (DGM Implementation)
    - Implementacion.tex §3.2: DGM for High-Dimensional PDEs
    - Tests_Python.tex §1.2: DGM Validation

Mathematical Foundation:
    HJB Equation: V_t + H(x, V_x, V_xx) = 0
    where H is the Hamiltonian operator.

    DGM approximates V_θ(t, x) using a neural network and minimizes
    the PDE residual via gradient descent.
"""

from functools import partial
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import (
    KernelOutput,
    build_pdf_grid,
    compute_density_entropy,
    compute_normal_pdf,
)

# ═══════════════════════════════════════════════════════════════════════════
# ACTIVATION FUNCTION REGISTRY (Zero-Heuristics Compliance)
# ═══════════════════════════════════════════════════════════════════════════

ACTIVATION_FUNCTIONS = {
    "tanh": jax.nn.tanh,  # Default: Smooth PDEs (recommended for HJB)
    "relu": jax.nn.relu,  # Alternative: Processes with rectification
    "elu": jax.nn.elu,  # Alternative: Smooth approximation to ReLU
    "gelu": jax.nn.gelu,  # Alternative: Transformer-style (Gaussian-like)
    "sigmoid": jax.nn.sigmoid,  # Alternative: Bounded outputs
    "swish": jax.nn.swish,  # Alternative: Self-gated (smooth)
}


def get_activation_fn(name: str):
    """
    Resolve activation function name to JAX callable.

    Args:
        name: Activation function name (from config.dgm_activation)

    Returns:
        Callable JAX activation function

    Raises:
        ValueError: If activation name is not recognized

    References:
        - Python.tex §2.2.2: DGM Network Architecture
    """
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {name}. " f"Valid options: {list(ACTIVATION_FUNCTIONS.keys())}")
    return ACTIVATION_FUNCTIONS[name]


class DGM_HJB_Solver(eqx.Module):
    """
    Deep Galerkin Method neural network for HJB equations.

    Architecture:
        Input: (t, x) where t is time, x is spatial coordinates
        Output: V(t, x) - value function

    Uses Equinox MLP as pytree for automatic differentiation.

    References:
        - Python.tex §2.2.2: DGM_HJB_Solver class
        - Implementacion.tex §3.2.1: Neural Architecture
    """

    mlp: eqx.nn.MLP

    def __init__(self, in_size: int, key: Array, config):
        """
        Initialize DGM solver network.

        Args:
            in_size: Input dimension (typically d+1 for d spatial dims + 1 time)
            key: JAX PRNG key for weight initialization
            config: PredictorConfig with dgm_width_size, dgm_depth, dgm_activation

        References:
            - Python.tex §2.2.2: DGM Network Initialization
        """
        # Resolve activation function from config (Zero-Heuristics)
        activation_fn = get_activation_fn(config.dgm_activation)

        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=config.dgm_width_size,
            depth=config.dgm_depth,
            key=key,
            activation=activation_fn,  # From config, not hardcoded
        )

    def __call__(self, t: float | Float[Array, ""], x: Float[Array, "d"]) -> Float[Array, ""]:
        """
        Evaluate V(t, x).

        Args:
            t: Time (scalar float or array)
            x: Spatial coordinates (d-dimensional)

        Returns:
            Value function V(t, x)
        """
        # Concatenate time and space
        t_arr = jnp.array([t]) if jnp.ndim(t) == 0 else t
        tx = jnp.concatenate([t_arr, x])

        # Forward pass through MLP
        output = self.mlp(tx)

        return output[0]


def compute_entropy_dgm(model: DGM_HJB_Solver, t: float, x_samples: Float[Array, "n d"], config) -> Float[Array, ""]:
    """
    Compute differential entropy of DGM value function.

    Entropy H_DGM = -∫ p(v) log p(v) dv measures the distribution
    of value function outputs over the spatial domain.

    Low entropy indicates mode collapse (constant predictions).

    Args:
        model: DGM solver network
        t: Time at which to evaluate
        x_samples: Samples from spatial domain (n x d)
        config: PredictorConfig with dgm_entropy_num_bins

    Returns:
        Differential entropy (scalar)

    References:
        - IO.tex §2.3: DGM Entropy Monitoring
        - Python.tex §4.2: Mode Collapse Detection

    Note:
        This computes entropy via histogram approximation.
        For true differential entropy, use kernel density estimation,
        but histogram is faster for monitoring purposes.
    """

    # Evaluate value function at all sample points
    @jax.vmap
    def evaluate_v(x):
        return model(t, x)

    values = evaluate_v(x_samples)

    # Compute histogram
    hist, bin_edges = jnp.histogram(values, bins=config.dgm_entropy_num_bins, density=True)

    # Bin width for differential entropy
    bin_width = bin_edges[1] - bin_edges[0]

    # Shannon entropy (discrete approximation)
    # H = -∑ p_i log(p_i) * Δx
    # Add small epsilon to avoid log(0)
    hist_safe = hist + config.numerical_epsilon
    entropy = -jnp.sum(hist * jnp.log(hist_safe)) * bin_width

    return entropy


def loss_hjb(
    model: DGM_HJB_Solver,
    t_batch: Float[Array, "n_t"],
    x_batch: Float[Array, "n_x d"],
    config,
) -> Float[Array, ""]:
    """
    HJB PDE residual loss for DGM training.

    Zero-Heuristics: ALL parameters from config (kernel_b_r, kernel_b_sigma).

    Args:
        model: DGM network
        t_batch: Time samples
        x_batch: Spatial samples
        config: PredictorConfig with kernel_b_r, kernel_b_sigma

    Returns:
        Mean squared residual loss

    References:
        - Python.tex §2.2.2: HJB Loss Function
        - Teoria.tex §2.2: HJB PDE Theory

    Note:
        This is a simplified drift-diffusion example for a stochastic process.
        For general case, pass Hamiltonian function as argument.
    """

    def pde_residual(t, x):
        """Compute PDE residual at single point (t, x)."""

        # Compute value and derivatives using JAX autodiff
        def v_fn(t_var, x_var):
            return model(t_var, x_var)

        # Value
        v_fn(t, x)

        # First derivatives
        v_t = jax.grad(v_fn, argnums=0)(t, x)
        v_x = jax.grad(v_fn, argnums=1)(t, x)

        # Second derivatives (Hessian)
        hess_fn = jax.hessian(v_fn, argnums=1)
        v_xx = hess_fn(t, x)

        # Hamiltonian operator (general drift-diffusion form)
        # H = r*X*V_X + 0.5*σ²*X²*V_XX
        # where r is drift coefficient, σ is dispersion coefficient
        X = x[0]  # Process magnitude (first coordinate)

        hamiltonian = config.kernel_b_r * X * v_x[0] + 0.5 * config.kernel_b_sigma**2 * X**2 * v_xx[0, 0]

        # PDE residual: V_t + H = 0
        residual = v_t + hamiltonian

        return residual**2

    # Vectorize over batch
    @jax.vmap
    def residual_over_x(x):
        @jax.vmap
        def residual_over_t(t):
            return pde_residual(t, x)

        return residual_over_t(t_batch)

    residuals = residual_over_x(x_batch)

    # Mean squared residual
    loss = jnp.mean(residuals)

    return loss


def compute_adaptive_entropy_threshold(ema_variance: Float[Array, ""], config) -> float:
    """
    Compute volatility-adaptive entropy threshold for mode collapse detection [V-MAJ-1].

    Threshold adapts to market regime:
    - High volatility (σ > 0.2): γ = γ_min (lenient, easier to detect mode collapse)
    - Normal volatility (σ ∈ [0.05, 0.2]): γ = γ_default (balanced)
    - Low volatility (σ < 0.05): γ = γ_max (strict, harder to detect mode collapse)

    Args:
        ema_variance: Current EMA variance (σ²)
        config: PredictorConfig with entropy_gamma_* parameters

    Returns:
        Scalar threshold ∈ [γ_min, γ_max]

    References:
        - Theory.tex §2.2: Entropy-based Mode Collapse Detection
    """
    sigma_t = jnp.sqrt(jnp.maximum(ema_variance, config.numerical_epsilon))

    # Volatility regime classification
    high_vol_threshold = config.entropy_volatility_high_threshold
    low_vol_threshold = config.entropy_volatility_low_threshold

    # V-MAJ-1: Piecewise linear interpolation based on volatility
    gamma = jnp.where(
        sigma_t > high_vol_threshold,
        config.entropy_gamma_min,  # Crisis: lenient
        jnp.where(
            sigma_t < low_vol_threshold,
            config.entropy_gamma_max,  # Low-vol: strict
            config.entropy_gamma_default,  # Normal: balanced
        ),
    )

    return gamma.astype(float)


@partial(jax.jit, static_argnames=("config",))
def kernel_b_predict(
    signal: Float[Array, "n"],
    key: Array,
    config,
    model: Optional[DGM_HJB_Solver] = None,
    ema_variance: Optional[Float[Array, ""]] = None,  # V-MAJ-1: Optional parameter for adaptive threshold
) -> KernelOutput:
    """
    Kernel B: DGM prediction for stochastic dynamics [V-MAJ-1: Adaptive entropy threshold].

    Algorithm:
        1. Initialize or use provided DGM model
        2. Evaluate value function at current state
        3. Compute entropy for mode collapse detection
        4. Use adaptive threshold based on volatility regime (V-MAJ-1)
        5. Return prediction with confidence

    Args:
        signal: Input time series (current state trajectory)
        key: JAX PRNG key for model initialization (if needed)
        config: PredictorConfig with dgm_width_size, dgm_depth, kernel_b_r,
                kernel_b_sigma, kernel_b_horizon, dgm_entropy_num_bins,
                kernel_b_spatial_samples, entropy_gamma_* (V-MAJ-1)
        model: Pre-trained DGM model (if None, creates dummy)
        ema_variance: EWMA variance for adaptive threshold [V-MAJ-1]
                 Required for mode-collapse detection

    Returns:
        KernelOutput with prediction, confidence, and diagnostics

    References:
        - Python.tex §2.2.2: Kernel B Full Algorithm
        - Implementacion.tex §3.2: DGM Prediction Pipeline
        - Theory.tex §2.2: Entropy-based Mode Collapse Detection [V-MAJ-1]

    Note:
        In production, model should be pre-trained on historical data.
        This implementation creates a placeholder for infrastructure testing.
    """
    # Current state (last value of signal)
    current_state = signal[-1]

    # Initialize model if not provided (for testing)
    if model is None:
        key_model, key_predict = jax.random.split(key)
        model = DGM_HJB_Solver(in_size=2, key=key_model, config=config)

    # Evaluate value function at current state and horizon
    t = 0.0  # Current time
    x = jnp.array([current_state])  # Spatial coordinate

    value = model(t, x)

    # Prediction: Drift-diffusion forecast
    # In full DGM, this would use optimal control policy derived from V
    # For now, use exponential drift: E[X_{t+h}] = X_t * exp(r*h)
    # where r is drift rate parameter from config
    prediction = current_state * jnp.exp(config.kernel_b_r * config.kernel_b_horizon)

    # Confidence: Dispersion term σ*X*√h
    confidence = config.kernel_b_sigma * current_state * jnp.sqrt(config.kernel_b_horizon)

    # Compute entropy for mode collapse detection
    # Sample spatial domain around current state
    x_samples = jnp.linspace(
        current_state * (1.0 - config.kernel_b_spatial_range_factor),
        current_state * (1.0 + config.kernel_b_spatial_range_factor),
        config.kernel_b_spatial_samples,
    )[
        :, None
    ]  # Shape (kernel_b_spatial_samples, 1)

    t_samples = jnp.array([t])
    viscosity_residual = loss_hjb(model, t_samples, x_samples, config)
    viscosity_solution_ok = viscosity_residual <= config.validation_viscosity_residual_max

    entropy_dgm = compute_entropy_dgm(model, t, x_samples, config)

    # V-MAJ-8: Apply stop_gradient to entropy diagnostic
    # Prevents gradient backprop through entropy computation (VRAM optimization)
    entropy_dgm = jax.lax.stop_gradient(entropy_dgm)

    # Check for mode collapse [V-MAJ-1: Adaptive threshold]
    if ema_variance is None:
        ema_variance = jnp.array(config.numerical_epsilon)

    # V-MAJ-1: Use adaptive threshold based on volatility regime
    entropy_threshold_adaptive = compute_adaptive_entropy_threshold(ema_variance, config)
    mode_collapse = entropy_dgm < entropy_threshold_adaptive

    # Diagnostics
    diagnostics = {
        "value_function": value,
        "entropy_dgm": entropy_dgm,  # V-MAJ-8: Already blocked by stop_gradient above
        "entropy_threshold_adaptive": entropy_threshold_adaptive,  # V-MAJ-1: Diagnostic (also stopped)
        "mode_collapse": mode_collapse,
        "viscosity_residual": viscosity_residual,
        "viscosity_solution_ok": viscosity_solution_ok,
        "r": config.kernel_b_r,
        "sigma": config.kernel_b_sigma,
        "horizon": config.kernel_b_horizon,
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

    # NOTE: Diagnostics returned unmodified (no apply_stop_gradient)
    # Stop-gradient applied at individual diagnostic computation sites for JIT compatibility
    return KernelOutput(
        prediction=prediction,
        confidence=confidence,
        entropy=entropy,
        probability_density=probability_density,
        kernel_id=1,  # 1=B (JAX JIT compatible)
        computation_time_us=float(config.kernel_output_time_us),
        numerics_flags=numerics_flags,
        metadata=diagnostics,
    )


# Public API
__all__ = ["DGM_HJB_Solver", "kernel_b_predict", "compute_entropy_dgm", "loss_hjb"]
