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

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import equinox as eqx
from typing import Optional

from .base import KernelOutput, apply_stop_gradient_to_diagnostics


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
    
    def __init__(
        self,
        in_size: int,
        key: Array,
        config
    ):
        """
        Initialize DGM solver network.
        
        Args:
            in_size: Input dimension (typically d+1 for d spatial dims + 1 time)
            key: JAX PRNG key for weight initialization
            config: PredictorConfig with dgm_width_size, dgm_depth
        
        References:
            - Python.tex §2.2.2: DGM Network Initialization
        """
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=config.dgm_width_size,
            depth=config.dgm_depth,
            key=key,
            activation=jax.nn.tanh  # Smooth activation for derivatives
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


@jax.jit
def compute_entropy_dgm(
    model: DGM_HJB_Solver,
    t: float,
    x_samples: Float[Array, "n d"],
    config
) -> Float[Array, ""]:
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
    hist_safe = hist + 1e-10
    entropy = -jnp.sum(hist * jnp.log(hist_safe)) * bin_width
    
    return entropy


@jax.jit
def loss_hjb(
    model: DGM_HJB_Solver,
    t_batch: Float[Array, "n_t"],
    x_batch: Float[Array, "n_x d"],
    config
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
        v_fn = lambda t_var, x_var: model(t_var, x_var)
        
        # Value
        v = v_fn(t, x)
        
        # First derivatives
        v_t = jax.grad(v_fn, argnums=0)(t, x)
        v_x = jax.grad(v_fn, argnums=1)(t, x)
        
        # Second derivatives (Hessian)
        hess_fn = jax.hessian(v_fn, argnums=1)
        v_xx = hess_fn(t, x)
        
        # Drift-diffusion equation Hamiltonian
        # H = mu*X*V_X + 0.5*sigma^2*X^2*V_XX
        X = x[0]  # Process value (first coordinate)
        
        hamiltonian = (
            config.kernel_b_r * X * v_x[0]
            + 0.5 * config.kernel_b_sigma ** 2 * X ** 2 * v_xx[0, 0]
        )
        
        # PDE residual: V_t + H = 0
        residual = v_t + hamiltonian
        
        return residual ** 2
    
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


@jax.jit
def kernel_b_predict(
    signal: Float[Array, "n"],
    key: Array,
    config,
    model: Optional[DGM_HJB_Solver] = None
) -> KernelOutput:
    """
    Kernel B: DGM prediction for stochastic dynamics.
    
    Algorithm:
        1. Initialize or use provided DGM model
        2. Evaluate value function at current state
        3. Compute entropy for mode collapse detection
        4. Return prediction with confidence
    
    Args:
        signal: Input time series (current state trajectory)
        key: JAX PRNG key for model initialization (if needed)
        config: PredictorConfig with dgm_width_size, dgm_depth, kernel_b_r, 
                kernel_b_sigma, kernel_b_horizon, dgm_entropy_num_bins, 
                kernel_b_spatial_samples
        model: Pre-trained DGM model (if None, creates dummy)
    
    Returns:
        KernelOutput with prediction, confidence, and diagnostics
    
    References:
        - Python.tex §2.2.2: Kernel B Full Algorithm
        - Implementacion.tex §3.2: DGM Prediction Pipeline
    
    Note:
        In production, model should be pre-trained on historical data.
        This implementation creates a placeholder for infrastructure testing.
    """
    # Current state (last value of signal)
    current_state = signal[-1]
    
    # Initialize model if not provided (for testing)
    if model is None:
        key_model, key_predict = jax.random.split(key)
        model = DGM_HJB_Solver(
            in_size=2, 
            key=key_model,
            config=config
        )
    
    # Evaluate value function at current state and horizon
    t = 0.0  # Current time
    x = jnp.array([current_state])  # Spatial coordinate
    
    value = model(t, x)
    
    # Prediction: Simple drift-diffusion forecast
    # In full DGM, this would use optimal control policy derived from V
    # For now, use drift-diffusion: E[X_{t+h}] = X_t * exp(mu*h)
    prediction = current_state * jnp.exp(config.kernel_b_r * config.kernel_b_horizon)
    
    # Confidence: Diffusion term σ*X*√h
    confidence = config.kernel_b_sigma * current_state * jnp.sqrt(config.kernel_b_horizon)
    
    # Compute entropy for mode collapse detection
    # Sample spatial domain around current state
    x_samples = jnp.linspace(
        current_state * 0.5,
        current_state * 1.5,
        config.kernel_b_spatial_samples
    )[:, None]  # Shape (kernel_b_spatial_samples, 1)
    
    entropy_dgm = compute_entropy_dgm(model, t, x_samples, config)
    
    # Check for mode collapse
    # Threshold: entropy should be > 0.5 * log(100) ≈ 2.3 for healthy distribution
    mode_collapse = entropy_dgm < 1.0
    
    # Diagnostics
    diagnostics = {
        "kernel_type": "B_DGM_Fokker_Planck",
        "value_function": value,
        "entropy_dgm": entropy_dgm,
        "mode_collapse": mode_collapse,
        "r": config.kernel_b_r,
        "sigma": config.kernel_b_sigma,
        "horizon": config.kernel_b_horizon
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
    "DGM_HJB_Solver",
    "kernel_b_predict",
    "compute_entropy_dgm",
    "loss_hjb"
]
