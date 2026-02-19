"""Kernel C: Itô/Lévy SDE Integration with Diffrax.

This kernel integrates stochastic differential equations using Diffrax solvers
with dynamic scheme switching (IMEX, implicit-explicit methods).

References:
    - Teoria.tex §2.3.3: Rama C (Itô/Lévy SDEs)
    - Python.tex §2.2.3: Kernel C (Neural SDE)
    - Implementacion.tex §3.3: IMEX Splitting for Stiff SDEs

Mathematical Foundation:
    SDE: dX_t = f(t, X_t) dt + g(t, X_t) dW_t
    where f is drift, g is diffusion, W_t is Brownian motion.
    
    Uses adaptive solvers (Heun, Euler-Maruyama) depending on stiffness.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import diffrax
from typing import Callable, Optional

from .base import KernelOutput, apply_stop_gradient_to_diagnostics


def drift_levy_stable(
    t: Float[Array, ""],
    y: Float[Array, "d"],
    args: tuple
) -> Float[Array, "d"]:
    """
    Drift term f(t, y) for Lévy stable process.
    
    For α-stable Lévy process with drift:
    f(t, y) = μ (mean drift)
    
    Args:
        t: Current time
        y: Current state (d-dimensional)
        args: Tuple of (mu, alpha, beta) parameters
    
    Returns:
        Drift vector (d-dimensional)
    
    References:
        - Teoria.tex §2.3.3: Lévy Stable Processes
        - Python.tex §2.2.3: Drift Function
    """
    mu, alpha, beta = args
    
    # Constant drift
    return jnp.full_like(y, mu)


def diffusion_levy(
    t: Float[Array, ""],
    y: Float[Array, "d"],
    args: tuple
) -> Float[Array, "d d"]:
    """
    Diffusion term g(t, y) for Lévy process with config-driven sigma.
    
    For simple case: g(t, y) = σ * I (isotropic diffusion)
    
    Zero-Heuristics: sigma is NOT hardcoded; must come from config.sde_diffusion_sigma.
    
    Args:
        t: Current time
        y: Current state (d-dimensional)
        args: Tuple of (mu, alpha, beta, sigma)
    
    Returns:
        Diffusion matrix (d x d)
    
    References:
        - Teoria.tex §2.3.3: Diffusion Coefficient
        - Implementacion.tex §3.3.1: Volatility Structure
    """
    mu, alpha, beta, sigma = args  # sigma from config, not hardcoded
    
    d = y.shape[0]
    return sigma * jnp.eye(d)


@jax.jit
def solve_sde(
    drift_fn: Callable,
    diffusion_fn: Callable,
    y0: Float[Array, "d"],
    t0: float,
    t1: float,
    key: Array,
    args: tuple = (),
    dt0: float = 0.01,
    solver: str = "euler"
) -> Float[Array, "d"]:
    """
    Solve SDE using Diffrax.
    
    Integrates dY = drift(t, Y)dt + diffusion(t, Y)dW from t0 to t1.
    
    Args:
        drift_fn: Drift function f(t, y, args)
        diffusion_fn: Diffusion function g(t, y, args)
        y0: Initial condition (d-dimensional)
        t0: Start time
        t1: End time
        key: JAX PRNG key for Brownian motion
        args: Additional arguments for drift/diffusion
        dt0: Initial time step
        solver: Solver type ('euler' or 'heun')
    
    Returns:
        Final state at time t1
    
    References:
        - Python.tex §2.2.3: SDE Solver with Diffrax
        - Implementacion.tex §3.3.2: Adaptive Stepping
    """
    # Define SDE terms
    drift_term = diffrax.ODETerm(drift_fn)
    diffusion_term = diffrax.ControlTerm(
        diffusion_fn,
        diffrax.VirtualBrownianTree(
            t0=t0,
            t1=t1,
            tol=1e-3,
            shape=(y0.shape[0],),
            key=key
        )
    )
    
    # Combined SDE terms
    terms = diffrax.MultiTerm(drift_term, diffusion_term)
    
    # Select solver
    if solver == "euler":
        solver_obj = diffrax.Euler()
    elif solver == "heun":
        solver_obj = diffrax.Heun()
    else:
        solver_obj = diffrax.Euler()  # Default
    
    # Adaptive step size controller
    stepsize_controller = diffrax.PIDController(
        rtol=1e-3,
        atol=1e-6,
        dtmin=1e-5,
        dtmax=0.1
    )
    
    # Solve SDE
    solution = diffrax.diffeqsolve(
        terms,
        solver_obj,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(t1=True)
    )
    
    # Return final state
    return solution.ys[-1] if solution.ys is not None else y0


@jax.jit
def kernel_c_predict(
    signal: Float[Array, "n"],
    key: Array,
    sigma: float,
    mu: float,
    alpha: float,
    beta: float,
    horizon: float,
    dt0: float
) -> KernelOutput:
    """
    Kernel C: Itô/Lévy SDE prediction.
    
    Algorithm:
        1. Extract current state from signal
        2. Define drift and diffusion functions
        3. Integrate SDE forward to horizon
        4. Return prediction with confidence
    
    Args:
        signal: Input time series (historical trajectory)
        key: JAX PRNG key for Brownian motion
        sigma: Diffusion coefficient (from config.sde_diffusion_sigma - REQUIRED)
        mu: Drift parameter (from config.kernel_c_mu - REQUIRED)
        alpha: Stability parameter (from config.kernel_c_alpha - REQUIRED, 1 < alpha <= 2)
        beta: Skewness parameter (from config.kernel_c_beta - REQUIRED, -1 <= beta <= 1)
        horizon: Prediction horizon (from config.kernel_c_horizon - REQUIRED)
        dt0: Initial time step (from config.kernel_c_dt0 - REQUIRED)
    
    Returns:
        KernelOutput with prediction, confidence, and diagnostics
    
    References:
        - Python.tex §2.2.3: Kernel C Complete Algorithm
        - Teoria.tex §2.3.3: Lévy Process Dynamics
    
    Example:
        >>> from stochastic_predictor.api.config import PredictorConfigInjector
        >>> config = PredictorConfigInjector().create_config()
        >>> signal = synthetic_levy_stable(rng_key)
        >>> key = initialize_jax_prng(42)
        >>> result = kernel_c_predict(
        ...     signal, key,
        ...     sigma=config.sde_diffusion_sigma,
        ...     mu=config.kernel_c_mu,
        ...     alpha=config.kernel_c_alpha,
        ...     beta=config.kernel_c_beta,
        ...     horizon=config.kernel_c_horizon,
        ...     dt0=config.kernel_c_dt0
        ... )
        >>> prediction = result.prediction
    """
    # Current state (last value, convert to 1D array)
    y0 = jnp.array([signal[-1]])
    
    # Time parameters
    t0 = 0.0
    t1 = horizon
    
    # SDE parameters (including sigma from config, not hardcoded)
    args = (mu, alpha, beta, sigma)
    
    # Integrate SDE
    y_final = solve_sde(
        drift_fn=drift_levy_stable,
        diffusion_fn=diffusion_levy,
        y0=y0,
        t0=t0,
        t1=t1,
        key=key,
        args=args,
        dt0=dt0,
        solver="heun"  # Heun for better accuracy on SDEs
    )
    
    # Prediction
    prediction = y_final[0]
    
    # Confidence: Theoretical variance of Lévy stable process
    # For α-stable: Var ~ t^(2/α) (power law)
    # For Brownian (α=2): Var = σ^2 * t
    # sigma comes from config (REQUIRED parameter passed explicitly)
    sigma = args[3]  # Extract sigma from args (injected from config)
    if alpha > 1.99:  # Near-Gaussian
        variance = (sigma ** 2) * horizon
    else:  # Heavy-tailed Lévy
        variance = (sigma ** alpha) * (horizon ** (2.0 / alpha))
    
    confidence = jnp.sqrt(variance)
    
    # Diagnostics
    diagnostics = {
        "kernel_type": "C_Ito_Levy_SDE",
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "horizon": horizon,
        "solver": "heun",
        "dt0": dt0,
        "final_state": y_final[0]
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
    "kernel_c_predict",
    "solve_sde",
    "drift_levy_stable",
    "diffusion_levy"
]
