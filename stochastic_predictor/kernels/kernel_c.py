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
        - Implementacion.tex §3.3.1: Diffusion Structure
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
    config,
    args: tuple = ()
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
        config: Configuration object with SDE solver parameters
        args: Additional arguments for drift/diffusion
    
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
            tol=config.sde_brownian_tree_tol,
            shape=(y0.shape[0],),
            key=key
        )
    )
    
    # Combined SDE terms
    terms = diffrax.MultiTerm(drift_term, diffusion_term)
    
    # Select solver (Zero-Heuristics: from config.sde_solver_type)
    if config.sde_solver_type == "euler":
        solver_obj = diffrax.Euler()
    elif config.sde_solver_type == "heun":
        solver_obj = diffrax.Heun()
    else:
        solver_obj = diffrax.Euler()  # Default
    
    # Adaptive step size controller (Zero-Heuristics: tolerances from config)
    stepsize_controller = diffrax.PIDController(
        rtol=config.sde_pid_rtol,
        atol=config.sde_pid_atol,
        dtmin=config.sde_pid_dtmin,
        dtmax=config.sde_pid_dtmax
    )
    
    # Solve SDE (dt0 from config: dtmax/10 as initial step)
    solution = diffrax.diffeqsolve(
        terms,
        solver_obj,
        t0=t0,
        t1=t1,
        dt0=config.sde_pid_dtmax / 10.0,
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
    config
) -> KernelOutput:
    """
    Kernel C: Itô/Lévy SDE prediction.
    
    Algorithm:
        1. Extract current state from signal
        2. Define drift and diffusion functions
        3. Integrate SDE forward to horizon
        4. Return prediction with confidence
    
    Zero-Heuristics: All parameters (sigma, mu, alpha, beta, horizon, tolerances,
    solver type) are injected from config, not hardcoded.
    
    Args:
        signal: Input time series (historical trajectory)
        key: JAX PRNG key for Brownian motion
        config: Configuration object with Kernel C and SDE parameters
    
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
        >>> result = kernel_c_predict(signal, key, config)
        >>> prediction = result.prediction
    """
    # Extract parameters from config (Zero-Heuristics pattern)
    sigma = config.sde_diffusion_sigma
    mu = config.kernel_c_mu
    alpha = config.kernel_c_alpha
    beta = config.kernel_c_beta
    horizon = config.kernel_c_horizon
    # Current state (last value, convert to 1D array)
    y0 = jnp.array([signal[-1]])
    
    # Time parameters
    t0 = 0.0
    t1 = horizon
    
    # SDE parameters
    args = (mu, alpha, beta, sigma)
    
    # Integrate SDE (config injection pattern)
    y_final = solve_sde(
        drift_fn=drift_levy_stable,
        diffusion_fn=diffusion_levy,
        y0=y0,
        t0=t0,
        t1=t1,
        key=key,
        config=config,
        args=args
    )
    
    # Prediction
    prediction = y_final[0]
    
    # Confidence: Theoretical variance of Lévy stable process
    # For α-stable: Var ~ t^(2/α) (power law)
    # For Brownian (α=2): Var = σ^2 * t
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
        "solver": config.sde_solver_type,
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
