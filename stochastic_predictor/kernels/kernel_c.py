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

from .base import (
    KernelOutput,
    apply_stop_gradient_to_diagnostics,
    build_pdf_grid,
    compute_density_entropy,
    compute_normal_pdf,
)


def estimate_stiffness(
    drift_fn: Callable,
    diffusion_fn: Callable,
    y: Float[Array, "d"],
    t: float,
    args: tuple,
    config
) -> Array:
    """
    Estimate stiffness ratio for dynamic solver selection.
    
    Stiffness metric: ||∇f|| / trace(g·g^T)
    where f is drift, g is diffusion.
    
    High ratio → stiff system (implicit solver required)
    Low ratio → non-stiff system (explicit solver sufficient)
    
    Args:
        drift_fn: Drift function f(t, y, args)
        diffusion_fn: Diffusion function g(t, y, args)
        y: Current state
        t: Current time
        args: Additional arguments
    
    Returns:
        Stiffness ratio (dimensionless)
    
    References:
        - Teoria.tex §2.3.3: Stiffness-Adaptive Schemes
        - Implementacion.tex §3.3.1: IMEX Splitting Criteria
    """
    # Compute drift Jacobian norm
    def drift_scalar(y_vec):
        return jnp.linalg.norm(drift_fn(t, y_vec, args))
    
    drift_grad = jax.grad(drift_scalar)(y)
    drift_jacobian_norm = jnp.linalg.norm(drift_grad)
    
    # Compute diffusion magnitude (trace of g·g^T)
    diffusion_matrix = diffusion_fn(t, y, args)
    diffusion_variance = jnp.trace(diffusion_matrix @ diffusion_matrix.T)
    
    # Stiffness ratio: drift strength / diffusion strength
    # Add small epsilon to prevent division by zero
    stiffness = drift_jacobian_norm / (jnp.sqrt(diffusion_variance) + config.numerical_epsilon)

    return jnp.asarray(stiffness)


def select_stiffness_solver(current_stiffness: float, config):
    """
    Dynamic solver selection based on Teoria.tex §2.3.3.
    
    Stiffness-adaptive scheme switching:
    - Low stiffness (< stiffness_low): Explicit Euler (fast, stable for non-stiff)
    - Medium stiffness (stiffness_low to stiffness_high): Heun (adaptive, balanced)
    - High stiffness (>= stiffness_high): Implicit Euler (stable for stiff systems)
    
    Args:
        current_stiffness: Estimated stiffness ratio
        config: Configuration with stiffness_low, stiffness_high thresholds
    
    Returns:
        Diffrax solver instance
    
    References:
        - Teoria.tex §2.3.3: IMEX Splitting for Stiff SDEs
        - Python.tex §2.2.3: Dynamic Scheme Switching
    """
    if current_stiffness < config.stiffness_low:
        return diffrax.Euler()  # Explicit - fast for non-stiff
    elif current_stiffness < config.stiffness_high:
        return diffrax.Heun()  # Adaptive - balanced
    else:
        return diffrax.ImplicitEuler()  # Implicit - stable for stiff


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
        args: Tuple of (mu, alpha, beta, sigma) parameters
    
    Returns:
        Drift vector (d-dimensional)
    
    References:
        - Teoria.tex §2.3.3: Lévy Stable Processes
        - Python.tex §2.2.3: Drift Function
    """
    # CRITICAL: Match arity with kernel_c_predict args packing (4 parameters)
    mu, alpha, beta, sigma = args
    
    # Constant drift (sigma not used in drift, but must unpack for consistency)
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


def sample_levy_jump_component(
    key: Array,
    horizon: float,
    config
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """
    Sample compound Poisson jump component for Levy process.

    Implements jump term:
        sum_{i=1}^{N_t} Y_i, where N_t ~ Poisson(lambda * t)

    Args:
        key: JAX PRNG key
        horizon: Integration horizon (t)
        config: PredictorConfig with jump intensity and size params

    Returns:
        (jump_sum, jump_count)

    References:
        - Theory.tex §2.3.4: Ito formula with jumps
    """
    max_events = int(config.kernel_c_jump_max_events)
    expected_count = config.kernel_c_jump_intensity * horizon

    key_count, key_sizes = jax.random.split(key)
    jump_count = jax.random.poisson(key_count, expected_count)
    jump_count = jnp.minimum(jump_count, max_events)

    jump_sizes = (
        config.kernel_c_jump_mean
        + config.kernel_c_jump_scale * jax.random.normal(key_sizes, (max_events,))
    )
    mask = jnp.arange(max_events) < jump_count
    jump_sum = jnp.sum(jump_sizes * mask)

    return jnp.asarray(jump_sum), jnp.asarray(jump_count)


def decompose_semimartingale(
    signal: Float[Array, "n"],
    dt: float
) -> tuple[Float[Array, ""], Float[Array, "n"], Float[Array, "n"]]:
    """
    Decompose signal into martingale and finite-variation components.

    Uses a drift estimate from mean increments:
        X_t = X_0 + M_t + A_t
        A_t = drift * t
        M_t = X_t - X_0 - A_t

    Args:
        signal: Input time series
        dt: Time step

    Returns:
        (drift_estimate, martingale_component, finite_variation_component)

    References:
        - Theory.tex §2.2.5: Semimartingale decomposition
    """
    increments = jnp.diff(signal)
    drift_estimate = jnp.mean(increments) / dt
    times = jnp.arange(signal.shape[0]) * dt
    finite_variation = drift_estimate * times
    martingale = signal - signal[0] - finite_variation
    return drift_estimate, martingale, finite_variation


def compute_information_drift(
    martingale_component: Float[Array, "n"],
    dt: float
) -> Float[Array, ""]:
    """
    Estimate information drift for filtration enlargement.

    Uses mean increment of martingale component as a proxy for alpha_t.

    Args:
        martingale_component: Martingale component from decomposition
        dt: Time step

    Returns:
        Estimated information drift

    References:
        - Theory.tex §2.1.6: Filtration enlargement (information drift)
    """
    increments = jnp.diff(martingale_component)
    return jnp.mean(increments) / dt


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
) -> tuple[Float[Array, "d"], Array, Float[Array, ""]]:
    """
    Solve SDE using Diffrax with XLA-compatible dynamic solver selection.
    
    Integrates dY = drift(t, Y)dt + diffusion(t, Y)dW from t0 to t1.
    
    COMPLIANCE: Python.tex §3.1 - Control Flow in Traced Contexts
    Uses jax.lax.cond for dynamic solver selection (not Python if statements).
    Returns solver_idx as jnp.int32 (not str) for XLA compatibility.
    
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
        tuple: (final_state, solver_idx, current_stiffness_metric)
        where solver_idx: 0=Euler, 1=Heun, 2=ImplicitEuler (as jnp.int32)
    
    References:
        - Python.tex §2.2.3: SDE Solver with Diffrax
        - Implementacion.tex §3.3.2: Adaptive Stepping
        - Python.tex §3.1: XLA-Compatible Control Flow
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
    
    # Adaptive step size controller (Zero-Heuristics: tolerances from config)
    stepsize_controller = diffrax.PIDController(
        rtol=config.sde_pid_rtol,
        atol=config.sde_pid_atol,
        dtmin=config.sde_pid_dtmin,
        dtmax=config.sde_pid_dtmax
    )
    
    # Initial step size from config
    dt0 = config.sde_pid_dtmax / config.sde_initial_dt_factor
    
    # Estimate current stiffness from drift/diffusion at initial state
    # CRITICAL: This is a JAX Tracer in @jax.jit context, NOT Python float
    # estimate_stiffness returns float, convert to Array for type consistency
    current_stiffness = jnp.asarray(estimate_stiffness(drift_fn, diffusion_fn, y0, t0, args, config))
    
    # XLA-COMPATIBLE DYNAMIC SOLVER SELECTION via jax.lax.cond
    # (NOT Python if statements which cause ConcretizationTypeError)
    
    def solve_with_euler(_):
        """Explicit Euler for non-stiff systems (solver_idx=0)"""
        sol = diffrax.diffeqsolve(
            terms, diffrax.Euler(), t0=t0, t1=t1, dt0=dt0, y0=y0, args=args,
            stepsize_controller=stepsize_controller, saveat=diffrax.SaveAt(t1=True)
        )
        y_final = sol.ys[-1] if sol.ys is not None else y0
        return y_final, jnp.array(0, dtype=jnp.int32)
        
    def solve_with_heun(_):
        """Heun method for medium-stiffness systems (solver_idx=1)"""
        sol = diffrax.diffeqsolve(
            terms, diffrax.Heun(), t0=t0, t1=t1, dt0=dt0, y0=y0, args=args,
            stepsize_controller=stepsize_controller, saveat=diffrax.SaveAt(t1=True)
        )
        y_final = sol.ys[-1] if sol.ys is not None else y0
        return y_final, jnp.array(1, dtype=jnp.int32)
        
    def solve_with_implicit(_):
        """Implicit Euler for stiff systems (solver_idx=2)"""
        sol = diffrax.diffeqsolve(
            terms, diffrax.ImplicitEuler(), t0=t0, t1=t1, dt0=dt0, y0=y0, args=args,
            stepsize_controller=stepsize_controller, saveat=diffrax.SaveAt(t1=True)
        )
        y_final = sol.ys[-1] if sol.ys is not None else y0
        return y_final, jnp.array(2, dtype=jnp.int32)
    
    # Pure JAX conditional branching (XLA-compatible)
    # if current_stiffness < stiffness_low: use Euler
    # elif current_stiffness < stiffness_high: use Heun
    # else: use ImplicitEuler
    def choose_medium_or_high(operand):
        # Inner cond for medium vs high stiffness
        return jax.lax.cond(
            current_stiffness < config.stiffness_high,
            solve_with_heun,
            solve_with_implicit,
            operand=operand
        )
    
    y_final, solver_idx = jax.lax.cond(
        current_stiffness < config.stiffness_low,
        solve_with_euler,
        choose_medium_or_high,
        operand=None
    )
    
    # Return final state, numeric solver index, and stiffness metric (all as Arrays/JAX types)
    return y_final, solver_idx, current_stiffness


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
    # solve_sde now returns (y_final, solver_idx, stiffness_metric) where solver_idx is jnp.int32
    key_sde, key_jump = jax.random.split(key)
    y_final, solver_idx, stiffness_metric = solve_sde(
        drift_fn=drift_levy_stable,
        diffusion_fn=diffusion_levy,
        y0=y0,
        t0=t0,
        t1=t1,
        key=key_sde,
        config=config,
        args=args
    )

    jump_sum, jump_count = sample_levy_jump_component(
        key=key_jump,
        horizon=horizon,
        config=config,
    )
    
    # Prediction
    prediction = y_final[0] + jump_sum
    
    # Confidence: Theoretical variance of Lévy stable process
    # For α-stable: Var ~ t^(2/α) (power law)
    # For Brownian (α=2): Var = σ^2 * t
    if alpha > config.kernel_c_alpha_gaussian_threshold:  # Near-Gaussian regime
        variance = (sigma ** 2) * horizon
    else:  # Heavy-tailed Lévy
        variance = (sigma ** alpha) * (horizon ** (2.0 / alpha))

    jump_variance = (
        config.kernel_c_jump_intensity
        * horizon
        * (config.kernel_c_jump_scale ** 2 + config.kernel_c_jump_mean ** 2)
    )
    variance = variance + jump_variance
    
    confidence = jnp.sqrt(variance)
    
    # Diagnostics
    drift_estimate, martingale_component, finite_variation = decompose_semimartingale(
        signal=signal,
        dt=config.sde_dt,
    )
    information_drift = compute_information_drift(
        martingale_component=martingale_component,
        dt=config.sde_dt,
    )

    diagnostics = {
        "kernel_type": "C_Ito_Levy_SDE",
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "horizon": horizon,
        "solver_idx": jnp.asarray(solver_idx),
        "stiffness_metric": jnp.asarray(stiffness_metric),
        "final_state": jnp.asarray(y_final[0]),
        "jump_count": jnp.asarray(jump_count),
        "jump_sum": jnp.asarray(jump_sum),
        "jump_variance": jnp.asarray(jump_variance),
        "semimartingale_drift": jnp.asarray(drift_estimate),
        "semimartingale_martingale": jnp.asarray(martingale_component[-1]),
        "semimartingale_finite_variation": jnp.asarray(finite_variation[-1]),
        "information_drift": jnp.asarray(information_drift),
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

    # Apply stop_gradient to diagnostics
    prediction, diagnostics = apply_stop_gradient_to_diagnostics(
        prediction, diagnostics
    )
    
    return KernelOutput(
        prediction=prediction,
        confidence=confidence,
        entropy=entropy,
        probability_density=probability_density,
        kernel_id="C",
        computation_time_us=jnp.array(config.kernel_output_time_us),
        numerics_flags=numerics_flags,
        metadata=diagnostics,
    )


# Public API
__all__ = [
    "kernel_c_predict",
    "solve_sde",
    "drift_levy_stable",
    "diffusion_levy",
    "sample_levy_jump_component",
    "decompose_semimartingale",
    "compute_information_drift",
]
