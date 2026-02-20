"""Kernel C: Itô/Lévy SDE Integration with Hybrid Schemes.

This kernel integrates stochastic differential equations using explicit,
implicit, and hybrid schemes based on real-time stiffness estimation.

References:
    - Teoria.tex §2.3.3: Rama C (Itô/Lévy SDEs)
    - Python.tex §2.2.3: Kernel C (Neural SDE)
    - Implementacion.tex §3.3: IMEX Splitting for Stiff SDEs
"""

import jax
import jax.numpy as jnp
from functools import partial
from jaxtyping import Array, Float
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
    dt: float,
    args: tuple,
    config
) -> Float[Array, ""]:
    """
    Estimate stiffness metric per Theory.tex §2.3.3.

    Uses diffusion Jacobian ratio and volatility-change term.
    """
    eps = config.sde_fd_epsilon
    y_dim = y.shape[0]

    diffusion_matrix = diffusion_fn(t, y, args)
    sigma_norm = jnp.linalg.norm(diffusion_matrix)

    basis = jnp.eye(y_dim)
    sigma_plus = jax.vmap(lambda e: diffusion_fn(t, y + eps * e, args))(basis)
    sigma_minus = jax.vmap(lambda e: diffusion_fn(t, y - eps * e, args))(basis)
    sigma_grad = (sigma_plus - sigma_minus) / (2.0 * eps)
    grad_norms = jnp.linalg.norm(sigma_grad.reshape(y_dim, -1), axis=1)
    max_grad = jnp.max(grad_norms)
    min_grad = jnp.max(jnp.array([jnp.min(grad_norms), config.numerical_epsilon]))
    stiffness_ratio = max_grad / min_grad

    drift = drift_fn(t, y, args)
    y_pred = y + drift * dt
    sigma_pred = diffusion_fn(t + dt, y_pred, args)
    sigma_pred_norm = jnp.linalg.norm(sigma_pred)
    dlog_sigma_dt = jnp.abs(
        jnp.log(sigma_pred_norm + config.numerical_epsilon)
        - jnp.log(sigma_norm + config.numerical_epsilon)
    ) / dt
    volatility_term = dlog_sigma_dt * dt

    stiffness_metric = jnp.maximum(stiffness_ratio, volatility_term)

    return jnp.asarray(stiffness_metric)


def select_stiffness_solver(current_stiffness: float, config):
    """Return solver regime label for debugging."""
    if current_stiffness < config.stiffness_low:
        return "explicit"
    if current_stiffness < config.stiffness_high:
        return "hybrid"
    return "implicit"


def drift_levy_stable(
    t: Float[Array, ""],
    y: Float[Array, "d"],
    args: tuple
) -> Float[Array, "d"]:
    """
    Drift term f(t, y) for Lévy stable process.
    """
    mu, alpha, beta, sigma = args
    return jnp.full_like(y, mu)


def diffusion_levy(
    t: Float[Array, ""],
    y: Float[Array, "d"],
    args: tuple
) -> Float[Array, "d d"]:
    """Diffusion term g(t, y) for Lévy process with config-driven sigma."""
    mu, alpha, beta, sigma = args
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
) -> tuple[Float[Array, "d"], Array, Float[Array, ""]]:
    """
    Solve SDE with explicit/implicit/hybrid transition per Theory.tex §2.3.3.

    Implements a predictor-corrector trapezoidal implicit step and hybrid
    convex mixing in the medium-stiffness regime.
    """
    n_steps = config.sde_numel_integrations
    horizon = t1 - t0
    dt = horizon / n_steps
    y_dim = y0.shape[0]

    noise = jax.random.normal(key, (n_steps, y_dim))

    def step(carry, noise_t):
        y, t, max_stiffness, solver_idx = carry
        dW = noise_t * jnp.sqrt(dt)

        stiffness_metric = estimate_stiffness(
            drift_fn=drift_fn,
            diffusion_fn=diffusion_fn,
            y=y,
            t=t,
            dt=dt,
            args=args,
            config=config,
        )

        drift = drift_fn(t, y, args)
        diffusion = diffusion_fn(t, y, args)
        diffusion_step = diffusion @ dW

        y_explicit = y + drift * dt + diffusion_step
        drift_pred = drift_fn(t + dt, y_explicit, args)
        y_implicit = y + 0.5 * dt * (drift + drift_pred) + diffusion_step

        lambda_mix = (stiffness_metric - config.stiffness_low) / (
            config.stiffness_high - config.stiffness_low
        )
        lambda_mix = jnp.clip(lambda_mix, 0.0, 1.0)

        use_explicit = stiffness_metric < config.stiffness_low
        use_implicit = stiffness_metric >= config.stiffness_high

        y_hybrid = (1.0 - lambda_mix) * y_explicit + lambda_mix * y_implicit
        y_next = jnp.where(
            use_explicit,
            y_explicit,
            jnp.where(use_implicit, y_implicit, y_hybrid),
        )

        solver_idx_next = jnp.where(
            use_explicit,
            jnp.array(0, dtype=jnp.int32),
            jnp.where(use_implicit, jnp.array(2, dtype=jnp.int32), jnp.array(1, dtype=jnp.int32)),
        )

        max_stiffness = jnp.maximum(max_stiffness, stiffness_metric)
        return (y_next, t + dt, max_stiffness, solver_idx_next), None

    init_carry = (y0, t0, jnp.array(0.0), jnp.array(0, dtype=jnp.int32))
    (y_final, _, max_stiffness, solver_idx), _ = jax.lax.scan(step, init_carry, noise)

    return y_final, solver_idx, max_stiffness


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


@partial(jax.jit, static_argnames=('config',))
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
