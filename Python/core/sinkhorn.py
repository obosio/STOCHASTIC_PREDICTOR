"""Sinkhorn utilities for Wasserstein fusion.

Implements volatility-coupled entropic regularization using the native
ott-jax library for maximum numerical stability and XLA efficiency.

References:
    - Stochastic_Predictor_Implementation.tex §2.4 (Golden Master: OTT-JAX)
    - Python.tex §2.1 (Precision Requirements)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from Python.api.types import PredictorConfig


@dataclass(frozen=True)
class SinkhornResult:
    """Sinkhorn solver outputs for diagnostics and fusion."""
    transport_matrix: Float[Array, "n n"]
    reg_ot_cost: Float[Array, ""]
    converged: Array
    epsilon: Float[Array, ""]
    max_err: Float[Array, ""]


def compute_sinkhorn_epsilon(
    ema_variance: Float[Array, "1"],
    config: PredictorConfig
) -> Float[Array, ""]:
    """
    Compute volatility-coupled Sinkhorn regularization.

    Formula:
        epsilon_t = max(epsilon_min, epsilon_0 * (1 + alpha * sigma_t))
    
    Apply stop_gradient to prevent backprop contamination (VRAM constraint).
    References: MIGRATION_AUTOTUNING_v1.0.md §4 (VRAM Constraint)
    """
    # Stop gradient on variance to avoid polluting neural net gradients
    ema_variance_sg = jax.lax.stop_gradient(ema_variance)
    sigma_t = jnp.sqrt(jnp.maximum(ema_variance_sg, config.numerical_epsilon))
    epsilon_t = config.sinkhorn_epsilon_0 * (1.0 + config.sinkhorn_alpha * sigma_t)
    return jax.lax.stop_gradient(jnp.maximum(config.sinkhorn_epsilon_min, epsilon_t))


def compute_cost_matrix(
    predictions: Float[Array, "n"],
    config: PredictorConfig
) -> Float[Array, "n n"]:
    """Compute pairwise cost matrix for kernel predictions."""
    diffs = predictions[:, None] - predictions[None, :]
    if config.sinkhorn_cost_type == "huber":
        delta = config.sinkhorn_huber_delta
        abs_diffs = jnp.abs(diffs)
        quadratic = 0.5 * jnp.square(diffs)
        linear = delta * (abs_diffs - 0.5 * delta)
        return jnp.where(abs_diffs <= delta, quadratic, linear)
    return jnp.square(diffs)


def volatility_coupled_sinkhorn(
    source_weights: Float[Array, "n"],
    target_weights: Float[Array, "n"],
    cost_matrix: Float[Array, "n n"],
    ema_variance: Float[Array, "1"],
    config: PredictorConfig
) -> SinkhornResult:
    """
    Run Sinkhorn with volatility-coupled epsilon using native OTT-JAX.
    
    COMPLIANCE: Golden Master (Implementacion.tex §2.4) requires ott-jax==0.4.5
    for numerically stable Wasserstein distance computation with optimal transport.
    
    Args:
        source_weights: Initial kernel weights [ρ_A, ρ_B, ρ_C, ρ_D]
        target_weights: Target simplex [0.25, 0.25, 0.25, 0.25]
        cost_matrix: Pairwise prediction distance matrix
        ema_variance: EWMA volatility estimate (scalar)
        config: Predictor configuration with Sinkhorn parameters
    
    Returns:
        SinkhornResult with transport matrix, cost, convergence flags
    
    References:
        - Implementacion.tex §2.4: Golden Master Dependencies (OTT-JAX)
        - Python.tex §2.1: Precision & Numerical Stability
        - Theory.tex §5.2: Optimal Transport Geometry
    """
    # Compute volatility-coupled epsilon as float (required by OTT API)
    epsilon_final_scalar = float(compute_sinkhorn_epsilon(ema_variance, config))
    
    # Instantiate OTT Geometry with cost matrix and dynamic epsilon
    # OTT-JAX automatically handles numerical stability via log-domain operations
    geom = geometry.Geometry(
        cost_matrix=cost_matrix,
        epsilon=epsilon_final_scalar,
        scale_cost="mean"  # Normalize costs for numerical robustness
    )
    
    # Create LinearProblem with marginal constraints (source and target weights)
    ot_prob = linear_problem.LinearProblem(
        geom,
        a=source_weights,
        b=target_weights
    )
    
    # Native OTT-JAX Sinkhorn solver with config-driven parameters
    solver = sinkhorn.Sinkhorn(
        max_iterations=int(config.sinkhorn_max_iter),
        threshold=float(config.validation_simplex_atol),
        inner_iterations=int(config.sinkhorn_inner_iterations)
    )
    
    # Dispatch to OTT solver with LinearProblem
    ott_result = solver(ot_prob)
    
    # Extract transport matrix and cost from OTT result
    transport = ott_result.matrix
    
    # Compute regularized OT cost (entropy penalty included by OTT)
    # Fallback to 0 if None
    reg_ot_cost = ott_result.reg_ot_cost if ott_result.reg_ot_cost is not None else jnp.array(0.0)
    
    # Convergence detection from OTT solver (numerical measure in dual variables)
    converged = jnp.asarray(ott_result.converged)
    
    # Extract max marginal error for diagnostics
    # OTT stores errors array; take max for simplex constraint violation
    max_err = jnp.max(jnp.asarray(ott_result.errors)) if ott_result.errors is not None else jnp.array(0.0)
    
    return SinkhornResult(
        transport_matrix=transport,
        reg_ot_cost=jnp.asarray(reg_ot_cost),
        converged=converged,
        epsilon=jnp.asarray(epsilon_final_scalar),
        max_err=jnp.asarray(max_err),
    )
