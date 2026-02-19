"""Sinkhorn utilities for Wasserstein fusion.

Implements volatility-coupled entropic regularization with a manual
Sinkhorn loop using jax.lax.scan for predictable XLA lowering.

References:
    - Stochastic_Predictor_Implementation.tex §2.4
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.scipy.special import logsumexp

from stochastic_predictor.api.types import PredictorConfig


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
    """
    sigma_t = jnp.sqrt(jnp.maximum(ema_variance, config.numerical_epsilon))
    epsilon_t = config.sinkhorn_epsilon_0 * (1.0 + config.sinkhorn_alpha * sigma_t)
    return jnp.maximum(config.sinkhorn_epsilon_min, epsilon_t)


def compute_cost_matrix(
    predictions: Float[Array, "n"],
    config: PredictorConfig
) -> Float[Array, "n n"]:
    """Compute pairwise squared-distance cost matrix for kernel predictions."""
    del config
    diffs = predictions[:, None] - predictions[None, :]
    return jnp.square(diffs)


def _smin(matrix: Float[Array, "n n"], epsilon: Float[Array, ""]) -> Float[Array, "n"]:
    """Log-domain soft minimum operator (stable Sinkhorn update)."""
    return -epsilon * logsumexp(-matrix / epsilon, axis=1)


def volatility_coupled_sinkhorn(
    source_weights: Float[Array, "n"],
    target_weights: Float[Array, "n"],
    cost_matrix: Float[Array, "n n"],
    ema_variance: Float[Array, "1"],
    config: PredictorConfig
) -> SinkhornResult:
    """Run Sinkhorn with volatility-coupled epsilon using jax.lax.scan."""
    log_a = jnp.log(jnp.maximum(source_weights, config.numerical_epsilon))
    log_b = jnp.log(jnp.maximum(target_weights, config.numerical_epsilon))

    f0 = jnp.zeros_like(source_weights)
    g0 = jnp.zeros_like(target_weights)

    def sinkhorn_step(carry, _):
        f, g = carry
        epsilon = compute_sinkhorn_epsilon(ema_variance, config)
        f = _smin(cost_matrix - g[None, :], epsilon) + log_a
        g = _smin(cost_matrix.T - f[None, :], epsilon) + log_b
        return (f, g), None

    (f_final, g_final), _ = jax.lax.scan(
        sinkhorn_step,
        (f0, g0),
        None,
        length=config.sinkhorn_max_iter
    )

    epsilon_final = compute_sinkhorn_epsilon(ema_variance, config)
    transport = jnp.exp((f_final[:, None] + g_final[None, :] - cost_matrix) / epsilon_final)
    safe_transport = jnp.maximum(transport, config.numerical_epsilon)
    entropy_term = jnp.sum(safe_transport * (jnp.log(safe_transport) - 1.0))
    reg_ot_cost = jnp.sum(transport * cost_matrix) + epsilon_final * entropy_term

    row_sums = jnp.sum(transport, axis=1)
    col_sums = jnp.sum(transport, axis=0)
    row_err = jnp.max(jnp.abs(row_sums - source_weights))
    col_err = jnp.max(jnp.abs(col_sums - target_weights))
    max_err = jnp.maximum(row_err, col_err)
    converged = jnp.asarray(max_err <= config.validation_simplex_atol)

    return SinkhornResult(
        transport_matrix=transport,
        reg_ot_cost=reg_ot_cost,
        converged=converged,
        epsilon=jnp.asarray(epsilon_final),
        max_err=jnp.asarray(max_err),
    )
