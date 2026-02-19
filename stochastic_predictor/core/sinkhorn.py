"""Sinkhorn utilities for Wasserstein fusion.

Implements volatility-coupled entropic regularization and
entropy-regularized OT via ott-jax.

References:
    - Predictor_Estocastico_Implementacion.tex §2.4
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from stochastic_predictor.api.types import PredictorConfig


@dataclass(frozen=True)
class SinkhornResult:
    """Sinkhorn solver outputs for diagnostics and fusion."""
    transport_matrix: Float[Array, "n n"]
    reg_ot_cost: Float[Array, ""]
    converged: bool
    epsilon: Float[Array, ""]


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


def run_sinkhorn(
    source_weights: Float[Array, "n"],
    target_weights: Float[Array, "n"],
    cost_matrix: Float[Array, "n n"],
    epsilon: Float[Array, ""],
    solver: Optional[sinkhorn.Sinkhorn] = None
) -> SinkhornResult:
    """Run entropy-regularized OT using ott-jax."""
    solver = solver or sinkhorn.Sinkhorn()
    epsilon_value = float(jnp.asarray(epsilon))
    geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=epsilon_value)
    problem = linear_problem.LinearProblem(geom, a=source_weights, b=target_weights)
    out = solver(problem)
    reg_ot_cost = out.reg_ot_cost
    if reg_ot_cost is None:
        reg_ot_cost = jnp.array(0.0)

    return SinkhornResult(
        transport_matrix=out.matrix,
        reg_ot_cost=jnp.asarray(reg_ot_cost),
        converged=bool(out.converged),
        epsilon=jnp.asarray(epsilon_value),
    )
