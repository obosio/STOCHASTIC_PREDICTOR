"""Wasserstein fusion for kernel ensemble outputs.

Combines heterogeneous kernel predictions using confidence-weighted
JKO updates with volatility-coupled Sinkhorn regularization.
"""

from dataclasses import dataclass
from typing import Iterable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from stochastic_predictor.api.types import PredictorConfig, PredictionResult
from stochastic_predictor.api.validation import validate_simplex
from stochastic_predictor.kernels.base import KernelOutput
from stochastic_predictor.core.sinkhorn import (
    SinkhornResult,
    compute_cost_matrix,
    volatility_coupled_sinkhorn,
)


@dataclass(frozen=True)
class FusionResult:
    """Outputs of the JKO fusion step."""
    fused_prediction: Float[Array, ""]
    updated_weights: Float[Array, "4"]
    free_energy: Float[Array, ""]
    sinkhorn_converged: Bool[Array, ""]
    sinkhorn_epsilon: Float[Array, ""]
    sinkhorn_transport: Float[Array, "4 4"]
    sinkhorn_max_err: Float[Array, ""]
    fisher_rao_distance: Float[Array, ""]


def compute_fisher_rao_distance(
    current_weights: Float[Array, "n"],
    target_weights: Float[Array, "n"],
    config: PredictorConfig
) -> Float[Array, ""]:
    """
    Compute Fisher-Rao distance on the probability simplex.

    Formula:
        d_FR(p, q) = 2 * arccos(sum_i sqrt(p_i * q_i))

    References:
        - Theory.tex §3.6: Fisher-Rao metric
    """
    eps = config.numerical_epsilon
    p = jnp.clip(current_weights, eps, 1.0)
    q = jnp.clip(target_weights, eps, 1.0)
    inner = jnp.sum(jnp.sqrt(p * q))
    inner = jnp.clip(inner, 0.0, 1.0)
    return 2.0 * jnp.arccos(inner)


def _normalize_confidences(
    confidences: Float[Array, "n"],
    config: PredictorConfig
) -> Float[Array, "n"]:
    """Normalize confidences into a simplex with numerical stability."""
    weights_raw = jnp.maximum(confidences, 0.0) + config.numerical_epsilon
    return weights_raw / jnp.sum(weights_raw)


def _jko_update_weights(
    current_weights: Float[Array, "n"],
    target_weights: Float[Array, "n"],
    config: PredictorConfig
) -> Float[Array, "n"]:
    """JKO proximal update via convex combination in simplex space."""
    updated = current_weights + config.learning_rate * (target_weights - current_weights)
    updated = jnp.maximum(updated, 0.0)
    updated = updated / jnp.sum(updated)
    return updated


def fuse_kernel_outputs(
    kernel_outputs: Iterable[KernelOutput],
    current_weights: Float[Array, "4"],
    ema_variance: Float[Array, "1"],
    config: PredictorConfig
) -> FusionResult:
    """
    Fuse kernel predictions with JKO-weighted Sinkhorn coupling.

    Args:
        kernel_outputs: Outputs from kernels A, B, C, D
        current_weights: Previous weights [rho_A, rho_B, rho_C, rho_D]
        ema_variance: EWMA variance (sigma_t^2) for volatility coupling
        config: PredictorConfig
    """
    predictions = jnp.array([ko.prediction for ko in kernel_outputs]).reshape(-1)
    confidences = jnp.array([ko.confidence for ko in kernel_outputs]).reshape(-1)

    target_weights = _normalize_confidences(confidences, config)

    cost_matrix = compute_cost_matrix(predictions, config)
    sinkhorn_result: SinkhornResult = volatility_coupled_sinkhorn(
        source_weights=current_weights,
        target_weights=target_weights,
        cost_matrix=cost_matrix,
        ema_variance=ema_variance,
        config=config,
    )

    fisher_rao_distance = compute_fisher_rao_distance(
        current_weights=current_weights,
        target_weights=target_weights,
        config=config,
    )

    updated_weights = _jko_update_weights(current_weights, target_weights, config)

    is_valid, msg = validate_simplex(updated_weights, config.validation_simplex_atol, "weights")
    if not is_valid:
        raise ValueError(msg)

    fused_prediction = jnp.sum(updated_weights * predictions)

    return FusionResult(
        fused_prediction=fused_prediction,
        updated_weights=updated_weights,
        free_energy=sinkhorn_result.reg_ot_cost,
        sinkhorn_converged=sinkhorn_result.converged,
        sinkhorn_epsilon=sinkhorn_result.epsilon,
        sinkhorn_transport=sinkhorn_result.transport_matrix,
        sinkhorn_max_err=sinkhorn_result.max_err,
        fisher_rao_distance=fisher_rao_distance,
    )
