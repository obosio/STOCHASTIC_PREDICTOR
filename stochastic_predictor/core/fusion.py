"""Wasserstein fusion for kernel ensemble outputs.

Combines heterogeneous kernel predictions using confidence-weighted
JKO updates with volatility-coupled Sinkhorn regularization.
"""

from dataclasses import dataclass
from typing import Iterable

import jax.numpy as jnp
from jaxtyping import Array, Float

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
    sinkhorn_converged: bool
    sinkhorn_epsilon: Float[Array, ""]
    sinkhorn_transport: Float[Array, "4 4"]


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

    updated_weights = _jko_update_weights(current_weights, target_weights, config)

    PredictionResult.validate_simplex(updated_weights, config.validation_simplex_atol)
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
    )
