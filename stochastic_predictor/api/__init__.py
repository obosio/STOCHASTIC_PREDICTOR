"""API Layer - External Contracts and Validation.

This layer exposes the predictor's public interface, including:
  - Data structures (types.py): PredictorConfig, MarketObservation, PredictionResult
  - PRNG management (prng.py): JAX threefry2x32 deterministic configuration
  - Input/output validation (validation.py): Sanitization and domain verification

Responsibilities per Clean Architecture:
  - Public contract definition (immutable dataclasses)
  - Strict type validation (jaxtyping)
  - Load shedding and inference degradation (TTL, Nyquist)
  - System global configuration

References:
  - Predictor_Estocastico_Python.tex §2.1, line 381: "api/ must contain exclusively
    global configuration, strict validation schemas, and degradation logic"
  - Predictor_Estocastico_API_Python.tex §1: Data Structures (Typing)
"""

# Import and re-export from submodules
from stochastic_predictor.api.types import (
    PredictorConfig,
    MarketObservation,
    PredictionResult,
    KernelOutput,
    InternalState,
    OperatingMode,
    KernelType,
    check_jax_config,
)

from stochastic_predictor.api.prng import (
    initialize_jax_prng,
    split_key,
    split_key_like,
    uniform_samples,
    normal_samples,
    exponential_samples,
    check_prng_state,
    verify_determinism,
)

from stochastic_predictor.api.validation import (
    validate_price,
    validate_timestamp,
    check_staleness,
    validate_shape,
    validate_finite,
    validate_simplex,
    validate_holder_exponent,
    validate_alpha_stable,
    validate_beta_stable,
    sanitize_array,
    warn_if_invalid,
)

# Consolidated public exports
__all__ = [
    # Types
    "PredictorConfig",
    "MarketObservation",
    "PredictionResult",
    "KernelOutput",
    "InternalState",
    "OperatingMode",
    "KernelType",
    "check_jax_config",
    # PRNG
    "initialize_jax_prng",
    "split_key",
    "split_key_like",
    "uniform_samples",
    "normal_samples",
    "exponential_samples",
    "check_prng_state",
    "verify_determinism",
    # Validation
    "validate_price",
    "validate_timestamp",
    "check_staleness",
    "validate_shape",
    "validate_finite",
    "validate_simplex",
    "validate_holder_exponent",
    "validate_alpha_stable",
    "validate_beta_stable",
    "sanitize_array",
    "warn_if_invalid",
]
