"""API Layer - External Contracts and Validation.

This layer exposes the predictor's public interface, including:
  - Data structures (types.py): PredictorConfig, ProcessState, PredictionResult
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
    ProcessState,
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
    validate_magnitude,
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
    ensure_float64,
    sanitize_external_observation,
    cast_array_to_float64,
)

from stochastic_predictor.api.schemas import (
    OperatingMode as OperatingModeSchema,
    ProcessStateSchema,
    KernelOutputSchema,
    TelemetryDataSchema,
    PredictionResultSchema,
    HealthCheckResponseSchema,
)

from stochastic_predictor.api.config import (
    ConfigManager,
    get_config,
    PredictorConfigInjector,
)

from stochastic_predictor.api.warmup import (
    warmup_all_kernels,
    warmup_kernel_a,
    warmup_kernel_b,
    warmup_kernel_c,
    warmup_kernel_d,
    warmup_with_retry,
    profile_warmup_and_recommend_timeout,
)

from stochastic_predictor.api.state_buffer import (
    update_signal_history,
    update_residual_buffer,
    batch_update_signal_history,
    update_cusum_statistics,
    update_ema_variance,
    atomic_state_update,
    reset_cusum_statistics,
)

# Consolidated public exports
__all__ = [
    # Types
    "PredictorConfig",
    "ProcessState",
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
    "validate_magnitude",
    "validate_timestamp",
    "check_staleness",
    "validate_shape",
    "validate_finite",
    "validate_simplex",
    "validate_holder_exponent",
    "validate_alpha_stable",
    "ensure_float64",
    "sanitize_external_observation",
    "cast_array_to_float64",
    "validate_beta_stable",
    "sanitize_array",
    "warn_if_invalid",
    # Schemas
    "OperatingModeSchema",
    "ProcessStateSchema",
    "KernelOutputSchema",
    "TelemetryDataSchema",
    "PredictionResultSchema",
    "HealthCheckResponseSchema",
    # Configuration
    "ConfigManager",
    "get_config",
    "PredictorConfigInjector",
    # Warmup (JIT Pre-compilation)
    "warmup_all_kernels",
    "warmup_kernel_a",
    "warmup_kernel_b",
    "profile_warmup_and_recommend_timeout",
    "warmup_kernel_c",
    "warmup_kernel_d",
    "warmup_with_retry",
    # State Buffer Management (Zero-Copy)
    "update_signal_history",
    "update_residual_buffer",
    "batch_update_signal_history",
    "update_cusum_statistics",
    "update_ema_variance",
    "atomic_state_update",
    "reset_cusum_statistics",
]
