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
  - Stochastic_Predictor_Python.tex §2.1, line 381: "api/ must contain exclusively
    global configuration, strict validation schemas, and degradation logic"
  - Stochastic_Predictor_API_Python.tex §1: Data Structures (Typing)
"""

from Python.api.config import ConfigManager, PredictorConfigInjector, get_config
from Python.api.prng import (
    check_prng_state,
    exponential_samples,
    initialize_jax_prng,
    normal_samples,
    split_key,
    split_key_like,
    uniform_samples,
    verify_determinism,
)
from Python.api.schemas import (
    HealthCheckResponseSchema,
    KernelOutputSchema,
)
from Python.api.schemas import OperatingMode as OperatingModeSchema
from Python.api.schemas import (
    PredictionResultSchema,
    ProcessStateSchema,
    TelemetryDataSchema,
)
from Python.api.state_buffer import (
    atomic_state_update,
    batch_update_signal_history,
    reset_cusum_statistics,
    update_cusum_statistics,
    update_ema_variance,
    update_residual_buffer,
    update_signal_history,
)

# Import and re-export from submodules
from Python.api.types import (
    InternalState,
    KernelOutput,
    KernelType,
    OperatingMode,
    PredictionResult,
    PredictorConfig,
    ProcessState,
    check_jax_config,
)
from Python.api.validation import (
    cast_array_to_float64,
    check_staleness,
    ensure_float64,
    sanitize_array,
    sanitize_external_observation,
    validate_alpha_stable,
    validate_beta_stable,
    validate_finite,
    validate_holder_exponent,
    validate_magnitude,
    validate_shape,
    validate_simplex,
    validate_timestamp,
    warn_if_invalid,
)
from Python.api.warmup import (
    profile_warmup_and_recommend_timeout,
    warmup_all_kernels,
    warmup_kernel_a,
    warmup_kernel_b,
    warmup_kernel_c,
    warmup_kernel_d,
    warmup_with_retry,
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
