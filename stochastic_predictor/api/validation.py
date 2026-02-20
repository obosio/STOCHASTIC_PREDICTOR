"""
Validation Functions for Predictor Input/Output.

This module provides domain validation, data sanitization, and anomaly detection
to ensure the integrity of the predictive pipeline.

References:
    - Stochastic_Predictor_API_Python.tex §1.2: validate_domain()
    - Stochastic_Predictor_IO.tex §2: Observation Protocol
    - Stochastic_Predictor_Implementation.tex §3: Quality Control
"""

from datetime import datetime
from typing import Union, Tuple, TYPE_CHECKING
import jax.numpy as jnp
from jaxtyping import Float, Array
import warnings

if TYPE_CHECKING:
    from stochastic_predictor.api.types import PredictorConfig


# ═══════════════════════════════════════════════════════════════════════════
# PROCESS STATE OBSERVATION VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_magnitude(
    magnitude: Union[float, Float[Array, "1"]],
    sigma_bound: float,
    sigma_val: float,
    allow_nan: bool
) -> Tuple[bool, str]:
    """
    Validate that a magnitude is within statistically reasonable bounds.
    
    Design: Detection of catastrophic outliers (> N sigma) that could
    indicate data feed errors or sensor malfunctions.
    
    Domain-Agnostic: Applies to any stochastic process without semantic assumptions.
    
    Zero-Heuristics Policy: All parameters MUST be passed from PredictorConfig.
    No default values to enforce configuration-driven operation.
    
    Args:
        magnitude: Value to validate (scalar or JAX array)
        sigma_bound: Maximum number of standard deviations allowed (from config.sigma_bound)
        sigma_val: Reference standard deviation (from config.sigma_val)
        allow_nan: If True, permits NaN values (from config.validation_finite_allow_nan)
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    References:
        - Stochastic_Predictor_API_Python.tex §1.2: ProcessState.validate_domain()
        
    Example:
        >>> from stochastic_predictor.api.validation import validate_magnitude
        >>> from stochastic_predictor.api.config import PredictorConfigInjector
        >>> config = PredictorConfigInjector().create_config()
        >>> is_valid, msg = validate_magnitude(
        ...     magnitude=jnp.array([100.5]),
        ...     sigma_bound=config.sigma_bound,
        ...     sigma_val=config.sigma_val
        ... )
        >>> assert is_valid
    """
    # Convert to JAX array if necessary
    magnitude_arr = jnp.asarray(magnitude)
    
    # Check NaN
    if jnp.any(jnp.isnan(magnitude_arr)):
        if not allow_nan:
            return False, "Signal magnitude contains NaN values"
    
    # Check infinities
    if jnp.any(jnp.isinf(magnitude_arr)):
        return False, "Signal magnitude contains infinite values"
    
    # Check bounds (outlier detection)
    threshold = sigma_bound * sigma_val
    if jnp.any(jnp.abs(magnitude_arr) > threshold):
        max_val = float(jnp.max(jnp.abs(magnitude_arr)))
        return False, (
            f"Signal magnitude outlier detected: |magnitude|={max_val:.2f} exceeds "
            f"{sigma_bound}σ threshold ({threshold:.2f})"
        )
    
    return True, "Valid"


def validate_timestamp(
    timestamp_ns: int,
    max_future_drift_ns: int,
    max_past_drift_ns: int
) -> Tuple[bool, str]:
    """
    Validate that a timestamp is in a reasonable range relative to current time.
    
    Zero-Heuristics Policy: All drift thresholds MUST be passed from PredictorConfig.
    No default values to enforce configuration-driven operation.
    
    Args:
        timestamp_ns: Unix timestamp in nanoseconds
        max_future_drift_ns: Maximum allowed future drift (from config.max_future_drift_ns)
        max_past_drift_ns: Maximum allowed past drift (from config.max_past_drift_ns)
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    References:
        - Stochastic_Predictor_API_Python.tex §3.1: Staleness TTL
        - Stochastic_Predictor_IO.tex §2.2: Timestamp Validation
        
    Example:
        >>> config = PredictorConfigInjector().create_config()
        >>> is_valid, msg = validate_timestamp(
        ...     timestamp_ns=time.time_ns(),
        ...     max_future_drift_ns=config.max_future_drift_ns,
        ...     max_past_drift_ns=config.max_past_drift_ns
        ... )
    """
    import time
    
    current_time_ns = time.time_ns()
    
    # Verify non-negative
    if timestamp_ns < 0:
        return False, f"Invalid timestamp: {timestamp_ns} (must be non-negative)"
    
    # Check future drift
    if timestamp_ns > current_time_ns + max_future_drift_ns:
        drift_s = (timestamp_ns - current_time_ns) / 1e9
        return False, f"Timestamp too far in future: {drift_s:.2f}s drift"
    
    # Check past drift
    if timestamp_ns < current_time_ns - max_past_drift_ns:
        drift_s = (current_time_ns - timestamp_ns) / 1e9
        return False, f"Timestamp too old: {drift_s:.2f}s drift"
    
    return True, "Valid"


def check_staleness(
    timestamp_ns: int,
    config: 'PredictorConfig'
) -> Tuple[bool, int]:
    """
    Verify if an observation is stale according to TTL from configuration.
    
    Zero-Heuristics Policy: TTL is NOT hardcoded; must come from PredictorConfig.
    
    Args:
        timestamp_ns: Observation timestamp
        config: PredictorConfig with staleness_ttl_ns parameter
        
    Returns:
        Tuple (is_stale: bool, age_ns: int)
        
    References:
        - Stochastic_Predictor_API_Python.tex §3.1: Staleness Policy (Staleness TTL)
        - types.py: PredictorConfig.staleness_ttl_ns
        
    Example:
        >>> import time
        >>> from stochastic_predictor.api.validation import check_staleness
        >>> from stochastic_predictor.api.config import PredictorConfigInjector
        >>> config = PredictorConfigInjector().create_config()
        >>> ts = time.time_ns() - 100_000_000  # 100ms ago
        >>> is_stale, age = check_staleness(ts, config)
        >>> ts = time.time_ns() - 1_000_000_000  # 1 second ago
        >>> is_stale, age = check_staleness(ts, ttl_ns=500_000_000)
        >>> assert is_stale  # Older than 500ms TTL
    """
    import time
    
    current_time_ns = time.time_ns()
    age_ns = current_time_ns - timestamp_ns
    is_stale = age_ns > config.staleness_ttl_ns
    
    return bool(is_stale), int(age_ns)


# ═══════════════════════════════════════════════════════════════════════════
# ARRAY AND SHAPE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_shape(
    array: jnp.ndarray,
    expected_shape: Tuple[int, ...],
    name: str
) -> Tuple[bool, str]:
    """
    Validate that an array has the expected shape.
    
    Zero-Heuristics Policy: All parameters MUST be passed explicitly.
    No default values to enforce configuration-driven operation.
    
    Args:
        array: JAX array to validate
        expected_shape: Expected shape (can contain -1 for variable dimensions)
        name: Array name (for error messages - REQUIRED)
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    Example:
        >>> import jax.numpy as jnp
        >>> from stochastic_predictor.api.validation import validate_shape
        >>> arr = jnp.zeros((10, 4))
        >>> is_valid, msg = validate_shape(arr, (10, 4), name="weights")
        >>> assert is_valid
    """
    actual_shape = array.shape
    
    # Validate number of dimensions
    if len(actual_shape) != len(expected_shape):
        return False, (
            f"{name}: Expected {len(expected_shape)} dimensions, "
            f"got {len(actual_shape)} (shape: {actual_shape})"
        )
    
    # Validate each dimension (ignore -1 as wildcard)
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected != -1 and actual != expected:
            return False, (
                f"{name}: Dimension {i} mismatch - expected {expected}, "
                f"got {actual} (full shape: {actual_shape})"
            )
    
    return True, "Valid"


def validate_finite(
    array: jnp.ndarray,
    name: str,
    allow_nan: bool,
    allow_inf: bool
) -> Tuple[bool, str]:
    """
    Validate that an array contains only finite values.
    
    Zero-Heuristics Policy: All parameters MUST be injected from PredictorConfig.
    No default values to enforce configuration-driven operation.
    
    Args:
        array: JAX array to validate
        name: Array name (for error messages) - from config.validation_finite_name
        allow_nan: If True, permits NaN values - from config.validation_finite_allow_nan
        allow_inf: If True, permits infinite values - from config.validation_finite_allow_inf
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    References:
        - Stochastic_Predictor_Tests_Python.tex §2.1: test_levy_cms_basic_properties
    """
    # Check NaN
    if not allow_nan and jnp.any(jnp.isnan(array)):
        nan_count = int(jnp.sum(jnp.isnan(array)))
        return False, f"{name}: Contains {nan_count} NaN value(s)"
    
    # Check infinities
    if not allow_inf and jnp.any(jnp.isinf(array)):
        inf_count = int(jnp.sum(jnp.isinf(array)))
        return False, f"{name}: Contains {inf_count} infinite value(s)"
    
    return True, "Valid"


def validate_simplex(
    weights: Float[Array, "N"],
    atol: float,
    name: str
) -> Tuple[bool, str]:
    """
    Validate that an array forms a simplex (sum = 1.0, all >= 0).
    
    Design: Critical for verifying JKO orchestrator weights [ρ_A, ρ_B, ρ_C, ρ_D].
    
    Zero-Heuristics Policy: All parameters MUST be injected from PredictorConfig.
    No default values to enforce configuration-driven operation.
    
    Args:
        weights: Weight array to validate
        atol: Absolute tolerance for sum - from config.validation_simplex_atol
        name: Array name (for error messages)
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    References:
        - Stochastic_Predictor_API_Python.tex §3.2: Simplex Constraint
        - types.py: PredictionResult.__post_init__()
        
    Example:
        >>> import jax.numpy as jnp
        >>> from stochastic_predictor.api.validation import validate_simplex
        >>> from stochastic_predictor.api.config import get_config
        >>> config = get_config()
        >>> weights = jnp.array([0.25, 0.25, 0.25, 0.25])
        >>> is_valid, msg = validate_simplex(weights, config.validation_simplex_atol, "weights")
        >>> assert is_valid
    """
    # Verify non-negativity
    if jnp.any(weights < 0):
        min_val = float(jnp.min(weights))
        return False, f"{name}: Contains negative value(s) (min={min_val:.6f})"
    
    # Verify sum
    total = float(jnp.sum(weights))
    if not jnp.isclose(total, 1.0, atol=atol):
        return False, (
            f"{name}: Does not sum to 1.0 (sum={total:.6f}, "
            f"error={abs(total - 1.0):.2e})"
        )
    
    return True, "Valid"


# ═══════════════════════════════════════════════════════════════════════════
# RANGE VALIDATION (DOMAIN CONSTRAINTS)
# ═══════════════════════════════════════════════════════════════════════════

def validate_holder_exponent(
    H: Union[float, Float[Array, "1"]],
    min_val: float,
    max_val: float
) -> Tuple[bool, str]:
    """
    Validate that a Holder exponent is in the specified range.
    
    Zero-Heuristics Policy: All parameters MUST be injected from PredictorConfig.
    No default values to enforce configuration-driven operation.
    
    Args:
        H: Holder exponent to validate
        min_val: Minimum allowed value - from config.validation_holder_exponent_min
        max_val: Maximum allowed value - from config.validation_holder_exponent_max
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    References:
        - Stochastic_Predictor_Theory.tex §2.1: Properties of Levy Processes
        - types.py: PredictorConfig.holder_threshold
    """
    H_val = float(jnp.asarray(H))
    
    if H_val < min_val or H_val > max_val:
        return False, (
            f"Holder exponent out of range: {H_val:.4f} not in [{min_val}, {max_val}]"
        )
    
    return True, "Valid"


def validate_alpha_stable(
    alpha: float,
    min_val: float,
    max_val: float,
    exclusive_bounds: bool
) -> Tuple[bool, str]:
    """
    Validate alpha parameter of Lévy stable distribution.
    
    Zero-Heuristics Policy: All parameters MUST be injected from PredictorConfig.
    No default values to enforce configuration-driven operation.
    
    Args:
        alpha: Stability parameter
        min_val: Minimum value - from config.validation_alpha_stable_min
        max_val: Maximum value - from config.validation_alpha_stable_max
        exclusive_bounds: If True, use (min, max] instead of [min, max] - from config.validation_alpha_stable_exclusive_bounds
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    References:
        - Stochastic_Predictor_Theory.tex §2.2: Alpha-Stable Distributions
        - Stochastic_Predictor_Tests_Python.tex §2.3: Property-Based Testing (alpha bounds)
        
    Note:
        alpha = 2.0 corresponds to Gaussian distribution
        alpha < 2.0 corresponds to heavy tails (Lévy)
    """
    if exclusive_bounds:
        if alpha <= min_val or alpha > max_val:
            return False, (
                f"Alpha parameter out of range: {alpha:.4f} not in ({min_val}, {max_val}]"
            )
    else:
        if alpha < min_val or alpha > max_val:
            return False, (
                f"Alpha parameter out of range: {alpha:.4f} not in [{min_val}, {max_val}]"
            )
    
    return True, "Valid"


def validate_beta_stable(
    beta: float,
    min_val: float,
    max_val: float
) -> Tuple[bool, str]:
    """
    Validate beta (skewness) parameter of stable distribution.
    
    Zero-Heuristics Policy: All parameters MUST be injected from PredictorConfig.
    No default values to enforce configuration-driven operation.
    
    Args:
        beta: Skewness parameter
        min_val: Minimum value - from config.validation_beta_stable_min
        max_val: Maximum value - from config.validation_beta_stable_max
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    References:
        - Stochastic_Predictor_Theory.tex §2.2: Levy Parametrization
        - Stochastic_Predictor_Tests_Python.tex §2.3: Hypothesis strategies (beta bounds)
    """
    if beta < min_val or beta > max_val:
        return False, (
            f"Beta parameter out of range: {beta:.4f} not in [{min_val}, {max_val}]"
        )
    
    return True, "Valid"


# ═══════════════════════════════════════════════════════════════════════════
# SANITIZATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def sanitize_array(
    array: jnp.ndarray,
    replace_nan: float,
    replace_inf: Union[float, None],
    clip_range: Union[Tuple[float, float], None]
) -> jnp.ndarray:
    """
    Sanitize an array by replacing NaN/Inf and applying clipping.
    
    Zero-Heuristics Policy: All parameters MUST be injected from PredictorConfig.
    No default values to enforce configuration-driven operation.
    
    Args:
        array: JAX array to sanitize
        replace_nan: Value to replace NaN (from config.sanitize_replace_nan_value) - None to preserve
        replace_inf: Value to replace Inf (from config.sanitize_replace_inf_value) - None to preserve
        clip_range: Tuple (min, max) for clipping (from config.sanitize_clip_range) - None to skip
        
    Returns:
        Sanitized JAX array
        
    Note: Use with caution - may mask underlying problems. All sanitization
        policies are determined by configuration, not heuristics.
    """
    result = array
    
    # Replace NaN
    if replace_nan is not None:
        result = jnp.where(jnp.isnan(result), replace_nan, result)
    
    # Replace Inf
    if replace_inf is not None:
        result = jnp.where(jnp.isinf(result), replace_inf, result)
    
    # Clipping
    if clip_range is not None:
        min_val, max_val = clip_range
        result = jnp.clip(result, min_val, max_val)
    
    return result


def warn_if_invalid(
    is_valid: bool,
    message: str,
    exception_type: Union[type, None]
) -> None:
    """
    Emit a warning or exception if validation fails.
    
    Zero-Heuristics Policy: exception_type MUST be specified explicitly.
    No default values to enforce configuration-driven operation.
    
    Args:
        is_valid: Validation result
        message: Error message
        exception_type: Exception class to raise if invalid (REQUIRED - pass None to use warnings)
        
    Example:
        >>> from stochastic_predictor.api.validation import validate_magnitude, warn_if_invalid
        >>> is_valid, msg = validate_magnitude(jnp.array([1000.0]), sigma_bound=5.0, sigma_val=1.0)
        >>> warn_if_invalid(is_valid, msg, exception_type=ValueError)
    """
    if not is_valid:
        if exception_type is not None:
            raise exception_type(message)
        else:
            warnings.warn(message, RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════
# PRECISION ENFORCEMENT (FLOAT64 CASTING)
# ═══════════════════════════════════════════════════════════════════════════

def ensure_float64(
    value: Union[float, int, Float[Array, "..."]]
) -> Float[Array, "..."]:
    """
    Explicitly cast value to float64 to prevent precision degradation.
    
    Critical for maintaining consistency with jax_enable_x64 = True.
    External data feeds (CSV, JSON, protobuf) may provide float32 data,
    causing silent precision loss. This function enforces float64.
    
    Args:
        value: Scalar or array to cast (int, float32, or float64)
    
    Returns:
        JAX array with dtype=float64
    
    Example:
        >>> import numpy as np
        >>> from stochastic_predictor.api.validation import ensure_float64
        >>> x_float32 = np.array([1.0, 2.0], dtype=np.float32)  # External source
        >>> x_float64 = ensure_float64(x_float32)  # Explicit cast
        >>> assert x_float64.dtype == jnp.float64
    
    References:
        - Stochastic_Predictor_API_Python.tex §5: Floating-Point Determinism
        - __init__.py: jax_enable_x64 = True enforcement
    """
    return jnp.asarray(value, dtype=jnp.float64)


def sanitize_external_observation(
    magnitude: Union[float, Float[Array, "1"]],
    timestamp_utc: datetime,
    state_tag: str | None = None,
    dispersion_proxy: Float[Array, "1"] | None = None,
) -> tuple[Float[Array, "1"], datetime, str | None, Float[Array, "1"] | None]:
    """
    Sanitize external observation to enforce float64 precision.
    
    Ensures that data from external feeds (market APIs, sensors, CSV files)
    undergoes explicit float64 casting BEFORE entering ProcessState.
    
    Prevents runtime precision degradation warnings and ensures bit-exact
    reproducibility across different data sources.
    
    Args:
        magnitude: Raw magnitude from external source (may be float32)
        timestamp_utc: UTC timestamp (datetime)
        state_tag: Optional process state label
        dispersion_proxy: Optional dispersion proxy (may be float32)
    
    Returns:
        Tuple of (magnitude_f64, timestamp_utc, state_tag, dispersion_proxy_f64)
    
    Example:
        >>> from datetime import datetime, timezone
        >>> from stochastic_predictor.api.validation import sanitize_external_observation
        >>> raw_magnitude = 123.45
        >>> raw_timestamp = datetime.now(timezone.utc)
        >>> mag_f64, ts_utc, tag, disp = sanitize_external_observation(
        ...     raw_magnitude, raw_timestamp
        ... )
        >>> from stochastic_predictor.api.types import ProcessState
        >>> obs = ProcessState(magnitude=mag_f64, timestamp_utc=ts_utc)
    
    References:
        - Stochastic_Predictor_Implementation.tex §6.4: External Data Feed Integration
        - types.py: ProcessState definition
    """
    # Cast magnitude to float64 explicitly
    magnitude_f64 = ensure_float64(magnitude)
    
    # Ensure magnitude is at least 1D array
    if magnitude_f64.ndim == 0:
        magnitude_f64 = jnp.expand_dims(magnitude_f64, axis=0)
    
    dispersion_proxy_f64 = None
    if dispersion_proxy is not None:
        dispersion_proxy_f64 = ensure_float64(dispersion_proxy)
        if dispersion_proxy_f64.ndim == 0:
            dispersion_proxy_f64 = jnp.expand_dims(dispersion_proxy_f64, axis=0)

    return magnitude_f64, timestamp_utc, state_tag, dispersion_proxy_f64


def cast_array_to_float64(
    array: Float[Array, "..."],
    warn_if_downcast: bool = True
) -> Float[Array, "..."]:
    """
    Cast JAX/NumPy array to float64 with optional downcast warning.
    
    Useful for internal buffers or intermediate computations that may
    inadvertently use float32 (e.g., from external libraries).
    
    Args:
        array: Input array (any dtype)
        warn_if_downcast: Emit warning if casting from higher precision
    
    Returns:
        Array with dtype=float64
    
    Example:
        >>> import jax.numpy as jnp
        >>> from stochastic_predictor.api.validation import cast_array_to_float64
        >>> x_f32 = jnp.array([1.0, 2.0], dtype=jnp.float32)
        >>> x_f64 = cast_array_to_float64(x_f32, warn_if_downcast=True)
        >>> # Warning emitted: "Downcasting from float32 to float64..."
    
    References:
        - __init__.py: jax_enable_x64 global configuration
    """
    if array.dtype == jnp.float64:
        return array  # Already float64, no-op
    
    if warn_if_downcast and array.dtype in (jnp.float32, jnp.float16):
        warnings.warn(
            f"Casting array from {array.dtype} to float64. "
            "Ensure external data sources provide float64 to avoid this overhead.",
            RuntimeWarning
        )
    
    return jnp.asarray(array, dtype=jnp.float64)


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Observation Validation
    "validate_magnitude",
    "validate_timestamp",
    "check_staleness",
    # Array Validation
    "validate_shape",
    "validate_finite",
    "validate_simplex",
    # Range Validation
    "validate_holder_exponent",
    "validate_alpha_stable",
    "validate_beta_stable",
    # Sanitization
    "sanitize_array",
    "warn_if_invalid",
    # Precision Enforcement (float64)
    "ensure_float64",
    "sanitize_external_observation",
    "cast_array_to_float64",
]
