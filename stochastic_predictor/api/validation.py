"""
Validation Functions for Predictor Input/Output.

This module provides domain validation, data sanitization, and anomaly detection
to ensure the integrity of the predictive pipeline.

References:
    - Predictor_Estocastico_API_Python.tex §1.2: validate_domain()
    - Predictor_Estocastico_IO.tex §2: Observation Protocol
    - Predictor_Estocastico_Implementacion.tex §3: Quality Control
"""

from typing import Union, Tuple
import jax.numpy as jnp
from jaxtyping import Float, Array
import warnings


# ═══════════════════════════════════════════════════════════════════════════
# MARKET OBSERVATION VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_price(
    price: Union[float, Float[Array, "1"]],
    sigma_bound: float,
    sigma_val: float,
    allow_nan: bool = False
) -> Tuple[bool, str]:
    """
    Validate that a price is within statistically reasonable bounds.
    
    Design: Detection of catastrophic outliers (> N sigma) that could
    indicate data feed errors or flash crashes.
    
    Zero-Heuristics Policy: All parameters MUST be passed from PredictorConfig.
    No default values to enforce configuration-driven operation.
    
    Args:
        price: Price to validate (scalar or JAX array)
        sigma_bound: Maximum number of standard deviations allowed (from config.sigma_bound)
        sigma_val: Reference standard deviation (from config.sigma_val)
        allow_nan: If True, permits NaN values (for missing data)
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    References:
        - API_Python.tex §1.2: MarketObservation.validate_domain()
        
    Example:
        >>> from stochastic_predictor.api.validation import validate_price
        >>> from stochastic_predictor.api.config import PredictorConfigInjector
        >>> config = PredictorConfigInjector().create_config()
        >>> is_valid, msg = validate_price(
        ...     price=jnp.array([100.5]),
        ...     sigma_bound=config.sigma_bound,
        ...     sigma_val=config.sigma_val
        ... )
        >>> assert is_valid
    """
    # Convert to JAX array if necessary
    price_arr = jnp.asarray(price)
    
    # Check NaN
    if jnp.any(jnp.isnan(price_arr)):
        if not allow_nan:
            return False, "Price contains NaN values"
    
    # Check infinities
    if jnp.any(jnp.isinf(price_arr)):
        return False, "Price contains infinite values"
    
    # Check bounds (outlier detection)
    threshold = sigma_bound * sigma_val
    if jnp.any(jnp.abs(price_arr) > threshold):
        max_val = float(jnp.max(jnp.abs(price_arr)))
        return False, (
            f"Price outlier detected: |price|={max_val:.2f} exceeds "
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
        - API_Python.tex §3.1: Staleness TTL
        - IO.tex §2.2: Timestamp Validation
        
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
    ttl_ns: int = 500_000_000  # 500ms default (from config.toml)
) -> Tuple[bool, int]:
    """
    Verify if an observation is stale according to TTL.
    
    Args:
        timestamp_ns: Observation timestamp
        ttl_ns: Time-To-Live in nanoseconds
        
    Returns:
        Tuple (is_stale: bool, age_ns: int)
        
    References:
        - API_Python.tex §3.1: Staleness Policy (Staleness TTL)
        - types.py: PredictorConfig.staleness_ttl_ns
        
    Example:
        >>> import time
        >>> from stochastic_predictor.api.validation import check_staleness
        >>> ts = time.time_ns() - 1_000_000_000  # 1 second ago
        >>> is_stale, age = check_staleness(ts, ttl_ns=500_000_000)
        >>> assert is_stale  # Older than 500ms TTL
    """
    import time
    
    current_time_ns = time.time_ns()
    age_ns = current_time_ns - timestamp_ns
    is_stale = age_ns > ttl_ns
    
    return bool(is_stale), int(age_ns)


# ═══════════════════════════════════════════════════════════════════════════
# ARRAY AND SHAPE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_shape(
    array: jnp.ndarray,
    expected_shape: Tuple[int, ...],
    name: str = "array"
) -> Tuple[bool, str]:
    """
    Validate that an array has the expected shape.
    
    Args:
        array: JAX array to validate
        expected_shape: Expected shape (can contain -1 for variable dimensions)
        name: Array name (for error messages)
        
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
    name: str = "array",
    allow_nan: bool = False,
    allow_inf: bool = False
) -> Tuple[bool, str]:
    """
    Validate that an array contains only finite values.
    
    Args:
        array: JAX array to validate
        name: Array name (for error messages)
        allow_nan: If True, permits NaN values
        allow_inf: If True, permits infinite values
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    References:
        - Tests_Python.tex §2.1: test_levy_cms_basic_properties
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
    atol: float = 1e-6,
    name: str = "weights"
) -> Tuple[bool, str]:
    """
    Validate that an array forms a simplex (sum = 1.0, all >= 0).
    
    Design: Critical for verifying JKO orchestrator weights [ρ_A, ρ_B, ρ_C, ρ_D].
    
    Args:
        weights: Weight array to validate
        atol: Absolute tolerance for sum
        name: Array name (for error messages)
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    References:
        - API_Python.tex §3.2: Simplex Constraint
        - types.py: PredictionResult.__post_init__()
        
    Example:
        >>> import jax.numpy as jnp
        >>> from stochastic_predictor.api.validation import validate_simplex
        >>> weights = jnp.array([0.25, 0.25, 0.25, 0.25])
        >>> is_valid, msg = validate_simplex(weights)
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
    min_val: float = 0.0,
    max_val: float = 1.0
) -> Tuple[bool, str]:
    """
    Validate that a Holder exponent is in the range [0, 1].
    
    Args:
        H: Holder exponent to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    References:
        - Teoria.tex §2.1: Properties of Lévy Processes
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
    min_val: float = 0.0,
    max_val: float = 2.0,
    exclusive_bounds: bool = True
) -> Tuple[bool, str]:
    """
    Validate alpha parameter of Lévy stable distribution.
    
    Args:
        alpha: Stability parameter
        min_val: Minimum value (default: 0.0)
        max_val: Maximum value (default: 2.0)
        exclusive_bounds: If True, use (min, max] instead of [min, max]
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    References:
        - Teoria.tex §2.2: Alpha-Stable Distributions
        - Tests_Python.tex §2.3: Property-Based Testing (alpha bounds)
        
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
    min_val: float = -1.0,
    max_val: float = 1.0
) -> Tuple[bool, str]:
    """
    Validate beta (skewness) parameter of stable distribution.
    
    Args:
        beta: Skewness parameter
        min_val: Minimum value (default: -1.0)
        max_val: Maximum value (default: 1.0)
        
    Returns:
        Tuple (is_valid: bool, error_message: str)
        
    References:
        - Teoria.tex §2.2: Lévy Parametrization
        - Tests_Python.tex §2.3: Hypothesis strategies (beta bounds)
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
    replace_nan: float = 0.0,
    replace_inf: Union[float, None] = None,
    clip_range: Union[Tuple[float, float], None] = None
) -> jnp.ndarray:
    """
    Sanitize an array by replacing NaN/Inf and applying clipping.
    
    Args:
        array: JAX array to sanitize
        replace_nan: Value to replace NaN (None to preserve)
        replace_inf: Value to replace Inf (None to preserve)
        clip_range: Tuple (min, max) for clipping (None to skip)
        
    Returns:
        Sanitized JAX array
        
    Note: Use with caution - may mask underlying problems
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
    exception_type: Union[type, None] = None
) -> None:
    """
    Emit a warning or exception if validation fails.
    
    Args:
        is_valid: Validation result
        message: Error message
        exception_type: If specified, raises exception instead of warning
        
    Example:
        >>> from stochastic_predictor.api.validation import validate_price, warn_if_invalid
        >>> is_valid, msg = validate_price(jnp.array([1000.0]), sigma_bound=5.0)
        >>> warn_if_invalid(is_valid, msg, exception_type=ValueError)
    """
    if not is_valid:
        if exception_type is not None:
            raise exception_type(message)
        else:
            warnings.warn(message, RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Observation Validation
    "validate_price",
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
]
