"""
Pydantic schemas for serialization, validation, and API contracts.

This module defines the data transfer objects (DTOs) used across the USP system.
Schemas enforce invariants at module boundaries and enable external validation.

References:
    - Predictor_Estocastico_API_Python.tex §1
    - Predictor_Estocastico_Implementacion.tex §3.1
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import jax.numpy as jnp
from jaxtyping import Float, Array
from pydantic import BaseModel, Field, validator


class OperatingMode(str, Enum):
    """
    Operating mode enumeration for the prediction system.
    
    Modes:
        INFERENCE: Real-time prediction without adaptation
        CALIBRATION: Gather statistics for hardware/environment characterization
        DIAGNOSTIC: Enable all telemetry (used during development)
    """
    INFERENCE = "inference"
    CALIBRATION = "calibration"
    DIAGNOSTIC = "diagnostic"


class MarketObservationSchema(BaseModel):
    """
    Market observation data transfer object.
    
    Schema for timestamped price observations from external market feeds.
    Includes temporal and domain validation.
    
    Attributes:
        price: Market price at observation time (positive, shape [1])
        timestamp_utc: UTC timestamp (ISO 8601)
        regime_tag: Optional market regime identifier (e.g., "bullish", "choppy")
        volatility_proxy: Optional realized volatility estimate for kernel tuning
    """
    price: Float[Array, "1"]
    timestamp_utc: datetime = Field(description="Observation time (UTC)")
    regime_tag: Optional[str] = Field(default=None, description="Market regime label")
    volatility_proxy: Optional[Float[Array, "1"]] = Field(
        default=None,
        description="Realized volatility estimate for Sinkhorn coupling"
    )
    
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True
    
    @validator("price")
    def validate_price_positive(cls, value):
        """Ensure price is strictly positive."""
        if jnp.any(value <= 0):
            raise ValueError(f"Price must be positive, got {value}")
        return value
    
    @validator("volatility_proxy")
    def validate_volatility_positive(cls, value):
        """Ensure volatility proxy (if provided) is positive."""
        if value is not None and jnp.any(value <= 0):
            raise ValueError(f"Volatility must be positive, got {value}")
        return value


class KernelOutputSchema(BaseModel):
    """
    Output contract for kernel computations.
    
    Each kernel (A, B, C, D) produces a normalized probability density estimate
    over the target variable space.
    
    Attributes:
        probability_density: Estimated probability density (summable to ~1.0)
        kernel_id: Which kernel produced this output ("A", "B", "C", or "D")
        computation_time_us: Execution time in microseconds
        numerics_flags: Diagnostic flags (NaN, Inf, stiffness warnings)
    """
    probability_density: Float[ArrayLike, "n_targets"]
    kernel_id: str = Field(description="Kernel identifier (A|B|C|D)")
    computation_time_us: float = Field(gt=0, description="Execution time in microseconds")
    numerics_flags: Dict[str, bool] = Field(
        default_factory=dict,
        description="Diagnostic flags: has_nan, has_inf, stiffness_warning, etc."
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator("kernel_id")
    def validate_kernel_id(cls, value):
        """Ensure kernel_id is a valid kernel."""
        if value not in ("A", "B", "C", "D"):
            raise ValueError(f"Invalid kernel_id: {value}")
        return value


class TelemetryDataSchema(BaseModel):
    """
    Internal telemetry snapshot for diagnostics and monitoring.
    
    Captured at each inference step for debugging and performance profiling.
    
    Attributes:
        step_index: Current prediction step index
        jax_device: JAX device being used (cpu, gpu:0, tpu:0, etc.)
        cusum_statistic: Current CUSUM test statistic (for anomaly detection)
        entropy_estimate: Shannon entropy from market regime
        sinkhorn_epsilon: Current regularization level (time-dependent)
        kernel_outputs: All kernel outputs for this step
        timestamp_utc: When telemetry was captured
    """
    step_index: int = Field(ge=0, description="Prediction step counter")
    jax_device: str = Field(description="JAX device identifier")
    cusum_statistic: Optional[float] = Field(default=None)
    entropy_estimate: Optional[float] = Field(default=None)
    sinkhorn_epsilon: Optional[float] = Field(default=None)
    kernel_outputs: Dict[str, KernelOutputSchema] = Field(default_factory=dict)
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True


class PredictionResultSchema(BaseModel):
    """
    Final prediction output contract.
    
    This is the system's external API response. Contains the fused prediction,
    confidence metrics, and optional telemetry.
    
    Attributes:
        target_prediction: Point estimate (mean or mode) of target variable
        confidence_lower: Lower confidence bound (e.g., 2.5th percentile)
        confidence_upper: Upper confidence bound (e.g., 97.5th percentile)
        operating_mode: Mode under which prediction was generated
        telemetry: Diagnostic telemetry (if requested)
        request_id: Unique request identifier for tracing
    """
    target_prediction: Float[ArrayLike, ""]
    confidence_lower: Float[ArrayLike, ""]
    confidence_upper: Float[ArrayLike, ""]
    operating_mode: OperatingMode
    telemetry: Optional[TelemetryDataSchema] = Field(default=None)
    request_id: Optional[str] = Field(default=None, description="Request trace ID")
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator("confidence_lower", always=True)
    def validate_bounds(cls, value, values):
        """Ensure confidence_lower <= target_prediction <= confidence_upper."""
        if "target_prediction" in values:
            target = values["target_prediction"]
            if jnp.any(value > target):
                raise ValueError(f"confidence_lower must be <= target_prediction")
        return value


class HealthCheckResponseSchema(BaseModel):
    """
    Health check response for system monitoring.
    
    Attributes:
        status: "healthy" or "degraded" or "unhealthy"
        version: Implementation version tag
        jax_config: JAX configuration snapshot
        uptime_seconds: System uptime in seconds
        last_inference_timestamp: When last prediction was made
    """
    status: str = Field(description="Health status")
    version: str = Field(description="Implementation version")
    jax_config: Dict[str, Any] = Field(description="JAX configuration snapshot")
    uptime_seconds: float = Field(ge=0)
    last_inference_timestamp: Optional[datetime] = Field(default=None)


__all__ = [
    "OperatingMode",
    "MarketObservationSchema",
    "KernelOutputSchema",
    "TelemetryDataSchema",
    "PredictionResultSchema",
    "HealthCheckResponseSchema",
]
