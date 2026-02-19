"""
Data Structures and Type Hints for the Universal Stochastic Predictor.

This module defines all immutable data structures used in the system,
ensuring strict dimensional typing via jaxtyping.

References:
    - Stochastic_Predictor_API_Python.tex §1: Data Structures (Typing)
    - Stochastic_Predictor_IO.tex §1: Configuration Vector
"""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from jaxtyping import Float, Array, Bool, PRNGKeyArray


# ═══════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER CONFIGURATION (Lambda)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PredictorConfig:
    """
    System Hyperparameter Vector Lambda (Complete Configuration).
    
    Design: Immutable (frozen=True) to guarantee thread-safety and 
    enable hashing (used in JAX JIT compilation cache).
    
    All parameters from config.toml are explicitly defined here to enforce
    zero hardcoded heuristics policy (Diamond Level Specification).
    
    References:
        - Stochastic_Predictor_API_Python.tex §1.1: Configuration (Lambda)
        - Stochastic_Predictor_IO.tex Table 1: Functional Parameters
        - config.toml: Project default values
    """
    # Snapshot Versioning (backward compatibility)
    schema_version: str = "1.0"
    
    # JKO Orchestrator (Optimal Transport)
    epsilon: float = 1e-3           # Entropic Regularization (Sinkhorn)
    learning_rate: float = 0.01     # Learning Rate (tau in JKO)
    sinkhorn_epsilon_min: float = 0.01  # Minimum epsilon for volatility coupling
    sinkhorn_epsilon_0: float = 0.1     # Base epsilon before coupling
    sinkhorn_alpha: float = 0.5         # Volatility coupling coefficient
    sinkhorn_max_iter: int = 200        # Max Sinkhorn iterations (scan length)
    
    # Entropy Monitoring (Mode Collapse Detection)
    entropy_window: int = 100       # Sliding window for entropy computation
    entropy_threshold: float = 0.8  # Minimum entropy threshold (deprecated, use entropy_gamma_*)
    entropy_gamma_min: float = 0.5  # Minimum gamma for crisis mode (lenient mode collapse detection)
    entropy_gamma_max: float = 1.0  # Maximum gamma for low-volatility mode (strict mode collapse detection)
    entropy_gamma_default: float = 0.8  # Default gamma for normal volatility regime
    
    # Kernel D (Log-Signatures)
    log_sig_depth: int = 3          # Truncation Depth (L)
    
    # Kernel A (WTMM + Fokker-Planck)
    wtmm_buffer_size: int = 128     # N_buf: Sliding memory
    besov_cone_c: float = 1.5       # Besov Cone of Influence
    kernel_ridge_lambda: float = 1e-6  # Ridge regularization parameter (Kernel A)
    
    # Kernel C (SDE Integration)
    stiffness_low: int = 100        # Threshold for explicit Euler-Maruyama
    stiffness_high: int = 1000      # Threshold for implicit trapezial
    sde_dt: float = 0.01            # Time step for SDE integration
    sde_numel_integrations: int = 100  # Number of integration steps
    sde_diffusion_sigma: float = 0.2    # Diffusion coefficient (Levy process volatility)
    
    # Circuit Breaker (Holder Singularity)
    holder_threshold: float = 0.4   # H_min: Critical threshold
    
    # CUSUM (Regime Change Detection)
    cusum_h: float = 5.0            # h: Drift threshold
    cusum_k: float = 0.5            # k: Slack (tolerance)
    grace_period_steps: int = 20    # Refractory period post-alarm
    residual_window_size: int = 252 # Rolling window size for kurtosis (annual)
    
    # Volatility Monitoring (EWMA)
    volatility_alpha: float = 0.1   # α: Exponential decay
    
    # Validation (Outlier Detection & Temporal Drift)
    sigma_bound: float = 20.0       # Black Swan threshold (N sigma)
    sigma_val: float = 1.0          # Reference standard deviation for outlier detection
    max_future_drift_ns: int = 1_000_000_000      # Max future drift: 1 second (clock skew tolerance)
    max_past_drift_ns: int = 86_400_000_000_000   # Max past drift: 24 hours (stale data threshold)
    
    # I/O Policies (Data Feed & Snapshots)
    data_feed_timeout: int = 30           # Timeout in seconds
    data_feed_max_retries: int = 3        # Maximum retry attempts
    snapshot_atomic_fsync: bool = True      # Force fsync for atomicity
    snapshot_compression: str = "none"      # Compression: "none", "gzip", "brotli"
    snapshot_format: str = "msgpack"        # Serialization: "msgpack" or "protobuf"
    snapshot_hash_algorithm: str = "sha256" # Hash: "sha256" or "crc32c"
    telemetry_hash_interval_steps: int = 1  # Emit parity hashes every N steps
    telemetry_buffer_capacity: int = 1024   # Max capacity of telemetry buffer (zero-heuristics injection)
    frozen_signal_min_steps: int = 5        # N_freeze: consecutive equal values
    frozen_signal_recovery_ratio: float = 0.1  # Ratio vs historical variance
    frozen_signal_recovery_steps: int = 2   # Consecutive recovery confirmations
    
    # Latency and Anti-Aliasing Policies
    staleness_ttl_ns: int = 500_000_000         # TTL: 500ms (degraded mode)
    besov_nyquist_interval_ns: int = 100_000_000 # Nyquist: 100ms (WTMM)
    inference_recovery_hysteresis: float = 0.8  # Recovery hysteresis factor
    
    # Kernel A Parameters (RKHS)
    kernel_a_bandwidth: float = 0.1             # Gaussian kernel bandwidth (smoothness)
    kernel_a_embedding_dim: int = 5             # Time-delay embedding dimension (Takens)
    kernel_a_min_variance: float = 1e-10        # Minimum variance clipping threshold (numerical stability)
    
    # Kernel B Parameters (DGM)
    dgm_width_size: int = 64                    # Hidden layer width for DGM network
    dgm_depth: int = 4                          # Number of hidden layers in DGM
    dgm_entropy_num_bins: int = 50              # Histogram bins for entropy monitoring
    dgm_activation: str = "tanh"                # Activation function: "tanh", "relu", "elu", "gelu"
    kernel_b_r: float = 0.05                    # Drift rate parameter (HJB Hamiltonian)
    kernel_b_sigma: float = 0.2                 # Dispersion coefficient (HJB diffusion term)
    kernel_b_horizon: float = 1.0               # Prediction horizon (HJB integration time)
    kernel_b_spatial_range_factor: float = 0.5  # Spatial sampling range factor (replaces hardcoded 0.5/1.5)
    
    # Kernel C Parameters (SDE)
    kernel_c_mu: float = 0.0                    # Drift (mean reversion rate)
    kernel_c_alpha: float = 1.8                 # Stability parameter (1 < alpha <= 2)
    kernel_c_beta: float = 0.0                  # Skewness parameter (-1 <= beta <= 1)
    kernel_c_horizon: float = 1.0               # Prediction horizon (integration time)
    kernel_c_dt0: float = 0.01                  # Initial time step (adaptive stepping)
    sde_initial_dt_factor: float = 10.0         # Safety factor for dt0 (dtmax / sde_initial_dt_factor)
    kernel_c_alpha_gaussian_threshold: float = 1.99  # Threshold for Gaussian regime detection (alpha > threshold)
    
    # Kernel D Parameters (Signatures)
    kernel_d_depth: int = 3                     # Log-signature truncation depth (L)
    kernel_d_alpha: float = 0.1                 # Signature extrapolation scaling factor
    kernel_d_confidence_base: float = 1.0       # Base factor for confidence calculation (base + sig_norm)
    
    # Base/Validation Parameters
    base_min_signal_length: int = 32            # Minimum required signal length
    signal_normalization_method: str = "zscore"  # Method: 'zscore' or 'minmax'
    numerical_epsilon: float = 1e-10            # Stability epsilon (divisions, logs, stiffness)
    warmup_signal_length: int = 100             # Representative signal length for JIT warm-up
    
    # Validation Constraints (Phase 5: Zero-Heuristics)
    validation_finite_allow_nan: bool = False         # Allow NaN in finite validation
    validation_finite_allow_inf: bool = False         # Allow Inf in finite validation
    validation_simplex_atol: float = 1e-6             # Tolerance for simplex constraint
    validation_holder_exponent_min: float = 0.0       # Min bound for Holder exponent
    validation_holder_exponent_max: float = 1.0       # Max bound for Holder exponent
    validation_alpha_stable_min: float = 0.0          # Min bound for alpha (stability)
    validation_alpha_stable_max: float = 2.0          # Max bound for alpha (stability)
    validation_alpha_stable_exclusive_bounds: bool = True  # Use strict inequalities for alpha
    validation_beta_stable_min: float = -1.0          # Min bound for beta (skewness)
    validation_beta_stable_max: float = 1.0           # Max bound for beta (skewness)
    sanitize_replace_nan_value: float = 0.0           # Replacement value for NaN in sanitization
    sanitize_replace_inf_value: Optional[float] = None  # Replacement value for Inf (None to preserve)
    sanitize_clip_range: Optional[tuple] = None       # Tuple (min, max) for clipping (None to skip)
    
    # Phase 6: SDE Integration Tolerances (Kernel C - Zero-Heuristics)
    sde_brownian_tree_tol: float = 1e-3               # Brownian tree tolerance for path generation
    sde_pid_rtol: float = 1e-3                        # Relative tolerance for PID controller
    sde_pid_atol: float = 1e-6                        # Absolute tolerance for PID controller
    sde_pid_dtmin: float = 1e-5                       # Minimum time step for PID controller
    sde_pid_dtmax: float = 0.1                        # Maximum time step for PID controller
    sde_solver_type: str = "heun"                     # SDE solver: "euler" or "heun"
    
    # Phase 6: Kernel B Hyperparameters (PDE/DGM - Zero-Heuristics)
    kernel_b_spatial_samples: int = 100               # Number of spatial sample points for entropy
    
    # Phase 6: Kernel D Hyperparameters (Signatures - Zero-Heuristics)
    kernel_d_confidence_scale: float = 0.1            # Scaling factor for signature confidence
    
    def __post_init__(self):
        """Validate mathematical invariants and configuration coherence."""
        # Simplex constraint implicit: learning_rate <= 1.0
        assert 0.0 < self.learning_rate <= 1.0, \
            f"learning_rate must be in (0, 1], got {self.learning_rate}"
        
        # Entropic regularization must be positive
        assert self.epsilon > 0, \
            f"epsilon must be > 0 (Sinkhorn), got {self.epsilon}"
        assert self.sinkhorn_epsilon_min > 0, \
            f"sinkhorn_epsilon_min must be > 0, got {self.sinkhorn_epsilon_min}"
        assert self.sinkhorn_epsilon_0 >= self.sinkhorn_epsilon_min, \
            f"sinkhorn_epsilon_0 must be >= epsilon_min, got {self.sinkhorn_epsilon_0}"
        assert 0.0 < self.sinkhorn_alpha <= 1.0, \
            f"sinkhorn_alpha must be in (0, 1], got {self.sinkhorn_alpha}"
        
        # Entropy monitoring constraints
        assert self.entropy_window > 0, \
            f"entropy_window must be > 0, got {self.entropy_window}"
        assert 0.0 < self.entropy_threshold <= 1.0, \
            f"entropy_threshold must be in (0, 1], got {self.entropy_threshold}"
        
        # Log-signature depth reasonable (exponential complexity)
        assert 1 <= self.log_sig_depth <= 5, \
            f"log_sig_depth must be in [1, 5], got {self.log_sig_depth}"
        
        # SDE integration parameters
        assert self.sde_dt > 0, \
            f"sde_dt must be > 0, got {self.sde_dt}"
        assert self.sde_numel_integrations > 0, \
            f"sde_numel_integrations must be > 0, got {self.sde_numel_integrations}"
        assert self.stiffness_low > 0 and self.stiffness_high > self.stiffness_low, \
            f"stiffness thresholds must satisfy 0 < low < high, got {self.stiffness_low}, {self.stiffness_high}"
        
        # Holder exponent bounds (stochastic processes)
        assert 0.0 < self.holder_threshold < 1.0, \
            f"holder_threshold must be in (0, 1), got {self.holder_threshold}"
        
        # CUSUM thresholds
        assert self.cusum_h > 0 and self.cusum_k >= 0, \
            "CUSUM thresholds must be non-negative"
        
        # Outlier detection & temporal drift
        assert self.sigma_bound > 0, \
            f"sigma_bound must be > 0, got {self.sigma_bound}"
        assert self.sigma_val > 0, \
            f"sigma_val must be > 0, got {self.sigma_val}"
        assert self.max_future_drift_ns > 0, \
            f"max_future_drift_ns must be > 0, got {self.max_future_drift_ns}"
        assert self.max_past_drift_ns > 0, \
            f"max_past_drift_ns must be > 0, got {self.max_past_drift_ns}"
        
        # I/O constraints
        assert self.data_feed_timeout > 0, \
            f"data_feed_timeout must be > 0, got {self.data_feed_timeout}"
        assert self.data_feed_max_retries >= 0, \
            f"data_feed_max_retries must be >= 0, got {self.data_feed_max_retries}"
        assert self.snapshot_compression in {"none", "gzip", "brotli"}, \
            f"snapshot_compression must be 'none', 'gzip', or 'brotli', got '{self.snapshot_compression}'"
        assert self.snapshot_format in {"msgpack", "protobuf"}, \
            f"snapshot_format must be 'msgpack' or 'protobuf', got '{self.snapshot_format}'"
        assert self.snapshot_hash_algorithm in {"sha256", "crc32c"}, \
            f"snapshot_hash_algorithm must be 'sha256' or 'crc32c', got '{self.snapshot_hash_algorithm}'"
        assert self.telemetry_hash_interval_steps > 0, \
            f"telemetry_hash_interval_steps must be > 0, got {self.telemetry_hash_interval_steps}"
        assert self.frozen_signal_min_steps >= 5, \
            f"frozen_signal_min_steps must be >= 5, got {self.frozen_signal_min_steps}"
        assert self.frozen_signal_recovery_ratio > 0, \
            f"frozen_signal_recovery_ratio must be > 0, got {self.frozen_signal_recovery_ratio}"
        assert self.frozen_signal_recovery_steps > 0, \
            f"frozen_signal_recovery_steps must be > 0, got {self.frozen_signal_recovery_steps}"
        
        # TTL/Nyquist coherence
        assert self.staleness_ttl_ns > 0 and self.besov_nyquist_interval_ns > 0, \
            "Latency timeouts must be positive"


# ═══════════════════════════════════════════════════════════════════════════
# INPUT/OUTPUT STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProcessState:
    """
    Predictor operational input (y_t, y_reference, tau).
    
    Design: Scalar fields (shape [1]) for compatibility with vmap
    in multi-asset architecture (vectorized batching).
    
    Domain-Agnostic: Applies to any stochastic process without semantic assumptions.
    
    References:
        - API_Python.tex §1.2: Operational Input
        - IO.tex §2: Observation Protocol
    """
    magnitude: Float[Array, "1"]    # y_t: Normalized or absolute magnitude
    reference: Float[Array, "1"]    # y_reference: Reference magnitude for comparison
    timestamp_ns: int               # Unix Epoch (nanoseconds)
    
    def validate_domain(
        self, 
        sigma_bound: float, 
        sigma_val: float
    ) -> bool:
        """
        Catastrophic Outlier Detection (> N sigma).
        
        Zero-Heuristics Policy: All parameters MUST be passed from PredictorConfig.
        No default values to enforce configuration-driven operation.
        
        Args:
            sigma_bound: Maximum number of standard deviations (from config.sigma_bound)
            sigma_val: Reference standard deviation (from config.sigma_val)
            
        Returns:
            bool: True if observation is within valid domain
            
        References:
            - API_Python.tex §1.2: validate_domain()
            
        Example:
            >>> config = PredictorConfigInjector().create_config()
            >>> obs = ProcessState(...)
            >>> is_valid = obs.validate_domain(
            ...     sigma_bound=config.sigma_bound,
            ...     sigma_val=config.sigma_val
            ... )
        """
        return bool(jnp.abs(self.magnitude) <= (sigma_bound * sigma_val))


@dataclass(frozen=True)
class PredictionResult:
    """
    System output (prediction + telemetry + control flags).
    
    Design: Scalar and vector fields following S_risk (risk vector)
    specification defined in IO.tex.
    
    References:
        - API_Python.tex §1.3: System Output
        - IO.tex §3: Risk State Vector (S_risk)
    """
    # Main Prediction
    predicted_next: Float[Array, "1"]       # y_{t+1} (Z-Score space)
    
    # State Telemetry (Basic S_risk)
    holder_exponent: Float[Array, "1"]      # H_t: Holder exponent
    cusum_drift: Float[Array, "1"]          # G^+: CUSUM statistic
    distance_to_collapse: Float[Array, "1"] # h - G^+: Safety margin
    free_energy: Float[Array, "1"]          # F: JKO energy
    
    # Advanced Telemetry (Extensions)
    kurtosis: Float[Array, "1"]             # κ_t: Empirical kurtosis
    dgm_entropy: Float[Array, "1"]          # H_DGM: Kernel B entropy
    adaptive_threshold: Float[Array, "1"]   # h_t: Adaptive CUSUM threshold
    
    # Orchestrator State (Weight Simplex)
    weights: Float[Array, "4"]              # [ρ_A, ρ_B, ρ_C, ρ_D]
    
    # Health and Control Flags (boolean)
    sinkhorn_converged: Bool[Array, "1"]    # JKO convergence
    degraded_inference_mode: bool           # TTL violation (freezing)
    emergency_mode: bool                    # H_t < H_min (singularity)
    regime_change_detected: bool            # CUSUM alarm (G+ > h_t)
    mode_collapse_warning: bool             # H_DGM < γ·H[g] (DGM collapse)
    
    # Consolidated Operating Mode
    mode: str  # "Standard" | "Robust" | "Emergency"
    
    def __post_init__(self):
        """
        Validate output (simplex constraint and flag coherence).
        
        Zero-Heuristics Compliance:
        Simplex validation uses config.validation_simplex_atol.
        Call validate_simplex() externally with config tolerance if needed.
        Basic validations (non-negativity, range checks) remain here.
        """
        # Weights non-negative
        assert jnp.all(self.weights >= 0.0), \
            "weights must be non-negative"
        
        # Holder exponent in valid range
        holder_val = float(self.holder_exponent)
        assert 0.0 <= holder_val <= 1.0, \
            f"holder_exponent must be in [0, 1], got {holder_val}"

        # Valid mode string
        valid_modes = {"Standard", "Robust", "Emergency"}
        assert self.mode in valid_modes, \
            f"mode must be one of {valid_modes}, got '{self.mode}'"
    
    @staticmethod
    def validate_simplex(weights: Float[Array, "4"], atol: float) -> None:
        """
        Validate simplex constraint with configurable tolerance.
        
        Args:
            weights: Weight array [ρ_A, ρ_B, ρ_C, ρ_D]
            atol: Absolute tolerance (use config.validation_simplex_atol)
        
        Raises:
            AssertionError: If weights don't sum to 1.0 within tolerance
        
        Example:
            >>> PredictionResult.validate_simplex(weights, config.validation_simplex_atol)
        """
        weights_sum = float(jnp.sum(weights))
        assert jnp.allclose(weights_sum, 1.0, atol=atol), \
            f"weights must form a simplex (sum=1.0 ± {atol}), got sum={weights_sum:.6f}"
        


# ═══════════════════════════════════════════════════════════════════════════
# AUXILIARY TYPES (Internal State Structures)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class KernelOutput:
    """
    Standardized output from a prediction kernel.
    
    Design: Common interface for all 4 kernels (A, B, C, D), allowing
    the JKO orchestrator to fuse them uniformly.
    
    References:
        - Python.tex §2: Module 2: Prediction Kernels
        - Implementacion.tex §2.1: Kernel Interface
    """
    prediction: Float[Array, "1"]       # Prediction y_{t+1}
    confidence: Float[Array, "1"]       # Confidence score [0, 1]
    entropy: Float[Array, "1"]          # Predictor entropy
    metadata: Optional[dict] = None     # Additional kernel-specific data


@dataclass
class InternalState:
    """
    Internal predictor state (JAX buffers resident in GPU/memory).
    
    Design: NOT frozen (mutable) because jnp.array fields are updated
    functionally via lax.dynamic_update_slice (Zero-Copy).
    
    Note: This structure is internal and NOT exposed in public API.
    
    References:
        - API_Python.tex §2: Predictor Architecture
        - Python.tex §6: State Management
    """
    # History Buffers (rolling windows)
    signal_history: Float[Array, "N"]       # Last N signal magnitudes
    residual_buffer: Float[Array, "N"]      # Last N prediction errors
    residual_window: Float[Array, "W"]      # Rolling window for kurtosis (252 steps)
    
    # Orchestrator Weights (simplex)
    rho: Float[Array, "4"]                  # [ρ_A, ρ_B, ρ_C, ρ_D]
    
    # CUSUM Statistics
    cusum_g_plus: Float[Array, "1"]         # G^+: Positive drift accumulated
    cusum_g_minus: Float[Array, "1"]        # G^-: Negative drift accumulated
    grace_counter: int                      # Refractory period counter
    adaptive_h_t: Float[Array, "1"]         # h_t: Kurtosis-adaptive CUSUM threshold
    
    # EWMA Volatility
    ema_variance: Float[Array, "1"]         # σ²_t: Exponential variance
    
    # Accumulated Telemetry
    kurtosis: Float[Array, "1"]             # κ_t: Empirical kurtosis
    holder_exponent: Float[Array, "1"]      # H_t: WTMM Holder
    dgm_entropy: Float[Array, "1"]          # H_DGM: Kernel B entropy
    mode_collapse_consecutive_steps: int    # V-MAJ-5: Counter for consecutive low-entropy steps
    
    # State Flags
    degraded_mode: bool                     # Degraded mode active
    emergency_mode: bool                    # Emergency mode active
    regime_changed: bool                    # Regime change detected
    
    # Timestamp Control
    last_update_ns: int                     # Last processed timestamp
    
    # PRNG State (threefry2x32)
    rng_key: PRNGKeyArray                   # JAX key for reproducibility


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS AND ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════

class OperatingMode:
    """
    Predictor operating modes.
    
    References:
        - API_Python.tex §3: Operating Modes
    """
    STANDARD = "Standard"       # Normal operation (all kernels active)
    ROBUST = "Robust"           # Degraded mode (weight freezing)
    EMERGENCY = "Emergency"     # Circuit breaker (H < H_min)


class KernelType:
    """
    Kernel identifiers.
    
    References:
        - Teoria.tex §2: Four Prediction Branches
        - Python.tex §2: Prediction Kernels
    """
    KERNEL_A = 0    # WTMM + Fokker-Planck (Lévy with drift)
    KERNEL_B = 1    # Deep Galerkin Method (HJB + DGM)
    KERNEL_C = 2    # Pure Monte Carlo (Stable Lévy CMS)
    KERNEL_D = 3    # Log-Signatures (Rough Paths)
    
    N_KERNELS = 4   # Total kernels in system


# ═══════════════════════════════════════════════════════════════════════════
# IMPORT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def check_jax_config() -> dict[str, bool]:
    """
    Verify that JAX is configured correctly per Golden Master.
    
    Returns:
        dict: Critical configuration states
        
    References:
        - Python.tex §1: JAX/XLA Stack
        - API_Python.tex §5: Floating-Point Determinism
    """
    import jax
    import os
    
    try:
        x64_enabled = bool(jax.config.read("jax_enable_x64"))
    except:
        x64_enabled = False
    
    try:
        cache_dir = jax.config.read("jax_compilation_cache_dir")
        has_cache = cache_dir is not None and cache_dir != ""
    except:
        has_cache = False
    
    return {
        "x64_enabled": x64_enabled,
        "deterministic_reductions": os.getenv("JAX_DETERMINISTIC_REDUCTIONS") == "1",
        "prng_threefry": os.getenv("JAX_DEFAULT_PRNG_IMPL") == "threefry2x32",
        "compilation_cache": has_cache,
    }


# Public exports
__all__ = [
    # Configuration
    "PredictorConfig",
    # Input/Output
    "ProcessState",
    "PredictionResult",
    # Internal Structures
    "KernelOutput",
    "InternalState",
    # Enumerations
    "OperatingMode",
    "KernelType",
    # Utilities
    "check_jax_config",
]
