"""
Data Structures and Type Hints for the Universal Stochastic Predictor.

This module defines all immutable data structures used in the system,
ensuring strict dimensional typing via jaxtyping.

References:
    - Stochastic_Predictor_API_Python.tex §1: Data Structures (Typing)
    - Stochastic_Predictor_IO.tex §1: Configuration Vector
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union
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

    # PRNG Defaults
    prng_seed: int = 42
    prng_split_count: int = 4
    
    # JKO Orchestrator (Optimal Transport)
    epsilon: float = 1e-3           # Entropic Regularization (Sinkhorn)
    learning_rate: float = 0.01     # Learning Rate (tau in JKO)
    jko_domain_length: float = 1.0  # Domain length for JKO scaling
    entropy_window_relaxation_factor: float = 5.0  # Relaxation multiplier for entropy window
    entropy_window_bounds_min: int = 10  # Minimum entropy window
    entropy_window_bounds_max: int = 500  # Maximum entropy window
    learning_rate_safety_factor: float = 0.8  # Safety factor for learning rate stability
    learning_rate_minimum: float = 1e-6  # Minimum learning rate
    sinkhorn_epsilon_min: float = 0.01  # Minimum epsilon for volatility coupling
    sinkhorn_epsilon_0: float = 0.1     # Base epsilon before coupling
    sinkhorn_alpha: float = 0.5         # Volatility coupling coefficient
    sinkhorn_max_iter: int = 200        # Max Sinkhorn iterations (scan length)
    sinkhorn_inner_iterations: int = 10 # Inner iterations for log-domain stability
    sinkhorn_cost_type: str = "squared"  # Cost type: "squared" or "huber"
    sinkhorn_huber_delta: float = 1.0    # Huber delta for robust cost
    
    # Entropy Monitoring (Mode Collapse Detection)
    entropy_window: int = 100       # Sliding window for entropy computation
    entropy_threshold: float = 0.8  # Minimum entropy threshold (deprecated, use entropy_gamma_*)
    entropy_gamma_min: float = 0.5  # Minimum gamma for crisis mode (lenient mode collapse detection)
    entropy_gamma_max: float = 1.0  # Maximum gamma for low-volatility mode (strict mode collapse detection)
    entropy_gamma_default: float = 0.8  # Default gamma for normal volatility regime
    entropy_volatility_low_threshold: float = 0.05   # Low-volatility sigma threshold
    entropy_volatility_high_threshold: float = 0.2   # High-volatility sigma threshold
    entropy_ratio_min: float = 0.1                  # Entropy ratio lower bound
    entropy_ratio_max: float = 10.0                 # Entropy ratio upper bound
    entropy_baseline_floor: float = 1e-6            # Baseline entropy floor
    entropy_scaling_trigger: float = 2.0            # Entropy ratio trigger for scaling
    mode_collapse_min_threshold: int = 10  # Minimum number of consecutive steps before mode collapse warning
    mode_collapse_window_ratio: float = 0.1  # Ratio of entropy_window for mode collapse warning threshold
    
    # Kernel D (Log-Signatures)
    log_sig_depth: int = 3          # Truncation Depth (L)
    kernel_d_load_shedding_depths: tuple[int, ...] = (2, 3, 5)
    
    # Kernel A (WTMM + Fokker-Planck)
    wtmm_buffer_size: int = 128     # N_buf: Sliding memory
    besov_cone_c: float = 1.5       # Besov Cone of Influence
    kernel_ridge_lambda: float = 1e-6  # Ridge regularization parameter (Kernel A)
    wtmm_num_scales: int = 16          # Number of WTMM scales
    wtmm_scale_min: float = 1.0        # Minimum WTMM scale
    wtmm_sigma: float = 1.0            # Morlet wavelet sigma
    wtmm_fc: float = 0.5               # Morlet wavelet central frequency
    wtmm_modulus_threshold: float = 0.01  # Modulus maxima threshold
    wtmm_max_link_distance: float = 2.0   # Max link distance for maxima chains
    wtmm_q_min: float = -2.0           # Min q for partition function
    wtmm_q_max: float = 2.0            # Max q for partition function
    wtmm_q_steps: int = 9              # Number of q values
    wtmm_h_min: float = 0.0            # Min Holder exponent grid
    wtmm_h_max: float = 1.5            # Max Holder exponent grid
    wtmm_h_steps: int = 151            # Holder exponent grid size
    wtmm_tau_default_scale: float = 0.5  # Default tau scale for non-finite q
    
    # Kernel C (SDE Integration)
    stiffness_low: int = 100        # Threshold for explicit Euler-Maruyama
    stiffness_high: int = 1000      # Threshold for implicit trapezial
    sde_dt: float = 0.01            # Time step for SDE integration
    sde_numel_integrations: int = 100  # Number of integration steps
    sde_diffusion_sigma: float = 0.2    # Diffusion coefficient (Levy process volatility)
    sde_fd_epsilon: float = 1e-6        # Finite-difference epsilon for stiffness Jacobian
    
    # Circuit Breaker (Holder Singularity)
    holder_threshold: float = 0.4   # H_min: Critical threshold
    robustness_dimension_threshold: float = 1.5  # Fractal dimension threshold
    robustness_force_kernel_d: bool = True  # Force kernel D on robustness trigger
    dgm_entropy_coupling_beta: float = 0.7  # DGM entropy-topology coupling beta
    dgm_max_capacity_factor: float = 4.0   # Max width*depth multiplier
    stiffness_calibration_c1: float = 25.0 # Calibration constant for stiffness low
    stiffness_calibration_c2: float = 250.0 # Calibration constant for stiffness high
    stiffness_min_low: float = 100.0       # Minimum low threshold
    stiffness_min_high: float = 1000.0     # Minimum high threshold
    holder_exponent_guard: float = 1e-3    # Guard against alpha -> 1
    
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
    telemetry_hash_algorithm: str = "sha256"  # Telemetry parity hash algorithm
    telemetry_adaptive_log_path: str = "io/adaptive_telemetry.jsonl"  # Adaptive telemetry log path
    telemetry_adaptive_window_size: int = 100  # Adaptive telemetry window size
    telemetry_placeholder_max_stiffness_metric: float = 0.0  # Placeholder max stiffness metric
    telemetry_placeholder_num_internal_iterations_mean: float = 0.0  # Placeholder mean iterations
    telemetry_placeholder_implicit_residual_norm_max: float = 0.0  # Placeholder residual norm
    telemetry_dashboard_title: str = "USP Telemetry Dashboard"  # Dashboard title
    telemetry_dashboard_width: int = 720  # Dashboard chart width
    telemetry_dashboard_height: int = 180  # Dashboard chart height
    telemetry_dashboard_spread_epsilon: float = 1e-9  # Chart spread epsilon
    telemetry_dashboard_recent_rows: int = 25  # Dashboard recent rows
    frozen_signal_variance_floor: float = 1e-12  # Minimum variance floor for recovery checks
    frozen_signal_min_steps: int = 5        # N_freeze: consecutive equal values
    frozen_signal_recovery_ratio: float = 0.1  # Ratio vs historical variance
    frozen_signal_recovery_steps: int = 2   # Consecutive recovery confirmations
    degraded_recovery_min_steps: int = 2    # Minimum steps before exiting degraded mode
    
    # Latency and Anti-Aliasing Policies
    staleness_ttl_ns: int = 500_000_000         # TTL: 500ms (degraded mode)
    besov_nyquist_interval_ns: int = 100_000_000 # Nyquist: 100ms (WTMM)
    inference_recovery_hysteresis: float = 0.8  # Recovery hysteresis factor
    
    # Kernel A Parameters (RKHS)
    kernel_a_bandwidth: float = 0.1             # Gaussian kernel bandwidth (smoothness)
    kernel_a_embedding_dim: int = 5             # Time-delay embedding dimension (Takens)
    kernel_a_min_wiener_hopf_order: int = 2      # Minimum Wiener-Hopf order
    kernel_a_min_variance: float = 1e-10        # Minimum variance clipping threshold (numerical stability)
    koopman_top_k: int = 5                      # Top-K Koopman spectral modes
    koopman_min_power: float = 1e-10            # Minimum spectral power for Koopman modes
    paley_wiener_integral_max: float = 100.0    # Max Paley-Wiener integral threshold
    
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
    kernel_c_jump_intensity: float = 0.05       # Levy jump intensity (events per unit time)
    kernel_c_jump_mean: float = 0.0             # Mean jump size
    kernel_c_jump_scale: float = 0.1            # Jump size scale (std dev)
    kernel_c_jump_max_events: int = 16          # Max jump events per step (static shape)
    
    # Kernel D Parameters (Signatures)
    kernel_d_depth: int = 3                     # Log-signature truncation depth (L)
    kernel_d_alpha: float = 0.1                 # Signature extrapolation scaling factor
    kernel_d_confidence_base: float = 1.0       # Base factor for confidence calculation (base + sig_norm)
    
    # Base/Validation Parameters
    base_min_signal_length: int = 32            # Minimum required signal length
    signal_sampling_interval: float = 1.0       # Sampling interval for FFT-based diagnostics
    signal_normalization_method: str = "zscore"  # Method: 'zscore' or 'minmax'
    numerical_epsilon: float = 1e-10            # Stability epsilon (divisions, logs, stiffness)
    warmup_signal_length: int = 100             # Representative signal length for JIT warm-up
    pdf_grid_min_z: float = -4.0                 # PDF grid lower bound (z-score)
    pdf_grid_max_z: float = 4.0                  # PDF grid upper bound (z-score)
    pdf_grid_num_points: int = 256               # PDF grid resolution
    pdf_min_sigma: float = 1e-6                  # Minimum sigma for PDF construction
    confidence_interval_z: float = 1.96          # Z-score for confidence bounds
    kernel_output_time_us: float = 1.0           # Placeholder runtime in microseconds
    kurtosis_min: float = 1.0                    # Minimum kurtosis clamp
    kurtosis_max: float = 100.0                  # Maximum kurtosis clamp
    kurtosis_reference: float = 3.0              # Reference kurtosis for adaptive CUSUM
    
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
    validation_viscosity_residual_max: float = 1e-3    # Max PDE residual for viscosity check
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
        if isinstance(self.kernel_d_load_shedding_depths, list):
            object.__setattr__(
                self,
                "kernel_d_load_shedding_depths",
                tuple(self.kernel_d_load_shedding_depths),
            )
        # Simplex constraint implicit: learning_rate <= 1.0
        assert 0.0 < self.learning_rate <= 1.0, \
            f"learning_rate must be in (0, 1], got {self.learning_rate}"
        assert self.jko_domain_length > 0.0, \
            f"jko_domain_length must be > 0, got {self.jko_domain_length}"
        assert self.entropy_window_relaxation_factor > 0.0, \
            "entropy_window_relaxation_factor must be > 0"
        assert self.entropy_window_bounds_min > 0, \
            "entropy_window_bounds_min must be > 0"
        assert self.entropy_window_bounds_max >= self.entropy_window_bounds_min, \
            "entropy_window_bounds_max must be >= entropy_window_bounds_min"
        assert 0.0 < self.learning_rate_safety_factor <= 1.0, \
            "learning_rate_safety_factor must be in (0, 1]"
        assert self.learning_rate_minimum > 0.0, \
            "learning_rate_minimum must be > 0"
        
        # Entropic regularization must be positive
        assert self.epsilon > 0, \
            f"epsilon must be > 0 (Sinkhorn), got {self.epsilon}"
        assert self.sinkhorn_epsilon_min > 0, \
            f"sinkhorn_epsilon_min must be > 0, got {self.sinkhorn_epsilon_min}"
        assert self.sinkhorn_epsilon_0 >= self.sinkhorn_epsilon_min, \
            f"sinkhorn_epsilon_0 must be >= epsilon_min, got {self.sinkhorn_epsilon_0}"
        assert 0.0 < self.sinkhorn_alpha <= 1.0, \
            f"sinkhorn_alpha must be in (0, 1], got {self.sinkhorn_alpha}"
        assert self.sinkhorn_inner_iterations > 0, \
            "sinkhorn_inner_iterations must be > 0"
        assert self.sinkhorn_cost_type in ("squared", "huber"), \
            "sinkhorn_cost_type must be 'squared' or 'huber'"
        assert self.sinkhorn_huber_delta > 0.0, \
            "sinkhorn_huber_delta must be > 0"
        
        # Entropy monitoring constraints
        assert self.entropy_window > 0, \
            f"entropy_window must be > 0, got {self.entropy_window}"
        assert 0.0 < self.entropy_threshold <= 1.0, \
            f"entropy_threshold must be in (0, 1], got {self.entropy_threshold}"
        assert 0.0 <= self.entropy_volatility_low_threshold < self.entropy_volatility_high_threshold, \
            "entropy_volatility thresholds must satisfy 0 <= low < high"
        assert 0.0 < self.entropy_ratio_min <= self.entropy_ratio_max, \
            "entropy_ratio bounds must satisfy 0 < min <= max"
        assert self.entropy_baseline_floor > 0.0, \
            "entropy_baseline_floor must be > 0"
        assert self.entropy_scaling_trigger > 0.0, \
            "entropy_scaling_trigger must be > 0"
        assert self.koopman_top_k > 0, \
            "koopman_top_k must be > 0"
        assert self.koopman_min_power > 0.0, \
            "koopman_min_power must be > 0"
        assert self.paley_wiener_integral_max > 0.0, \
            "paley_wiener_integral_max must be > 0"
        assert self.wtmm_num_scales > 0, \
            "wtmm_num_scales must be > 0"
        assert self.wtmm_scale_min > 0.0, \
            "wtmm_scale_min must be > 0"
        assert self.wtmm_sigma > 0.0, \
            "wtmm_sigma must be > 0"
        assert self.wtmm_fc > 0.0, \
            "wtmm_fc must be > 0"
        assert self.wtmm_modulus_threshold > 0.0, \
            "wtmm_modulus_threshold must be > 0"
        assert self.wtmm_max_link_distance > 0.0, \
            "wtmm_max_link_distance must be > 0"
        assert self.wtmm_q_steps > 1, \
            "wtmm_q_steps must be > 1"
        assert self.wtmm_q_max > self.wtmm_q_min, \
            "wtmm_q_max must be > wtmm_q_min"
        assert self.wtmm_h_steps > 1, \
            "wtmm_h_steps must be > 1"
        assert self.wtmm_h_max > self.wtmm_h_min, \
            "wtmm_h_max must be > wtmm_h_min"
        assert self.wtmm_tau_default_scale > 0.0, \
            "wtmm_tau_default_scale must be > 0"
        assert self.kernel_a_min_wiener_hopf_order > 0, \
            "kernel_a_min_wiener_hopf_order must be > 0"
        
        # Log-signature depth reasonable (exponential complexity)
        assert 3 <= self.log_sig_depth <= 5, \
            f"log_sig_depth must be in [3, 5], got {self.log_sig_depth}"
        
        # SDE integration parameters
        assert self.sde_dt > 0, \
            f"sde_dt must be > 0, got {self.sde_dt}"
        assert self.sde_numel_integrations > 0, \
            f"sde_numel_integrations must be > 0, got {self.sde_numel_integrations}"
        
        # CFL Condition (Courant-Friedrichs-Lewy): Theory.tex §2.3.3
        # Stochastic CFL: Δt < 2/λ_max(J_b + J_σ²)
        # Practical bound (safety margin C_safe ≈ 0.9):
        # dt_computed = horizon / n_steps must respect dtmax with safety margin
        dt_upper_bound = self.sde_pid_dtmax * 0.9  # C_safe safety margin
        assert dt_upper_bound > 0.0, \
            "CFL validation: sde_pid_dtmax must allow safe timesteps"
        
        assert self.stiffness_low > 0 and self.stiffness_high > self.stiffness_low, \
            f"stiffness thresholds must satisfy 0 < low < high, got {self.stiffness_low}, {self.stiffness_high}"
        assert self.sde_fd_epsilon > 0.0, \
            "sde_fd_epsilon must be > 0"
        assert self.kernel_c_jump_intensity >= 0.0, \
            "kernel_c_jump_intensity must be >= 0"
        assert self.kernel_c_jump_scale >= 0.0, \
            "kernel_c_jump_scale must be >= 0"
        assert self.kernel_c_jump_max_events > 0, \
            "kernel_c_jump_max_events must be > 0"
        
        # Holder exponent bounds (stochastic processes)
        assert 0.0 < self.holder_threshold < 1.0, \
            f"holder_threshold must be in (0, 1), got {self.holder_threshold}"
        assert self.robustness_dimension_threshold > 0.0, \
            "robustness_dimension_threshold must be > 0"
        assert 0.0 < self.dgm_entropy_coupling_beta <= 1.0, \
            "dgm_entropy_coupling_beta must be in (0, 1]"
        assert self.dgm_max_capacity_factor >= 1.0, \
            "dgm_max_capacity_factor must be >= 1"
        assert self.stiffness_calibration_c1 > 0.0, \
            "stiffness_calibration_c1 must be > 0"
        assert self.stiffness_calibration_c2 > 0.0, \
            "stiffness_calibration_c2 must be > 0"
        assert self.stiffness_min_low > 0.0, \
            "stiffness_min_low must be > 0"
        assert self.stiffness_min_high > self.stiffness_min_low, \
            "stiffness_min_high must be > stiffness_min_low"
        assert 0.0 < self.holder_exponent_guard < 1.0, \
            "holder_exponent_guard must be in (0, 1)"

        assert self.telemetry_buffer_capacity > 0, \
            "telemetry_buffer_capacity must be > 0"
        assert self.telemetry_hash_algorithm in ("sha256", "crc32c"), \
            "telemetry_hash_algorithm must be 'sha256' or 'crc32c'"
        assert self.telemetry_adaptive_log_path, \
            "telemetry_adaptive_log_path must be non-empty"
        assert self.signal_sampling_interval > 0.0, \
            "signal_sampling_interval must be > 0"
        assert self.degraded_recovery_min_steps > 0, \
            "degraded_recovery_min_steps must be > 0"
        assert self.telemetry_adaptive_window_size > 0, \
            "telemetry_adaptive_window_size must be > 0"
        assert self.telemetry_dashboard_width > 0, \
            "telemetry_dashboard_width must be > 0"
        assert self.telemetry_dashboard_height > 0, \
            "telemetry_dashboard_height must be > 0"
        assert self.telemetry_dashboard_spread_epsilon > 0.0, \
            "telemetry_dashboard_spread_epsilon must be > 0"
        assert self.telemetry_dashboard_recent_rows > 0, \
            "telemetry_dashboard_recent_rows must be > 0"
        assert self.frozen_signal_variance_floor > 0.0, \
            "frozen_signal_variance_floor must be > 0"
        
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

        # Stability parameter validation bounds
        assert self.validation_alpha_stable_min >= 0.0, \
            "validation_alpha_stable_min must be >= 0"
        assert self.validation_alpha_stable_max <= 2.0, \
            "validation_alpha_stable_max must be <= 2"
        assert self.validation_beta_stable_min >= -1.0, \
            "validation_beta_stable_min must be >= -1"
        assert self.validation_beta_stable_max <= 1.0, \
            "validation_beta_stable_max must be <= 1"
        assert self.validation_viscosity_residual_max > 0.0, \
            "validation_viscosity_residual_max must be > 0"
        
        # TTL/Nyquist coherence
        assert self.staleness_ttl_ns > 0 and self.besov_nyquist_interval_ns > 0, \
            "Latency timeouts must be positive"

        # PRNG defaults
        assert self.prng_seed >= 0, "prng_seed must be >= 0"
        assert self.prng_split_count >= 4, (
            "prng_split_count must be >= number of kernels"
        )

        # PDF grid and confidence bounds
        assert self.pdf_grid_num_points > 1, "pdf_grid_num_points must be > 1"
        assert self.pdf_grid_min_z < self.pdf_grid_max_z, "pdf_grid_min_z must be < pdf_grid_max_z"
        assert self.pdf_min_sigma > 0.0, "pdf_min_sigma must be > 0"
        assert self.confidence_interval_z > 0.0, "confidence_interval_z must be > 0"
        assert self.kernel_output_time_us > 0.0, "kernel_output_time_us must be > 0"

        # Kurtosis bounds
        assert self.kurtosis_min > 0.0, "kurtosis_min must be > 0"
        assert self.kurtosis_max > self.kurtosis_min, "kurtosis_max must be > kurtosis_min"
        assert self.kurtosis_reference > 0.0, "kurtosis_reference must be > 0"

        # Load shedding depths
        if not self.kernel_d_load_shedding_depths:
            raise AssertionError("kernel_d_load_shedding_depths must be non-empty")
        if not all(depth > 0 for depth in self.kernel_d_load_shedding_depths):
            raise AssertionError("kernel_d_load_shedding_depths must be positive")


# ═══════════════════════════════════════════════════════════════════════════
# INPUT/OUTPUT STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProcessState:
    """
    Predictor operational input (y_t, tau_utc).
    
    Design: Scalar fields (shape [1]) for compatibility with vmap
    in multi-asset architecture (vectorized batching).
    
    Domain-Agnostic: Applies to any stochastic process without semantic assumptions.
    
    References:
        - API_Python.tex §1.2: Operational Input
        - IO.tex §2: Observation Protocol
    """
    magnitude: Float[Array, "1"]
    timestamp_utc: datetime
    state_tag: Optional[str] = None
    dispersion_proxy: Optional[Float[Array, "1"]] = None

    @property
    def timestamp_ns(self) -> int:
        """Nanosecond epoch derived from UTC timestamp."""
        timestamp = self.timestamp_utc
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return int(timestamp.timestamp() * 1e9)
    
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
    System output for external API contracts.
    
    References:
        - API_Python.tex §1.3: System Output
        - IO.tex §3: Risk State Vector (S_risk)
    
    Note: operating_mode is Array (int32) in core (XLA-compatible); convert to string in API layer.
    """
    reference_prediction: Float[Array, ""]
    confidence_lower: Float[Array, ""]
    confidence_upper: Float[Array, ""]
    operating_mode: Array  # int32 scalar
    telemetry: Optional[object] = None
    request_id: Optional[str] = None

    def __post_init__(self):
        lower = float(self.confidence_lower)
        upper = float(self.confidence_upper)
        ref = float(self.reference_prediction)
        assert lower <= ref <= upper, (
            "confidence_lower must be <= reference_prediction <= confidence_upper"
        )
        


# ═══════════════════════════════════════════════════════════════════════════
# AUXILIARY TYPES (Internal State Structures)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class KernelOutput:
    """
    API-facing kernel output contract.
    """
    probability_density: Float[Array, "n_targets"]
    kernel_id: str
    computation_time_us: float
    numerics_flags: dict
    entropy: Optional[Float[Array, ""]] = None


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
    grace_counter: Union[int, Array]        # Refractory period counter
    adaptive_h_t: Float[Array, "1"]         # h_t: Kurtosis-adaptive CUSUM threshold
    
    # EWMA Volatility
    ema_variance: Float[Array, "1"]         # σ²_t: Exponential variance
    
    # Accumulated Telemetry
    kurtosis: Float[Array, "1"]             # κ_t: Empirical kurtosis
    holder_exponent: Float[Array, "1"]      # H_t: WTMM Holder
    dgm_entropy: Float[Array, "1"]          # H_DGM: Kernel B entropy
    mode_collapse_consecutive_steps: Union[int, Array]    # V-MAJ-5: Counter for consecutive low-entropy steps
    degraded_mode_recovery_counter: Union[int, Array]     # V-MAJ-7: Steps needed to recover from degraded mode
    
    # V-MAJ-7: Level 4 Autonomy Adaptive Telemetry
    baseline_entropy: Float[Array, "1"]     # H_baseline: Reference entropy for κ = H_current / H_baseline
    solver_explicit_count: Union[int, Array]              # N_explicit: Explicit SDE solver steps in window
    solver_implicit_count: Union[int, Array]              # N_implicit: Implicit SDE solver steps in window
    architecture_scaling_events: Union[int, Array]        # N_scale: DGM architecture scaling events
    
    # State Flags
    degraded_mode: Union[bool, Array]       # Degraded mode active
    emergency_mode: Union[bool, Array]      # Emergency mode active
    regime_changed: Union[bool, Array]      # Regime change detected
    
    # Timestamp Control
    last_update_ns: Union[int, Array]       # Last processed timestamp
    
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
    
    Note: Core returns integers; API layer converts to strings.
    """
    INFERENCE = 0
    CALIBRATION = 1
    DIAGNOSTIC = 2
    
    @staticmethod
    def to_string(mode: int) -> str:
        """Convert integer mode to API string (host-side only)."""
        if mode == 0:
            return "inference"
        elif mode == 1:
            return "calibration"
        elif mode == 2:
            return "diagnostic"
        return "inference"


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
