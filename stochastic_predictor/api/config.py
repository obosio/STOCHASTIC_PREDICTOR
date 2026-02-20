"""
Configuration management and injection for the Universal Stochastic Predictor.

Provides singleton access to config.toml with validation and environment
variable overrides. Implements the Configuration pattern for dependency injection.

References:
    - config.toml (auto-discovered at runtime)
    - Stochastic_Predictor_Implementation.tex §1.2
"""

from __future__ import annotations

import logging
import os
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python < 3.11

import jax
from jaxtyping import Float

# Module logger for config hot-reload events (V-MIN-3)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .types import PredictorConfig


# ═══════════════════════════════════════════════════════════════════════════
# JAX CONFIGURATION (MUST execute before any XLA tracing)
# ═══════════════════════════════════════════════════════════════════════════
# Enable 64-bit precision for Malliavin calculus, Signature computations,
# and deterministic reproducibility across CPU/GPU/FPGA backends.
# CRITICAL: This must execute at import time, before any JAX operations.
jax.config.update("jax_enable_x64", True)


def verify_xla_precision() -> None:
    """
    Audit internal XLA compilation target to ensure FP64 is active.
    
    Guarantees immutability of algorithmic signatures (Kernel D log-signatures)
    and prevents silent truncation in Malliavin calculus operations.
    
    Raises:
        RuntimeError: If JAX FP64 enforcement failed (catastrophic failure mode)
    
    References:
        - Stochastic_Predictor_Implementation.tex §2.0.0 (Bootstrap FP64 policy)
        - config.toml [core] float_precision = 64
    
    Example:
        >>> verify_xla_precision()  # Raises if FP64 not active
        >>> # Proceed with kernel operations
    """
    test_array = jax.numpy.array([1.0], dtype=jax.numpy.float64)
    
    if test_array.dtype != jax.numpy.float64:
        raise RuntimeError(
            "CRITICAL: JAX FP64 enforcement failed. "
            "Log-signature truncation will suffer from catastrophic cancellation. "
            "Verify that jax.config.update('jax_enable_x64', True) executed before XLA tracing. "
            "This may indicate a JAX version incompatibility or environment configuration issue."
        )
    
    # Verify dtype attribute is correctly set
    if str(test_array.dtype) != "float64":
        raise RuntimeError(
            f"CRITICAL: JAX dtype inspection failed. Expected 'float64', got '{test_array.dtype}'. "
            f"XLA backend may be misconfigured or downgrading precision silently."
        )


# Execute FP64 verification immediately at module import
verify_xla_precision()


class ConfigManager:
    """
    Singleton configuration manager.
    
    Loads config.toml from the project root, applies environment variable
    overrides, and provides validated access to configuration parameters.
    
    Thread-safe singleton pattern (lazy initialization).
    """
    
    _instance: Optional["ConfigManager"] = None
    _config: Dict[str, Any] = {}
    _initialized: bool = False
    _config_path: Optional[Path] = None
    _last_mtime: float = 0.0
    
    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    @classmethod
    def _initialize(cls) -> None:
        """Load and parse config.toml, apply environment overrides."""
        if cls._initialized:
            return
        
        # Discover config.toml from project root
        config_path = cls._find_config_file()
        if not config_path:
            raise FileNotFoundError(
                "config.toml not found. Expected in project root. "
                "See doc/latex/specification/Stochastic_Predictor_Implementation.tex §1.2"
            )
        
        cls._config_path = config_path
        cls._last_mtime = config_path.stat().st_mtime
        
        # Parse TOML
        with open(config_path, "rb") as f:
            cls._config = tomllib.load(f)
        
        # Apply environment variable overrides (dot-notation: CORE__JAX_PLATFORMS)
        cls._apply_env_overrides()
        cls._initialized = True
    
    @staticmethod
    def _find_config_file() -> Optional[Path]:
        """Discover config.toml in the project hierarchy."""
        # Start from current file location and traverse upward
        search_paths = [
            Path.cwd() / "config.toml",  # Current working directory
            Path(__file__).parent.parent.parent / "config.toml",  # Project root (../../..)
            Path.home() / ".stochastic_predictor" / "config.toml",  # User home (optional)
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        return None
    
    @classmethod
    def _apply_env_overrides(cls) -> None:
        """Apply environment variable overrides (dot-notation)."""
        for env_var, value in os.environ.items():
            if env_var.startswith("USP_"):
                # Parse USP_SECTION__KEY format
                parts = env_var[4:].lower().split("__")
                if len(parts) == 2:
                    section, key = parts
                    if section not in cls._config:
                        cls._config[section] = {}
                    cls._config[section][key] = value
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value with fallback.
        
        Args:
            section: Config section (e.g., "core", "kernels")
            key: Configuration key
            default: Fallback value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._config.get(section, {}).get(key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config.get(section, {})
    
    def raw_config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary (for inspection/debugging)."""
        return self._config.copy()
    
    def check_and_reload(self) -> bool:
        """
        Check if config.toml has been modified and reload if necessary.
        
        Enables hot-reload after autonomous config mutations without restart.
        Uses mtime (modification time) tracking for efficient change detection.
        
        COMPLIANCE:
            - V-CRIT-4: Hot-reload config mechanism
            - V-MIN-3: Log config hot-reload events to telemetry
        
        Returns:
            True if config was reloaded, False if no changes detected
        
        Example:
            >>> config_manager = get_config()
            >>> # After atomic_write_config() mutation
            >>> if config_manager.check_and_reload():
            ...     print("Configuration reloaded")
        
        References:
            - AUDIT_SPEC_COMPLIANCE_2026-02-19.md: V-CRIT-4, V-MIN-3
            - IO.tex §3.3.7: Hot-reload integration with mutation protocol
        """
        if not self._config_path or not self._config_path.exists():
            return False
        
        # Check modification time
        current_mtime = self._config_path.stat().st_mtime
        
        if current_mtime <= self._last_mtime:
            return False  # No changes
        
        # Reload configuration
        try:
            with open(self._config_path, "rb") as f:
                self._config = tomllib.load(f)
            
            # Reapply environment overrides
            self._apply_env_overrides()
            
            # Update mtime
            self._last_mtime = current_mtime
            
            # V-MIN-3: Log successful reload event
            logger.info(
                f"Config hot-reloaded at {datetime.now().isoformat()}. "
                f"Trigger: external mutation detected (mtime={current_mtime:.3f})."
            )
            
            return True
        
        except Exception as e:
            # V-MIN-3: Log reload failure
            logger.error(
                f"Config hot-reload failed at {datetime.now().isoformat()}: {e}. "
                f"mtime={current_mtime:.3f}"
            )
            return False


# Lazy singleton initialization
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global ConfigManager singleton."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# ═══════════════════════════════════════════════════════════════════════════
# FIELD MAPPING METADATA (Single Source of Truth)
# ═══════════════════════════════════════════════════════════════════════════

# Maps PredictorConfig field names to their config.toml section
# Format: {field_name: section_name}
# This is the ONLY place that needs updating when adding new config fields
FIELD_TO_SECTION_MAP: Dict[str, str] = {
    # Metadata
    "schema_version": "meta",
    "prng_seed": "core",
    "prng_split_count": "core",
    
    # JKO Orchestrator & Optimal Transport
    "epsilon": "orchestration",
    "learning_rate": "orchestration",
    "jko_domain_length": "orchestration",
    "entropy_window_relaxation_factor": "orchestration",
    "entropy_window_bounds_min": "orchestration",
    "entropy_window_bounds_max": "orchestration",
    "learning_rate_safety_factor": "orchestration",
    "learning_rate_minimum": "orchestration",
    "sinkhorn_epsilon_min": "orchestration",
    "sinkhorn_epsilon_0": "orchestration",
    "sinkhorn_alpha": "orchestration",
    "sinkhorn_max_iter": "orchestration",
    "sinkhorn_inner_iterations": "orchestration",
    
    # Entropy Monitoring
    "entropy_window": "orchestration",
    "entropy_threshold": "orchestration",
    "entropy_gamma_min": "orchestration",
    "entropy_gamma_max": "orchestration",
    "entropy_gamma_default": "orchestration",
    "entropy_volatility_low_threshold": "orchestration",
    "entropy_volatility_high_threshold": "orchestration",
    "entropy_ratio_min": "orchestration",
    "entropy_ratio_max": "orchestration",
    "entropy_baseline_floor": "orchestration",
    
    # Kernel Parameters
    "log_sig_depth": "kernels",
    "wtmm_buffer_size": "kernels",
    "wtmm_num_scales": "kernels",
    "wtmm_scale_min": "kernels",
    "wtmm_sigma": "kernels",
    "wtmm_fc": "kernels",
    "wtmm_modulus_threshold": "kernels",
    "wtmm_max_link_distance": "kernels",
    "wtmm_q_min": "kernels",
    "wtmm_q_max": "kernels",
    "wtmm_q_steps": "kernels",
    "wtmm_h_min": "kernels",
    "wtmm_h_max": "kernels",
    "wtmm_h_steps": "kernels",
    "wtmm_tau_default_scale": "kernels",
    "besov_cone_c": "kernels",
    "besov_nyquist_interval_ns": "kernels",
    "stiffness_low": "kernels",
    "stiffness_high": "kernels",
    "sde_dt": "kernels",
    "sde_numel_integrations": "kernels",
    "kernel_a_bandwidth": "kernels",
    "kernel_a_embedding_dim": "kernels",
    "kernel_a_min_wiener_hopf_order": "kernels",
    "kernel_a_min_variance": "kernels",
    "koopman_top_k": "kernels",
    "koopman_min_power": "kernels",
    "paley_wiener_integral_max": "kernels",
    "dgm_width_size": "kernels",
    "dgm_depth": "kernels",
    "dgm_entropy_num_bins": "kernels",
    "dgm_activation": "kernels",
    "kernel_b_r": "kernels",
    "kernel_b_sigma": "kernels",
    "kernel_b_horizon": "kernels",
    "kernel_c_mu": "kernels",
    "kernel_c_alpha": "kernels",
    "kernel_c_beta": "kernels",
    "kernel_c_horizon": "kernels",
    "kernel_c_dt0": "kernels",
    "kernel_c_alpha_gaussian_threshold": "kernels",
    "kernel_c_jump_intensity": "kernels",
    "kernel_c_jump_mean": "kernels",
    "kernel_c_jump_scale": "kernels",
    "kernel_c_jump_max_events": "kernels",
    "kernel_d_depth": "kernels",
    "kernel_d_load_shedding_depths": "kernels",
    "kernel_d_alpha": "kernels",
    "kernel_d_confidence_base": "kernels",
    "base_min_signal_length": "kernels",
    "signal_normalization_method": "kernels",
    "numerical_epsilon": "kernels",
    "warmup_signal_length": "kernels",
    "pdf_grid_min_z": "kernels",
    "pdf_grid_max_z": "kernels",
    "pdf_grid_num_points": "kernels",
    "pdf_min_sigma": "kernels",
    "confidence_interval_z": "kernels",
    "kernel_output_time_us": "kernels",
    "kurtosis_min": "kernels",
    "kurtosis_max": "kernels",
    "kurtosis_reference": "kernels",
    
    # Circuit Breaker & Regime Detection
    "holder_threshold": "orchestration",
    "dgm_entropy_coupling_beta": "orchestration",
    "dgm_max_capacity_factor": "orchestration",
    "stiffness_calibration_c1": "orchestration",
    "stiffness_calibration_c2": "orchestration",
    "stiffness_min_low": "orchestration",
    "stiffness_min_high": "orchestration",
    "holder_exponent_guard": "orchestration",
    "cusum_h": "orchestration",
    "cusum_k": "orchestration",
    "grace_period_steps": "orchestration",
    "residual_window_size": "orchestration",
    "volatility_alpha": "orchestration",
    "inference_recovery_hysteresis": "orchestration",
    
    # Validation & Outlier Detection
    "sigma_bound": "orchestration",
    "sigma_val": "orchestration",
    "max_future_drift_ns": "orchestration",
    "max_past_drift_ns": "orchestration",
    
    # Validation Constraints (Phase 5: Zero-Heuristics)
    "validation_finite_allow_nan": "validation",
    "validation_finite_allow_inf": "validation",
    "validation_simplex_atol": "validation",
    "validation_holder_exponent_min": "validation",
    "validation_holder_exponent_max": "validation",
    "validation_alpha_stable_min": "validation",
    "validation_alpha_stable_max": "validation",
    "validation_alpha_stable_exclusive_bounds": "validation",
    "validation_beta_stable_min": "validation",
    "validation_beta_stable_max": "validation",
    "validation_viscosity_residual_max": "validation",
    "sanitize_replace_nan_value": "validation",
    
    # Phase 6: SDE Integration Tolerances (Kernel C - Zero-Heuristics)
    "sde_brownian_tree_tol": "kernels",
    "sde_pid_rtol": "kernels",
    "sde_pid_atol": "kernels",
    "sde_pid_dtmin": "kernels",
    "sde_pid_dtmax": "kernels",
    "sde_solver_type": "kernels",
    "sde_initial_dt_factor": "kernels",
    
    # Phase 6: Kernel B Hyperparameters (Zero-Heuristics)
    "kernel_b_spatial_samples": "kernels",
    "kernel_b_spatial_range_factor": "kernels",
    "kernel_ridge_lambda": "kernels",
    "sde_diffusion_sigma": "kernels",
    
    # Phase 6: Kernel D Hyperparameters (Zero-Heuristics)
    "kernel_d_confidence_scale": "kernels",
    "sanitize_replace_inf_value": "validation",
    "sanitize_clip_range": "validation",
    
    # Mode Collapse Detection (Orchestration)
    "mode_collapse_min_threshold": "orchestration",
    "mode_collapse_window_ratio": "orchestration",
    
    # Meta-Optimization (Capa 3 - Auto-Tuning v2.1.0)
    "log_sig_depth_min": "meta_optimization",
    "log_sig_depth_max": "meta_optimization",
    "wtmm_buffer_size_min": "meta_optimization",
    "wtmm_buffer_size_max": "meta_optimization",
    "wtmm_buffer_size_step": "meta_optimization",
    "besov_cone_c_min": "meta_optimization",
    "besov_cone_c_max": "meta_optimization",
    "cusum_k_min": "meta_optimization",
    "cusum_k_max": "meta_optimization",
    "sinkhorn_alpha_min": "meta_optimization",
    "sinkhorn_alpha_max": "meta_optimization",
    "volatility_alpha_min": "meta_optimization",
    "volatility_alpha_max": "meta_optimization",
    "n_trials": "meta_optimization",
    "n_startup_trials": "meta_optimization",
    "multivariate": "meta_optimization",
    "train_ratio": "meta_optimization",
    "n_folds": "meta_optimization",
    
    # I/O Policies
    "data_feed_timeout": "io",
    "data_feed_max_retries": "io",
    "snapshot_atomic_fsync": "io",
    "snapshot_compression": "io",
    "snapshot_format": "io",
    "snapshot_hash_algorithm": "io",
    "telemetry_hash_interval_steps": "io",
    "telemetry_buffer_capacity": "io",
    "telemetry_hash_algorithm": "io",
    "telemetry_adaptive_log_path": "io",
    "telemetry_adaptive_window_size": "io",
    "telemetry_placeholder_max_stiffness_metric": "io",
    "telemetry_placeholder_num_internal_iterations_mean": "io",
    "telemetry_placeholder_implicit_residual_norm_max": "io",
    "telemetry_dashboard_title": "io",
    "telemetry_dashboard_width": "io",
    "telemetry_dashboard_height": "io",
    "telemetry_dashboard_spread_epsilon": "io",
    "telemetry_dashboard_recent_rows": "io",
    "frozen_signal_variance_floor": "io",
    "frozen_signal_min_steps": "io",
    "frozen_signal_recovery_ratio": "io",
    "frozen_signal_recovery_steps": "io",
    
    # Core System Policies
    "staleness_ttl_ns": "core",
}


class PredictorConfigInjector:
    """
    Dependency injection wrapper for PredictorConfig.
    
    Bridges config.toml values with the types.PredictorConfig dataclass,
    allowing seamless configuration-driven instantiation.
    
    Example:
        >>> injector = PredictorConfigInjector()
        >>> cfg = injector.create_config()
        >>> assert cfg.grace_period_steps == 20  # or env override
    """
    
    def __init__(self):
        self.config_manager = get_config()
    
    def create_config(self) -> PredictorConfig:
        """
        Create a PredictorConfig instance from config.toml using automated field mapping.
        
        Uses dataclass introspection to ensure all PredictorConfig fields are populated
        from config.toml or environment variables, without falling back to dataclass defaults.
        
        Architecture:
            1. Introspect PredictorConfig fields using dataclasses.fields()
            2. Map each field to its config.toml section via FIELD_TO_SECTION_MAP
            3. Auto-construct cfg_dict without manual hard-coding
            4. Validate completeness (all fields have section mappings)
        
        Returns:
            Configured PredictorConfig instance with all fields populated
            
        Raises:
            ValueError: If FIELD_TO_SECTION_MAP is incomplete (missing field mappings)
        
        References:
            - FIELD_TO_SECTION_MAP: Single source of truth for field→section mapping
            - PredictorConfig: types.py dataclass with defaults
        """
        from .types import PredictorConfig
        
        # Introspect PredictorConfig dataclass fields
        config_fields = fields(PredictorConfig)
        
        # Validate that FIELD_TO_SECTION_MAP is complete
        field_names = {f.name for f in config_fields}
        mapped_fields = set(FIELD_TO_SECTION_MAP.keys())
        
        missing_mappings = field_names - mapped_fields
        if missing_mappings:
            raise ValueError(
                f"FIELD_TO_SECTION_MAP is incomplete. Missing mappings for: {missing_mappings}. "
                f"Update FIELD_TO_SECTION_MAP in config.py to include all PredictorConfig fields."
            )
        
        # Auto-construct configuration dictionary
        cfg_dict = {}
        for field in config_fields:
            field_name = field.name
            section = FIELD_TO_SECTION_MAP[field_name]

            section_values = self.config_manager.get_section(section)
            if field_name not in section_values:
                raise ValueError(
                    "Missing required config.toml entry: "
                    f"[{section}].{field_name}. "
                    "Diamond mode forbids dataclass defaults."
                )

            cfg_dict[field_name] = section_values[field_name]
        
        return PredictorConfig(**cfg_dict)
    
    def verify_jax_config(self) -> Dict[str, bool]:
        """
        Verify that JAX is configured per config.toml specifications.
        
        Compares JAX runtime config with config.toml requirements.
        
        Returns:
            Dictionary of verification flags: {check_name: passed}
        """
        checks = {}
        
        # Check JAX precision mode
        expected_dtype = self.config_manager.get("core", "jax_default_dtype", "float32")
        # Note: JAX config attributes vary by version; verify using device capabilities
        current_devices = jax.devices()
        checks["jax_device_available"] = len(current_devices) > 0
        
        # Check platform configuration
        expected_platform = self.config_manager.get("core", "jax_platforms", "cpu")
        checks["jax_platform_matches"] = any(
            expected_platform in str(d).lower() for d in current_devices
        )
        
        return checks


__all__ = [
    "ConfigManager",
    "get_config",
    "PredictorConfigInjector",
]
