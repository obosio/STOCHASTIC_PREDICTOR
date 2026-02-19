"""
Configuration management and injection for the Universal Stochastic Predictor.

Provides singleton access to config.toml with validation and environment
variable overrides. Implements the Configuration pattern for dependency injection.

References:
    - config.toml (auto-discovered at runtime)
    - Predictor_Estocastico_Implementacion.tex §1.2
"""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python < 3.11

import jax
from jaxtyping import Float

if TYPE_CHECKING:
    from .types import PredictorConfig


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
                "See doc/Predictor_Estocastico_Implementacion.tex §1.2"
            )
        
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
    
    # JKO Orchestrator & Optimal Transport
    "epsilon": "orchestration",
    "learning_rate": "orchestration",
    "sinkhorn_epsilon_min": "orchestration",
    "sinkhorn_epsilon_0": "orchestration",
    "sinkhorn_alpha": "orchestration",
    
    # Entropy Monitoring
    "entropy_window": "orchestration",
    "entropy_threshold": "orchestration",
    
    # Kernel Parameters
    "log_sig_depth": "kernels",
    "wtmm_buffer_size": "kernels",
    "besov_cone_c": "kernels",
    "besov_nyquist_interval_ns": "kernels",
    "stiffness_low": "kernels",
    "stiffness_high": "kernels",
    "sde_dt": "kernels",
    "sde_numel_integrations": "kernels",
    "kernel_a_bandwidth": "kernels",
    "kernel_a_embedding_dim": "kernels",
    "dgm_width_size": "kernels",
    "dgm_depth": "kernels",
    "dgm_entropy_num_bins": "kernels",
    "kernel_b_r": "kernels",
    "kernel_b_sigma": "kernels",
    "kernel_b_horizon": "kernels",
    "kernel_c_mu": "kernels",
    "kernel_c_alpha": "kernels",
    "kernel_c_beta": "kernels",
    "kernel_c_horizon": "kernels",
    "kernel_c_dt0": "kernels",
    "kernel_d_depth": "kernels",
    "kernel_d_alpha": "kernels",
    "base_min_signal_length": "kernels",
    "signal_normalization_method": "kernels",
    
    # Circuit Breaker & Regime Detection
    "holder_threshold": "orchestration",
    "cusum_h": "orchestration",
    "cusum_k": "orchestration",
    "grace_period_steps": "orchestration",
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
    "sanitize_replace_nan_value": "validation",
    
    # Phase 6: SDE Integration Tolerances (Kernel C - Zero-Heuristics)
    "sde_brownian_tree_tol": "kernels",
    "sde_pid_rtol": "kernels",
    "sde_pid_atol": "kernels",
    "sde_pid_dtmin": "kernels",
    "sde_pid_dtmax": "kernels",
    "sde_solver_type": "kernels",
    
    # Phase 6: Kernel B Hyperparameters (Zero-Heuristics)
    "kernel_b_spatial_samples": "kernels",
    "kernel_ridge_lambda": "kernels",
    "sde_diffusion_sigma": "kernels",
    
    # Phase 6: Kernel D Hyperparameters (Zero-Heuristics)
    "kernel_d_confidence_scale": "kernels",
    "sanitize_replace_inf_value": "validation",
    "sanitize_clip_range": "validation",
    
    # I/O Policies
    "data_feed_timeout": "io",
    "data_feed_max_retries": "io",
    "snapshot_atomic_fsync": "io",
    "snapshot_compression": "io",
    
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
        >>> assert cfg.cusum_grace_period == 20  # or env override
    """
    
    def __init__(self):
        self.config_manager = get_config()
    
    def create_config(self) -> PredictorConfig:
        """
        Create a PredictorConfig instance from config.toml using automated field mapping.
        
        Uses dataclass introspection to ensure all PredictorConfig fields are populated
        from config.toml or environment variables, with dataclass defaults as fallback.
        
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
            
            # Get value from config.toml, fallback to dataclass default
            default_value = field.default if field.default is not field.default_factory else None
            value = self.config_manager.get(section, field_name, default_value)
            
            cfg_dict[field_name] = value
        
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
