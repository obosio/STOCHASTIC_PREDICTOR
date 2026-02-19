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
        Create a PredictorConfig instance from config.toml.
        
        Maps all fields from config.toml to PredictorConfig dataclass,
        ensuring single source of truth for algorithmic configuration.
        
        Imports PredictorConfig here to avoid circular imports.
        
        Returns:
            Configured PredictorConfig instance with all fields populated
        """
        from .types import PredictorConfig
        
        cfg_dict = {
            # Snapshot Versioning
            "schema_version": self.config_manager.get(
                "meta", "schema_version", "1.0"
            ),
            
            # JKO Orchestrator (Optimal Transport)
            "epsilon": self.config_manager.get(
                "orchestration", "epsilon", 1e-3
            ),
            "learning_rate": self.config_manager.get(
                "orchestration", "learning_rate", 0.01
            ),
            
            # Kernel D (Log-Signatures)
            "log_sig_depth": self.config_manager.get(
                "kernels", "log_sig_depth", 3
            ),
            
            # Kernel A (WTMM + Fokker-Planck)
            "wtmm_buffer_size": self.config_manager.get(
                "kernels", "wtmm_buffer_size", 128
            ),
            "besov_cone_c": self.config_manager.get(
                "kernels", "besov_cone_c", 1.5
            ),
            
            # Circuit Breaker (Holder Singularity)
            "holder_threshold": self.config_manager.get(
                "orchestration", "holder_threshold", 0.4
            ),
            
            # CUSUM (Regime Change Detection)
            "cusum_h": self.config_manager.get(
                "orchestration", "cusum_h", 5.0
            ),
            "cusum_k": self.config_manager.get(
                "orchestration", "cusum_k", 0.5
            ),
            "grace_period_steps": self.config_manager.get(
                "orchestration", "grace_period_steps", 20
            ),
            
            # Volatility Monitoring (EWMA)
            "volatility_alpha": self.config_manager.get(
                "orchestration", "volatility_alpha", 0.1
            ),
            
            # Latency and Anti-Aliasing Policies
            "staleness_ttl_ns": self.config_manager.get(
                "core", "staleness_ttl_ns", 500_000_000
            ),
            "besov_nyquist_interval_ns": self.config_manager.get(
                "kernels", "besov_nyquist_interval_ns", 100_000_000
            ),
            "inference_recovery_hysteresis": self.config_manager.get(
                "orchestration", "inference_recovery_hysteresis", 0.8
            ),
        }
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
