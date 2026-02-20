"""Bayesian Meta-Optimization (Learning to Learn).

Implements derivative-free hyperparameter search using Gaussian Processes (TPE)
to optimize structural parameters (log-signature depth, buffer sizes, etc.)
via walk-forward validation.

References:
    - MIGRATION_AUTOTUNING_v1.0.md §3.2: Capa 3 Meta-Optimization
    - Implementation.tex §8.3: Bayesian Hyperparameter Tuning
"""

from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass, fields
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
import pickle
import hashlib
from pathlib import Path
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python < 3.11

try:
    import optuna  # type: ignore[import-not-found]
    from optuna.samplers import TPESampler  # type: ignore[import-not-found]
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore
    TPESampler = None  # type: ignore

from Python.api.types import PredictorConfig
from Python.api.config import FIELD_TO_SECTION_MAP, get_config
from Python.io.config_mutation import (
    atomic_write_config,
    ConfigMutationError,
    MutationRateLimiter,
)


class IntegrityError(Exception):
    """Raised when TPE checkpoint integrity verification fails (SHA-256 mismatch)."""
    pass


@dataclass
class MetaOptimizationConfig:
    """
    Configuration for meta-optimization search space and constraints.
    
    Supports two-tier optimization:
        - Fast Tuning: 6 params, 50 trials (~2 hours)
        - Deep Tuning: 20+ params, 500 trials (~10-30 days)
    
    COMPLIANCE: V-CRIT-6 - Deep Tuning search space (20+ params)
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # STRUCTURAL PARAMETERS (Deep Tuning - Low Frequency)
    # ═══════════════════════════════════════════════════════════════════
    
    # Log-Signature Depth (Kernel D)
    log_sig_depth_min: int = 2
    log_sig_depth_max: int = 5
    
    # WTMM Buffer Size (Kernel C)
    wtmm_buffer_size_min: int = 64
    wtmm_buffer_size_max: int = 512
    wtmm_buffer_size_step: int = 64
    
    # Besov Cone Constraint (Kernel C)
    besov_cone_c_min: float = 1.0
    besov_cone_c_max: float = 3.0
    
    # DGM Architecture (Kernel A) - NEW
    dgm_width_size_min: int = 32
    dgm_width_size_max: int = 256
    dgm_width_size_step: int = 32  # Must be power of 2
    
    dgm_depth_min: int = 2
    dgm_depth_max: int = 6
    
    dgm_entropy_num_bins_min: int = 20
    dgm_entropy_num_bins_max: int = 100
    
    # SDF Solver Thresholds (Kernel B) - NEW
    stiffness_low_min: float = 50.0
    stiffness_low_max: float = 500.0
    
    stiffness_high_min: float = 500.0
    stiffness_high_max: float = 5000.0
    
    # SDE Integration Parameters - NEW
    sde_dt_min: float = 0.001
    sde_dt_max: float = 0.1
    
    sde_numel_integrations_min: int = 50
    sde_numel_integrations_max: int = 200
    
    sde_diffusion_sigma_min: float = 0.05
    sde_diffusion_sigma_max: float = 0.5
    
    # Kernel Ridge Regularization - NEW
    kernel_ridge_lambda_min: float = 1e-8
    kernel_ridge_lambda_max: float = 1e-3
    
    # ═══════════════════════════════════════════════════════════════════
    # SENSITIVITY PARAMETERS (Fast Tuning - High Frequency)
    # ═══════════════════════════════════════════════════════════════════
    
    # CUSUM Detection
    cusum_k_min: float = 0.1
    cusum_k_max: float = 1.0
    
    cusum_h_min: float = 2.0
    cusum_h_max: float = 10.0
    
    cusum_grace_period_steps_min: int = 5
    cusum_grace_period_steps_max: int = 100
    
    # Sinkhorn Regularization
    sinkhorn_alpha_min: float = 0.1
    sinkhorn_alpha_max: float = 1.0
    
    sinkhorn_epsilon_min_min: float = 0.001
    sinkhorn_epsilon_min_max: float = 0.1
    
    sinkhorn_epsilon_0_min: float = 0.05
    sinkhorn_epsilon_0_max: float = 0.5
    
    # Volatility Coupling
    volatility_alpha_min: float = 0.05
    volatility_alpha_max: float = 0.3
    
    # JKO Wasserstein Flow - NEW
    learning_rate_min: float = 0.001
    learning_rate_max: float = 0.1
    
    entropy_window_min: int = 50
    entropy_window_max: int = 500
    
    entropy_threshold_min: float = 0.5
    entropy_threshold_max: float = 0.95
    
    # Holder Exponent Threshold - NEW
    holder_threshold_min: float = 0.2
    holder_threshold_max: float = 0.65
    
    # ═══════════════════════════════════════════════════════════════════
    # OPTIMIZATION CONTROL
    # ═══════════════════════════════════════════════════════════════════
    
    n_trials: int = 50  # Fast Tuning default
    n_startup_trials: int = 10
    multivariate: bool = True
    
    # Deep Tuning mode (set n_trials=500 for Deep Tuning)
    enable_deep_tuning: bool = False
    
    # Walk-forward validation
    train_ratio: float = 0.7
    n_folds: int = 5


def load_meta_optimization_config() -> MetaOptimizationConfig:
    """
    Load meta-optimization defaults from config.toml.
    
    Zero-Heuristics: All metaparameter defaults are config-driven.
    """
    config_manager = get_config()
    section = config_manager.get_section("meta_optimization")
    meta_fields = [field.name for field in fields(MetaOptimizationConfig)]
    missing = [name for name in meta_fields if name not in section]
    if missing:
        raise ValueError(
            "Missing required [meta_optimization] entries in config.toml: "
            f"{sorted(missing)}"
        )
    values = {name: section[name] for name in meta_fields}
    return MetaOptimizationConfig(**values)


def _coerce_float(value: Any, label: str) -> float:
    """Coerce config values to float for delta validation."""
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    raise ConfigMutationError(
        f"Non-numeric value for '{label}' in config mutation: {value}"
    )


@dataclass
class OptimizationResult:
    """Results of meta-optimization run."""
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    study: Optional[Any]  # optuna.Study if available


class BayesianMetaOptimizer:
    """
    Derivative-Free Meta-Optimization using Gaussian Processes (TPE).
    
    Ensures structural hyperparameters evolve to fit process topology
    via walk-forward validation (no look-ahead bias).
    
    References:
        - MIGRATION_AUTOTUNING_v1.0.md §3.2
        - Implementation.tex §8.3: TPE Sampler
    
    Example:
        >>> def evaluator(params):
        ...     # Run walk-forward validation with params
        ...     return mean_squared_error
        >>> 
        >>> optimizer = BayesianMetaOptimizer(evaluator)
        >>> result = optimizer.optimize(n_trials=50)
        >>> best_config = result.best_params
    """
    
    def __init__(
        self,
        walk_forward_evaluator: Callable[[Dict[str, Any]], float],
        meta_config: Optional[MetaOptimizationConfig] = None,
        base_config: Optional[PredictorConfig] = None,
    ):
        """
        Initialize meta-optimizer.
        
        Args:
            walk_forward_evaluator: Function that takes params dict and returns
                generalization error (lower is better). Must ensure causal
                evaluation (no look-ahead).
            meta_config: Search space configuration
            base_config: Base PredictorConfig to extend with optimized params
        
        Raises:
            ImportError: If optuna is not installed
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for meta-optimization. "
                "Install with: pip install optuna>=3.0"
            )
        
        self.evaluator = walk_forward_evaluator
        self.meta_config = meta_config or load_meta_optimization_config()
        self.base_config = base_config
        self.study: Optional[Any] = None
    
    def _objective(self, trial: Any) -> float:
        """
        Objective function for Optuna optimization.
        
        Defines the search space mapping to PredictorConfig parameters.
        Returns generalization error (to be minimized).
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Generalization error from walk-forward validation
        """
        # Structural parameters (discrete)
        log_sig_depth = trial.suggest_int(
            "log_sig_depth",
            self.meta_config.log_sig_depth_min,
            self.meta_config.log_sig_depth_max,
        )
        
        wtmm_buffer_size = trial.suggest_int(
            "wtmm_buffer_size",
            self.meta_config.wtmm_buffer_size_min,
            self.meta_config.wtmm_buffer_size_max,
            step=self.meta_config.wtmm_buffer_size_step,
        )
        
        besov_cone_c = trial.suggest_float(
            "besov_cone_c",
            self.meta_config.besov_cone_c_min,
            self.meta_config.besov_cone_c_max,
        )
        
        # Sensitivity parameters (continuous)
        cusum_k = trial.suggest_float(
            "cusum_k",
            self.meta_config.cusum_k_min,
            self.meta_config.cusum_k_max,
        )
        
        sinkhorn_alpha = trial.suggest_float(
            "sinkhorn_alpha",
            self.meta_config.sinkhorn_alpha_min,
            self.meta_config.sinkhorn_alpha_max,
        )
        
        volatility_alpha = trial.suggest_float(
            "volatility_alpha",
            self.meta_config.volatility_alpha_min,
            self.meta_config.volatility_alpha_max,
        )
        
        # Assemble candidate parameter dictionary (Fast Tuning baseline)
        candidate_params = {
            "log_sig_depth": log_sig_depth,
            "wtmm_buffer_size": wtmm_buffer_size,
            "besov_cone_c": besov_cone_c,
            "cusum_k": cusum_k,
            "sinkhorn_alpha": sinkhorn_alpha,
            "volatility_alpha": volatility_alpha,
        }
        
        # ═══════════════════════════════════════════════════════════════
        # DEEP TUNING: Add 14+ structural parameters
        # ═══════════════════════════════════════════════════════════════
        if self.meta_config.enable_deep_tuning:
            # DGM Architecture
            dgm_width_size = trial.suggest_int(
                "dgm_width_size",
                self.meta_config.dgm_width_size_min,
                self.meta_config.dgm_width_size_max,
                step=self.meta_config.dgm_width_size_step,
            )
            dgm_depth = trial.suggest_int(
                "dgm_depth",
                self.meta_config.dgm_depth_min,
                self.meta_config.dgm_depth_max,
            )
            dgm_entropy_num_bins = trial.suggest_int(
                "dgm_entropy_num_bins",
                self.meta_config.dgm_entropy_num_bins_min,
                self.meta_config.dgm_entropy_num_bins_max,
            )
            
            # SDF Solver Thresholds
            stiffness_low = trial.suggest_float(
                "stiffness_low",
                self.meta_config.stiffness_low_min,
                self.meta_config.stiffness_low_max,
            )
            stiffness_high = trial.suggest_float(
                "stiffness_high",
                self.meta_config.stiffness_high_min,
                self.meta_config.stiffness_high_max,
            )
            
            # SDE Integration
            sde_dt = trial.suggest_float(
                "sde_dt",
                self.meta_config.sde_dt_min,
                self.meta_config.sde_dt_max,
                log=True,  # Log-uniform sampling
            )
            sde_numel_integrations = trial.suggest_int(
                "sde_numel_integrations",
                self.meta_config.sde_numel_integrations_min,
                self.meta_config.sde_numel_integrations_max,
            )
            sde_diffusion_sigma = trial.suggest_float(
                "sde_diffusion_sigma",
                self.meta_config.sde_diffusion_sigma_min,
                self.meta_config.sde_diffusion_sigma_max,
            )
            
            # Kernel Ridge Regularization
            kernel_ridge_lambda = trial.suggest_float(
                "kernel_ridge_lambda",
                self.meta_config.kernel_ridge_lambda_min,
                self.meta_config.kernel_ridge_lambda_max,
                log=True,  # Log-uniform sampling
            )
            
            # CUSUM Extended
            cusum_h = trial.suggest_float(
                "cusum_h",
                self.meta_config.cusum_h_min,
                self.meta_config.cusum_h_max,
            )
            cusum_grace_period_steps = trial.suggest_int(
                "cusum_grace_period_steps",
                self.meta_config.cusum_grace_period_steps_min,
                self.meta_config.cusum_grace_period_steps_max,
            )
            
            # Sinkhorn Extended
            sinkhorn_epsilon_min = trial.suggest_float(
                "sinkhorn_epsilon_min",
                self.meta_config.sinkhorn_epsilon_min_min,
                self.meta_config.sinkhorn_epsilon_min_max,
                log=True,
            )
            sinkhorn_epsilon_0 = trial.suggest_float(
                "sinkhorn_epsilon_0",
                self.meta_config.sinkhorn_epsilon_0_min,
                self.meta_config.sinkhorn_epsilon_0_max,
            )
            
            # JKO Wasserstein Flow
            learning_rate = trial.suggest_float(
                "learning_rate",
                self.meta_config.learning_rate_min,
                self.meta_config.learning_rate_max,
                log=True,
            )
            entropy_window = trial.suggest_int(
                "entropy_window",
                self.meta_config.entropy_window_min,
                self.meta_config.entropy_window_max,
            )
            entropy_threshold = trial.suggest_float(
                "entropy_threshold",
                self.meta_config.entropy_threshold_min,
                self.meta_config.entropy_threshold_max,
            )
            
            # Holder Exponent Threshold
            holder_threshold = trial.suggest_float(
                "holder_threshold",
                self.meta_config.holder_threshold_min,
                self.meta_config.holder_threshold_max,
            )
            
            # Add to candidate params
            candidate_params.update({
                "dgm_width_size": dgm_width_size,
                "dgm_depth": dgm_depth,
                "dgm_entropy_num_bins": dgm_entropy_num_bins,
                "stiffness_low": stiffness_low,
                "stiffness_high": stiffness_high,
                "sde_dt": sde_dt,
                "sde_numel_integrations": sde_numel_integrations,
                "sde_diffusion_sigma": sde_diffusion_sigma,
                "kernel_ridge_lambda": kernel_ridge_lambda,
                "cusum_h": cusum_h,
                "cusum_grace_period_steps": cusum_grace_period_steps,
                "sinkhorn_epsilon_min": sinkhorn_epsilon_min,
                "sinkhorn_epsilon_0": sinkhorn_epsilon_0,
                "learning_rate": learning_rate,
                "entropy_window": entropy_window,
                "entropy_threshold": entropy_threshold,
                "holder_threshold": holder_threshold,
            })
        
        # Evaluate strictly via Walk-Forward to prevent look-ahead bias
        generalization_error = self.evaluator(candidate_params)
        
        return generalization_error

    def export_best_params_to_config(
        self,
        config_path: str,
        trigger: str = "MetaOptimization",
        rate_limiter: Optional[MutationRateLimiter] = None,
    ) -> None:
        """
        Export best parameters to config.toml using atomic mutation protocol.

        Args:
            config_path: Path to config.toml
            trigger: Audit log trigger label
            rate_limiter: Optional MutationRateLimiter for guardrails

        Raises:
            ValueError: If optimization has not run or mapping missing
            ConfigMutationError: If rate limiting blocks mutation
        """
        if self.study is None:
            raise ValueError("No optimization results available. Run optimize() first.")

        best_params = dict(self.study.best_params)
        dotted_params: Dict[str, Any] = {}
        for key, value in best_params.items():
            if key not in FIELD_TO_SECTION_MAP:
                raise ValueError(f"Missing FIELD_TO_SECTION_MAP entry for '{key}'")
            section = FIELD_TO_SECTION_MAP[key]
            dotted_params[f"{section}.{key}"] = value

        delta: Dict[str, tuple[float, float]] = {}
        if rate_limiter is not None:
            can_mutate, reason = rate_limiter.can_mutate()
            if not can_mutate:
                raise ConfigMutationError(f"Mutation blocked: {reason}")

            with open(config_path, "rb") as config_file:
                current_config = tomllib.load(config_file)
            for dotted_key, new_value in dotted_params.items():
                parts = dotted_key.split(".")
                current = current_config
                for part in parts:
                    current = current[part]
                current_val = _coerce_float(current, dotted_key)
                new_val = _coerce_float(new_value, dotted_key)
                delta[dotted_key] = (current_val, new_val)

            valid, reason = rate_limiter.validate_delta(delta)
            if not valid:
                raise ConfigMutationError(f"Delta rejected: {reason}")

        atomic_write_config(
            Path(config_path),
            dotted_params,
            trigger=trigger,
            best_objective=float(self.study.best_value),
        )

        if rate_limiter is not None:
            rate_limiter.record_mutation(delta)
    
    def optimize(
        self,
        n_trials: Optional[int] = None,
        direction: str = "minimize",
    ) -> OptimizationResult:
        """
        Run Gaussian Process optimization to find optimal system personality.
        
        Args:
            n_trials: Number of optimization trials (default: from meta_config)
            direction: "minimize" or "maximize" (default: minimize error)
        
        Returns:
            OptimizationResult with best parameters and study object
        
        Example:
            >>> result = optimizer.optimize(n_trials=100)
            >>> print(f"Best MSE: {result.best_value:.6f}")
            >>> print(f"Best params: {result.best_params}")
        """
        if optuna is None or TPESampler is None:
            raise ImportError(
                "Optuna is required for meta-optimization. "
                "Install with: pip install optuna>=3.0"
            )

        # COMPLIANCE: Zero-Heuristics - explicit validation of n_trials, no or-pattern fallback
        if n_trials is None:
            if self.meta_config is None:
                raise ValueError(
                    "n_trials is None and meta_config is not available. "
                    "Zero-Heuristics policy forbids implicit fallback patterns."
                )
            n_trials = self.meta_config.n_trials
        if n_trials <= 0:
            raise ValueError(f"n_trials must be > 0, got {n_trials}")
        
        # Create Optuna study with TPE sampler (multivariate Gaussian Process)
        # COMPLIANCE: Zero-Heuristics - explicit prng_seed validation, no default fallback
        if self.base_config is not None and self.base_config.prng_seed is not None:
            seed = self.base_config.prng_seed
        else:
            # Fallback: must read explicit config value, not silent default
            core_section = get_config().get_section("core")
            if "prng_seed" not in core_section:
                raise ValueError(
                    "Missing required config: [core].prng_seed. "
                    "Zero-Heuristics policy forbids silent defaults."
                )
            seed = core_section["prng_seed"]
        sampler = TPESampler(
            multivariate=self.meta_config.multivariate,
            n_startup_trials=self.meta_config.n_startup_trials,
            seed=seed,
        )
        
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            study_name="USP_MetaOptimization",
        )
        assert self.study is not None
        
        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            show_progress_bar=True,
        )
        
        return OptimizationResult(
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            n_trials=len(self.study.trials),
            study=self.study,
        )
    
    def get_best_config(self) -> Optional[PredictorConfig]:
        """
        Generate PredictorConfig with optimized parameters.
        
        Returns:
            PredictorConfig with best parameters, or None if not optimized yet
        """
        if self.study is None or self.base_config is None:
            return None
        
        from dataclasses import replace
        
        return replace(
            self.base_config,
            **self.study.best_params
        )
    
    def save_study(self, path: str) -> None:
        """
        Save TPE checkpoint with SHA-256 integrity verification.
        
        Enables resumable Deep Tuning campaigns (500 trials over weeks).
        Pickle serialization + SHA-256 hash stored as .sha256 sidecar file.
        
        Args:
            path: Output path for checkpoint (e.g., "io/snapshots/study_campaign_001.pkl")
        
        Raises:
            ValueError: If study is None (no optimization run yet)
        
        Example:
            >>> optimizer.optimize(n_trials=50)
            >>> optimizer.save_study("io/snapshots/baseline_tuning.pkl")
            # Creates: baseline_tuning.pkl + baseline_tuning.pkl.sha256
        
        References:
            - Implementation.tex §5.4.2: TPE Checkpoint Protocol
        """
        if self.study is None:
            raise ValueError("Cannot save study: no optimization has run yet")
        
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize study with pickle
        checkpoint_bytes = pickle.dumps(self.study)
        
        # Compute SHA-256 integrity hash
        sha256_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        
        # Write checkpoint file
        with open(checkpoint_path, "wb") as f:
            f.write(checkpoint_bytes)
        
        # Write integrity hash as sidecar file
        sha256_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
        with open(sha256_path, "w") as f:
            f.write(sha256_hash)
    
    @classmethod
    def load_study(
        cls,
        path: str,
        walk_forward_evaluator: Callable[[Dict[str, Any]], float],
        meta_config: Optional[MetaOptimizationConfig] = None,
        base_config: Optional[PredictorConfig] = None,
    ) -> "BayesianMetaOptimizer":
        """
        Load TPE checkpoint with SHA-256 integrity verification.
        
        Enables resuming Deep Tuning campaigns after interruption.
        Verifies SHA-256 hash before deserializing to prevent corrupted state.
        
        Args:
            path: Checkpoint file path (e.g., "io/snapshots/study_campaign_001.pkl")
            walk_forward_evaluator: Evaluator function for resuming optimization
            meta_config: Search space configuration
            base_config: Base PredictorConfig
        
        Returns:
            BayesianMetaOptimizer instance with loaded study
        
        Raises:
            FileNotFoundError: If checkpoint or .sha256 file missing
            IntegrityError: If SHA-256 verification fails
        
        Example:
            >>> optimizer = BayesianMetaOptimizer.load_study(
            ...     "io/snapshots/baseline_tuning.pkl",
            ...     evaluator_func
            ... )
            >>> optimizer.optimize(n_trials=50)  # Resume from checkpoint
        
        References:
            - Implementation.tex §5.4.2: TPE Checkpoint Protocol
        """
        checkpoint_path = Path(path)
        sha256_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
        
        # Ensure both checkpoint and hash file exist
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not sha256_path.exists():
            raise FileNotFoundError(f"Integrity hash not found: {sha256_path}")
        
        # Read checkpoint bytes
        with open(checkpoint_path, "rb") as f:
            checkpoint_bytes = f.read()
        
        # Read expected hash
        with open(sha256_path, "r") as f:
            expected_hash = f.read().strip()
        
        # Verify integrity
        actual_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        if actual_hash != expected_hash:
            raise IntegrityError(
                f"Checkpoint integrity verification failed:\n"
                f"  Expected SHA-256: {expected_hash}\n"
                f"  Actual SHA-256:   {actual_hash}\n"
                f"  File: {checkpoint_path}"
            )
        
        # Deserialize study
        study = pickle.loads(checkpoint_bytes)
        
        # Create optimizer instance
        optimizer = cls(
            walk_forward_evaluator=walk_forward_evaluator,
            meta_config=meta_config,
            base_config=base_config,
        )
        optimizer.study = study
        
        return optimizer
    
    def generate_optimization_report(self) -> str:
        """
        Generate human-readable optimization summary with parameter importance.
        
        COMPLIANCE: V-MIN-2 - Actionable insights from meta-optimization
        
        Returns:
            Formatted report with:
                - Best hyperparameters
                - Objective value
                - Parameter importance ranking (via fANOVA if available)
                - Convergence status
                - Trial count
        
        Example:
            >>> optimizer.optimize(n_trials=100)
            >>> report = optimizer.generate_optimization_report()
            >>> print(report)
            ================================================================================
            Meta-Optimization Summary
            ================================================================================
            Study Name: USP_MetaOptimization
            Tier: deep_tuning
            Total Trials: 100
            Best Value: 0.004512
            
            Best Hyperparameters:
              log_sig_depth                  = 4
              wtmm_buffer_size               = 256
              dgm_width_size                 = 128
              dgm_depth                      = 4
              ...
            
            Parameter Importance (fANOVA):
              log_sig_depth                  0.4523
              dgm_depth                      0.2341
              wtmm_buffer_size               0.1245
              ...
            ================================================================================
        
        References:
            - AUDIT_SPEC_COMPLIANCE_2026-02-19.md: V-MIN-2
        """
        if self.study is None:
            return "No optimization run yet. Call optimize() first."
        
        report = []
        report.append("=" * 80)
        report.append("Meta-Optimization Summary")
        report.append("=" * 80)
        report.append(f"Study Name: {self.study.study_name}")
        
        # Determine tier from study structure
        if len(self.study.best_params) <= 6:
            tier = "fast_tuning"
        else:
            tier = "deep_tuning"
        report.append(f"Tier: {tier}")
        
        report.append(f"Total Trials: {len(self.study.trials)}")
        report.append(f"Best Value: {self.study.best_value:.6f}")
        report.append("")
        report.append("Best Hyperparameters:")
        
        # Sort parameters alphabetically for consistency
        for param, value in sorted(self.study.best_params.items()):
            if isinstance(value, float):
                value_str = f"{value:.6f}"
            else:
                value_str = str(value)
            report.append(f"  {param:30s} = {value_str}")
        
        # Parameter importance (requires optuna.importance module)
        try:
            if optuna is None:
                raise ImportError("optuna is not available")
            importance = optuna.importance.get_param_importances(self.study)
            
            report.append("")
            report.append("Parameter Importance (fANOVA):")
            report.append("  (Shows relative contribution to objective variance)")
            report.append("")
            
            # Sort by importance descending, show top 10
            sorted_importance = sorted(importance.items(), key=lambda x: -x[1])[:10]
            for param, score in sorted_importance:
                report.append(f"  {param:30s} {score:.4f}")
        
        except Exception:
            # Importance calculation may fail for small n_trials
            report.append("")
            report.append("Parameter Importance: Not available (requires ≥20 completed trials)")
        
        report.append("=" * 80)
        return "\n".join(report)


def walk_forward_split(
    data_length: int,
    train_ratio: float = 0.7,
    n_folds: int = 5,
    data: Optional[np.ndarray] = None,
    stratify_by_volatility: bool = False,
    min_fold_size: int = 100,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward validation splits (strictly causal).
    
    COMPLIANCE: V-MAJ-8 - Walk-Forward Split Stratification
    Implementation.tex §5.3 - Causal Cross-Validation
    
    Ensures no look-ahead bias by always training on past data and
    validating on future data. Optionally stratifies by volatility regime
    to ensure validation folds span diverse market conditions.
    
    Args:
        data_length: Total length of time series
        train_ratio: Initial training set ratio (default 0.7)
        n_folds: Number of validation folds (default 5)
        data: Time series data (required if stratify_by_volatility=True)
        stratify_by_volatility: If True, ensure each fold contains diverse
                                volatility samples (default False)
        min_fold_size: Minimum samples per fold; skip smaller folds (default 100)
    
    Returns:
        List of (train_indices, val_indices) tuples
    
    Stratification Criteria (if enabled):
        - Each validation fold must contain ≥10% samples from:
            * Low volatility regime (σ² < 33rd percentile)
            * Medium volatility regime (33rd ≤ σ² ≤ 67th percentile)
            * High volatility regime (σ² > 67th percentile)
        - If stratification fails, falls back to non-stratified split with warning
    
    Example:
        >>> # Non-stratified (legacy behavior)
        >>> splits = walk_forward_split(1000, train_ratio=0.7, n_folds=5)
        >>> 
        >>> # Stratified by volatility
        >>> data = np.random.randn(1000)
        >>> splits = walk_forward_split(
        ...     data_length=1000,
        ...     data=data,
        ...     stratify_by_volatility=True,
        ...     min_fold_size=100
        ... )
        >>> for train_idx, val_idx in splits:
        ...     # Each validation fold contains diverse volatility samples
        ...     pass
    
    References:
        - Implementation.tex §5.3: Causal Cross-Validation (Walk-Forward)
        - MIGRATION_AUTOTUNING_v1.0.md §4: Walk-Forward Isolation
        - AUDIT_SPEC_COMPLIANCE_2026-02-19.md: V-MAJ-8
    """
    initial_train_size = int(data_length * train_ratio)
    fold_size = (data_length - initial_train_size) // n_folds
    
    # Validate fold size
    if fold_size < min_fold_size:
        raise ValueError(
            f"Fold size {fold_size} < minimum {min_fold_size}. "
            f"Reduce n_folds or increase data_length. "
            f"Current: data_length={data_length}, n_folds={n_folds}, "
            f"train_ratio={train_ratio:.2f}"
        )
    
    # Compute volatility tertiles if stratification requested
    volatility = None
    volatility_tertiles = None
    
    if stratify_by_volatility:
        if data is None:
            raise ValueError(
                "stratify_by_volatility=True requires data argument"
            )
        
        # Compute rolling volatility (20-period window)
        volatility = compute_rolling_volatility(data, window=20)
        volatility_tertiles = np.percentile(volatility, [33, 67])
    
    splits = []
    for i in range(n_folds):
        train_end = initial_train_size + i * fold_size
        val_start = train_end
        val_end = min(val_start + fold_size, data_length)
        
        train_indices = np.arange(0, train_end)
        val_indices = np.arange(val_start, val_end)
        
        # Skip too-small validation sets
        if len(val_indices) < min_fold_size:
            continue
        
        # Validate stratification if requested
        if stratify_by_volatility and volatility is not None and volatility_tertiles is not None:
            val_volatility = volatility[val_indices]
            
            # Count samples in each regime
            low_vol_count = np.sum(val_volatility < volatility_tertiles[0])
            mid_vol_count = np.sum(
                (val_volatility >= volatility_tertiles[0]) & 
                (val_volatility <= volatility_tertiles[1])
            )
            high_vol_count = np.sum(val_volatility > volatility_tertiles[1])
            
            # Require at least 10% representation from each regime
            total_val = len(val_indices)
            low_pct = low_vol_count / total_val
            mid_pct = mid_vol_count / total_val
            high_pct = high_vol_count / total_val
            
            min_representation = 0.10
            
            if (low_pct < min_representation or 
                mid_pct < min_representation or 
                high_pct < min_representation):
                # Stratification failed for this fold - emit warning but include anyway
                import warnings
                warnings.warn(
                    f"Fold {i}: Insufficient volatility diversity "
                    f"(low={low_pct:.1%}, mid={mid_pct:.1%}, high={high_pct:.1%}). "
                    f"Required ≥{min_representation:.0%} per regime. Including fold anyway.",
                    UserWarning
                )
        
        splits.append((train_indices, val_indices))
    
    return splits


def compute_rolling_volatility(
    data: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """
    Compute rolling volatility (standard deviation) for stratification.
    
    COMPLIANCE: V-MAJ-8 Helper Function
    
    Args:
        data: Time series data
        window: Rolling window size (default 20)
    
    Returns:
        Rolling volatility array (same length as data, initial values filled)
    
    Example:
        >>> data = np.random.randn(1000)
        >>> volatility = compute_rolling_volatility(data, window=20)
        >>> volatility.shape  # (1000,)
    """
    # Pad with zeros for initial window
    volatility = np.zeros(len(data))
    
    for i in range(window, len(data)):
        window_data = data[i - window:i]
        volatility[i] = np.std(window_data)
    
    # Fill initial values with first computed volatility
    if len(data) >= window:
        volatility[:window] = volatility[window]
    
    return volatility


class AsyncMetaOptimizer:
    """
    Asynchronous wrapper for BayesianMetaOptimizer I/O operations.
    
    Prevents checkpoint writes from blocking telemetry emission and main
    compute thread. Uses ThreadPoolExecutor for non-blocking save_study()
    and load_study() operations.
    
    COMPLIANCE: V-CRIT-3 - Async I/O wrapper for checkpoint persistence
    
    Example:
        >>> async_optimizer = AsyncMetaOptimizer(
        ...     walk_forward_evaluator,
        ...     max_workers=2
        ... )
        >>> result = async_optimizer.optimize(n_trials=50)
        >>> 
        >>> # Non-blocking save
        >>> future = async_optimizer.save_study_async("io/snapshots/study.pkl")
        >>> # Continue compute work while save happens in background
        >>> run_telemetry_emission()
        >>> future.result()  # Wait for completion when needed
    
    References:
        - AUDIT_SPEC_COMPLIANCE_2026-02-19.md: V-CRIT-3
        - Implementation.tex §5.4.2: Async checkpoint protocol
    """
    
    def __init__(
        self,
        walk_forward_evaluator: Callable[[Dict[str, Any]], float],
        meta_config: Optional[MetaOptimizationConfig] = None,
        base_config: Optional[PredictorConfig] = None,
        max_workers: int = 2,
    ):
        """
        Initialize async meta-optimizer wrapper.
        
        Args:
            walk_forward_evaluator: Evaluator function for optimization
            meta_config: Search space configuration
            base_config: Base PredictorConfig
            max_workers: ThreadPoolExecutor worker count (default: 2)
        """
        self.optimizer = BayesianMetaOptimizer(
            walk_forward_evaluator,
            meta_config,
            base_config,
        )
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_saves: list[Future] = []
    
    def optimize(
        self,
        n_trials: Optional[int] = None,
        direction: str = "minimize",
    ) -> OptimizationResult:
        """
        Run optimization (delegates to underlying BayesianMetaOptimizer).
        
        Args:
            n_trials: Number of optimization trials
            direction: "minimize" or "maximize"
        
        Returns:
            OptimizationResult with best parameters
        """
        return self.optimizer.optimize(n_trials, direction)
    
    def get_best_config(self) -> Optional[PredictorConfig]:
        """Generate PredictorConfig with optimized parameters."""
        return self.optimizer.get_best_config()
    
    def save_study_async(self, path: str) -> Future:
        """
        Save TPE checkpoint asynchronously (non-blocking).
        
        Submits save operation to thread pool, allowing compute thread
        to continue without blocking on disk I/O.
        
        Args:
            path: Output path for checkpoint
        
        Returns:
            Future object for save operation status
        
        Example:
            >>> future = async_optimizer.save_study_async("io/study.pkl")
            >>> # Continue compute work...
            >>> future.result()  # Wait for completion
        """
        future = self.executor.submit(self.optimizer.save_study, path)
        self._pending_saves.append(future)
        return future
    
    def wait_all_saves(self, timeout: Optional[float] = None) -> None:
        """
        Wait for all pending save operations to complete.
        
        Args:
            timeout: Maximum wait time in seconds (None = wait forever)
        
        Raises:
            TimeoutError: If timeout exceeded
        """
        for future in self._pending_saves:
            future.result(timeout=timeout)
        self._pending_saves.clear()
    
    @classmethod
    def load_study_async(
        cls,
        path: str,
        walk_forward_evaluator: Callable[[Dict[str, Any]], float],
        meta_config: Optional[MetaOptimizationConfig] = None,
        base_config: Optional[PredictorConfig] = None,
        max_workers: int = 2,
    ) -> Future:
        """
        Load TPE checkpoint asynchronously (returns Future).
        
        Args:
            path: Checkpoint file path
            walk_forward_evaluator: Evaluator function
            meta_config: Search space configuration
            base_config: Base PredictorConfig
            max_workers: ThreadPoolExecutor worker count
        
        Returns:
            Future[AsyncMetaOptimizer] resolving to loaded optimizer
        
        Example:
            >>> future = AsyncMetaOptimizer.load_study_async(
            ...     "io/study.pkl",
            ...     evaluator_func
            ... )
            >>> async_optimizer = future.result()
        """
        executor = ThreadPoolExecutor(max_workers=1)
        
        def _load():
            sync_optimizer = BayesianMetaOptimizer.load_study(
                path,
                walk_forward_evaluator,
                meta_config,
                base_config,
            )
            async_optimizer = cls(
                walk_forward_evaluator,
                meta_config,
                base_config,
                max_workers,
            )
            async_optimizer.optimizer = sync_optimizer
            return async_optimizer
        
        return executor.submit(_load)
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown thread pool executor.
        
        Args:
            wait: If True, wait for pending tasks to complete
        """
        self.executor.shutdown(wait=wait)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (auto-shutdown)."""
        self.shutdown(wait=True)


# Public API
__all__ = [
    "BayesianMetaOptimizer",
    "AsyncMetaOptimizer",
    "MetaOptimizationConfig",
    "OptimizationResult",
    "IntegrityError",
    "walk_forward_split",
    "compute_rolling_volatility",
]

