"""Bayesian Meta-Optimization (Learning to Learn).

Implements derivative-free hyperparameter search using Gaussian Processes (TPE)
to optimize structural parameters (log-signature depth, buffer sizes, etc.)
via walk-forward validation.

References:
    - MIGRATION_AUTOTUNING_v1.0.md §3.2: Capa 3 Meta-Optimization
    - Implementation.tex §8.3: Bayesian Hyperparameter Tuning
"""

from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore
    TPESampler = None  # type: ignore

from stochastic_predictor.api.types import PredictorConfig


@dataclass
class MetaOptimizationConfig:
    """Configuration for meta-optimization search space and constraints."""
    
    # Structural parameters (high impact, low frequency tuning)
    log_sig_depth_min: int = 2
    log_sig_depth_max: int = 5
    
    wtmm_buffer_size_min: int = 64
    wtmm_buffer_size_max: int = 512
    wtmm_buffer_size_step: int = 64
    
    besov_cone_c_min: float = 1.0
    besov_cone_c_max: float = 3.0
    
    # Sensitivity parameters (medium impact)
    cusum_k_min: float = 0.1
    cusum_k_max: float = 1.0
    
    sinkhorn_alpha_min: float = 0.1
    sinkhorn_alpha_max: float = 1.0
    
    volatility_alpha_min: float = 0.05
    volatility_alpha_max: float = 0.3
    
    # Optimization control
    n_trials: int = 50
    n_startup_trials: int = 10
    multivariate: bool = True
    
    # Walk-forward validation
    train_ratio: float = 0.7
    n_folds: int = 5


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
        self.meta_config = meta_config or MetaOptimizationConfig()
        self.base_config = base_config
        self.study: Optional[optuna.Study] = None
    
    def _objective(self, trial: optuna.Trial) -> float:
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
        
        # Assemble candidate parameter dictionary
        candidate_params = {
            "log_sig_depth": log_sig_depth,
            "wtmm_buffer_size": wtmm_buffer_size,
            "besov_cone_c": besov_cone_c,
            "cusum_k": cusum_k,
            "sinkhorn_alpha": sinkhorn_alpha,
            "volatility_alpha": volatility_alpha,
        }
        
        # Evaluate strictly via Walk-Forward to prevent look-ahead bias
        generalization_error = self.evaluator(candidate_params)
        
        return generalization_error
    
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
        n_trials = n_trials or self.meta_config.n_trials
        
        # Create Optuna study with TPE sampler (multivariate Gaussian Process)
        sampler = TPESampler(
            multivariate=self.meta_config.multivariate,
            n_startup_trials=self.meta_config.n_startup_trials,
            seed=42,  # For reproducibility
        )
        
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            study_name="USP_MetaOptimization",
        )
        
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


def walk_forward_split(
    data_length: int,
    train_ratio: float = 0.7,
    n_folds: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward validation splits (strictly causal).
    
    Ensures no look-ahead bias by always training on past data and
    validating on future data.
    
    Args:
        data_length: Total length of time series
        train_ratio: Initial training set ratio
        n_folds: Number of validation folds
    
    Returns:
        List of (train_indices, val_indices) tuples
    
    Example:
        >>> splits = walk_forward_split(1000, train_ratio=0.7, n_folds=5)
        >>> for train_idx, val_idx in splits:
        ...     # Train on past, validate on future
        ...     pass
    
    References:
        - MIGRATION_AUTOTUNING_v1.0.md §4: Walk-Forward Isolation
    """
    initial_train_size = int(data_length * train_ratio)
    fold_size = (data_length - initial_train_size) // n_folds
    
    splits = []
    for i in range(n_folds):
        train_end = initial_train_size + i * fold_size
        val_start = train_end
        val_end = min(val_start + fold_size, data_length)
        
        train_indices = np.arange(0, train_end)
        val_indices = np.arange(val_start, val_end)
        
        if len(val_indices) > 0:  # Skip empty validation sets
            splits.append((train_indices, val_indices))
    
    return splits


# Public API
__all__ = [
    "BayesianMetaOptimizer",
    "MetaOptimizationConfig",
    "OptimizationResult",
    "walk_forward_split",
]
