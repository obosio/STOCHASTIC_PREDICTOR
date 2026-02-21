"""Core Orchestration Layer

Orchestrates the prediction pipeline: SIA identification → multi-branch prediction → adaptive fusion.

Responsibilities:
  - SIA (System Identification Archive) execution
  - Branch routing (A, B, C, D kernel selection)
  - JKO Wasserstein fusion with volatility-coupled Sinkhorn
  - CUSUM regime change detection with grace period
  - Entropy monitoring and diagnostics

See: doc/latex/specification/Stochastic_Predictor_Python.tex §2 - Orchestration Layer

Key algorithms:
  - Dynamic SDE scheme transition: doc/latex/specification/Stochastic_Predictor_Theory.tex §2.3.3
  - Volatility-coupled Sinkhorn: doc/latex/specification/Stochastic_Predictor_Implementation.tex §2.4
  - CUSUM grace period: doc/latex/specification/Stochastic_Predictor_API_Python.tex §3.2
  - JAX stop_gradient optimization: doc/latex/specification/Stochastic_Predictor_Python.tex §3.1

Expected module structure:
  - orchestrator.py: Main orchestration logic
  - fusion.py: Wasserstein transport fusion (JKO + Sinkhorn)
  - sinkhorn.py: Volatility-coupled entropic regularization
  - meta_optimizer.py: Bayesian meta-optimization utilities
"""

from .fusion import FusionResult, fuse_kernel_outputs
from .meta_optimizer import (
    AsyncMetaOptimizer,
    BayesianMetaOptimizer,
    IntegrityError,
    MetaOptimizationConfig,
    OptimizationResult,
    walk_forward_split,
)
from .orchestrator import (
    OrchestrationResult,
    apply_host_architecture_scaling,
    compute_adaptive_jko_params,
    compute_adaptive_stiffness_thresholds,
    compute_entropy_ratio,
    initialize_batched_states,
    initialize_state,
    orchestrate_step,
    orchestrate_step_batch,
    scale_dgm_architecture,
)
from .sinkhorn import SinkhornResult, compute_sinkhorn_epsilon

__all__ = [
    "AsyncMetaOptimizer",
    "BayesianMetaOptimizer",
    "FusionResult",
    "IntegrityError",
    "MetaOptimizationConfig",
    "OptimizationResult",
    "OrchestrationResult",
    "SinkhornResult",
    "compute_adaptive_jko_params",
    "compute_adaptive_stiffness_thresholds",
    "apply_host_architecture_scaling",
    "compute_entropy_ratio",
    "compute_sinkhorn_epsilon",
    "fuse_kernel_outputs",
    "initialize_batched_states",
    "initialize_state",
    "orchestrate_step",
    "orchestrate_step_batch",
    "scale_dgm_architecture",
    "walk_forward_split",
]
