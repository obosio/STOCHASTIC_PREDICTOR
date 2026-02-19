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
  - sia.py: System identification module
  - fusion.py: Wasserstein transport fusion (JKO + Sinkhorn)
  - cusum.py: Change point detection
  - entropy.py: Entropy monitoring and state vector
"""

from .fusion import FusionResult, fuse_kernel_outputs
from .orchestrator import OrchestrationResult, initialize_state, orchestrate_step
from .sinkhorn import SinkhornResult, compute_sinkhorn_epsilon
from .meta_optimizer import (
    BayesianMetaOptimizer,
    MetaOptimizationConfig,
    OptimizationResult,
    walk_forward_split,
)

__all__ = [
    "BayesianMetaOptimizer",
    "FusionResult",
    "MetaOptimizationConfig",
    "OptimizationResult",
    "OrchestrationResult",
    "SinkhornResult",
    "compute_sinkhorn_epsilon",
    "fuse_kernel_outputs",
    "initialize_state",
    "orchestrate_step",
    "walk_forward_split",
]
