"""Kernels - XLA Engines for Prediction Branches.

Pure XLA-compiled prediction engines for four branches of the system.

Architecture:
  - Branch A (Hilbert): RKHS projections for smooth processes
  - Branch B (Markov): Deep Generative Models via Fokker-Planck equations
  - Branch C (Ito/Levy): Differentiable SDE integration with dynamic scheme switching
  - Branch D (Rough Paths): Signature-based topology for low Holder regularity

See: doc/latex/specification/Stochastic_Predictor_Python.tex §2 - XLA Engine Layer

Each kernel must:
  - Accept time series and return predictions
  - Support JAX transformations (jit, vmap, grad)
  - Implement stop_gradient for non-trainable diagnostics
  - Reference exact theory section in docstrings

Branch specifications:
  - A (Hilbert): doc/latex/specification/Stochastic_Predictor_Python.tex (search for Branch A)
  - B (Fokker-Planck): doc/latex/specification/Stochastic_Predictor_Implementation.tex + doc/latex/specification/Stochastic_Predictor_Python.tex
  - C (Ito/Levy): doc/latex/specification/Stochastic_Predictor_Theory.tex §2.3.3 + doc/latex/specification/Stochastic_Predictor_Python.tex
  - D (Signatures): doc/latex/specification/Stochastic_Predictor_Python.tex + doc/latex/specification/Stochastic_Predictor_Theory.tex §5

Expected module structure:
  - kernel_a.py: Hilbert/RKHS kernel
  - kernel_b.py: Fokker-Planck/DGM kernel (Equinox Neural ODE)
  - kernel_c.py: Ito/Levy kernel (Diffrax SDE solver)
  - kernel_d.py: Signatures kernel (Signax)
  - base.py: Base classes and utilities
"""

from .base import (
    KernelOutput,
    PredictionKernel,
    apply_stop_gradient_to_diagnostics,
    validate_kernel_input,
    compute_signal_statistics,
    normalize_signal
)

from .kernel_a import (
    kernel_a_predict,
    gaussian_kernel,
    compute_gram_matrix,
    kernel_ridge_regression,
    create_embedding
)

from .kernel_b import (
    DGM_HJB_Solver,
    kernel_b_predict,
    compute_entropy_dgm,
    loss_hjb
)

from .kernel_c import (
    kernel_c_predict,
    solve_sde,
    drift_levy_stable,
    diffusion_levy
)

from .kernel_d import (
    kernel_d_predict,
    compute_log_signature,
    create_path_augmentation,
    predict_from_signature
)

__all__ = [
    # Base
    "KernelOutput",
    "PredictionKernel",
    "apply_stop_gradient_to_diagnostics",
    "validate_kernel_input",
    "compute_signal_statistics",
    "normalize_signal",
    
    # Kernel A (RKHS)
    "kernel_a_predict",
    "gaussian_kernel",
    "compute_gram_matrix",
    "kernel_ridge_regression",
    "create_embedding",
    
    # Kernel B (DGM)
    "DGM_HJB_Solver",
    "kernel_b_predict",
    "compute_entropy_dgm",
    "loss_hjb",
    
    # Kernel C (SDE)
    "kernel_c_predict",
    "solve_sde",
    "drift_levy_stable",
    "diffusion_levy",
    
    # Kernel D (Signatures)
    "kernel_d_predict",
    "compute_log_signature",
    "create_path_augmentation",
    "predict_from_signature",
]

