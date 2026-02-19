"""Kernels - XLA Motors for Prediction Branches

Pure XLA-compiled prediction engines for 4 branches of the system.

Architecture:
  - Rama A (Hilbert): RKHS projections for smooth processes
  - Rama B (Markov): Deep Generative Models via Fokker-Planck equations
  - Rama C (Itô/Lévy): Differentiable SDE integration with dynamic scheme switching
  - Rama D (Rough Paths): Signature-based topology for low Hölder regularity

See: doc/Predictor_Estocastico_Python.tex §2 - Capa de Motores XLA

Each kernel must:
  - Accept time series and return predictions
  - Support JAX transformations (jit, vmap, grad)
  - Implement stop_gradient for non-trainable diagnostics
  - Reference exact theory section in docstrings

Branch specifications:
  - A (Hilbert): doc/Predictor_Estocastico_Python.tex (searched for Rama A)
  - B (Fokker-Planck): doc/Predictor_Estocastico_Implementacion.tex + doc/Predictor_Estocastico_Python.tex
  - C (Itô/Lévy): doc/Predictor_Estocastico_Teoria.tex §2.3.3 + doc/Predictor_Estocastico_Python.tex
  - D (Signatures): doc/Predictor_Estocastico_Python.tex + doc/Predictor_Estocastico_Teoria.tex §5

Expected module structure:
  - kernel_a.py: Hilbert/RKHS kernel
  - kernel_b.py: Fokker-Planck/DGM kernel (Equinox Neural ODE)
  - kernel_c.py: Itô/Lévy kernel (Diffrax SDE solver)
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

