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
