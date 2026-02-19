"""Universal Stochastic Predictor - Root Package.

This package implements the complete specification for the Universal Stochastic Predictor,
a system for prediction of dynamic processes with unknown underlying probability law.

References:
  - doc/Predictor_Estocastico_Python.tex §2: Physical Directory Architecture
  - doc/Predictor_Estocastico_Teoria.tex: Mathematical Foundations

Architecture (5-Layer Clean Architecture):
  - api/: Exposure layer (facade, configuration, load shedding)
  - core/: Orchestration layer (JKO, Sinkhorn, entropy monitoring)
  - kernels/: XLA Motors (kernels A, B, C, D for prediction)
  - io/: Physical I/O layer (atomic snapshots, channel management)
  - tests/: External validation (outside main package)

Version: v1.0.0-Diamond-Spec
Date: February 18, 2026
"""

__version__ = "1.0.0-diamond-implementation"
__author__ = "Stochastic Predictor Development Consortium"

