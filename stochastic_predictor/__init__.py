"""Universal Stochastic Predictor - Root Package

This package implements the complete specification for the Universal Stochastic Predictor,
a system for prediction of dynamic processes with unknown underlying probability law.

See: doc/Predictor_Estocastico_Python.tex §2 - Arquitectura Física de Directorios

Architecture (5-Tier Clean):
  - api/: Exposure layer (facade, config, load shedding)
  - core/: Orchestration layer (JKO, Sinkhorn, entropy monitoring)
  - kernels/: XLA Motors (A,B,C,D kernels for prediction)
  - io/: Physical I/O layer (atomic snapshots, channel management)
  - tests/: External validation (outside main package)

Version: v1.0.0-Diamond-Spec
Date: 18 de febrero de 2026
"""

__version__ = "1.0.0-diamond-implementation"
__author__ = "Consorcio de Desarrollo de Meta-Predicción Adaptativa"
