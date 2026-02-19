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

# ═══════════════════════════════════════════════════════════════════════════
# JAX CONFIGURATION (CRITICAL - Must execute before any JAX imports)
# ═══════════════════════════════════════════════════════════════════════════

import jax

# Enable float64 precision globally (MANDATORY per Python.tex §1.3)
# Required for:
# - Malliavin derivative stability (Kernel C)
# - Hölder exponent precision (Kernel A - WTMM)
# - Sinkhorn convergence under extreme conditions (JKO Orchestrator)
# - Path signature accuracy (Kernel D)
jax.config.update('jax_enable_x64', True)

# ═══════════════════════════════════════════════════════════════════════════
# PACKAGE METADATA
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "1.0.0-diamond-implementation"
__author__ = "Stochastic Predictor Development Consortium"

