"""Universal Stochastic Predictor - Root Package.

This package implements the complete specification for the Universal Stochastic Predictor,
a system for prediction of dynamic processes with unknown underlying probability law.

References:
  - doc/latex/specification/Stochastic_Predictor_Python.tex §2: Physical Directory Architecture
  - doc/latex/specification/Stochastic_Predictor_Theory.tex: Mathematical Foundations

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

import os

import jax

# ─────────────────────────────────────────────────────────────────────────────
# 1. NUMERICAL PRECISION (MANDATORY per Stochastic_Predictor_Python.tex §1.3, Stochastic_Predictor_API_Python.tex §5)
# ─────────────────────────────────────────────────────────────────────────────

# Enable float64 precision globally
# Required for:
# - Malliavin derivative stability (Kernel C - SDE)
# - Holder exponent precision (Kernel A - WTMM)
# - Sinkhorn convergence under extreme conditions (JKO Orchestrator, epsilon -> 0)
# - Path signature accuracy (Kernel D - rough paths with H < 0.5)
jax.config.update("jax_enable_x64", True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DETERMINISTIC EXECUTION (MANDATORY per Stochastic_Predictor_Tests_Python.tex §1.1)
# ─────────────────────────────────────────────────────────────────────────────

# Force threefry2x32 PRNG implementation for bit-exact parity
# Must be set BEFORE any JAX operations (prevents runtime warnings in prng.py)
os.environ["JAX_DEFAULT_PRNG_IMPL"] = "threefry2x32"

# Force deterministic reductions for hardware parity (CPU/GPU/TPU)
# Ensures bit-exact reproducibility across different backends
os.environ["JAX_DETERMINISTIC_REDUCTIONS"] = "1"

# XLA GPU deterministic operations
# Guarantees identical results across runs on GPU hardware
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPILATION CACHE (JIT Consistency)
# ─────────────────────────────────────────────────────────────────────────────

# Enable compilation cache to ensure JIT consistency across sessions
# Prevents recompilation overhead and guarantees identical XLA lowering
cache_dir = os.getenv("USP_JAX_CACHE_DIR", "/tmp/jax_cache")
jax.config.update("jax_compilation_cache_dir", cache_dir)

# ═══════════════════════════════════════════════════════════════════════════
# PACKAGE METADATA
# ═══════════════════════════════════════════════════════════════════════════

__version__ = "1.0.0-diamond-implementation"
__author__ = "Stochastic Predictor Development Consortium"
