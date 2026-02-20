# Universal Stochastic Predictor (USP)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Level%204%20Autonomy%20Complete-brightgreen.svg)
![Version](https://img.shields.io/badge/version-v2.2.0-brightgreen.svg)
![Release](https://img.shields.io/badge/release-Testing%20Phase-blue.svg)

## Description

Universal stochastic prediction system that operates on dynamic processes with unknown underlying probability laws.

This repository contains:

- Full technical specification: 7 LaTeX documents (3000+ lines, 1.73 MB PDFs)
- Implementation scaffold: validated 5-layer structure (Diamond level)
- Golden Master: strict dependency pinning (`==`)
- Implementation code: active development (branch `implementation/base-jax`)

## Branch Versioning

| Branch | Convention | Description |
| ---- | ---------- | ----------- |
| `main` | `spec/v1.x.x` | Specification phase (immutable, frozen) |
| `implementation/base-jax` | `impl/v2.x.x` | Implementation phases (active development) |

### Current Tags

**Specification**:

- `spec/v1.0.0` - Pure specification (v1.0.0-Specification equivalent)

**Implementation**:

- `impl/v2.0.0` - Initial 5-tier architecture scaffold (Bootstrap)
- `impl/v2.0.1` - Phase 1 Complete: Full API Layer (types, PRNG, validation, schemas, config)
- `impl/v2.0.2` - Phase 2 Complete: Kernels A, B, C, D (RKHS, DGM, SDE, Signatures)
- `impl/v2.0.3` - Phase 3 Complete: Core Orchestration (JKO, Sinkhorn)
- `impl/v2.0.4` - Phase 4 Complete: I/O Layer (ingestion, snapshots, telemetry, credentials)
- `impl/v2.1.0` - Auto-Tuning Phase: Level 4 Autonomy Implementation
- `test/v2.2.0` - Testing Phase: XLA Compliance & Golden Master Enforcement [CURRENT]
  - **Critical Fixes**: XLA control flow violations in Kernel C (ConcretizationTypeError resolution)
  - **Type System**: Full JAX Array/bool compatibility for vmap purity
  - **Golden Master**: Enforced OTT-JAX==0.4.5 for Optimal Transport
  - **Pydantic V2**: Full migration from @validator → @field_validator
  - **Ready for Testing**: All VSCode errors resolved, comprehensive audit passed
- `impl/v2.5.x` - Pending: Phase 5 - Tests and hardening
- `impl/v3.0.0` - Pending: First production-ready version

**Legacy** (for history):

- `v0.0.1-spec-complete`
- `v1.0.0-Specification`
- `v1.1.0-Implementation-Scaffold`

## Key Capabilities of the Specified System

### Multicore Architecture

1. **Identification Engine (SIA)**: topological characterization via WTMM, stationarity detection, Holder exponent estimation, entropy computation.

2. **Specialized Prediction Kernels**:
   - **Branch A (Hilbert)**: RKHS
   - **Branch B (Fokker-Planck)**: DGM/Neural ODEs
   - **Branch C (Ito/Levy)**: differentiable stochastic differential equations
   - **Branch D (Signatures)**: rough paths topological analysis

3. **Adaptive Orchestrator**: Wasserstein transport with JKO scheme, CUSUM change detection.

4. **Auto-Tuning System (v2.1.0 - Level 4 Autonomy)**:
   - **Layer 1**: Automatic entropy reset on regime change (max-entropy JKO restart)
   - **Layer 2**: Real-time parameter adaptation (kurtosis-coupled CUSUM, volatility-coupled Sinkhorn)
   - **Layer 3**: Bayesian meta-optimization (TPE/Optuna for structural hyperparameters)
   - **Level 4 Capabilities**:
     - Adaptive DGM architecture scaling (entropy-driven capacity adjustments)
     - Hölder-informed stiffness thresholds (rough path detection)
     - Regime-dependent JKO flow parameters (volatility-adaptive learning rate)
     - Configuration mutation safety guardrails (rate limiting, degradation auto-rollback)
     - Adaptive telemetry monitoring (solver frequency tracking, architecture scaling events)
   - Walk-forward validation with volatility stratification (no look-ahead bias)
   - Checkpoint persistence with SHA-256 integrity verification
   - Stop-gradient diagnostics (VRAM-optimized gradient flow)

## Specified Tech Stack

### Golden Master (Dependency Pinning Required)

```bash
JAX          == 0.4.20
Equinox      == 0.11.2
Diffrax      == 0.4.1
Signax       == 0.1.4
OTT-JAX      == 0.4.5
PyWavelets   == 1.4.1
Python       == 3.10.12
```

**Critical restriction**: versions must be pinned with `==`. No `>=` or `-U`. See [doc/latex/specification/Stochastic_Predictor_Python.tex](doc/latex/specification/Stochastic_Predictor_Python.tex).

### Required 5-Layer Architecture

For future implementations:

```bash
Python/
|-- api/          # Facade, config, load shedding
|-- core/         # JKO, Sinkhorn, monitoring
|-- kernels/      # XLA engines (A,B,C,D)
|-- io/           # Physical I/O, atomic snapshots
`-- tests/        # External validation
```

See [doc/latex/specification/Stochastic_Predictor_Python.tex](doc/latex/specification/Stochastic_Predictor_Python.tex).

### Security Policies

- Forbidden: hardcoded credentials
- Required: environment variable injection (`.env`)
- `.gitignore` rule: `.env`, `secrets/`, `*.log`

See [doc/latex/specification/Stochastic_Predictor_IO.tex](doc/latex/specification/Stochastic_Predictor_IO.tex).

### CI/CD Environment Validation

Before pytest, validate Golden Master:

```bash
EXPECTED_JAX=$(grep "^jax==" requirements.txt | cut -d'=' -f3)
ACTUAL_JAX=$(python -c "import jax; print(jax.__version__)")
[[ "$EXPECTED_JAX" == "$ACTUAL_JAX" ]] || exit 1
```

See [doc/latex/specification/Stochastic_Predictor_Tests_Python.tex](doc/latex/specification/Stochastic_Predictor_Tests_Python.tex).

## Documentation

Seven LaTeX documents compiled into PDFs in `doc/pdf/specification/`:

| Document | Lines | Content |
| -------- | ----- | ------- |
| Theory.tex | 500+ | Mathematical foundations, stochastic processes, optimal transport |
| Implementation.tex | 800+ | Algorithms, Sinkhorn dynamics coupled to volatility |
| Python.tex | 1700+ | JAX/Python stack, 5-layer architecture, technical specs |
| API_Python.tex | 685+ | High-level API, CUSUM grace period |
| IO.tex | 292+ | I/O interface, security policies |
| Tests_Python.tex | 1623+ | Test suite, CI/CD validation, environment |
| Test_Cases.tex | 400+ | Additional test cases |

### Compilation (Automatic)

The `compile.sh` script automatically detects and compiles all LaTeX source files:

```bash
cd doc

# Show options
./compile.sh help

# Compile documents with changes
./compile.sh --all

# Force full rebuild (ignore timestamps)
./compile.sh --all --force

# Compile a specific document
./compile.sh Stochastic_Predictor_Python.tex

# Clean build artifacts
./compile.sh clean
```

**Automatic structure:**

- Source: `latex/specification/*.tex` -> Output: `pdf/specification/*.pdf`
- The script is folder-agnostic and works with any subfolder under `latex/`

For details, see [doc/README.md](doc/README.md).

## Current Status

### PHASE: v2.1.0 Complete - Level 4 Autonomy

**Active branch**: `implementation/base-jax`
**Current tag**: `impl/v2.1.0` (pending commit)
**Date**: 19 Feb 2026

Completed (Phases 1-4 + Level 4 Autonomy):

- 7 LaTeX specification documents (1.73 MB PDFs)
- 5-layer structure implemented (`api/`, `core/`, `kernels/`, `io/`, `tests/`)
- **API layer materialized**:
  - `types.py` (544 lines): PredictorConfig, InternalState (extended with Level 4 telemetry counters)
  - `prng.py` (301 lines): JAX threefry2x32 deterministic PRNG management
  - `validation.py` (467 lines): input/output domain validation
  - `schemas.py` (330 lines): Pydantic models for serialization
  - `config.py` (433 lines): ConfigManager singleton with hot-reload support
  - `state_buffer.py`, `warmup.py`: stop_gradient on buffer stats to reduce VRAM
- **Kernels layer materialized**:
  - `kernels/base.py`: normalization and shared utilities
  - `kernel_a.py`: RKHS (Gaussian kernel ridge)
  - `kernel_b.py`: DGM PDE solver (HJB)
  - `kernel_c.py`: SDE integration (Levy)
  - `kernel_d.py`: path signatures
- **Core orchestration materialized**:
  - `core/sinkhorn.py`: Sinkhorn scan-based OT with volatility coupling
  - `core/fusion.py`: JKO fusion and free-energy tracking
  - `core/orchestrator.py`: state updates, degraded modes, **adaptive functions** (entropy ratio, DGM scaling, stiffness thresholds, JKO params)
  - `core/meta_optimizer.py`: BayesianMetaOptimizer with TPE, walk-forward stratification
- **IO layer materialized**:
  - `io/telemetry.py`: TelemetryBuffer, AdaptiveTelemetry collection
  - `io/loaders.py`, `io/validators.py`, `io/snapshots.py`, `io/credentials.py`
  - `io/config_mutation.py` (NEW): MutationRateLimiter, DegradationMonitor with audit trail
- **Level 4 Autonomy** (8/8 V-MAJ violations addressed):
  - V-MAJ-1: Adaptive DGM architecture (entropy-driven scaling) ✅
  - V-MAJ-2: Hölder-informed stiffness thresholds ✅
  - V-MAJ-3: Regime-dependent JKO flow parameters ✅
  - V-MAJ-4: Configuration mutation rate limiting ✅
  - V-MAJ-5: Degradation detection with auto-rollback ✅
  - V-MAJ-6: Checkpoint resumption tests (deferred to testing phase) ⏸️
  - V-MAJ-7: Adaptive telemetry monitoring ✅
  - V-MAJ-8: Walk-forward stratification (already compliant) ✅
- **Supporting tools**:
  - `examples/run_deep_tuning.py`: Deep Tuning example (500 trials, checkpoint resumption)
  - `scripts/migrate_config.py`: Config migration utility (v2.0.x → v2.1.0)
  - `benchmarks/bench_adaptive_vs_fixed.py`: Adaptive vs fixed hyperparameter comparison
  - `.github/workflows/test_meta_optimization.yml`: CI/CD regression tests
- Golden Master strict dependency pinning (`==`)
- Documentation: Implementation v2.1.0 (Bootstrap, API, Core, IO) in LaTeX
- Security policies (.env, .gitignore)
- Centralized config (config.toml) with locked parameter protection
- Full tech stack (JAX 0.4.20 + Equinox 0.11.2 + Diffrax 0.4.1 + Pydantic 2.0.0)
- 100% English code enforcement (language policy verified)

Deferred to future phases:

- Unit tests (testing phase)
- SIA engine full integration (Kernel A WTMM)
- Production deployment infrastructure
- Visualization dashboard (GAP-6)

### IO Guidelines (Phase 4)

- **TelemetryBuffer**: the orchestrator emits a telemetry buffer at the end of each step for out-of-thread consumption.
- **Deterministic logging**: record sha256 hashes of $\rho$ weights and OT cost at configurable intervals for CPU/GPU parity.
- **Snapshots**: MessagePack serialization with hash verification and atomic write-then-rename protocol.
- **Validation**: catastrophic outlier, frozen signal, and TTL staleness gate with degraded mode.

This repository is ready for active development with a validated scaffold and a rigorous specification baseline.

## Key Concepts

- **Multifractal Analysis (WTMM)**: local singularity detection
- **Adaptive Optimal Transport**: dynamic regularization coupled to volatility
- **Dynamic SDE Schemes**: automatic Euler -> implicit transition based on stiffness
- **Gradient Truncation**: XLA optimization for SIA/CUSUM (30-50% VRAM)
- **CUSUM Grace Period**: post-regime-change refractory period (10-60 steps)
- **Rough Paths Theory**: signatures for processes with $H \le 1/2$
- **Circuit Breaker**: protection when $H < H_{\min}$, activates Branch D

See LaTeX documents for full derivations and pseudocode.

## Contributions

This repository is a specification baseline. Contributions focus on:

- Specification improvements: corrections, clarifications, mathematical extensions
- Technical review: algorithm validation, inconsistency detection
- Future use: foundation for JAX implementations, other languages, etc.

See [CONTRIBUTING.md](CONTRIBUTING.md) before contributing.

## Authors

Adaptive Meta-Prediction Development Consortium

## License

[MIT License](LICENSE)

## Acknowledgments

Specification integrates JAX, Equinox, Diffrax, Signax, PyWavelets, OTT-JAX.

---

v2.1.0-Level4-Autonomy-Complete: Adaptive architecture, config mutation safety, meta-optimization
Guaranteed stack: JAX==0.4.20 | Equinox==0.11.2 | Diffrax==0.4.1 | Signax==0.1.4 | OTT-JAX==0.4.5 | Pydantic==2.0.0
Active branch: `implementation/base-jax` - Level 4 Autonomy complete (8/8 V-MAJ addressed, 5/6 GAP implemented)
