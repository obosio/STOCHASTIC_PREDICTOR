# Testing Plan for v2.1.0-RC1

**Branch**: `testing/v2.1.0-RC1`  
**Tag**: `impl/v2.1.0-RC1`  
**Status**: Ready for QA/Validation Phase  
**Date**: February 19, 2026

---

## Objective

Validate v2.1.0-RC1 (Release Candidate 1) across all critical dimensions before advancing to Phase 5 (comprehensive test suite) and v3.0.0 (production release).

**Pre-Testing Status**:

- ✅ 5 critical XLA/JAX bugs fixed
- ✅ Golden Master dependencies enforced
- ✅ Type system 100% aligned
- ✅ 13-point architecture audit passed
- ✅ Zero VSCode errors

---

## Testing Checklist

### 1. Functional Testing

#### 1.1 Core Modules

- [ ] **API Layer** (`stochastic_predictor/api/`)
  - [ ] `types.py`: All dataclasses instantiable (PredictorConfig, ProcessState, PredictionResult, InternalState)
  - [ ] `schemas.py`: Pydantic V2 validators execute (all @field_validator patterns)
  - [ ] `validation.py`: Input validation rejects outliers, NaN, Inf appropriately
  - [ ] `config.py`: Configuration loading from `config.toml` works
  - [ ] `prng.py`: PRNG key splitting yields deterministic sequences
  - [ ] `warmup.py`: JIT warmup completes without errors

#### 1.2 Kernels

- [ ] **Kernel A** (WTMM+Fokker-Planck)
  - [ ] `kernel_a_predict()` returns valid KernelOutput
  - [ ] Holder exponent computed [0, 1]
  - [ ] JIT compilation succeeds on 100-point signal
  
- [ ] **Kernel B** (DGM)
  - [ ] `kernel_b_predict()` entropy computation valid
  - [ ] Architecture scaling triggered on high entropy ratio (κ > 2.0)
  - [ ] DGM network forward pass completes

- [ ] **Kernel C** (SDE Solver)
  - [ ] **XLA Control Flow**: `solve_sde()` compiles under @jax.jit
  - [ ] **Stiffness Selection**: Euler/Heun/ImplicitEuler selection via jax.lax.cond works
  - [ ] **Solver Index**: Returns jnp.int32 (0, 1, 2) not string
  - [ ] Drift/diffusion functions respect 4-arity args tuple

- [ ] **Kernel D** (Signatures)
  - [ ] `kernel_d_predict()` creates path augmentation with float64 precision
  - [ ] Log-signature truncation depth 3 executes

#### 1.3 Core Orchestration

- [ ] **Sinkhorn**
  - [ ] Native OTT-JAX solver instantiates correctly
  - [ ] LinearProblem formulation with marginal constraints
  - [ ] Convergence check returns boolean
  - [ ] Transport matrix forms valid coupling

- [ ] **Fusion**
  - [ ] `fuse_kernel_outputs()` produces weight simplex [0.25, 0.25, 0.25, 0.25]
  - [ ] JKO cost computation valid

- [ ] **Orchestrator**
  - [ ] `orchestrate_step()` single prediction completes
  - [ ] `orchestrate_step_batch()` vmap execution on B=10 batch
  - [ ] Type annotations match actual returns (PyTree vs Array)

#### 1.4 I/O Layer

- [ ] **Ingestion**
  - [ ] `evaluate_ingestion()` flags outliers, frozen signals, stale data
  
- [ ] **Snapshots**
  - [ ] Atomic write+hash (SHA-256) succeeds
  - [ ] Serialization (msgpack) round-trips without loss

- [ ] **Telemetry**
  - [ ] Buffer enqueue/dequeue non-blocking
  - [ ] Parity hash computation

---

### 2. Type System Validation

#### 2.1 JAX/XLA Compatibility

- [ ] **VMap Compatibility**
  - [ ] `orchestrate_step_batch()` vmap-able with B=1..100 assets
  - [ ] PredictionResult fields properly typed for batching
  - [ ] Bool flags typed as `Bool[Array, "1"]` not Python bool

- [ ] **JIT Compilation**
  - [ ] All @jax.jit functions compile to XLA without error
  - [ ] No ConcretizationTypeError or AbstractTracer issues
  - [ ] Control flow via jax.lax.cond (not Python if)

#### 2.2 Type Annotations

- [ ] **Return Types**: orchestrate_step_batch returns `tuple[PredictionResult, InternalState]`
- [ ] **Array Types**: All numerical outputs use jaxtyping annotations
- [ ] **PRNG Keys**: PRNGKeyArray used consistently

---

### 3. Golden Master Compliance

#### 3.1 Dependency Enforcement

- [ ] **Exact Versions** (must be `==` not `>=`)
  - [ ] JAX == 0.4.20
  - [ ] OTT-JAX == 0.4.5 (enforced in Sinkhorn)
  - [ ] Diffrax == 0.4.1
  - [ ] Equinox == 0.11.2
  - [ ] Signax == 0.1.4
  - [ ] Python == 3.10.12

#### 3.2 Precision Enforcement

- [ ] **FP64 Enabled**
  - [ ] JAX config: `enable_x64=True`
  - [ ] All numerical arrays use float64
  - [ ] Signature/Malliavin calculations in float64

#### 3.3 Zero-Heuristics Policy

- [ ] All parameters from `config.toml` (no hardcoded values)
- [ ] No implicit defaults (except where explicitly stated)

---

### 4. Performance & Resource Testing

#### 4.1 Computation Speed

- [ ] Single orchestration step: < 10ms (CPU)
- [ ] Vmap batch (B=100): < 500ms (CPU)
- [ ] JIT compilation cache hit: < 1ms

#### 4.2 Memory Efficiency

- [ ] **VRAM Optimization**: `jax.lax.stop_gradient()` on diagnostics
- [ ] Batching doesn't leak gradients to unrelated dimensions
- [ ] No excessive allocations during vmap execution

#### 4.3 Numerical Stability

- [ ] Log-sumexp (Sinkhorn) doesn't NaN/Inf
- [ ] SDE solver adaptive stepping at high stiffness
- [ ] Lévy process path generation doesn't diverge

---

### 5. Edge Cases & Error Handling

#### 5.1 Signal Validation

- [ ] Very short signal (n=5): Rejected or handled gracefully
- [ ] Signal with NaN/Inf: Rejected (catastrophic outlier detection)
- [ ] Zero-variance signal: Handled without division by zero
- [ ] Stale signal (TTL violation): Degraded mode activation

#### 5.2 Numerical Edge Cases

- [ ] Extreme stiffness (ratio → ∞): ImplicitEuler selected
- [ ] Near-zero entropy: No log(0) in Sinkhorn
- [ ] Highly skewed Lévy α ← 0 or α → 2: SDE stable

#### 5.3 Batch Limits

- [ ] B=1 (single asset): Works
- [ ] B=1000 (large batch): Completes or reports memory limit gracefully

---

### 6. Backward Compatibility

- [ ] v2.0.4 → v2.1.0-RC1 upgrade: No breaking changes
- [ ] Configuration files (config.toml) still load
- [ ] Snapshot format: Compatible with v2.0.4 (if applicable)
- [ ] API signatures: Identical (type annotations only change)

---

### 7. Documentation Consistency

- [ ] CHANGELOG.md reflects all 5 fixes
- [ ] RELEASE_NOTES.md user-friendly and accurate
- [ ] README.md version badge shows RC1
- [ ] [TESTING.md](TESTING.md) this file exists and is current

---

## Test Environment Setup

```bash
# Clone testing branch
git clone -b testing/v2.1.0-RC1 https://github.com/obosio/STOCHASTIC_PREDICTOR.git
cd STOCHASTIC_PREDICTOR

# Install
pip install -e .

# Verify dependencies (exact versions)
pip freeze | grep -E "jax|ott-jax|diffrax|equinox|signax"

# Verify JAX config
python3 -c "import jax; print(jax.config.read('jax_enable_x64'))"
```

---

## Test Results Template

For each test subsection, record:

```markdown
### [SUBSECTION]
**Status**: PASS | FAIL | SKIP
**Duration**: X.Xs
**Notes**: 
- Item 1: PASS
- Item 2: PASS
- Item 3: FAIL - [reason]
```

---

## Known Limitations (Acceptable for RC1)

1. **Phase 5**: No comprehensive pytest suite yet (pending next phase)
2. **Performance**: No production benchmarks (preliminary estimates only)
3. **Multi-GPU**: Single-device testing; multi-GPU pending
4. **Integration**: No external data source testing (synthetic data only)

---

## Escalation & Bugs

**If you find issues**:

1. Record in detail (test name, input, error message)
2. Check if error is listed in [Known Limitations](#known-limitations-acceptable-for-rc1)
3. If new bug: Open GitHub issue with tag `RC1-testing`
4. Critical bugs (blocks all testing): Contact maintainer

---

## Success Criteria

**RC1 passes if**:

- ✅ All checks in sections 1-4 marked PASS
- ✅ Edge cases (section 5) handled gracefully
- ✅ Backward compatibility (section 6) verified
- ✅ Documentation (section 7) current
- ✅ Zero critical/blocking bugs

**If met** → Advance to Phase 5 (comprehensive test suite)  
**If not met** → Create RC2 with fixes

---

## Timeline

- **RC1 Testing Window**: February 19 - March 5, 2026
- **Bug Fix Window**: March 6 - March 12, 2026
- **RC2 Release**: March 13, 2026 (if bugfixes needed)
- **v3.0.0 Target**: April 2026

---

## Contact

Questions? Refer to:

- [CHANGELOG.md](CHANGELOG.md) - Technical details
- [RELEASE_NOTES.md](RELEASE_NOTES.md) - User guide
- [README.md](README.md) - Architecture overview
- [doc/](doc/) - Full specification
