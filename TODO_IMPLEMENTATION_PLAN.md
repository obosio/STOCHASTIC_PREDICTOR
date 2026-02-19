# TODO: IMPLEMENTATION PLAN v2.0.5

**Status:** READY FOR IMPLEMENTATION  
**Start Date:** 19 de febrero de 2026  
**Objective:** Fix all audit violations (3 critical + 8 major + 5 gaps)  
**Total Effort:** 14 days | **Timeline:** 2-3 weeks (team of 2-3)

---

## AUDIT FINDINGS SUMMARY

From AUDIT_REPORT_v1.1:

| Category | Count | Status |
| --- | --- | --- |
| **Critical Violations** | 3 | 🔴 BLOCK release |
| **Major Violations** | 8 | 🟠 Must fix v2.0.5 |
| **Implementation Gaps** | 5 | 🟡 Phase 2 |
| **Overall Conformity** | 87% | ✅ Good baseline |

### RTM Coverage: 95%

- 23/24 files mapped to .tex sections
- 6 files with violations documented
- 0 orphan code (everything has spec)

### VRAM Budget: ✅ COMPLIANT

- 26 instances of stop_gradient() implemented
- 30-50% savings on all backends (A100/H100/TPU/CPU)
- 2 diagnostics need stop_gradient() (dgm_entropy, holder_exponent)

### Deterministic Parity: ✅ CONFIGURED

- threefry2x32 PRNG: Implemented
- Deterministic reductions: Implemented  
- GPU ops: Implemented
- parity_hashes() function: Implemented
- Need: Cross-hardware test suite

---

# PHASE 1: CRITICAL VIOLATIONS (P0) - 6 DAYS

## V-CRIT-1: CUSUM Kurtosis Adjustment

**File:** `api/state_buffer.py::update_cusum_statistics()` (L155)  
**Spec:** `Implementation.tex §2.3 Algorithm 2.2`  
**Formula:** $h_t = k \cdot \sigma_t \cdot (1 + \ln(\kappa_t / 3))$  
**Current:** Static threshold, NO kurtosis computation

### Tasks

- [ ] Add `residual_window: Float[1024]` to `api/types.py::InternalState`
- [ ] Implement `compute_rolling_kurtosis(residual_window) → float` in `api/state_buffer.py`
  - κ_t bounded [1.0, 100.0]
  - Test: κ_gaussian ≈ 3, κ_heavy_tail > 4
- [ ] Update `update_cusum_statistics()`:
  - Shift and append residual to window
  - Compute κ_t
  - Apply formula: `h_t = config.cusum_k * sigma_t * (1 + ln(max(κ/3, 3)))`
- [ ] Implement grace period logic:
  - If alarm AND grace_counter==0 → set grace_counter = config.grace_period_steps
  - Decrement each step
  - If grace_counter > 0 → suppress alarm
- [ ] Add to config.toml if missing: `residual_window_size = 252`
- [ ] Test: Verify grace period blocks consecutive alarms

**Acceptance:** CUSUM test passes, grace period works

---

## V-CRIT-2: Sinkhorn Volatility Coupling

**File:** `core/sinkhorn.py::compute_sinkhorn_epsilon()` (L~80)  
**Spec:** `Implementation.tex §2.4.2 Algorithm 2.4`  
**Formula:** $\epsilon_t = \max(\epsilon_{min}, \epsilon_0 \cdot (1 + \alpha \cdot \sigma_t))$  
**Current:** Constant epsilon, ignores `config.sinkhorn_alpha`

### Tasks

- [ ] Verify `api/state_buffer.py::update_ema_variance()` exists and working
  - Formula: `EMA_t = λ * e_t^2 + (1-λ) * EMA_{t-1}` where λ=volatility_alpha
- [ ] Implement `compute_sinkhorn_epsilon_dynamic(ema_variance, config) → float`:
  - `sigma_t = sqrt(max(ema_var, 1e-10))`
  - `epsilon = max(epsilon_min, epsilon_0 * (1 + alpha * sigma_t))`
  - Test: ε increases with σ, respects bounds
- [ ] Update `core/sinkhorn.py` to use dynamic epsilon instead of config constant
- [ ] Update `core/fusion.py::fuse_kernel_outputs()` signature:
  - Add parameter: `ema_variance: Float[Array, ""]`
  - Pass to dynamic epsilon computation
- [ ] Update `core/orchestrator.py::orchestrate_step()`:
  - Pass `state.ema_variance` to fusion function
- [ ] Add to config.toml if missing:

  ```toml
  sinkhorn_epsilon_0 = 0.1
  sinkhorn_epsilon_min = 0.01
  sinkhorn_alpha = 0.5
  ```

- [ ] Test: Verify ε behavior with low/high volatility scenarios

**Acceptance:** Sinkhorn test passes, epsilon adapts correctly

---

## V-CRIT-3: Grace Period Logic

**File:** `core/orchestrator.py::orchestrate_step()`  
**Spec:** `Implementation.tex §2.5 Logic 2.5.3`  
**Current:** grace_counter field exists but never used

### Tasks

- [ ] Grace period already implemented in V-CRIT-1 (update_cusum_statistics)
- [ ] In orchestrator, capture tuple from update_cusum_statistics:

  ```python
  updated_state, should_alarm, h_t = update_cusum_statistics(...)
  ```

- [ ] Only emit regime change event if `should_alarm == True`
- [ ] Test: Verify sequential alarm suppression works

**Acceptance:** grace_counter properly tracks and suppresses alarms

---

## Checkpoint P0 Complete

Before proceeding to P1, validate:

- [ ] All P0 tests pass
- [ ] JIT compilation successful on all modules
- [ ] Config parameters properly injected
- [ ] Ready for P1 work

---

# PHASE 2: MAJOR VIOLATIONS (P1) - 8 DAYS

## V-MAJ-1: Entropy Threshold Adaptive Range

**File:** `kernels/kernel_b.py::kernel_b_predict()` (L~350)  
**Spec:** `Theory.tex §2.2 Theorem` - requires $\gamma \in [0.5, 1.0]$ range  
**Current:** Fixed `entropy_threshold = 0.8`

### Tasks

- [ ] Add to `api/types.py::PredictorConfig`:

  ```python
  entropy_gamma_min: float = 0.5
  entropy_gamma_max: float = 1.0
  entropy_gamma_default: float = 0.8
  ```

- [ ] Implement `compute_adaptive_entropy_threshold(H_terminal, regime) → float`:
  - Crisis: γ_min (lenient)
  - Low_vol: γ_max (strict)
  - Normal: γ_default
- [ ] Update kernel_b to use adaptive threshold instead of constant
- [ ] Test: Verify threshold adapts by regime

**Acceptance:** Entropy threshold tests pass

---

## V-MAJ-2,3,4: State Field Updates

**Files:** Various (orchestrator, kernel_b, state_buffer)  
**Problem:** State fields initialized but never updated (kurtosis, dgm_entropy, holder_exponent)

### Tasks

- [ ] `state.kurtosis` ← updated in V-CRIT-1 (CUSUM kurtosis computation)
- [ ] `state.dgm_entropy` ← assign from kernel_b output in orchestrator
- [ ] `state.holder_exponent` ← compute via WTMM in kernel_a (see P2.1)
- [ ] Ensure all updated to PredictionResult for telemetry

**Acceptance:** State fields properly tracked in telemetry

---

## V-MAJ-5: Mode Collapse Detection

**File:** `core/orchestrator.py`  
**Problem:** Binary per-step detection, no window accumulation

### Tasks

- [ ] Add `mode_collapse_consecutive_steps: int = 0` to InternalState
- [ ] In orchestrate_step():
  - If `dgm_entropy < threshold` → counter++, else counter=0
  - Emit warning if counter >= 10 (configurable)
- [ ] Test: Verify warning triggers after N consecutive steps

**Acceptance:** Mode collapse warning tests pass

---

## V-MAJ-6: Frozen Signal Recovery Ratio

**File:** `io/validators.py::detect_frozen_recovery()`  
**Problem:** Parameter defined in config but never used

### V-MAJ-6 Tasks

- [ ] Pass `config.frozen_signal_recovery_ratio` to detection function
- [ ] Use in recovery threshold calculation
- [ ] Wire up in orchestrator's frozen signal detection

**Acceptance:** Recovery ratio properly applied

---

## V-MAJ-7: Degraded Mode Hysteresis

**File:** `core/orchestrator.py::orchestrate_step()`  
**Problem:** No hysteresis in mode switching

### V-MAJ-7 Tasks

- [ ] Implement hysteresis:
  - If already degraded: recover threshold < normal threshold × recovery_hysteresis
  - If normal: degrade threshold > normal threshold
- [ ] Test: Verify no oscillation between modes

**Acceptance:** Mode hysteresis tests pass

---

## V-MAJ-8: Add stop_gradient to dgm_entropy

**File:** `kernels/kernel_b.py::compute_entropy_dgm()`  
**Problem:** dgm_entropy diagnostic leaks to gradient graph

### V-MAJ-8 Tasks

- [ ] Wrap output: `dgm_entropy_stopped = jax.lax.stop_gradient(dgm_entropy)`
- [ ] Return stopped version
- [ ] Verify VRAM savings in analysis

**Acceptance:** VRAM analysis updated, gradient graph verified

---

# PHASE 3: IMPLEMENTATION GAPS (P2) - 8 DAYS

## P2.1: WTMM Complete Implementation

**File:** `kernels/kernel_a.py`  
**Status:** Incomplete - has CWT but missing maxima linking and spectrum

### P2.1 Tasks

- [ ] Implement `link_wavelet_maxima()`: Chain detection across scales
- [ ] Implement `compute_singularity_spectrum()`: Legendre transform
- [ ] Integrate into kernel_a_predict()
- [ ] Extract holder_exponent as max of spectrum
- [ ] Test: Verify on Brownian motion (H≈0.5) and fBm

**Acceptance:** WTMM tests pass

---

## P2.2: Adaptive SDE Stiffness

**File:** `kernels/kernel_c.py`  
**Status:** Basic SDE but no stiffness adaptation

### P2.2 Tasks

- [ ] Implement `estimate_local_stiffness()` via Jacobian eigenvalues
- [ ] Implement stiffness-based scheme selection:
  - High stiffness → implicit Euler-Maruyama
  - Medium → hybrid interpolation
  - Low → explicit
- [ ] Integrate into kernel_c SDE loop
- [ ] Test: Verify scheme selection works

**Acceptance:** SDE stiffness tests pass

---

## P2.3: Telemetry Buffer Integration

**File:** `io/telemetry.py` + `core/orchestrator.py`  
**Status:** Buffer exists but not wired into orchestration

### Tasks

- [ ] Create background consumer thread for telemetry buffer
- [ ] In orchestrate_step(), enqueue TelemetryRecord with:
  - weights (ρ)
  - holder_exponent
  - kurtosis
  - sinkhorn_epsilon
- [ ] Thread consumes asynchronously (non-blocking)
- [ ] Test: Verify no blocking on orchestration

**Acceptance:** Telemetry integration tests pass

---

## P2.4: Snapshot Persistence

**File:** `io/snapshots.py`  
**Status:** Snapshots work but incomplete field coverage

### Tasks

- [ ] Add WTMM buffer to snapshot
- [ ] Add mode_collapse_consecutive_steps
- [ ] Add ema_variance
- [ ] Add residual_window (via signal_history proxy)
- [ ] Test: Verify restore consistency

**Acceptance:** Snapshot restore tests pass

---

## P2.5: Test Suite Scaffolding

**Files:** `tests/test_*.py`  
**Status:** Currently empty

### Tasks

- [ ] Create `test_cusum_kurtosis.py` - unit tests for P0.1
- [ ] Create `test_sinkhorn_dynamic.py` - unit tests for P0.2
- [ ] Create `test_entropy_adaptive.py` - unit tests for P1.1
- [ ] Create `test_wtmm.py` - unit tests for P2.1
- [ ] Create `test_cpu_gpu_parity.py` - cross-hardware validation
- [ ] Create `test_e2e_integration.py` - end-to-end scenarios

**Acceptance:** Test suite passes on all platforms

---

# PHASE 4: VALIDATION (2-3 days)

- [ ] Re-audit all P0 violations → all FIXED
- [ ] Validate all P1 violations → all FIXED  
- [ ] Integration testing: E2E scenarios on mixed hardware (CPU+GPU)
- [ ] Final compliance report
- [ ] Generate v2.0.5 release notes

---

## TIMELINE & CHECKPOINTS

### Week 1: Critical Path (P0)

- [ ] Mon-Tue: V-CRIT-1 (CUSUM) complete + tested
- [ ] Wed: V-CRIT-2 (Sinkhorn) complete + tested
- [ ] Thu: V-CRIT-3 (Grace) complete + tested
- [ ] Fri: Checkpoint P0 - all tests passing

### Week 2: Major Violations (P1)

- [ ] Mon-Tue: V-MAJ-1,2,3 complete
- [ ] Wed: V-MAJ-4,5,6 complete
- [ ] Thu: V-MAJ-7,8 complete
- [ ] Fri: Checkpoint P1 - all tests passing

### Week 3: Gaps & Validation (P2 + Phase 4)

- [ ] Mon-Tue: P2.1 (WTMM) complete
- [ ] Wed-Thu: P2.2,3,4 complete
- [ ] Fri: P2.5 + Validation complete
- [ ] Release ready v2.0.5

---

## SUCCESS CRITERIA

✅ All 3 critical violations fixed  
✅ All 8 major violations fixed  
✅ 5 implementation gaps completed  
✅ Test suite 100% pass rate (all backends)  
✅ CPU/GPU parity validated with SHA256 hashes  
✅ Conformity score ≥ 98%  
✅ Ready for production deployment

**Versioning:** v2.0.5 release tag after validation complete
