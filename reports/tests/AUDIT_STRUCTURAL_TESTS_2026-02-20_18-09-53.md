# Structural Test Coverage Audit - Execution Report

**Date:** 2026-02-20 18:09:53 | **Project:** USP | **Execution:** Post-NEW-2 Fix  
**Python:** 3.13.12 | **JAX:** 0.4.20 (FP64 enabled) | **pytest:** 9.0.2

---

## Executive Summary

| Metric | Value | Status |
| --- | --- | --- |
| **Total Tests** | 73 | - |
| **Passed** | 62 | ✅ |
| **Failed** | 11 | ⚠️ |
| **Success Rate** | 84.9% | - |
| **Execution Time** | 11.24s | - |
| **Coverage** | 100.0% (95/95 functions) | ✅ |
| **Gaps** | 0 | ✅ |
| **Orphans** | 0 | ✅ |

---

## Critical Fix Applied: NEW-2 (list → tuple conversion)

### Status: ✅ RESOLVED

**Location:** `stochastic_predictor/api/config.py:548-566`

**Change:**

```text
```

**Result:** PredictorConfig no longer contains unhashable list fields  
**Impact:** Eliminates "Non-hashable static arguments" error completely

### Evidence of Resolution

**Previous Error (❌ NEW-2):**

```text
TypeError: unhashable type: 'list'
Cannot hash object of type PredictorConfig with field
sanitize_clip_range=[-1000000.0, 1000000.0]
```

**New Error (✅ Different Root Cause):**

```text
ValueError: inconsistent size for core dimension 'm': 94 vs 95
  File: kernel_a.py:687 in kernel_ridge_regression
  at: alpha = jnp.linalg.solve(K_reg, y_train)
```

**Conclusion:** JAX hashing now works! Configuration passes static_argnames validation.

---

## Test Results Summary

### Passed Tests (62/73)

✅ All API configuration, validation, PRNG, and type tests passing:

- **TestAPIConfig:** 4 passed
- **TestAPIValidation:** 18 passed  
- **TestAPITypes:** 8 passed
- **TestAPISchemas:** 3 passed
- **TestAPIPRNG:** 8 passed
- **TestBasicSetup:** 2 passed
- **TestCoreOrchestrator:** 7 passed
- **TestAPIStateBuffer:** 8 passed
- **TestCoverageValidation:** 1 passed

### Failed Tests (11/73)

⚠️ **Error Type Changed:** No longer JAX hashing issues. Now algorithmic dimension mismatches.

| Test | Class | New Failure Type |
| --- | --- | --- |
| test_kernel_a_predict | TestKernelA | MATRIX-DIM-1 |
| test_kernel_b_predict | TestKernelB | MATRIX-DIM-1 |
| test_kernel_c_predict | TestKernelC | MATRIX-DIM-1 |
| test_kernel_d_predict | TestKernelD | MATRIX-DIM-1 |
| test_warmup_kernel_a | TestAPIWarmup | MATRIX-DIM-1 |
| test_warmup_kernel_b | TestAPIWarmup | MATRIX-DIM-1 |
| test_warmup_kernel_c | TestAPIWarmup | MATRIX-DIM-1 |
| test_warmup_kernel_d | TestAPIWarmup | MATRIX-DIM-1 |
| test_warmup_all_kernels | TestAPIWarmup | MATRIX-DIM-1 |
| test_warmup_with_retry | TestAPIWarmup | MATRIX-DIM-1 |
| test_profile_warmup_and_recommend_timeout | TestAPIWarmup | MATRIX-DIM-1 |

---

## New Defect Detected: MATRIX-DIM-1 (Algorithmic)

### Error Signature

```text
ValueError: inconsistent size for core dimension 'm': 94 vs 95
  on vectorized function with excluded=frozenset()
  and signature='(m,m),(m)->(m)'
```

### Root Cause

Matrix solve operation in kernel_ridge_regression receives:

- **K_reg:** 94×94 matrix
- **y_train:** 95-element vector
- **Expected:** Both must have matching first dimension

### Affected Functions

**Primary:**

- `stochastic_predictor.kernels.kernel_a.kernel_ridge_regression()` (line 687)

**Cascading:**

- kernel_a/b/c/d_predict() all call kernel_ridge_regression
- warmup_kernel_* functions trigger predict on startup

### Problem Analysis

The dimension mismatch occurs because:

1. Data preparation creates y_train with shape (95,)
2. Kernel matrix K_reg computed as shape (94, 94)
3. Likely: off-by-one error in training data build or kernel matrix padding

### Sample Traceback

```text
File: stochastic_predictor/kernels/kernel_a.py, line 687
  alpha = jnp.linalg.solve(K_reg, y_train)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ValueError: inconsistent size for core dimension 'm': 94 vs 95
  signature='(m,m),(m)->(m)'
```

---

## Compliance Status

| Item | Status | Notes |
| --- | --- | --- |
| Meta-validator Coverage | ✅ 100% (95/95) | All public functions under test |
| Structural Gaps | ✅ 0 | No untested public symbols |
| Orphan Tests | ✅ 0 | All tests match real functions |
| NEW-2 Resolution | ✅ FIXED | list → tuple conversion successful |
| MATRIX-DIM-1 Issue | ❌ OPEN | Kernel dimension mismatch (new) |

---

## Progression Summary

| Phase | Issue | Status | Resolution |
| --- | --- | --- | --- |
| Phase 1 | Orphan tests | ✅ Fixed | Test name alignment |
| Phase 2 | PROD-1 (JAX decorator) | ✅ Fixed | Removed @jax.jit from validate_kernel_input |
| Phase 3 | PROD-2 (missing config fields) | ✅ Fixed | Added fields to config.toml |
| Phase 4 | NEW-1 (TracerBoolConversionError) | ✅ Fixed | Removed @jax.jit decorator |
| Phase 5 | **NEW-2 (unhashable lists)** | **✅ FIXED** | **list → tuple conversion** |
| Phase 6 | MATRIX-DIM-1 (new) | ❌ Open | Requires kernel logic investigation |

---

## Audit Trail

**Session:** 2026-02-20 18:09:53  
**Execution Command:**

```bash
pytest tests/structure/test_structural_execution.py --tb=no -q
```

**Key Change Applied:**

```python
# Before: Lists in config
sanitize_clip_range: list = [-1000000.0, 1000000.0]

# After: Tuples in config (hashable for JAX)
sanitize_clip_range: tuple = (-1000000.0, 1000000.0)
```

**Meta-Validator Result:**

```bash
python tests/structure/validate_coverage.py
```

✅ 100% coverage (95/95), 0 gaps, 0 orphans

---

## Recommendations

### Immediate (P0)

1. ✅ **NEW-2 Fix Verified** - list → tuple conversion working
2. Investigate MATRIX-DIM-1 cause in kernel_ridge_regression
3. Check if training data shape differs from kernel matrix shape

### Investigation Points

- Line 687 in kernel_a.py: y_train shape vs K_reg shape
- Data normalization or padding logic
- Potential off-by-one in kernel matrix construction

---

## Assessment

**Major Achievement:** ✅ JAX compatibility issue (NEW-2) completely resolved through tuple conversion. This unblocks kernel execution beyond configuration stage.

**Remaining Work:** MATRIX-DIM-1 is a legitimate algorithmic issue uncovered by NEW-2 fix. Kernels now run JAX code successfully; data shape mismatch now visible.

---

**Report Generated:** 2026-02-20 18:09:53  
**Framework:** pytest 9.0.2 | pytest-jaxtyping 0.3.9 | JAX 0.4.20 (FP64)  
**Execution Duration:** 11.24 seconds  
**Status:** ✅ Major blockers resolved | ⚠️ New algorithmic issue detected
