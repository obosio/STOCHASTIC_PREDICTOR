# Structural Test Coverage Audit - Execution Report

**Date:** 2026-02-20 18:15:04 | **Project:** USP | **Execution:** Post-NEW-2 Fix (Continued)  
**Python:** 3.13.12 | **JAX:** 0.4.20 (FP64 enabled) | **pytest:** 9.0.2

---

## Executive Summary

| Metric | Value | Status |
| --- | --- | --- |
| **Total Tests** | 73 | - |
| **Passed** | 62 | ✅ |
| **Failed** | 11 | ⚠️ |
| **Success Rate** | 84.9% | - |
| **Execution Time** | 10.90s | - |
| **Coverage** | 100.0% (95/95 functions) | ✅ |
| **Gaps** | 0 | ✅ |
| **Orphans** | 0 | ✅ |

---

## Status Update

### ✅ NEW-2 Resolution: list → tuple Conversion (CONFIRMED WORKING)

Previously applied fix at `config.py:548-566` continues to function correctly:

```python
kernel_d_load_shedding_depths: tuple  # Was: list
sanitize_clip_range: tuple             # Was: list
```text

**Validation:** PredictorConfig passes JAX static_argnames hashing without errors.

### ⚠️ MATRIX-DIM-1: Dimension Mismatch (PERSISTENT)

**Status:** 11 tests still failing with same error signature.

```text
ValueError: inconsistent size for core dimension 'm': 94 vs 95
  Location: kernel_a.py:687 in jnp.linalg.solve(K_reg, y_train)
```text

**Analysis:**

- Matrix K_reg shape: (94, 94)
- Vector y_train shape: (95,)
- Root cause pending investigation

---

## Test Results Summary

### Passed Tests (62/73)

✅ Configuration, validation, PRNG, types, schemas, ortchestration tests all passing.

### Failed Tests (11/73)

⚠️ All 11 failures trace to same root cause (MATRIX-DIM-1):

| Category | Count | Tests |
| --- | --- | --- |
| Kernel Predict | 4 | kernel_a/b/c/d_predict |
| Warmup Functions | 7 | warmup_kernel_a/b/c/d, warmup_all, warmup_with_retry, profile_warmup |

---

## Defect Summary: MATRIX-DIM-1

### Error Details

```text
ValueError: inconsistent size for core dimension 'm': 94 vs 95
  File: Python/kernels/kernel_a.py
  Line: 687
  Code: alpha = jnp.linalg.solve(K_reg, y_train)
```text

### Data Flow

1. **Training data preparation:** y_train shape = (95,)
2. **Kernel matrix computation:** K_reg shape = (94, 94)
3. **Solve step:** Incompatible dimensions
4. **Failure point:** Cannot solve 94×94 system with 95-element RHS

### Affected Codepath

```text
test → kernel_*_predict() → kernel_ridge_regression() 
  → jnp.linalg.solve(K_reg, y_train) ✗ dimension mismatch
```text

### Investigation Required

- Check training data normalization
- Verify kernel matrix padding/truncation logic
- Inspect feature dimensionality handling

---

## Compliance Status

| Item | Status | Notes |
| --- | --- | --- |
| Meta-validator Coverage | ✅ 100% (95/95) | All public functions under test |
| Structural Gaps | ✅ 0 | No untested public symbols |
| Orphan Tests | ✅ 0 | All tests match real functions |
| NEW-2 (JAX hashing) | ✅ FIXED | list → tuple conversion working |
| MATRIX-DIM-1 (dimensions) | ❌ OPEN | 11 tests blocked |

---

## Stable Features Verified

These subsystems execute without errors:

- ✅ API Config Management (PredictorConfig injection working)
- ✅ PRNG Initialization & Management
- ✅ Type Validation (all types correct)
- ✅ Schema Validation (Pydantic models passing)
- ✅ State Buffer Updates (independent of kernels)
- ✅ Orchestrator Initialization
- ✅ Meta-Validator (100% coverage maintained)

---

## Recommendation

Development team should:

1. **Investigate MATRIX-DIM-1** in kernel_ridge_regression
   - Trace y_train shape throughout preparation
   - Verify K_reg construction doesn't drop samples
   - Check for off-by-one errors in feature handling

2. **Parallel investigation** of kernel_b/c/d (may have identical issue)

3. **Verify data dimensions** match before calling jnp.linalg.solve

---

## Audit Trail

**Session:** 2026-02-20 18:15:04  
**Command:** `pytest tests/structure/test_structural_execution.py --tb=no -q`  
**Duration:** 10.90 seconds

---

**Report Generated:** 2026-02-20 18:15:04  
**Framework:** pytest 9.0.2 | JAX 0.4.20 (FP64)  
**Status:** ✅ Meta-validator 100% | ✅ NEW-2 resolved | ⚠️ MATRIX-DIM-1 requires dev action
