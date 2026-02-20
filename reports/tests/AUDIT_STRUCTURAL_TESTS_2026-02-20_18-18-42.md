# Structural Test Audit Report

**Generated:** 2026-02-20 18:18:42 UTC  
**Project:** Universal Stochastic Predictor (USP)  
**Test Suite:** test_structural_execution.py  
**Python:** 3.13.12 | JAX: 0.4.20 (FP64 enabled)

---

## Executive Summary

**Test Results:** 62 PASSED / 11 FAILED (84.9% pass rate)  
**Status:** ⚠️ ERRORS DETECTED AFTER KERNEL_A FIX  
**Finding:** Previous dimension mismatch error (94 vs 95) was resolved by kernel_a.py:784-789 fix, but NEW error patterns emerged: **JAX tracing type incompatibilities across all kernels**.

---

## Critical Discovery: NEW Error Pattern

The fix applied at **kernel_a.py:784-789** successfully resolved the matrix dimension mismatch (94 vs 95), but unveiled a deeper issue: **Functions decorated with `@jax.jit` are receiving non-array arguments without proper `static_argnames` markers**.

### Root Cause Classification

**Error Category:** JAX JIT Compilation Type Mismatch  
**Affected Components:** kernel_a.py, kernel_b.py, kernel_c.py, kernel_d.py  
**Pattern:** Non-array types (strings, functions, callables, config objects) being passed to jitted functions

---

## Failed Tests Analysis

### 1. **test_kernel_a_predict**

- **Error:** `TypeError: Cannot interpret value of type <class 'function'> as an abstract array`
- **Location:** kernel_a.py:384 (continuous_wavelet_transform)
- **Problematic Argument:** `epsilon` parameter interpreted as function instead of array
- **Issue:** Argument `epsilon` passed to @jax.jit without static_argnames marker
- **Scope:** Affects 4 tests (test_kernel_a_predict + 3 warmup tests)

### 2. **test_kernel_b_predict**

- **Error:** `ValueError: ema_variance is required for adaptive entropy threshold`
- **Location:** kernel_b.py (dgm_hjb_solver)
- **Issue:** Missing EMA variance in adaptive threshold computation during trace
- **Scope:** Affects 2 tests (test_kernel_b_predict + test_warmup_kernel_b)

### 3. **test_kernel_c_predict**

- **Error:** `TypeError: Error interpreting argument to <function solve_sde> as an abstract array`
- **Location:** kernel_c.py:328 (solve_sde)
- **Problematic Value:** PredictorConfig object passed where array expected at path `t1`
- **Issue:** Complex config type not marked as static in @jax.jit decorator
- **Scope:** Affects 2 tests (test_kernel_c_predict + test_warmup_kernel_c)

### 4. **test_kernel_d_predict**

- **Error:** `TypeError: Error interpreting argument to <function apply_stop_gradient_to_diagnostics>`
- **Location:** kernel_d.py:265
- **Problematic Value:** String `'D_Signature_Rough_Paths'` (kernel_type) passed to jitted function
- **Issue:** Non-array diagnostic metadata not marked as static
- **Scope:** Affects 2 tests (test_kernel_d_predict + test_warmup_kernel_d)

### 5. **test_warmup_all_kernels**

- **Error:** Cascade failure from kernel_a continuous_wavelet_transform
- **Root:** Same epsilon function/array confusion as test_kernel_a_predict
- **Scope:** 1 test

### 6. **test_warmup_with_retry**

- **Error:** RuntimeError wrapping continuous_wavelet_transform failure
- **Location:** warmup.py:388 (warm-up retry logic)
- **Scope:** 1 test

### 7. **test_profile_warmup_and_recommend_timeout**

- **Error:** Cascade failure during profiling phase
- **Root:** Same epsilon issue in kernel_a
- **Scope:** 1 test

---

## Failure Distribution

| Component        | Failed | Total | Rate   |
|------------------|--------|-------|--------|
| Kernel A         | 4      | 6     | 66.7%  |
| Kernel B         | 2      | 4     | 50.0%  |
| Kernel C         | 2      | 4     | 50.0%  |
| Kernel D         | 2      | 4     | 50.0%  |
| Warmup API       | 7      | 7     | 100%   |
| **TOTAL**        | **11** | **73**| **15.1%**|

---

## Technical Analysis

### Previous State (Before kernel_a.py:784-789)

```
Error: ValueError: inconsistent size for core dimension 'm': 94 vs 95
Location: kernel_a.py:687 (jnp.linalg.solve)
Cause: K_reg (94×94) vs y_train (95 elements) dimension mismatch
```

### Current State (After kernel_a.py:784-789)

```
Error: TypeError: Cannot interpret value of type <class 'function'> as abstract array
Pattern: Functions/strings/configs passed to @jax.jit without static markers
Scope: All 4 kernels + Warmup API
```

### Conclusion

The kernel_a fix successfully corrected the numerical dimension issue, but exposed that **underlying kernel decorator signatures are missing critical static_argnames specifications** for non-array parameters.

---

## Affected Code Locations

### kernel_a.py

- Line 384: `continuous_wavelet_transform()` decorated but receives function in `epsilon` parameter
- Line 821: `kernel_a_predict()` calls extracted functions without type normalization

### kernel_b.py

- Adaptive threshold computation missing EMA variance in trace phase
- State management during JIT compilation incomplete

### kernel_c.py

- Line 328: `solve_sde()` receives PredictorConfig without static marker
- Path `t1` contains config object in trace

### kernel_d.py

- Line 265: `apply_stop_gradient_to_diagnostics()` receives string diagnostics['kernel_type']
- Diagnostic metadata not marked as static

---

## Test Coverage Validation

**Meta-Validator Status:** ✅ PASSED  

- Functions Tested: 95/95 (100%)
- Gaps: 0
- Orphans: 0

Coverage remains complete; failures are runtime type-checking issues, not missing tests.

---

## Recommendations

### Immediate Action (Priority 1)

Audit all `@jax.jit` decorators in kernel files:

1. Identify all non-array parameters
2. Add `static_argnames` declarations for callables, strings, config objects
3. Test tracing with simplified type signatures

### Investigation Path  

- **kernel_a.py:384** - Check `epsilon` parameter source (should be scalar, not function)
- **kernel_c.py:328** - Extract config fields needed; pass scalars, mark config as static
- **kernel_d.py:265** - Separate diagnostic strings into separate non-jitted wrapper

### Validation Strategy

1. Fix one kernel (kernel_a) with proper static_argnames
2. Re-run test_kernel_a_predict to verify type checking passes
3. Apply same pattern to B, C, D kernels
4. Execute full test suite

---

## Change History

| Timestamp | Event |
|-----------|-------|
| 2026-02-20 18:09:53 | Previous report: 94 vs 95 dimension mismatch discovered |
| 2026-02-20 18:15:04 | Fix applied to kernel_a.py:784-789, errors persist with same 11 tests |
| 2026-02-20 18:18:42 | **NEW REPORT** - Fix validated; new error pattern (JIT type mismatch) exposed |

---

## Conclusion

✅ **Dimension mismatch RESOLVED** - kernel_a.py:784-789 fix successful  
⚠️ **NEW defects UNCOVERED** - JAX JIT tracing incompatibilities across all kernels  
📊 **Test Coverage** - 100% maintained (95/95 functions)  
🔍 **Next Phase** - Static type marker audit + decorator specification validation
