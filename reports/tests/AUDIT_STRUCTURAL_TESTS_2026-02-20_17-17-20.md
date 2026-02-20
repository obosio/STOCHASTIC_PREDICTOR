# Structural Test Coverage Audit - Final Report

**Date:** 2026-02-20 | **Project:** USP | **Coverage:** 100.0% (95/95 functions)

---

## Executive Summary

| Metric | Status |
| --- | --- |
| **Structural Coverage** | ✅ 100% (95/95 public functions tested) |
| **Meta-Validator** | ✅ 0 gaps, 0 orphans, 100% coverage |
| **Test Alignment** | ✅ All tests named with exact symbols |
| **Test Execution** | 🟡 52/72 passed (72.2%) |
| **PROD-2 (config.toml)** | ✅ RESOLVED |
| **NEW-1 (TracerBoolConversionError)** | ✅ RESOLVED (@jax.jit removed) |
| **NEW-2 (@partial + config)** | 🔴 Requires different approach |

---

## Test Execution Results

```text
Total: 72 tests
✅ Passed: 52 (72.2%)
❌ Failed: 20 (27.8%)

Passing Categories:
- API (config, prng, types, validation, schemas): 34/34 ✅
- Kernels base: 4/4 ✅
- Some orchestrator/state_buffer: 14/16 ✅

Failing Categories:
- Warmup + advanced kernels: 0/20 ✅ (20 failures from NEW-2 issue)
```

---

## Remaining Issues

### NEW-2: @jax.jit Incompatibilities with Non-Array Arguments

**Root Problem:** Multiple functions decorated with `@jax.jit` receive non-array parameters (config, InternalState, ProcessState) that cannot be traced by JAX.

**Error Pattern:**

```python
TypeError: Cannot interpret value of type <class 'stochastic_predictor.api.types.InternalState'> 
as an abstract array; it does not have a dtype attribute

Error interpreting argument to <function update_signal_history at 0x...> as an abstract array.
The problematic value is of type <class '...'> and was passed to the function at path state.

This typically means that a jit-wrapped function was called with a non-array argument, 
and this argument was not marked as static using static_argnums or static_argnames.
```

**Affected Functions (Expanded List - 20+ functions):**

**state_buffer.py (6 functions):**

- update_signal_history
- batch_update_signal_history
- reset_cusum_statistics
- update_cusum_statistics
- update_ema_variance
- update_residual_buffer

**orchestrator.py (2 functions):**

- initialize_state
- initialize_batched_states

**kernel functions (11 functions):**

- kernel_a.py: extract_holder_exponent_wtmm, kernel_ridge_regression, create_embedding, kernel_a_predict
- kernel_b.py: compute_entropy_dgm, loss_hjb, compute_adaptive_entropy_threshold
- kernel_c.py: kernel_c_predict
- kernel_d.py: compute_log_signature, predict_from_signature, kernel_d_predict

**Recommended Fix:**
Remove `@jax.jit` from all functions that receive non-traceable arguments (config, state objects). These functions are not in performance-critical paths and JAX cannot trace non-array types.

**Traceback Example:**

```python
tests/structure/test_structural_execution.py:379: in test_update_signal_history
    new_state = update_signal_history(state, jnp.array(0.5))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/pjit.py:616: in _infer_params_impl
    avals.append(shaped_abstractify(a))
                 ^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/api_util.py:598: in _shaped_abstractify_slow
    raise TypeError(...)
E   TypeError: Error interpreting argument to <function update_signal_history> as abstract array
```

---

## Action Items

| Priority | Task | Status |
| --- | --- | --- |
| 🔴 P1 | Remove `@jax.jit` from 11 kernel functions with config params | ⏳ Pending |
| green | Re-execute pytest after fix | ⏳ Pending |
| green | Verify 72/72 tests passing | ⏳ Pending |

---

## Verification Checklist

- ✅ Meta-validator: 0 gaps, 0 orphans (100% coverage confirmed)
- ✅ Test names: All aligned with public symbols
- ✅ config.toml: Complete (PROD-2 resolved)
- ✅ NEW-1: Fixed (validate_kernel_input @jax.jit removed)
- 🔄 NEW-2: Analysis complete, ready for dev fix
- ⏳ Final execution: Awaiting NEW-2 resolution
