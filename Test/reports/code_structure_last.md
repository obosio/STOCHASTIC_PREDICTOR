# 🧪 Code Structure Tests Report

**Generated:** 2026-02-21 15:19:14 UTC

## 📊 Executive Summary

❌ **Overall Status:** FAIL

| Metric | Value |
| --- | --- |
| Total Tests | 79 |
| Passed | 67 (84.8%) |
| Failed | 12 (15.2%) |
| Warnings | 2 |
| Duration | 10.57s |
| Exit Code | 1 |

---

## 📝 Detailed Test Output

⚠️ **Attention:** Some tests failed. Review the output below.

```text
============================= test session starts ==============================
platform darwin -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0 --
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/.venv/bin/python
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR
plugins: jaxtyping-0.3.9, hypothesis-6.151.9, cov-7.0.0
collecting ... collected 79 items

Test/scripts/code_structure.py::TestBasicSetup::test_get_config PASSED   [  1%]
Test/scripts/code_structure.py::TestBasicSetup::test_initialize_jax_prng PASSED [  2%]
Test/scripts/code_structure.py::TestAPIConfig::test_get_config PASSED    [  3%]
Test/scripts/code_structure.py::TestAPIConfig::test_PredictorConfigInjector PASSED [  5%]
Test/scripts/code_structure.py::TestAPIConfig::test_ConfigManager PASSED [  6%]
Test/scripts/code_structure.py::TestAPIConfig::test_PredictorConfig PASSED [  7%]
Test/scripts/code_structure.py::TestAPIPRNG::test_initialize_jax_prng PASSED [  8%]
Test/scripts/code_structure.py::TestAPIPRNG::test_split_key PASSED       [ 10%]
Test/scripts/code_structure.py::TestAPIPRNG::test_split_key_like PASSED  [ 11%]
Test/scripts/code_structure.py::TestAPIPRNG::test_uniform_samples PASSED [ 12%]
Test/scripts/code_structure.py::TestAPIPRNG::test_normal_samples PASSED  [ 13%]
Test/scripts/code_structure.py::TestAPIPRNG::test_exponential_samples PASSED [ 15%]
Test/scripts/code_structure.py::TestAPIPRNG::test_check_prng_state PASSED [ 16%]
Test/scripts/code_structure.py::TestAPIPRNG::test_verify_determinism PASSED [ 17%]
Test/scripts/code_structure.py::TestAPITypes::test_ProcessState PASSED   [ 18%]
Test/scripts/code_structure.py::TestAPITypes::test_KernelType PASSED     [ 20%]
Test/scripts/code_structure.py::TestAPITypes::test_OperatingMode PASSED  [ 21%]
Test/scripts/code_structure.py::TestAPITypes::test_check_jax_config PASSED [ 22%]
Test/scripts/code_structure.py::TestAPIValidation::test_validate_magnitude PASSED [ 24%]
Test/scripts/code_structure.py::TestAPIValidation::test_validate_timestamp PASSED [ 25%]
Test/scripts/code_structure.py::TestAPIValidation::test_check_staleness PASSED [ 26%]
Test/scripts/code_structure.py::TestAPIValidation::test_validate_shape PASSED [ 27%]
Test/scripts/code_structure.py::TestAPIValidation::test_validate_finite PASSED [ 29%]
Test/scripts/code_structure.py::TestAPIValidation::test_sanitize_array PASSED [ 30%]
Test/scripts/code_structure.py::TestAPIValidation::test_ensure_float64 PASSED [ 31%]
Test/scripts/code_structure.py::TestAPIValidation::test_cast_array_to_float64 PASSED [ 32%]
Test/scripts/code_structure.py::TestAPIValidation::test_validate_holder_exponent PASSED [ 34%]
Test/scripts/code_structure.py::TestAPIValidation::test_validate_alpha_stable PASSED [ 35%]
Test/scripts/code_structure.py::TestAPIValidation::test_validate_beta_stable PASSED [ 36%]
Test/scripts/code_structure.py::TestAPIValidation::test_validate_simplex PASSED [ 37%]
Test/scripts/code_structure.py::TestAPIValidation::test_sanitize_external_observation PASSED [ 39%]
Test/scripts/code_structure.py::TestAPIValidation::test_warn_if_invalid PASSED [ 40%]
Test/scripts/code_structure.py::TestAPISchemas::test_ProcessStateSchema PASSED [ 41%]
Test/scripts/code_structure.py::TestAPISchemas::test_OperatingModeSchema PASSED [ 43%]
Test/scripts/code_structure.py::TestAPIStateBuffer::test_update_signal_history PASSED [ 44%]
Test/scripts/code_structure.py::TestAPIStateBuffer::test_batch_update_signal_history PASSED [ 45%]
Test/scripts/code_structure.py::TestAPIStateBuffer::test_reset_cusum_statistics PASSED [ 46%]
Test/scripts/code_structure.py::TestAPIStateBuffer::test_update_ema_variance PASSED [ 48%]
Test/scripts/code_structure.py::TestAPIStateBuffer::test_update_residual_buffer PASSED [ 49%]
Test/scripts/code_structure.py::TestCoreOrchestrator::test_initialize_state PASSED [ 50%]
Test/scripts/code_structure.py::TestCoreOrchestrator::test_initialize_batched_states PASSED [ 51%]
Test/scripts/code_structure.py::TestCoreOrchestrator::test_compute_entropy_ratio PASSED [ 53%]
Test/scripts/code_structure.py::TestCoreOrchestrator::test_scale_dgm_architecture PASSED [ 54%]
Test/scripts/code_structure.py::TestCoreOrchestrator::test_compute_adaptive_stiffness_thresholds PASSED [ 55%]
Test/scripts/code_structure.py::TestCoreOrchestrator::test_compute_adaptive_jko_params PASSED [ 56%]
Test/scripts/code_structure.py::TestCoreFusion::test_compute_sinkhorn_epsilon PASSED [ 58%]
Test/scripts/code_structure.py::TestCoreMetaOptimizer::test_walk_forward_split FAILED [ 59%]
Test/scripts/code_structure.py::TestKernelsBase::test_apply_stop_gradient_to_diagnostics PASSED [ 60%]
Test/scripts/code_structure.py::TestKernelsBase::test_compute_signal_statistics PASSED [ 62%]
Test/scripts/code_structure.py::TestKernelsBase::test_normalize_signal PASSED [ 63%]
Test/scripts/code_structure.py::TestKernelsBase::test_validate_kernel_input PASSED [ 64%]
Test/scripts/code_structure.py::TestKernelA::test_gaussian_kernel PASSED [ 65%]
Test/scripts/code_structure.py::TestKernelA::test_compute_gram_matrix PASSED [ 67%]
Test/scripts/code_structure.py::TestKernelA::test_kernel_ridge_regression PASSED [ 68%]
Test/scripts/code_structure.py::TestKernelA::test_create_embedding PASSED [ 69%]
Test/scripts/code_structure.py::TestKernelA::test_kernel_a_predict FAILED [ 70%]
Test/scripts/code_structure.py::TestKernelB::test_dgm_hjb_solver PASSED  [ 72%]
Test/scripts/code_structure.py::TestKernelB::test_kernel_b_predict FAILED [ 73%]
Test/scripts/code_structure.py::TestKernelC::test_drift_levy_stable PASSED [ 74%]
Test/scripts/code_structure.py::TestKernelC::test_diffusion_levy PASSED  [ 75%]
Test/scripts/code_structure.py::TestKernelC::test_kernel_c_predict FAILED [ 77%]
Test/scripts/code_structure.py::TestKernelD::test_create_path_augmentation PASSED [ 78%]
Test/scripts/code_structure.py::TestKernelD::test_compute_log_signature PASSED [ 79%]
Test/scripts/code_structure.py::TestKernelD::test_predict_from_signature PASSED [ 81%]
Test/scripts/code_structure.py::TestKernelD::test_kernel_d_predict FAILED [ 82%]
Test/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_a FAILED [ 83%]
Test/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_b FAILED [ 84%]
Test/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_c FAILED [ 86%]
Test/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_d FAILED [ 87%]
Test/scripts/code_structure.py::TestAPIWarmup::test_warmup_all_kernels FAILED [ 88%]
Test/scripts/code_structure.py::TestAPIWarmup::test_warmup_with_retry FAILED [ 89%]
Test/scripts/code_structure.py::TestAPIWarmup::test_profile_warmup_and_recommend_timeout FAILED [ 91%]
Test/scripts/code_structure.py::TestIOModuleImportable::test_config_mutation_module_exists PASSED [ 92%]
Test/scripts/code_structure.py::TestIOModuleImportable::test_credentials_module_exists PASSED [ 93%]
Test/scripts/code_structure.py::TestIOModuleImportable::test_dashboard_module_exists PASSED [ 94%]
Test/scripts/code_structure.py::TestIOModuleImportable::test_loaders_module_exists PASSED [ 96%]
Test/scripts/code_structure.py::TestIOModuleImportable::test_snapshots_module_exists PASSED [ 97%]
Test/scripts/code_structure.py::TestIOModuleImportable::test_telemetry_module_exists PASSED [ 98%]
Test/scripts/code_structure.py::TestIOModuleImportable::test_validators_module_exists PASSED [100%]

=================================== FAILURES ===================================
________________ TestCoreMetaOptimizer.test_walk_forward_split _________________
Test/scripts/code_structure.py:537: in test_walk_forward_split
    assert abs(actual_ratio - train_ratio) < 0.05, (
E   AssertionError: Split 0 ratio mismatch: 0.921 vs expected 0.700
E   assert 0.2210526315789474 < 0.05
E    +  where 0.2210526315789474 = abs((0.9210526315789473 - 0.7))
______________________ TestKernelA.test_kernel_a_predict _______________________
Test/scripts/code_structure.py:606: in test_kernel_a_predict
    output = kernel_a_predict(signal, prng_key, config_obj)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:797: in kernel_a_predict
    wiener_hopf_filter = compute_wiener_hopf_filter(
Python/kernels/kernel_a.py:518: in compute_wiener_hopf_filter
    autocorr = autocorr_full[n - 1 : n - 1 + order + 1] / n
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/array_methods.py:1050: in op
    return getattr(self.aval, f"_{name}")(self, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/array_methods.py:646: in _getitem
    return lax_numpy._rewriting_take(self, item)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:11939: in _rewriting_take
    return _gather(arr, treedef, static_idx, dynamic_idx, indices_are_sorted,
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:11948: in _gather
    indexer = _index_to_gather(shape(arr), idx)  # shared with _scatter_update
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:12194: in _index_to_gather
    raise IndexError(msg)
E   IndexError: Array slice indices must have static start/stop/step to be used with NumPy indexing syntax. Found
  slice(99, Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace>, None). To index a statically sized
  array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does not support dynamically sized arrays
  within JIT compiled functions).
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
______________________ TestKernelB.test_kernel_b_predict _______________________
Test/scripts/code_structure.py:621: in test_kernel_b_predict
    output = kernel_b_predict(signal, prng_key, config_obj)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Value 'B' with type <class 'str'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
______________________ TestKernelC.test_kernel_c_predict _______________________
Test/scripts/code_structure.py:647: in test_kernel_c_predict
    output = kernel_c_predict(signal, prng_key, config_obj)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Value 'C' with type <class 'str'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
______________________ TestKernelD.test_kernel_d_predict _______________________
Test/scripts/code_structure.py:676: in test_kernel_d_predict
    output = kernel_d_predict(signal, prng_key, config_obj)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Value 'D' with type <class 'str'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
______________________ TestAPIWarmup.test_warmup_kernel_a ______________________
Test/scripts/code_structure.py:685: in test_warmup_kernel_a
    time_ms = warmup_kernel_a(config_obj, prng_key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:62: in warmup_kernel_a
    _ = kernel_a_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:797: in kernel_a_predict
    wiener_hopf_filter = compute_wiener_hopf_filter(
Python/kernels/kernel_a.py:518: in compute_wiener_hopf_filter
    autocorr = autocorr_full[n - 1 : n - 1 + order + 1] / n
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/array_methods.py:1050: in op
    return getattr(self.aval, f"_{name}")(self, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/array_methods.py:646: in _getitem
    return lax_numpy._rewriting_take(self, item)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:11939: in _rewriting_take
    return _gather(arr, treedef, static_idx, dynamic_idx, indices_are_sorted,
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:11948: in _gather
    indexer = _index_to_gather(shape(arr), idx)  # shared with _scatter_update
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:12194: in _index_to_gather
    raise IndexError(msg)
E   IndexError: Array slice indices must have static start/stop/step to be used with NumPy indexing syntax. Found
  slice(99, Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace>, None). To index a statically sized
  array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does not support dynamically sized arrays
  within JIT compiled functions).
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
______________________ TestAPIWarmup.test_warmup_kernel_b ______________________
Test/scripts/code_structure.py:690: in test_warmup_kernel_b
    time_ms = warmup_kernel_b(config_obj, prng_key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:94: in warmup_kernel_b
    _ = kernel_b_predict(dummy_signal, key, config, ema_variance=jnp.array(config.numerical_epsilon))
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Value 'B' with type <class 'str'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
______________________ TestAPIWarmup.test_warmup_kernel_c ______________________
Test/scripts/code_structure.py:695: in test_warmup_kernel_c
    time_ms = warmup_kernel_c(config_obj, prng_key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:128: in warmup_kernel_c
    _ = kernel_c_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Value 'C' with type <class 'str'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
______________________ TestAPIWarmup.test_warmup_kernel_d ______________________
Test/scripts/code_structure.py:700: in test_warmup_kernel_d
    time_ms = warmup_kernel_d(config_obj, prng_key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:160: in warmup_kernel_d
    _ = kernel_d_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Value 'D' with type <class 'str'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
____________________ TestAPIWarmup.test_warmup_all_kernels _____________________
Test/scripts/code_structure.py:705: in test_warmup_all_kernels
    results = warmup_all_kernels(config_obj, key=prng_key, verbose=False)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:304: in warmup_all_kernels
    timings["kernel_a"] = warmup_kernel_a(config, keys[0])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:62: in warmup_kernel_a
    _ = kernel_a_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:797: in kernel_a_predict
    wiener_hopf_filter = compute_wiener_hopf_filter(
Python/kernels/kernel_a.py:518: in compute_wiener_hopf_filter
    autocorr = autocorr_full[n - 1 : n - 1 + order + 1] / n
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/array_methods.py:1050: in op
    return getattr(self.aval, f"_{name}")(self, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/array_methods.py:646: in _getitem
    return lax_numpy._rewriting_take(self, item)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:11939: in _rewriting_take
    return _gather(arr, treedef, static_idx, dynamic_idx, indices_are_sorted,
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:11948: in _gather
    indexer = _index_to_gather(shape(arr), idx)  # shared with _scatter_update
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:12194: in _index_to_gather
    raise IndexError(msg)
E   IndexError: Array slice indices must have static start/stop/step to be used with NumPy indexing syntax. Found
  slice(99, Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace>, None). To index a statically sized
  array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does not support dynamically sized arrays
  within JIT compiled functions).
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
_____________________ TestAPIWarmup.test_warmup_with_retry _____________________
Python/api/warmup.py:368: in warmup_with_retry
    timings = warmup_all_kernels(config, verbose=verbose)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:304: in warmup_all_kernels
    timings["kernel_a"] = warmup_kernel_a(config, keys[0])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:62: in warmup_kernel_a
    _ = kernel_a_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:797: in kernel_a_predict
    wiener_hopf_filter = compute_wiener_hopf_filter(
Python/kernels/kernel_a.py:518: in compute_wiener_hopf_filter
    autocorr = autocorr_full[n - 1 : n - 1 + order + 1] / n
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/array_methods.py:1050: in op
    return getattr(self.aval, f"_{name}")(self, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/array_methods.py:646: in _getitem
    return lax_numpy._rewriting_take(self, item)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:11939: in _rewriting_take
    return _gather(arr, treedef, static_idx, dynamic_idx, indices_are_sorted,
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:11948: in _gather
    indexer = _index_to_gather(shape(arr), idx)  # shared with _scatter_update
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:12194: in _index_to_gather
    raise IndexError(msg)
E   IndexError: Array slice indices must have static start/stop/step to be used with NumPy indexing syntax. Found
  slice(99, Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace>, None). To index a statically sized
  array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does not support dynamically sized arrays
  within JIT compiled functions).
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.

The above exception was the direct cause of the following exception:
Test/scripts/code_structure.py:710: in test_warmup_with_retry
    results = warmup_with_retry(config_obj, max_retries=1, verbose=False)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:377: in warmup_with_retry
    raise RuntimeError(f"Warm-up failed after {max_retries} attempts: {e}") from e
E   RuntimeError: Warm-up failed after 1 attempts: Array slice indices must have static start/stop/step to be used with
  NumPy indexing syntax. Found slice(99, Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace>, None). To
  index a statically sized array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does not support
  dynamically sized arrays within JIT compiled functions).
___________ TestAPIWarmup.test_profile_warmup_and_recommend_timeout ____________
Test/scripts/code_structure.py:715: in test_profile_warmup_and_recommend_timeout
    results = profile_warmup_and_recommend_timeout(config_obj, verbose=False)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:431: in profile_warmup_and_recommend_timeout
    timings = warmup_all_kernels(config, verbose=verbose)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:304: in warmup_all_kernels
    timings["kernel_a"] = warmup_kernel_a(config, keys[0])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:62: in warmup_kernel_a
    _ = kernel_a_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:797: in kernel_a_predict
    wiener_hopf_filter = compute_wiener_hopf_filter(
Python/kernels/kernel_a.py:518: in compute_wiener_hopf_filter
    autocorr = autocorr_full[n - 1 : n - 1 + order + 1] / n
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/array_methods.py:1050: in op
    return getattr(self.aval, f"_{name}")(self, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/array_methods.py:646: in _getitem
    return lax_numpy._rewriting_take(self, item)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:11939: in _rewriting_take
    return _gather(arr, treedef, static_idx, dynamic_idx, indices_are_sorted,
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:11948: in _gather
    indexer = _index_to_gather(shape(arr), idx)  # shared with _scatter_update
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:12194: in _index_to_gather
    raise IndexError(msg)
E   IndexError: Array slice indices must have static start/stop/step to be used with NumPy indexing syntax. Found
  slice(99, Traced<ShapedArray(int64[], weak_type=True)>with<DynamicJaxprTrace>, None). To index a statically sized
  array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does not support dynamically sized arrays
  within JIT compiled functions).
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
=============================== warnings summary ===============================
.venv/lib/python3.13/site-packages/_pytest/config/__init__.py:1303
/Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/.venv/lib/python3.13/site-packages/_pytest/config/__init__.py:1303:
  PytestAssertRewriteWarning: Module already imported so cannot be rewritten; jaxtyping
    self._mark_plugins_for_rewrite(hook, disable_autoload)

Test/scripts/code_structure.py::TestAPIValidation::test_warn_if_invalid
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/api/validation.py:458:
  RuntimeWarning: test warning
    warnings.warn(message, RuntimeWarning)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED Test/scripts/code_structure.py::TestCoreMetaOptimizer::test_walk_forward_split
FAILED Test/scripts/code_structure.py::TestKernelA::test_kernel_a_predict - I...
FAILED Test/scripts/code_structure.py::TestKernelB::test_kernel_b_predict - T...
FAILED Test/scripts/code_structure.py::TestKernelC::test_kernel_c_predict - T...
FAILED Test/scripts/code_structure.py::TestKernelD::test_kernel_d_predict - T...
FAILED Test/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_a - ...
FAILED Test/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_b - ...
FAILED Test/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_c - ...
FAILED Test/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_d - ...
FAILED Test/scripts/code_structure.py::TestAPIWarmup::test_warmup_all_kernels
FAILED Test/scripts/code_structure.py::TestAPIWarmup::test_warmup_with_retry
FAILED Test/scripts/code_structure.py::TestAPIWarmup::test_profile_warmup_and_recommend_timeout
================== 12 failed, 67 passed, 2 warnings in 10.57s ==================

```

---

## 🎯 Final Summary

❌ **12 test(s) failed out of 79.**

**Recommended Actions:**

1. Review failed test details in the output above
2. Fix the underlying code issues
3. Re-run tests to verify fixes

⚠️ **2 warning(s) detected** - consider addressing them.

**Test Duration:** 10.57 seconds

**Report generated at:** 2026-02-21 15:19:14 UTC
