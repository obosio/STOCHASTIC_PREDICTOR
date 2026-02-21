# 🧪 Code Structure Tests Report

**Generated:** 2026-02-21 12:53:13 UTC

## 📊 Executive Summary

❌ **Overall Status:** FAIL

| Metric | Value |
| --- | --- |
| Total Tests | 79 |
| Passed | 67 (84.8%) |
| Failed | 12 (15.2%) |
| Warnings | 2 |
| Duration | 9.91s |
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

tests/scripts/code_structure.py::TestBasicSetup::test_get_config PASSED  [  1%]
tests/scripts/code_structure.py::TestBasicSetup::test_initialize_jax_prng PASSED [  2%]
tests/scripts/code_structure.py::TestAPIConfig::test_get_config PASSED   [  3%]
tests/scripts/code_structure.py::TestAPIConfig::test_PredictorConfigInjector PASSED [  5%]
tests/scripts/code_structure.py::TestAPIConfig::test_ConfigManager PASSED [  6%]
tests/scripts/code_structure.py::TestAPIConfig::test_PredictorConfig PASSED [  7%]
tests/scripts/code_structure.py::TestAPIPRNG::test_initialize_jax_prng PASSED [  8%]
tests/scripts/code_structure.py::TestAPIPRNG::test_split_key PASSED      [ 10%]
tests/scripts/code_structure.py::TestAPIPRNG::test_split_key_like PASSED [ 11%]
tests/scripts/code_structure.py::TestAPIPRNG::test_uniform_samples PASSED [ 12%]
tests/scripts/code_structure.py::TestAPIPRNG::test_normal_samples PASSED [ 13%]
tests/scripts/code_structure.py::TestAPIPRNG::test_exponential_samples PASSED [ 15%]
tests/scripts/code_structure.py::TestAPIPRNG::test_check_prng_state PASSED [ 16%]
tests/scripts/code_structure.py::TestAPIPRNG::test_verify_determinism PASSED [ 17%]
tests/scripts/code_structure.py::TestAPITypes::test_ProcessState PASSED  [ 18%]
tests/scripts/code_structure.py::TestAPITypes::test_KernelType PASSED    [ 20%]
tests/scripts/code_structure.py::TestAPITypes::test_OperatingMode PASSED [ 21%]
tests/scripts/code_structure.py::TestAPITypes::test_check_jax_config PASSED [ 22%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_magnitude PASSED [ 24%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_timestamp PASSED [ 25%]
tests/scripts/code_structure.py::TestAPIValidation::test_check_staleness PASSED [ 26%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_shape PASSED [ 27%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_finite PASSED [ 29%]
tests/scripts/code_structure.py::TestAPIValidation::test_sanitize_array PASSED [ 30%]
tests/scripts/code_structure.py::TestAPIValidation::test_ensure_float64 PASSED [ 31%]
tests/scripts/code_structure.py::TestAPIValidation::test_cast_array_to_float64 PASSED [ 32%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_holder_exponent PASSED [ 34%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_alpha_stable PASSED [ 35%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_beta_stable PASSED [ 36%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_simplex PASSED [ 37%]
tests/scripts/code_structure.py::TestAPIValidation::test_sanitize_external_observation PASSED [ 39%]
tests/scripts/code_structure.py::TestAPIValidation::test_warn_if_invalid PASSED [ 40%]
tests/scripts/code_structure.py::TestAPISchemas::test_ProcessStateSchema PASSED [ 41%]
tests/scripts/code_structure.py::TestAPISchemas::test_OperatingModeSchema PASSED [ 43%]
tests/scripts/code_structure.py::TestAPIStateBuffer::test_update_signal_history PASSED [ 44%]
tests/scripts/code_structure.py::TestAPIStateBuffer::test_batch_update_signal_history PASSED [ 45%]
tests/scripts/code_structure.py::TestAPIStateBuffer::test_reset_cusum_statistics PASSED [ 46%]
tests/scripts/code_structure.py::TestAPIStateBuffer::test_update_ema_variance PASSED [ 48%]
tests/scripts/code_structure.py::TestAPIStateBuffer::test_update_residual_buffer PASSED [ 49%]
tests/scripts/code_structure.py::TestCoreOrchestrator::test_initialize_state PASSED [ 50%]
tests/scripts/code_structure.py::TestCoreOrchestrator::test_initialize_batched_states PASSED [ 51%]
tests/scripts/code_structure.py::TestCoreOrchestrator::test_compute_entropy_ratio PASSED [ 53%]
tests/scripts/code_structure.py::TestCoreOrchestrator::test_scale_dgm_architecture PASSED [ 54%]
tests/scripts/code_structure.py::TestCoreOrchestrator::test_compute_adaptive_stiffness_thresholds PASSED [ 55%]
tests/scripts/code_structure.py::TestCoreOrchestrator::test_compute_adaptive_jko_params PASSED [ 56%]
tests/scripts/code_structure.py::TestCoreFusion::test_compute_sinkhorn_epsilon PASSED [ 58%]
tests/scripts/code_structure.py::TestCoreMetaOptimizer::test_walk_forward_split FAILED [ 59%]
tests/scripts/code_structure.py::TestKernelsBase::test_apply_stop_gradient_to_diagnostics PASSED [ 60%]
tests/scripts/code_structure.py::TestKernelsBase::test_compute_signal_statistics PASSED [ 62%]
tests/scripts/code_structure.py::TestKernelsBase::test_normalize_signal PASSED [ 63%]
tests/scripts/code_structure.py::TestKernelsBase::test_validate_kernel_input PASSED [ 64%]
tests/scripts/code_structure.py::TestKernelA::test_gaussian_kernel PASSED [ 65%]
tests/scripts/code_structure.py::TestKernelA::test_compute_gram_matrix PASSED [ 67%]
tests/scripts/code_structure.py::TestKernelA::test_kernel_ridge_regression PASSED [ 68%]
tests/scripts/code_structure.py::TestKernelA::test_create_embedding PASSED [ 69%]
tests/scripts/code_structure.py::TestKernelA::test_kernel_a_predict FAILED [ 70%]
tests/scripts/code_structure.py::TestKernelB::test_dgm_hjb_solver PASSED [ 72%]
tests/scripts/code_structure.py::TestKernelB::test_kernel_b_predict FAILED [ 73%]
tests/scripts/code_structure.py::TestKernelC::test_drift_levy_stable PASSED [ 74%]
tests/scripts/code_structure.py::TestKernelC::test_diffusion_levy PASSED [ 75%]
tests/scripts/code_structure.py::TestKernelC::test_kernel_c_predict FAILED [ 77%]
tests/scripts/code_structure.py::TestKernelD::test_create_path_augmentation PASSED [ 78%]
tests/scripts/code_structure.py::TestKernelD::test_compute_log_signature PASSED [ 79%]
tests/scripts/code_structure.py::TestKernelD::test_predict_from_signature PASSED [ 81%]
tests/scripts/code_structure.py::TestKernelD::test_kernel_d_predict FAILED [ 82%]
tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_a FAILED [ 83%]
tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_b FAILED [ 84%]
tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_c FAILED [ 86%]
tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_d FAILED [ 87%]
tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_all_kernels FAILED [ 88%]
tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_with_retry FAILED [ 89%]
tests/scripts/code_structure.py::TestAPIWarmup::test_profile_warmup_and_recommend_timeout FAILED [ 91%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_config_mutation_module_exists PASSED [ 92%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_credentials_module_exists PASSED [ 93%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_dashboard_module_exists PASSED [ 94%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_loaders_module_exists PASSED [ 96%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_snapshots_module_exists PASSED [ 97%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_telemetry_module_exists PASSED [ 98%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_validators_module_exists PASSED [100%]

=================================== FAILURES ===================================
________________ TestCoreMetaOptimizer.test_walk_forward_split _________________
tests/scripts/code_structure.py:537: in test_walk_forward_split
    assert abs(actual_ratio - train_ratio) < 0.05, (
E   AssertionError: Split 0 ratio mismatch: 0.921 vs expected 0.700
E   assert 0.2210526315789474 < 0.05
E    +  where 0.2210526315789474 = abs((0.9210526315789473 - 0.7))
______________________ TestKernelA.test_kernel_a_predict _______________________
tests/scripts/code_structure.py:606: in test_kernel_a_predict
    output = kernel_a_predict(signal, prng_key, config_obj)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:826: in kernel_a_predict
    koopman_freqs, koopman_powers = compute_koopman_spectrum(
Python/kernels/kernel_a.py:475: in compute_koopman_spectrum
    top_power, top_idx = jax.lax.top_k(power, top_k)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   jax.errors.TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
E   The error occurred while tracing the function compute_koopman_spectrum at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_a.py:448 for jit.
  This concrete value was not available in Python because it depends on the value of the argument sampling_interval.
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerBoolConversionError
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
______________________ TestKernelB.test_kernel_b_predict _______________________
tests/scripts/code_structure.py:621: in test_kernel_b_predict
    output = kernel_b_predict(signal, prng_key, config_obj)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Value 'B' with type <class 'str'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
______________________ TestKernelC.test_kernel_c_predict _______________________
tests/scripts/code_structure.py:647: in test_kernel_c_predict
    output = kernel_c_predict(signal, prng_key, config_obj)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Value 'C' with type <class 'str'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
______________________ TestKernelD.test_kernel_d_predict _______________________
tests/scripts/code_structure.py:676: in test_kernel_d_predict
    output = kernel_d_predict(signal, prng_key, config_obj)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Value 'D' with type <class 'str'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
______________________ TestAPIWarmup.test_warmup_kernel_a ______________________
tests/scripts/code_structure.py:685: in test_warmup_kernel_a
    time_ms = warmup_kernel_a(config_obj, prng_key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:62: in warmup_kernel_a
    _ = kernel_a_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:826: in kernel_a_predict
    koopman_freqs, koopman_powers = compute_koopman_spectrum(
Python/kernels/kernel_a.py:475: in compute_koopman_spectrum
    top_power, top_idx = jax.lax.top_k(power, top_k)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   jax.errors.TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
E   The error occurred while tracing the function compute_koopman_spectrum at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_a.py:448 for jit.
  This concrete value was not available in Python because it depends on the value of the argument sampling_interval.
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerBoolConversionError
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
______________________ TestAPIWarmup.test_warmup_kernel_b ______________________
tests/scripts/code_structure.py:690: in test_warmup_kernel_b
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
tests/scripts/code_structure.py:695: in test_warmup_kernel_c
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
tests/scripts/code_structure.py:700: in test_warmup_kernel_d
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
tests/scripts/code_structure.py:705: in test_warmup_all_kernels
    results = warmup_all_kernels(config_obj, key=prng_key, verbose=False)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:313: in warmup_all_kernels
    timings["kernel_a"] = warmup_kernel_a(config, keys[0])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:62: in warmup_kernel_a
    _ = kernel_a_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:826: in kernel_a_predict
    koopman_freqs, koopman_powers = compute_koopman_spectrum(
Python/kernels/kernel_a.py:475: in compute_koopman_spectrum
    top_power, top_idx = jax.lax.top_k(power, top_k)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   jax.errors.TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
E   The error occurred while tracing the function compute_koopman_spectrum at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_a.py:448 for jit.
  This concrete value was not available in Python because it depends on the value of the argument sampling_interval.
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerBoolConversionError
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
_____________________ TestAPIWarmup.test_warmup_with_retry _____________________
Python/api/warmup.py:379: in warmup_with_retry
    timings = warmup_all_kernels(config, verbose=verbose)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:313: in warmup_all_kernels
    timings["kernel_a"] = warmup_kernel_a(config, keys[0])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:62: in warmup_kernel_a
    _ = kernel_a_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:826: in kernel_a_predict
    koopman_freqs, koopman_powers = compute_koopman_spectrum(
Python/kernels/kernel_a.py:475: in compute_koopman_spectrum
    top_power, top_idx = jax.lax.top_k(power, top_k)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   jax.errors.TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
E   The error occurred while tracing the function compute_koopman_spectrum at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_a.py:448 for jit.
  This concrete value was not available in Python because it depends on the value of the argument sampling_interval.
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerBoolConversionError
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.

The above exception was the direct cause of the following exception:
tests/scripts/code_structure.py:710: in test_warmup_with_retry
    results = warmup_with_retry(config_obj, max_retries=1, verbose=False)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:388: in warmup_with_retry
    raise RuntimeError(
E   RuntimeError: Warm-up failed after 1 attempts: Attempted boolean conversion of traced array with shape bool[].
E   The error occurred while tracing the function compute_koopman_spectrum at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_a.py:448 for jit.
  This concrete value was not available in Python because it depends on the value of the argument sampling_interval.
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerBoolConversionError
___________ TestAPIWarmup.test_profile_warmup_and_recommend_timeout ____________
tests/scripts/code_structure.py:715: in test_profile_warmup_and_recommend_timeout
    results = profile_warmup_and_recommend_timeout(config_obj, verbose=False)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:445: in profile_warmup_and_recommend_timeout
    timings = warmup_all_kernels(config, verbose=verbose)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:313: in warmup_all_kernels
    timings["kernel_a"] = warmup_kernel_a(config, keys[0])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:62: in warmup_kernel_a
    _ = kernel_a_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:826: in kernel_a_predict
    koopman_freqs, koopman_powers = compute_koopman_spectrum(
Python/kernels/kernel_a.py:475: in compute_koopman_spectrum
    top_power, top_idx = jax.lax.top_k(power, top_k)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   jax.errors.TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
E   The error occurred while tracing the function compute_koopman_spectrum at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_a.py:448 for jit.
  This concrete value was not available in Python because it depends on the value of the argument sampling_interval.
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerBoolConversionError
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
=============================== warnings summary ===============================
.venv/lib/python3.13/site-packages/_pytest/config/__init__.py:1303
/Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/.venv/lib/python3.13/site-packages/_pytest/config/__init__.py:1303:
  PytestAssertRewriteWarning: Module already imported so cannot be rewritten; jaxtyping
    self._mark_plugins_for_rewrite(hook, disable_autoload)

tests/scripts/code_structure.py::TestAPIValidation::test_warn_if_invalid
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/api/validation.py:502:
  RuntimeWarning: test warning
    warnings.warn(message, RuntimeWarning)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/scripts/code_structure.py::TestCoreMetaOptimizer::test_walk_forward_split
FAILED tests/scripts/code_structure.py::TestKernelA::test_kernel_a_predict - ...
FAILED tests/scripts/code_structure.py::TestKernelB::test_kernel_b_predict - ...
FAILED tests/scripts/code_structure.py::TestKernelC::test_kernel_c_predict - ...
FAILED tests/scripts/code_structure.py::TestKernelD::test_kernel_d_predict - ...
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_a
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_b
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_c
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_d
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_all_kernels
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_with_retry
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_profile_warmup_and_recommend_timeout
================== 12 failed, 67 passed, 2 warnings in 9.91s ===================

```

---

## 🎯 Final Summary

❌ **12 test(s) failed out of 79.**

**Recommended Actions:**

1. Review failed test details in the output above
2. Fix the underlying code issues
3. Re-run tests to verify fixes

⚠️ **2 warning(s) detected** - consider addressing them.

**Test Duration:** 9.91 seconds

**Report generated at:** 2026-02-21 12:53:13 UTC
