# 🧪 Code Structure Tests Report

**Generated:** 2026-02-21 01:20:35 UTC

## 📊 Executive Summary

❌ **Overall Status:** FAIL

| Metric | Value |
| --- | --- |
| Total Tests | 79 |
| Passed | 67 (84.8%) |
| Failed | 12 (15.2%) |
| Warnings | 3 |
| Duration | 10.53s |
| Exit Code | 1 |

---

## 📝 Detailed Test Output

⚠️ **Attention:** Some tests failed. Review the output below.

```text
=================================================== test session starts
  ====================================================
platform darwin -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0 --
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/.venv/bin/python
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR
plugins: jaxtyping-0.3.9, hypothesis-6.151.9, cov-7.0.0
collecting ... collected 79 items

tests/scripts/code_structure.py::TestBasicSetup::test_get_config PASSED                                              [
  1%]
tests/scripts/code_structure.py::TestBasicSetup::test_initialize_jax_prng PASSED                                     [
  2%]
tests/scripts/code_structure.py::TestAPIConfig::test_get_config PASSED                                               [
  3%]
tests/scripts/code_structure.py::TestAPIConfig::test_PredictorConfigInjector PASSED                                  [
  5%]
tests/scripts/code_structure.py::TestAPIConfig::test_ConfigManager PASSED                                            [
  6%]
tests/scripts/code_structure.py::TestAPIConfig::test_PredictorConfig PASSED                                          [
  7%]
tests/scripts/code_structure.py::TestAPIPRNG::test_initialize_jax_prng PASSED                                        [
  8%]
tests/scripts/code_structure.py::TestAPIPRNG::test_split_key PASSED                                                  [
  10%]
tests/scripts/code_structure.py::TestAPIPRNG::test_split_key_like PASSED                                             [
  11%]
tests/scripts/code_structure.py::TestAPIPRNG::test_uniform_samples PASSED                                            [
  12%]
tests/scripts/code_structure.py::TestAPIPRNG::test_normal_samples PASSED                                             [
  13%]
tests/scripts/code_structure.py::TestAPIPRNG::test_exponential_samples PASSED                                        [
  15%]
tests/scripts/code_structure.py::TestAPIPRNG::test_check_prng_state PASSED                                           [
  16%]
tests/scripts/code_structure.py::TestAPIPRNG::test_verify_determinism PASSED                                         [
  17%]
tests/scripts/code_structure.py::TestAPITypes::test_ProcessState PASSED                                              [
  18%]
tests/scripts/code_structure.py::TestAPITypes::test_KernelType PASSED                                                [
  20%]
tests/scripts/code_structure.py::TestAPITypes::test_OperatingMode PASSED                                             [
  21%]
tests/scripts/code_structure.py::TestAPITypes::test_check_jax_config PASSED                                          [
  22%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_magnitude PASSED                                   [
  24%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_timestamp PASSED                                   [
  25%]
tests/scripts/code_structure.py::TestAPIValidation::test_check_staleness PASSED                                      [
  26%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_shape PASSED                                       [
  27%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_finite PASSED                                      [
  29%]
tests/scripts/code_structure.py::TestAPIValidation::test_sanitize_array PASSED                                       [
  30%]
tests/scripts/code_structure.py::TestAPIValidation::test_ensure_float64 PASSED                                       [
  31%]
tests/scripts/code_structure.py::TestAPIValidation::test_cast_array_to_float64 PASSED                                [
  32%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_holder_exponent PASSED                             [
  34%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_alpha_stable PASSED                                [
  35%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_beta_stable PASSED                                 [
  36%]
tests/scripts/code_structure.py::TestAPIValidation::test_validate_simplex PASSED                                     [
  37%]
tests/scripts/code_structure.py::TestAPIValidation::test_sanitize_external_observation PASSED                        [
  39%]
tests/scripts/code_structure.py::TestAPIValidation::test_warn_if_invalid PASSED                                      [
  40%]
tests/scripts/code_structure.py::TestAPISchemas::test_ProcessStateSchema PASSED                                      [
  41%]
tests/scripts/code_structure.py::TestAPISchemas::test_OperatingModeSchema PASSED                                     [
  43%]
tests/scripts/code_structure.py::TestAPIStateBuffer::test_update_signal_history PASSED                               [
  44%]
tests/scripts/code_structure.py::TestAPIStateBuffer::test_batch_update_signal_history PASSED                         [
  45%]
tests/scripts/code_structure.py::TestAPIStateBuffer::test_reset_cusum_statistics PASSED                              [
  46%]
tests/scripts/code_structure.py::TestAPIStateBuffer::test_update_ema_variance PASSED                                 [
  48%]
tests/scripts/code_structure.py::TestAPIStateBuffer::test_update_residual_buffer PASSED                              [
  49%]
tests/scripts/code_structure.py::TestCoreOrchestrator::test_initialize_state PASSED                                  [
  50%]
tests/scripts/code_structure.py::TestCoreOrchestrator::test_initialize_batched_states PASSED                         [
  51%]
tests/scripts/code_structure.py::TestCoreOrchestrator::test_compute_entropy_ratio PASSED                             [
  53%]
tests/scripts/code_structure.py::TestCoreOrchestrator::test_scale_dgm_architecture PASSED                            [
  54%]
tests/scripts/code_structure.py::TestCoreOrchestrator::test_compute_adaptive_stiffness_thresholds PASSED             [
  55%]
tests/scripts/code_structure.py::TestCoreOrchestrator::test_compute_adaptive_jko_params PASSED                       [
  56%]
tests/scripts/code_structure.py::TestCoreFusion::test_compute_sinkhorn_epsilon PASSED                                [
  58%]
tests/scripts/code_structure.py::TestCoreMetaOptimizer::test_walk_forward_split FAILED                               [
  59%]
tests/scripts/code_structure.py::TestKernelsBase::test_apply_stop_gradient_to_diagnostics PASSED                     [
  60%]
tests/scripts/code_structure.py::TestKernelsBase::test_compute_signal_statistics PASSED                              [
  62%]
tests/scripts/code_structure.py::TestKernelsBase::test_normalize_signal PASSED                                       [
  63%]
tests/scripts/code_structure.py::TestKernelsBase::test_validate_kernel_input PASSED                                  [
  64%]
tests/scripts/code_structure.py::TestKernelA::test_gaussian_kernel PASSED                                            [
  65%]
tests/scripts/code_structure.py::TestKernelA::test_compute_gram_matrix PASSED                                        [
  67%]
tests/scripts/code_structure.py::TestKernelA::test_kernel_ridge_regression PASSED                                    [
  68%]
tests/scripts/code_structure.py::TestKernelA::test_create_embedding PASSED                                           [
  69%]
tests/scripts/code_structure.py::TestKernelA::test_kernel_a_predict FAILED                                           [
  70%]
tests/scripts/code_structure.py::TestKernelB::test_dgm_hjb_solver PASSED                                             [
  72%]
tests/scripts/code_structure.py::TestKernelB::test_kernel_b_predict FAILED                                           [
  73%]
tests/scripts/code_structure.py::TestKernelC::test_drift_levy_stable PASSED                                          [
  74%]
tests/scripts/code_structure.py::TestKernelC::test_diffusion_levy PASSED                                             [
  75%]
tests/scripts/code_structure.py::TestKernelC::test_kernel_c_predict FAILED                                           [
  77%]
tests/scripts/code_structure.py::TestKernelD::test_create_path_augmentation PASSED                                   [
  78%]
tests/scripts/code_structure.py::TestKernelD::test_compute_log_signature PASSED                                      [
  79%]
tests/scripts/code_structure.py::TestKernelD::test_predict_from_signature PASSED                                     [
  81%]
tests/scripts/code_structure.py::TestKernelD::test_kernel_d_predict FAILED                                           [
  82%]
tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_a FAILED                                          [
  83%]
tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_b FAILED                                          [
  84%]
tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_c FAILED                                          [
  86%]
tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_d FAILED                                          [
  87%]
tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_all_kernels FAILED                                       [
  88%]
tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_with_retry FAILED                                        [
  89%]
tests/scripts/code_structure.py::TestAPIWarmup::test_profile_warmup_and_recommend_timeout FAILED                     [
  91%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_config_mutation_module_exists PASSED                   [
  92%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_credentials_module_exists PASSED                       [
  93%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_dashboard_module_exists PASSED                         [
  94%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_loaders_module_exists PASSED                           [
  96%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_snapshots_module_exists PASSED                         [
  97%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_telemetry_module_exists PASSED                         [
  98%]
tests/scripts/code_structure.py::TestIOModuleImportable::test_validators_module_exists PASSED
  [100%]

========================================================= FAILURES
  =========================================================
______________________________________ TestCoreMetaOptimizer.test_walk_forward_split
  _______________________________________
tests/scripts/code_structure.py:537: in test_walk_forward_split
    assert abs(actual_ratio - train_ratio) < 0.05, (
E   AssertionError: Split 0 ratio mismatch: 0.921 vs expected 0.700
E   assert 0.2210526315789474 < 0.05
E    +  where 0.2210526315789474 = abs((0.9210526315789473 - 0.7))
____________________________________________ TestKernelA.test_kernel_a_predict
  _____________________________________________
tests/scripts/code_structure.py:606: in test_kernel_a_predict
    output = kernel_a_predict(signal, prng_key, config_obj)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:821: in kernel_a_predict
    holder_exponent_estimate = extract_holder_exponent_wtmm(signal_normalized, config)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:421: in extract_holder_exponent_wtmm
    holder_exponent_raw, spectrum_max = compute_singularity_spectrum(
Python/kernels/kernel_a.py:319: in compute_singularity_spectrum
    h_range = jnp.linspace(h_min, h_max, h_steps)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:6925: in linspace
    num = core.concrete_dim_or_error(num, "'num' argument of jnp.linspace")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array
  with shape int64[]
E   'num' argument of jnp.linspace
E   The error occurred while tracing the function compute_singularity_spectrum at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_a.py:284 for jit.
  This concrete value was not available in Python because it depends on the value of the argument h_steps.
E   
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
____________________________________________ TestKernelB.test_kernel_b_predict
  _____________________________________________
tests/scripts/code_structure.py:621: in test_kernel_b_predict
    output = kernel_b_predict(signal, prng_key, config_obj)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_b.py:403: in kernel_b_predict
    entropy_threshold_adaptive = compute_adaptive_entropy_threshold(ema_variance, config)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_b.py:310: in compute_adaptive_entropy_threshold
    return float(gamma)
           ^^^^^^^^^^^^
E   jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array
  with shape float64[]
E   The problem arose with the `float` function. If trying to convert the data type of a value, try using
  `x.astype(float)` or `jnp.array(x, float)` instead.
E   The error occurred while tracing the function kernel_b_predict at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_b.py:313 for jit.
  This value became a tracer due to JAX operations on these lines:
E   
E     operation a:f64[] = convert_element_type[new_dtype=float64 weak_type=False] b
E       from line
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_b.py:190:16
  (compute_entropy_dgm)
E   
E     operation a:f64[] = convert_element_type[new_dtype=float64 weak_type=True] b
E       from line
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_b.py:400:23
  (kernel_b_predict)
E   
E     operation a:f64[] = max b c
E       from line
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_b.py:293:23
  (compute_adaptive_entropy_threshold)
E   
E     operation a:bool[] = gt b c
E       from line
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_b.py:301:8
  (compute_adaptive_entropy_threshold)
E   
E     operation a:bool[] = lt b c
E       from line
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_b.py:304:12
  (compute_adaptive_entropy_threshold)
E   
E   (Additional originating lines are not shown.)
E   
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
____________________________________________ TestKernelC.test_kernel_c_predict
  _____________________________________________
tests/scripts/code_structure.py:647: in test_kernel_c_predict
    output = kernel_c_predict(signal, prng_key, config_obj)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_c.py:408: in kernel_c_predict
    prediction, diagnostics = apply_stop_gradient_to_diagnostics(
Python/kernels/base.py:165: in apply_stop_gradient_to_diagnostics
    diagnostics_stopped = jax.tree_map(jax.lax.stop_gradient, diagnostics)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Value 'C_Ito_Levy_SDE' with type <class 'str'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
____________________________________________ TestKernelD.test_kernel_d_predict
  _____________________________________________
tests/scripts/code_structure.py:676: in test_kernel_d_predict
    output = kernel_d_predict(signal, prng_key, config_obj)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_d.py:265: in kernel_d_predict
    prediction, diagnostics = apply_stop_gradient_to_diagnostics(
Python/kernels/base.py:165: in apply_stop_gradient_to_diagnostics
    diagnostics_stopped = jax.tree_map(jax.lax.stop_gradient, diagnostics)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Value 'D_Signature_Rough_Paths' with type <class 'str'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
____________________________________________ TestAPIWarmup.test_warmup_kernel_a
  ____________________________________________
tests/scripts/code_structure.py:685: in test_warmup_kernel_a
    time_ms = warmup_kernel_a(config_obj, prng_key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:62: in warmup_kernel_a
    _ = kernel_a_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:821: in kernel_a_predict
    holder_exponent_estimate = extract_holder_exponent_wtmm(signal_normalized, config)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:421: in extract_holder_exponent_wtmm
    holder_exponent_raw, spectrum_max = compute_singularity_spectrum(
Python/kernels/kernel_a.py:319: in compute_singularity_spectrum
    h_range = jnp.linspace(h_min, h_max, h_steps)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:6925: in linspace
    num = core.concrete_dim_or_error(num, "'num' argument of jnp.linspace")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array
  with shape int64[]
E   'num' argument of jnp.linspace
E   The error occurred while tracing the function compute_singularity_spectrum at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_a.py:284 for jit.
  This concrete value was not available in Python because it depends on the value of the argument h_steps.
E   
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
____________________________________________ TestAPIWarmup.test_warmup_kernel_b
  ____________________________________________
tests/scripts/code_structure.py:690: in test_warmup_kernel_b
    time_ms = warmup_kernel_b(config_obj, prng_key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:94: in warmup_kernel_b
    _ = kernel_b_predict(dummy_signal, key, config, ema_variance=jnp.array(config.numerical_epsilon))
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_b.py:403: in kernel_b_predict
    entropy_threshold_adaptive = compute_adaptive_entropy_threshold(ema_variance, config)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_b.py:310: in compute_adaptive_entropy_threshold
    return float(gamma)
           ^^^^^^^^^^^^
E   jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array
  with shape float64[]
E   The problem arose with the `float` function. If trying to convert the data type of a value, try using
  `x.astype(float)` or `jnp.array(x, float)` instead.
E   The error occurred while tracing the function kernel_b_predict at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_b.py:313 for jit.
  This value became a tracer due to JAX operations on these lines:
E   
E     operation a:f64[] = convert_element_type[new_dtype=float64 weak_type=False] b
E       from line
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_b.py:190:16
  (compute_entropy_dgm)
E   
E     operation a:f64[] = max b c
E       from line
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_b.py:293:23
  (compute_adaptive_entropy_threshold)
E   
E     operation a:bool[] = gt b c
E       from line
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_b.py:301:8
  (compute_adaptive_entropy_threshold)
E   
E     operation a:bool[] = lt b c
E       from line
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_b.py:304:12
  (compute_adaptive_entropy_threshold)
E   
E     operation a:f64[] = pjit[
E     name=_where
E     jaxpr={ lambda ; b:bool[] c:f64[] d:f64[]. let
E         e:f64[] = select_n b d c
E       in (e,) }
E   ] f g h
E       from line
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_b.py:303:8
  (compute_adaptive_entropy_threshold)
E   
E   (Additional originating lines are not shown.)
E   
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
____________________________________________ TestAPIWarmup.test_warmup_kernel_c
  ____________________________________________
tests/scripts/code_structure.py:695: in test_warmup_kernel_c
    time_ms = warmup_kernel_c(config_obj, prng_key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:128: in warmup_kernel_c
    _ = kernel_c_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_c.py:408: in kernel_c_predict
    prediction, diagnostics = apply_stop_gradient_to_diagnostics(
Python/kernels/base.py:165: in apply_stop_gradient_to_diagnostics
    diagnostics_stopped = jax.tree_map(jax.lax.stop_gradient, diagnostics)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Value 'C_Ito_Levy_SDE' with type <class 'str'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
____________________________________________ TestAPIWarmup.test_warmup_kernel_d
  ____________________________________________
tests/scripts/code_structure.py:700: in test_warmup_kernel_d
    time_ms = warmup_kernel_d(config_obj, prng_key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:160: in warmup_kernel_d
    _ = kernel_d_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_d.py:265: in kernel_d_predict
    prediction, diagnostics = apply_stop_gradient_to_diagnostics(
Python/kernels/base.py:165: in apply_stop_gradient_to_diagnostics
    diagnostics_stopped = jax.tree_map(jax.lax.stop_gradient, diagnostics)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   TypeError: Value 'D_Signature_Rough_Paths' with type <class 'str'> is not a valid JAX type
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
__________________________________________ TestAPIWarmup.test_warmup_all_kernels
  ___________________________________________
tests/scripts/code_structure.py:705: in test_warmup_all_kernels
    results = warmup_all_kernels(config_obj, key=prng_key, verbose=False)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:313: in warmup_all_kernels
    timings["kernel_a"] = warmup_kernel_a(config, keys[0])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:62: in warmup_kernel_a
    _ = kernel_a_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:821: in kernel_a_predict
    holder_exponent_estimate = extract_holder_exponent_wtmm(signal_normalized, config)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:421: in extract_holder_exponent_wtmm
    holder_exponent_raw, spectrum_max = compute_singularity_spectrum(
Python/kernels/kernel_a.py:319: in compute_singularity_spectrum
    h_range = jnp.linspace(h_min, h_max, h_steps)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:6925: in linspace
    num = core.concrete_dim_or_error(num, "'num' argument of jnp.linspace")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array
  with shape int64[]
E   'num' argument of jnp.linspace
E   The error occurred while tracing the function compute_singularity_spectrum at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_a.py:284 for jit.
  This concrete value was not available in Python because it depends on the value of the argument h_steps.
E   
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
___________________________________________ TestAPIWarmup.test_warmup_with_retry
  ___________________________________________
Python/api/warmup.py:379: in warmup_with_retry
    timings = warmup_all_kernels(config, verbose=verbose)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:313: in warmup_all_kernels
    timings["kernel_a"] = warmup_kernel_a(config, keys[0])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:62: in warmup_kernel_a
    _ = kernel_a_predict(dummy_signal, key, config)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:821: in kernel_a_predict
    holder_exponent_estimate = extract_holder_exponent_wtmm(signal_normalized, config)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:421: in extract_holder_exponent_wtmm
    holder_exponent_raw, spectrum_max = compute_singularity_spectrum(
Python/kernels/kernel_a.py:319: in compute_singularity_spectrum
    h_range = jnp.linspace(h_min, h_max, h_steps)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:6925: in linspace
    num = core.concrete_dim_or_error(num, "'num' argument of jnp.linspace")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array
  with shape int64[]
E   'num' argument of jnp.linspace
E   The error occurred while tracing the function compute_singularity_spectrum at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_a.py:284 for jit.
  This concrete value was not available in Python because it depends on the value of the argument h_steps.
E   
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.

The above exception was the direct cause of the following exception:
tests/scripts/code_structure.py:710: in test_warmup_with_retry
    results = warmup_with_retry(config_obj, max_retries=1, verbose=False)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/api/warmup.py:388: in warmup_with_retry
    raise RuntimeError(
E   RuntimeError: Warm-up failed after 1 attempts: Abstract tracer value encountered where concrete value is expected:
  traced array with shape int64[]
E   'num' argument of jnp.linspace
E   The error occurred while tracing the function compute_singularity_spectrum at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_a.py:284 for jit.
  This concrete value was not available in Python because it depends on the value of the argument h_steps.
E   
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
_________________________________ TestAPIWarmup.test_profile_warmup_and_recommend_timeout
  __________________________________
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
Python/kernels/kernel_a.py:821: in kernel_a_predict
    holder_exponent_estimate = extract_holder_exponent_wtmm(signal_normalized, config)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Python/kernels/kernel_a.py:421: in extract_holder_exponent_wtmm
    holder_exponent_raw, spectrum_max = compute_singularity_spectrum(
Python/kernels/kernel_a.py:319: in compute_singularity_spectrum
    h_range = jnp.linspace(h_min, h_max, h_steps)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/jax/_src/numpy/lax_numpy.py:6925: in linspace
    num = core.concrete_dim_or_error(num, "'num' argument of jnp.linspace")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array
  with shape int64[]
E   'num' argument of jnp.linspace
E   The error occurred while tracing the function compute_singularity_spectrum at
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/kernel_a.py:284 for jit.
  This concrete value was not available in Python because it depends on the value of the argument h_steps.
E   
E   See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
E   --------------------
E   For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set
  JAX_TRACEBACK_FILTERING=off to include these.
===================================================== warnings summary
  =====================================================
.venv/lib/python3.13/site-packages/_pytest/config/__init__.py:1303
/Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/.venv/lib/python3.13/site-packages/_pytest/config/__init__.py:1303:
  PytestAssertRewriteWarning: Module already imported so cannot be rewritten; jaxtyping
    self._mark_plugins_for_rewrite(hook, disable_autoload)

tests/scripts/code_structure.py::TestAPIValidation::test_warn_if_invalid
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/api/validation.py:502:
  RuntimeWarning: test warning
    warnings.warn(message, RuntimeWarning)

tests/scripts/code_structure.py::TestKernelsBase::test_apply_stop_gradient_to_diagnostics
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/kernels/base.py:165:
  DeprecationWarning: jax.tree_map is deprecated: use jax.tree.map (jax v0.4.25 or newer) or jax.tree_util.tree_map (any
  JAX version).
    diagnostics_stopped = jax.tree_map(jax.lax.stop_gradient, diagnostics)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================================================= short test summary info
  ==================================================
FAILED tests/scripts/code_structure.py::TestCoreMetaOptimizer::test_walk_forward_split - AssertionError: Split 0 ratio
  mi...
FAILED tests/scripts/code_structure.py::TestKernelA::test_kernel_a_predict - jax.errors.ConcretizationTypeError:
  Abstract...
FAILED tests/scripts/code_structure.py::TestKernelB::test_kernel_b_predict - jax.errors.ConcretizationTypeError:
  Abstract...
FAILED tests/scripts/code_structure.py::TestKernelC::test_kernel_c_predict - TypeError: Value 'C_Ito_Levy_SDE' with type
  ...
FAILED tests/scripts/code_structure.py::TestKernelD::test_kernel_d_predict - TypeError: Value 'D_Signature_Rough_Paths'
  w...
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_a - jax.errors.ConcretizationTypeError:
  Abstrac...
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_b - jax.errors.ConcretizationTypeError:
  Abstrac...
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_c - TypeError: Value 'C_Ito_Levy_SDE' with
  type...
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_kernel_d - TypeError: Value 'D_Signature_Rough_Paths'
  ...
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_all_kernels - jax.errors.ConcretizationTypeError:
  Abst...
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_warmup_with_retry - RuntimeError: Warm-up failed after 1
  atte...
FAILED tests/scripts/code_structure.py::TestAPIWarmup::test_profile_warmup_and_recommend_timeout -
  jax.errors.Concretizat...
======================================== 12 failed, 67 passed, 3 warnings in 10.53s
  ========================================

```

---

## 🎯 Final Summary

❌ **12 test(s) failed out of 79.**

**Recommended Actions:**

1. Review failed test details in the output above
2. Fix the underlying code issues
3. Re-run tests to verify fixes

⚠️ **3 warning(s) detected** - consider addressing them.

**Test Duration:** 10.53 seconds

**Report generated at:** 2026-02-21 01:20:35 UTC
