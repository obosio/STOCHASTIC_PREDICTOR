# 🧪 Code Structure Tests Report

**Generated:** 2026-02-21 15:58:08 UTC

## 📊 Executive Summary

✅ **Overall Status:** PASS

| Metric | Value |
| --- | --- |
| Total Tests | 79 |
| Passed | 79 (100.0%) |
| Failed | 0 (0.0%) |
| Warnings | 2 |
| Duration | 12.72s |
| Exit Code | 0 |

---

## 📝 Detailed Test Output

```text
============================= test session starts ==============================
platform darwin -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0 --
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/.venv/bin/python
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Test/scripts
plugins: jaxtyping-0.3.9, hypothesis-6.151.9, cov-7.0.0
collecting ... collected 79 items

code_structure.py::TestBasicSetup::test_get_config PASSED                [  1%]
code_structure.py::TestBasicSetup::test_initialize_jax_prng PASSED       [  2%]
code_structure.py::TestAPIConfig::test_get_config PASSED                 [  3%]
code_structure.py::TestAPIConfig::test_PredictorConfigInjector PASSED    [  5%]
code_structure.py::TestAPIConfig::test_ConfigManager PASSED              [  6%]
code_structure.py::TestAPIConfig::test_PredictorConfig PASSED            [  7%]
code_structure.py::TestAPIPRNG::test_initialize_jax_prng PASSED          [  8%]
code_structure.py::TestAPIPRNG::test_split_key PASSED                    [ 10%]
code_structure.py::TestAPIPRNG::test_split_key_like PASSED               [ 11%]
code_structure.py::TestAPIPRNG::test_uniform_samples PASSED              [ 12%]
code_structure.py::TestAPIPRNG::test_normal_samples PASSED               [ 13%]
code_structure.py::TestAPIPRNG::test_exponential_samples PASSED          [ 15%]
code_structure.py::TestAPIPRNG::test_check_prng_state PASSED             [ 16%]
code_structure.py::TestAPIPRNG::test_verify_determinism PASSED           [ 17%]
code_structure.py::TestAPITypes::test_ProcessState PASSED                [ 18%]
code_structure.py::TestAPITypes::test_KernelType PASSED                  [ 20%]
code_structure.py::TestAPITypes::test_OperatingMode PASSED               [ 21%]
code_structure.py::TestAPITypes::test_check_jax_config PASSED            [ 22%]
code_structure.py::TestAPIValidation::test_validate_magnitude PASSED     [ 24%]
code_structure.py::TestAPIValidation::test_validate_timestamp PASSED     [ 25%]
code_structure.py::TestAPIValidation::test_check_staleness PASSED        [ 26%]
code_structure.py::TestAPIValidation::test_validate_shape PASSED         [ 27%]
code_structure.py::TestAPIValidation::test_validate_finite PASSED        [ 29%]
code_structure.py::TestAPIValidation::test_sanitize_array PASSED         [ 30%]
code_structure.py::TestAPIValidation::test_ensure_float64 PASSED         [ 31%]
code_structure.py::TestAPIValidation::test_cast_array_to_float64 PASSED  [ 32%]
code_structure.py::TestAPIValidation::test_validate_holder_exponent PASSED [ 34%]
code_structure.py::TestAPIValidation::test_validate_alpha_stable PASSED  [ 35%]
code_structure.py::TestAPIValidation::test_validate_beta_stable PASSED   [ 36%]
code_structure.py::TestAPIValidation::test_validate_simplex PASSED       [ 37%]
code_structure.py::TestAPIValidation::test_sanitize_external_observation PASSED [ 39%]
code_structure.py::TestAPIValidation::test_warn_if_invalid PASSED        [ 40%]
code_structure.py::TestAPISchemas::test_ProcessStateSchema PASSED        [ 41%]
code_structure.py::TestAPISchemas::test_OperatingModeSchema PASSED       [ 43%]
code_structure.py::TestAPIStateBuffer::test_update_signal_history PASSED [ 44%]
code_structure.py::TestAPIStateBuffer::test_batch_update_signal_history PASSED [ 45%]
code_structure.py::TestAPIStateBuffer::test_reset_cusum_statistics PASSED [ 46%]
code_structure.py::TestAPIStateBuffer::test_update_ema_variance PASSED   [ 48%]
code_structure.py::TestAPIStateBuffer::test_update_residual_buffer PASSED [ 49%]
code_structure.py::TestCoreOrchestrator::test_initialize_state PASSED    [ 50%]
code_structure.py::TestCoreOrchestrator::test_initialize_batched_states PASSED [ 51%]
code_structure.py::TestCoreOrchestrator::test_compute_entropy_ratio PASSED [ 53%]
code_structure.py::TestCoreOrchestrator::test_scale_dgm_architecture PASSED [ 54%]
code_structure.py::TestCoreOrchestrator::test_compute_adaptive_stiffness_thresholds PASSED [ 55%]
code_structure.py::TestCoreOrchestrator::test_compute_adaptive_jko_params PASSED [ 56%]
code_structure.py::TestCoreFusion::test_compute_sinkhorn_epsilon PASSED  [ 58%]
code_structure.py::TestCoreMetaOptimizer::test_walk_forward_split PASSED [ 59%]
code_structure.py::TestKernelsBase::test_apply_stop_gradient_to_diagnostics PASSED [ 60%]
code_structure.py::TestKernelsBase::test_compute_signal_statistics PASSED [ 62%]
code_structure.py::TestKernelsBase::test_normalize_signal PASSED         [ 63%]
code_structure.py::TestKernelsBase::test_validate_kernel_input PASSED    [ 64%]
code_structure.py::TestKernelA::test_gaussian_kernel PASSED              [ 65%]
code_structure.py::TestKernelA::test_compute_gram_matrix PASSED          [ 67%]
code_structure.py::TestKernelA::test_kernel_ridge_regression PASSED      [ 68%]
code_structure.py::TestKernelA::test_create_embedding PASSED             [ 69%]
code_structure.py::TestKernelA::test_kernel_a_predict PASSED             [ 70%]
code_structure.py::TestKernelB::test_dgm_hjb_solver PASSED               [ 72%]
code_structure.py::TestKernelB::test_kernel_b_predict PASSED             [ 73%]
code_structure.py::TestKernelC::test_drift_levy_stable PASSED            [ 74%]
code_structure.py::TestKernelC::test_diffusion_levy PASSED               [ 75%]
code_structure.py::TestKernelC::test_kernel_c_predict PASSED             [ 77%]
code_structure.py::TestKernelD::test_create_path_augmentation PASSED     [ 78%]
code_structure.py::TestKernelD::test_compute_log_signature PASSED        [ 79%]
code_structure.py::TestKernelD::test_predict_from_signature PASSED       [ 81%]
code_structure.py::TestKernelD::test_kernel_d_predict PASSED             [ 82%]
code_structure.py::TestAPIWarmup::test_warmup_kernel_a PASSED            [ 83%]
code_structure.py::TestAPIWarmup::test_warmup_kernel_b PASSED            [ 84%]
code_structure.py::TestAPIWarmup::test_warmup_kernel_c PASSED            [ 86%]
code_structure.py::TestAPIWarmup::test_warmup_kernel_d PASSED            [ 87%]
code_structure.py::TestAPIWarmup::test_warmup_all_kernels PASSED         [ 88%]
code_structure.py::TestAPIWarmup::test_warmup_with_retry PASSED          [ 89%]
code_structure.py::TestAPIWarmup::test_profile_warmup_and_recommend_timeout PASSED [ 91%]
code_structure.py::TestIOModuleImportable::test_config_mutation_module_exists PASSED [ 92%]
code_structure.py::TestIOModuleImportable::test_credentials_module_exists PASSED [ 93%]
code_structure.py::TestIOModuleImportable::test_dashboard_module_exists PASSED [ 94%]
code_structure.py::TestIOModuleImportable::test_loaders_module_exists PASSED [ 96%]
code_structure.py::TestIOModuleImportable::test_snapshots_module_exists PASSED [ 97%]
code_structure.py::TestIOModuleImportable::test_telemetry_module_exists PASSED [ 98%]
code_structure.py::TestIOModuleImportable::test_validators_module_exists PASSED [100%]

=============================== warnings summary ===============================
../../.venv/lib/python3.13/site-packages/_pytest/config/__init__.py:1303
/Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/.venv/lib/python3.13/site-packages/_pytest/config/__init__.py:1303:
  PytestAssertRewriteWarning: Module already imported so cannot be rewritten; jaxtyping
    self._mark_plugins_for_rewrite(hook, disable_autoload)

code_structure.py::TestAPIValidation::test_warn_if_invalid
  /Users/obosio/Library/CloudStorage/Dropbox/OCTA/Projects/STOCHASTIC_PREDICTOR/Python/api/validation.py:458:
  RuntimeWarning: test warning
    warnings.warn(message, RuntimeWarning)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 79 passed, 2 warnings in 12.72s ========================

```

---

## 🎯 Final Summary

✅ **All 79 tests passed!** Code structure validated successfully.

⚠️ **2 warning(s) detected** - consider addressing them.

**Test Duration:** 12.72 seconds

**Report generated at:** 2026-02-21 15:58:08 UTC
