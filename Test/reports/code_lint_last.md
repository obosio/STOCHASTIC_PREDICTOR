# Code Quality Report

## Metadata

| Field | Value |
| --- | --- |
| Report ID | code_quality |
| Timestamp (UTC) | 2026-02-22T00:42:43.064076+00:00 |
| Status | PASS |
| Source | Test/scripts/code_lint.py |
| Framework Version | 2.1.0 |
| Notes | Unified code quality: black, isort, flake8, mypy (3-level: BLOCKING/ERROR/WARNING) |

## Execution Summary

| Metric | Value |
| --- | --- |
| Blocking Issues | 0 |
| Error Issues | 0 |
| Warning Issues | 47 |
| Total Issues | 47 |
| Files Scanned | 63 |

## Scope

### Folders

- Python
- Test

### Files

- Python/__init__.py
- Python/api/__init__.py
- Python/api/config.py
- Python/api/prng.py
- Python/api/schemas.py
- Python/api/state_buffer.py
- Python/api/types.py
- Python/api/validation.py
- Python/api/warmup.py
- Python/core/__init__.py
- Python/core/fusion.py
- Python/core/meta_optimizer.py
- Python/core/orchestrator.py
- Python/core/sinkhorn.py
- Python/io/__init__.py
- Python/io/config_mutation.py
- Python/io/credentials.py
- Python/io/dashboard.py
- Python/io/loaders.py
- Python/io/snapshots.py
- Python/io/telemetry.py
- Python/io/validators.py
- Python/kernels/__init__.py
- Python/kernels/base.py
- Python/kernels/kernel_a.py
- Python/kernels/kernel_b.py
- Python/kernels/kernel_c.py
- Python/kernels/kernel_d.py
- Test/conftest.py
- Test/framework/__init__.py
- Test/framework/discovery.py
- Test/framework/generator.py
- Test/framework/inspector.py
- Test/framework/reports.py
- Test/run_tests.py
- Test/scripts/code_alignment.py
- Test/scripts/code_lint.py
- Test/scripts/code_structure.py
- Test/scripts/dependency_check.py
- Test/scripts/tests_generation.py
- Test/tests/unit/api/test_config.py
- Test/tests/unit/api/test_prng.py
- Test/tests/unit/api/test_schemas.py
- Test/tests/unit/api/test_state_buffer.py
- Test/tests/unit/api/test_types.py
- Test/tests/unit/api/test_validation.py
- Test/tests/unit/api/test_warmup.py
- Test/tests/unit/core/test_fusion.py
- Test/tests/unit/core/test_meta_optimizer.py
- Test/tests/unit/core/test_orchestrator.py
- Test/tests/unit/core/test_sinkhorn.py
- Test/tests/unit/io/test_config_mutation.py
- Test/tests/unit/io/test_credentials.py
- Test/tests/unit/io/test_dashboard.py
- Test/tests/unit/io/test_loaders.py
- Test/tests/unit/io/test_snapshots.py
- Test/tests/unit/io/test_telemetry.py
- Test/tests/unit/io/test_validators.py
- Test/tests/unit/kernels/test_base.py
- Test/tests/unit/kernels/test_kernel_a.py
- Test/tests/unit/kernels/test_kernel_b.py
- Test/tests/unit/kernels/test_kernel_c.py
- Test/tests/unit/kernels/test_kernel_d.py

### Modules

- None

### Functions

- None

### Classes

- None

## Details

### Tools

black, isort, flake8, mypy

### Classification

BLOCKING (won't execute) / ERROR (executes but buggy) / WARNING (style)

## Issues & Warnings

### Blocking Issues

- No blocking issues

### Error Issues

- No error issues

### Warning Issues

- [lint] Test/scripts/code_alignment.py:115 F824 F824 `global CHANGED_FILES_CACHE` is unused: name is never assigned in scope
- [lint] Test/scripts/code_structure.py:41 F401 F401 'typing.Dict' imported but unused
- [lint] Test/scripts/code_structure.py:41 F401 F401 'typing.List' imported but unused
- [lint] Test/scripts/code_structure.py:41 F401 F401 'typing.Set' imported but unused
- [lint] Test/scripts/code_structure.py:62 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:63 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:64 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:69 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:72 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:73 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:83 F401 F401 'Python.api.schemas.HealthCheckResponseSchema' imported but unused
- [lint] Test/scripts/code_structure.py:83 F401 F401 'Python.api.schemas.KernelOutputSchema' imported but unused
- [lint] Test/scripts/code_structure.py:83 F401 F401 'Python.api.schemas.PredictionResultSchema' imported but unused
- [lint] Test/scripts/code_structure.py:83 F401 F401 'Python.api.schemas.TelemetryDataSchema' imported but unused
- [lint] Test/scripts/code_structure.py:83 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:90 F401 F401 'Python.api.state_buffer.atomic_state_update' imported but unused
- [lint] Test/scripts/code_structure.py:90 F401 F401 'Python.api.state_buffer.update_cusum_statistics' imported but unused
- [lint] Test/scripts/code_structure.py:90 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:99 F401 F401 'Python.api.types.PredictionResult' imported but unused
- [lint] Test/scripts/code_structure.py:99 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:109 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:125 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:134 F401 F401 'Python.core.fusion.FusionResult' imported but unused
- [lint] Test/scripts/code_structure.py:134 F401 F401 'Python.core.fusion.fuse_kernel_outputs' imported but unused
- [lint] Test/scripts/code_structure.py:134 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:135 F401 F401 'Python.core.meta_optimizer.AsyncMetaOptimizer' imported but unused
- [lint] Test/scripts/code_structure.py:135 F401 F401 'Python.core.meta_optimizer.BayesianMetaOptimizer' imported but unused
- [lint] Test/scripts/code_structure.py:135 F401 F401 'Python.core.meta_optimizer.IntegrityError' imported but unused
- [lint] Test/scripts/code_structure.py:135 F401 F401 'Python.core.meta_optimizer.MetaOptimizationConfig' imported but unused
- [lint] Test/scripts/code_structure.py:135 F401 F401 'Python.core.meta_optimizer.OptimizationResult' imported but unused
- [lint] Test/scripts/code_structure.py:135 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:145 F401 F401 'Python.core.orchestrator.OrchestrationResult' imported but unused
- [lint] Test/scripts/code_structure.py:145 F401 F401 'Python.core.orchestrator.apply_host_architecture_scaling' imported but unused
- [lint] Test/scripts/code_structure.py:145 F401 F401 'Python.core.orchestrator.orchestrate_step' imported but unused
- [lint] Test/scripts/code_structure.py:145 F401 F401 'Python.core.orchestrator.orchestrate_step_batch' imported but unused
- [lint] Test/scripts/code_structure.py:145 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:157 F401 F401 'Python.core.sinkhorn.SinkhornResult' imported but unused
- [lint] Test/scripts/code_structure.py:157 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:160 F401 F401 'Python.kernels.base.PredictionKernel' imported but unused
- [lint] Test/scripts/code_structure.py:160 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:167 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:174 F401 F401 'Python.kernels.kernel_b.compute_entropy_dgm' imported but unused
- [lint] Test/scripts/code_structure.py:174 F401 F401 'Python.kernels.kernel_b.loss_hjb' imported but unused
- [lint] Test/scripts/code_structure.py:174 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:180 F401 F401 'Python.kernels.kernel_c.solve_sde' imported but unused
- [lint] Test/scripts/code_structure.py:180 E402 E402 module level import not at top of file
- [lint] Test/scripts/code_structure.py:186 E402 E402 module level import not at top of file
