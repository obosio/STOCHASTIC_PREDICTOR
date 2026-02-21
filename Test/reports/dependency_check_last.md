# Dependency Report

## Introduction

This report inventories all project dependencies across production, testing, and documentation layers. It verifies requirements consistency and tracks package versions pinned by the project.

## Execution Summary

| Metric | Count |
| ------ | ----- |
| Total Unique Packages | 22 |
| Status | ✅ PASS |

## Requirements Details

### Production Layer

**Requirement File:** `Python/requirements.txt`

**Package Count:** 14

**Packages:**

- `PyWavelets`
- `diffrax`
- `equinox`
- `jax`
- `jaxlib`
- `jaxtyping`
- `numpy`
- `optax`
- `ott-jax`
- `pandas`
- `pydantic`
- `scipy`
- `signax`
- `tomli`

### Testing Layer

**Requirement File:** `Test/requirements.txt`

**Package Count:** 8

**Packages:**

- `black`
- `flake8`
- `isort`
- `matplotlib`
- `mypy`
- `pytest`
- `pytest-cov`
- `seaborn`

### Documentation Layer

**Requirement File:** `Doc/requirements.txt`

**Package Count:** 0

No packages.

## Debug Information

All requirements files found and parsed successfully. No conflicts detected.
