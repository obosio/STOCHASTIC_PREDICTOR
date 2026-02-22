# Dependency Check Report

## Metadata

| Field | Value |
| --- | --- |
| Report ID | dependency_check |
| Timestamp (UTC) | 2026-02-22T00:42:41.953046+00:00 |
| Status | PASS |
| Source | Test/scripts/dependency_check.py |
| Framework Version | 2.1.0 |
| Notes | Requirements vs imports and installed packages (3-level: BLOCKING/ERROR/WARNING) |

## Execution Summary

| Metric | Value |
| --- | --- |
| Total Unique Packages | 30 |
| Blocking Issues | 0 |
| Error Issues | 0 |
| Warning Issues | 2 |
| Missing in Production | 0 |
| Missing in Testing | 0 |
| Version Mismatches | 0 |
| Unresolved Imports | 0 |

## Scope

### Folders

- Python
- Test

### Files

- Python/requirements.txt
- Test/requirements.txt

### Modules

- brotli
- equinox
- google_crc32c
- jax
- jaxtyping
- msgpack
- numpy
- optuna
- ott
- pydantic
- pytest
- signax
- toml
- tomli
- yaml

### Functions

- None

### Classes

- None

## Details

| Layer | Package Count |
| --- | --- |
| Production | 19 |
| Testing | 30 |

## Issues & Warnings

### Blocking Issues

- No blocking issues

### Error Issues

- No error issues

### Warning Issues

- [dependency] extra_in_production_requirements: 6
- [dependency] extra_in_testing_requirements: 15

## Extras

### Missing In Production Requirements

- No findings

### Missing In Testing Requirements

- No findings

### Extra In Production Requirements

- diffrax
- jaxlib
- optax
- pandas
- pywavelets
- scipy

### Extra In Testing Requirements

- black
- diffrax
- flake8
- flake8-pyproject
- isort
- jaxlib
- matplotlib
- mypy
- optax
- pandas
- pytest-cov
- pywavelets
- scipy
- seaborn
- types-pyyaml

### Not Installed Production

- No findings

### Not Installed Testing

- No findings

### Unresolved Production Imports

- No findings

### Unresolved Testing Imports

- No findings

### Version Mismatches

- No findings
