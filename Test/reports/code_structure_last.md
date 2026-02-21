# Code Structure Report

## Introduction

This report provides a comprehensive analysis of the project's code structure, documenting all modules, files, and lines of code across production and test layers.

## Execution Summary

| Metric | Count |
| ------ | ----- |
| Total Modules | 23 |
| Total Files | 27 |
| Total Lines | 10,192 |
| Status | ✅ PASS |

## Code Inventory by Layer

### API

**Location:** `Python/api`

**Metrics:**

- Modules: 7
- Files: 8
- Lines: 3,435

**Files:**

- `__init__.py` (150 lines)
- `config.py` (599 lines)
- `prng.py` (281 lines)
- `schemas.py` (210 lines)
- `state_buffer.py` (399 lines)
- `types.py` (705 lines)
- `validation.py` (612 lines)
- `warmup.py` (479 lines)

### CORE

**Location:** `Python/core`

**Metrics:**

- Modules: 4
- Files: 5
- Lines: 2,560

**Files:**

- `__init__.py` (71 lines)
- `fusion.py` (129 lines)
- `meta_optimizer.py` (1174 lines)
- `orchestrator.py` (1051 lines)
- `sinkhorn.py` (135 lines)

### IO

**Location:** `Python/io`

**Metrics:**

- Modules: 7
- Files: 8
- Lines: 1,893

**Files:**

- `__init__.py` (112 lines)
- `config_mutation.py` (728 lines)
- `credentials.py` (34 lines)
- `dashboard.py` (235 lines)
- `loaders.py` (123 lines)
- `snapshots.py` (188 lines)
- `telemetry.py` (384 lines)
- `validators.py` (89 lines)

### KERNELS

**Location:** `Python/kernels`

**Metrics:**

- Modules: 5
- Files: 6
- Lines: 2,304

**Files:**

- `__init__.py` (90 lines)
- `base.py` (235 lines)
- `kernel_a.py` (877 lines)
- `kernel_b.py` (431 lines)
- `kernel_c.py` (408 lines)
- `kernel_d.py` (263 lines)

### TESTS

**Location:** `Test/tests`

**Metrics:**

- Modules: 0
- Files: 0
- Lines: 0

## Debug Information

**Analysis Timestamp:** 2026-02-21T15:15:25.640446

All layers scanned successfully. No structural issues detected.
