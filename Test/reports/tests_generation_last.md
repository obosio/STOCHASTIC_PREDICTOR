# Test Execution Report

## Introduction

This report documents all automated tests executed against the project codebase. Tests are auto-generated from discovered modules and executed using pytest framework.

## Execution Summary

| Metric | Value |
| ------ | ----- |
| Status | ❌ FAILED |
| Exit Code | 1 |
| Framework Version | 2.1.0 |
| Timestamp | 2026-02-21 15:15:25 |

## Test Coverage

**Layers Tested:**

- `Python/api/` - API layer with 7 modules
- `Python/core/` - Core layer with 4 modules
- `Python/io/` - IO layer with 7 modules
- `Python/kernels/` - Kernels layer with 5 modules

**Test Approach:**

- Auto-generated smoke tests for all discovered callables
- 157 total tests generated and executed
- Tests use pytest framework with custom markers per layer

## Debug Information

⚠️ Tests failed with exit code 1.

**Troubleshooting:**

1. Review full pytest output for detailed error messages
2. Check if all dependencies are installed: `pip install -r Test/requirements.txt`
3. Some tests may be skipped due to missing optional dependencies
4. See pytest.ini for test configuration and markers
