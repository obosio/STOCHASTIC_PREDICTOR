# Test Scripts Compliance Audit

**Date:** 2026-02-20  
**Scope:** Verify existing test scripts against applicable TESTING_AUDIT_POLICIES  
**Scripts Audited:**

- `tests/scripts/code_structure.py` (executor)
- `tests/scripts/tests_coverage.py` (validator)
- `tests/scripts/tests_start.sh` (orchestrator)

---

## Applicable Testing Policies for Test Infrastructure

Only policies that directly relate to how tests are structured, run, and validated apply to the test scripts themselves:

| Policy ID | Policy Name | Applicable? | Reason |
|-----------|-------------|-------------|--------|
| #20-32 | Hot-start, snapshots, determinism, etc. | NO | These test USP code, not test infrastructure |
| #33 | Walk-forward validation | YES | Tests should use walk-forward splits |
| #34 | Bayesian optimization efficiency | NO | Tests result, not test requirement |
| #39 | Code coverage ≥ 90% | YES | tests_coverage.py must enforce this |
| #40 | Reproducibility with fixed seed | YES | All randomness must be seeded |
| #41 | State isolation between tests | YES | Tests must not couple/interfere |
| #42 | Performance/runtime limits | YES | Full suite must run < 5 minutes |
| #43 | Numerical parity CPU/GPU | NO | Tested by code_structure.py, not a framework requirement |
| #44 | JIT cache warmup | YES | Tests should warmup JAX |
| #45 | Atomic TOML mutation | NO | Config is read-only in tests |

---

## Individual Script Audit

### SCRIPT: tests_coverage.py (Structural Coverage Validator)

**Purpose:** Validate 95+ functions have test coverage, 0 gaps, 0 orphans

**Applicable Policies:**

- #39: Coverage threshold enforcement
- #41: Test organization and cataloging

**Compliance Findings:**

| Finding | Status | Details |
|---------|--------|---------|
| **Hardcoded Coverage Threshold (Line 293)** | ⚠️ WARNING | `assert result["coverage"] == 100.0` - Cannot be configured from config.toml |
| **Policy Source** | ✅ PASS | Self-contained, no external file dependency |
| **Gap/Orphan Detection** | ✅ PASS | Proper analysis of coverage gaps and orphaned tests |
| **JSON Output** | ✅ PASS | Timestamps and proper structure |
| **Determinism** | ✅ PASS | No randomness involved |

**VIOLATION DETAILS:**

**Policy #39 - Coverage Threshold Must Be Configurable**

- **Current State:** Line 293 hardcodes `100.0%` requirement
- **Spec Requirement:** Coverage threshold should derive from config or policy, not hardcoded
- **Impact:** Medium - Cannot adjust thresholds without code modification
- **Fix Required:** YES

---

### SCRIPT: code_structure.py (Test Executor)

**Purpose:** Execute 100+ unit/integration tests covering all Python/ modules

**Applicable Policies:**

- #40: Reproducibility with fixed seeds
- #41: State isolation between test classes
- #44: JAX JIT warmup before tests
- #42: Runtime performance < 5 minutes

**Compliance Findings:**

| Finding | Status | Details |
|---------|--------|---------|
| **X64 Precision Enabled (Lines 30-32)** | ✅ PASS | JAX x64 enabled BEFORE JAX import - correct |
| **Fixed Seed Usage (Line 116, 176, 235)** | ✅ PASS | Seeds for reproducibility (seed=42, seed=123) |
| **Module Fixture Scope (Line 115)** | ⚠️ WARNING | `scope="module"` can cause state coupling across tests in same module |
| **JAX Initialize (Line 116)** | ✅ PASS | `initialize_jax_prng(seed=42)` creates deterministic key |
| **Test Class Isolation** | ✅ PASS | Separate test classes for different modules |
| **Warmup Missing** | ⚠️ WARNING | No explicit JAX JIT warmup before tests (tests will cause first-time compilation) |
| **Performance Tracking** | ❌ FAIL | No runtime measurement/validation against < 5 minute target |

**VIOLATION DETAILS:**

**Policy #44 - JIT Cache Warmup Before Tests**

- **Current State:** Tests execute without warmup phase
- **Spec Requirement:** Warmup must precompile all depths M ∈ {2,3,5} before first test
- **Impact:** High - First tests incur ~200ms JIT compilation stalls
- **Fix Required:** YES - Add warmup_all_kernels() call in fixture setup

**Policy #42 - Full Suite Runtime < 5 Minutes**

- **Current State:** No runtime measurement implemented
- **Spec Requirement:** Must assert full suite completes within 5 minutes
- **Impact:** Medium - Unknown if tests actually meet SLA requirement
- **Fix Required:** YES - Add elapsed time tracking + assertion

**Policy #41 - State Isolation**

- **Current State:** Module-scoped fixture may carry state between tests
- **Spec Requirement:** Each test should start with clean state
- **Impact:** Low-Medium - Risk of test interdependencies
- **Fix Required:** CONDITIONALLY - Review if state is truly isolated

---

### SCRIPT: tests_start.sh (Orchestrator)

**Purpose:** Sequentially execute compliance → coverage → execution with fail-fast

**Applicable Policies:**

- #40: Reproducibility (exit codes, deterministic execution)
- #41: Proper test sequencing
- #42: Overall execution time tracking
- #44: JAX warmup before running code_structure.py

**Compliance Findings:**

| Finding | Status | Details |
|---------|--------|---------|
| **Fail-Fast Strategy (Line 158: set -o pipefail)** | ✅ PASS | Stops on first failure |
| **Deterministic Execution Order** | ✅ PASS | Sequential: compliance → coverage → execute |
| **Output Timestamps (Implicit)** | ✅ PASS | Each script produces timestamped JSON |
| **Environment Setup** | ⚠️ WARNING | No explicit JAX_ENABLE_X64 or JAX_DETERMINISTIC_REDUCTIONS setup |
| **PRNG Seeding** | ❌ FAIL | Shell script does not seed PRNG; relies on Python scripts |
| **Total Runtime Assertion** | ❌ FAIL | No validation that total time < 5 minutes |

**VIOLATION DETAILS:**

**Policy #40 - Deterministic Execution Environment**

- **Current State:** Shell script does not configure JAX environment variables
- **Spec Requirement:** Must set JAX_ENABLE_X64, JAX_DETERMINISTIC_REDUCTIONS before tests
- **Impact:** Medium - Tests may use different precision/reduction settings
- **Fix Required:** YES - Export environment variables in tests_start.sh

**Policy #42 - Overall Runtime SLA**

- **Current State:** Bash script runs tests but doesn't measure total time
- **Spec Requirement:** Must validate full suite < 5 minutes
- **Impact:** Medium - Unknown if orchestrator meets performance SLA
- **Fix Required:** YES - Add timing measurement and assertion in tests_start.sh

---

## Summary of Violations

### CRITICAL (Must Fix Before Next Release)

❌ **code_structure.py Policy #44 - Missing JAX JIT Warmup**

- First-time compilations will cause ~200ms stalls in tests
- Solution: Add `warmup_all_kernels()` in pytest fixture

❌ **tests_start.sh Policy #40 - Missing Environment Setup**

- JAX environment not configured consistently
- Solution: Export JAX_DETERMINISTIC_REDUCTIONS and guarantee x64

### HIGH (Should Fix)

⚠️ **tests_coverage.py Policy #39 - Hardcoded Coverage Threshold**

- Line 293 hardcodes 100.0% instead of reading from config
- Solution: Read threshold from CODE_AUDIT_POLICIES_SPECIFICATION.md or environment

⚠️ **code_structure.py Policy #42 - No Runtime SLA Tracking**

- No measurement that tests complete within 5 minutes
- Solution: Add timing instrumentation to main() function

⚠️ **tests_start.sh Policy #42 - No Total Runtime Assertion**

- Orchestrator doesn't validate SLA compliance
- Solution: Add total elapsed time check at end

### MEDIUM (Conditional Fixes)

⚠️ **code_structure.py Policy #41 - Module-Scoped Fixtures**

- Fixture scope="module" could couple state across tests
- Analysis: Verify if shared config_obj and prng_key create test interdependencies
- Recommendation: Change to scope="function" for full isolation

---

## Compliance Matrix

| Script | P#40 | P#41 | P#42 | P#44 | Overall |
|--------|------|------|------|------|---------|
| code_structure.py | ✅ | ⚠️ | ❌ | ❌ | 🔴 FAIL |
| tests_coverage.py | ✅ | ✅ | ✅ | ✅ | ✅ PASS* |
| tests_start.sh | ❌ | ✅ | ❌ | N/A | 🔴 FAIL |

*tests_coverage.py PASS with warning on Policy #39 (hardcoded threshold)

---

## Recommendations

### Phase 1: Critical Fixes (Before Production)

1. **code_structure.py - Add JAX Warmup**

   ```python
   @pytest.fixture(scope="session")
   def warmup_jax():
       """Warmup JAX JIT compiler before running tests."""
       from Python.api.warmup import warmup_all_kernels
       warmup_all_kernels()
   ```

2. **tests_start.sh - Set Environment**

   ```bash
   export JAX_ENABLE_X64=1
   export JAX_DETERMINISTIC_REDUCTIONS=1
   ```

3. **tests_start.sh - Track Total Time**

   ```bash
   START_TIME=$(date +%s)
   # ... run all tests ...
   END_TIME=$(date +%s)
   ELAPSED=$((END_TIME - START_TIME))
   if [ $ELAPSED -gt 300 ]; then
       print_failure "Tests took ${ELAPSED}s (> 5 min SLA)"
   fi
   ```

### Phase 2: High Priority (Before Next Sprint)

1. **tests_coverage.py - Make Threshold Configurable**
   - Read from environment variable: `TEST_COVERAGE_THRESHOLD=100.0`
   - Default to 100% if unset

2. **code_structure.py - Add Runtime Instrumentation**
   - Measure elapsed time for full suite
   - Print report: `Tests completed in 47.3s (SLA: < 300s)`

### Phase 3: Medium Priority (Backlog)

1. **code_structure.py - Evaluate Fixture Scope**
   - Audit whether module-scoped fixtures create test coupling
   - Recommendation: Change to function scope for safety

---

## Conclusion

**Current Status:** 🔴 **2 of 3 scripts FAIL compliance**

**Root Cause:** Recent scripts (code_structure.py, tests_start.sh) were created before testing policies were fully defined.

**Path to Compliance:** ~4-6 hours of work to implement critical and high-priority fixes.

**No Blocking Issues:** All violations are code-only fixes; no architectural changes needed.
