# Audit Policies & Evaluation Criteria

## Extracted from Official Specification (Theory, Implementation, I/O)

**Document Version:** 1.0  
**Last Updated:** 2026-02-20  
**Audit Scope:** USP v2.1.0-RC1  
**Policy Strictness:** ZERO-HEURISTICS (STRICT: 0% Fallback Tolerance)

---

## Executive Summary

This document formalizes **23 mandatory policies** extracted from the mathematical specification across three core specification documents:

- **Theory.tex:** 7 policies (mathematical foundations, theory-code alignment)
- **Implementation.tex:** 10 policies (numerical schemes, validation, optimization)
- **I/O.tex:** 6 policies (data handling, security, persistence, configuration)

All policies are enforced at **CRITICAL** or **HIGH** severity. Any violation triggers audit FAILURE.

---

## 1. ZERO-HEURISTICS POLICY (Policy #1)

**Source:** `Copilot-Instructions.md` + `Specification (I/O.tex § Configuration Mutation)`

**Policy Statement:**

```text
Every numerical parameter, threshold, and algorithmic decision MUST be 
derived from configuration (config.toml), never hardcoded into source.
NO implicit fallbacks, defaults, or "safe" magic constants are permitted.
Fail-fast on missing configuration.
```

**Audit Criteria:**

- ❌ **CRITICAL:** Any hardcoded numeric constant used as a threshold (e.g., `if x > 0.5:`)
- ❌ **CRITICAL:** `.get(key, default)` patterns with fallback values
- ❌ **CRITICAL:** Conditional `or` expressions masking defaults (e.g., `config.value or 1.0`)
- ❌ **CRITICAL:** `try/except` blocks silently catching missing config keys
- ✅ **PASS:** Explicit `config.get(section, key)` with ValueError on missing
- ✅ **PASS:** `assert config[key] is not None` before use
- ✅ **PASS:** ConfigManager validation schema with `jsonschema` strict mode

**Config Dependence Traceability:**
All 200+ parameters must be traceable to `stochastic_predictor/api/config.py` `FIELD_TO_SECTION_MAP`.

**Severity:** CRITICAL  
**Remediation:** Replace all fallback patterns with explicit validation.

---

## 2. CONFIGURATION IMMUTABILITY POLICY (Policy #2)

**Source:** `I/O.tex § Invariant Protection: Locked Configuration Subsections`

**Policy Statement:**

```text
The following config.toml subsections are STRICTLY IMMUTABLE during 
autonomous self-calibration and must NOT appear in optimizer search space:

[core] Section:
  - float_precision (must remain 64)
  - jax_platform (must remain locked at deployment value)

[io] Section:
  - snapshot_path (changing orphans existing snapshots)
  - telemetry_buffer_maxlen (requires memory reallocation)
  - credentials_vault_path (exposes secrets)

[security] Section:
  - telemetry_hash_interval_steps (breaks parity validation)
  - snapshot_integrity_hash_algorithm (invalidates existing snapshots)
  - allowed_mutation_rate_per_hour (prevents optimizer runaway)

[meta_optimization] Section:
  - max_deep_tuning_iterations (prevents infinite loops)
  - checkpoint_path (breaks resumability)
  - mutation_protocol_version (semantic versioning lock)
```

**Audit Criteria:**

- ❌ **CRITICAL:** Meta-optimizer references any locked parameter in search space
- ❌ **CRITICAL:** Configuration mutation writes to locked subsections
- ✅ **PASS:** Locked parameters registered in optimizer exclusion list
- ✅ **PASS:** Mutation validation schema rejects locked parameter updates
- ✅ **PASS:** Audit log shows no mutations to locked subsections

**Severity:** CRITICAL  
**Remediation:** Update optimizer search space definition and validation schema.

---

## 3. VALIDATION SCHEMA ENFORCEMENT (Policy #3)

**Source:** `I/O.tex § Configuration Mutation Protocol § Validation Schema`

**Policy Statement:**

```text
Before ANY configuration mutation is committed, the merged configuration 
must pass strict schema validation including:

1. Type checking (int, float, str)
2. Range validation [min, max]
3. Constraint checks (e.g., "must be power of 2", "must be < stiffness_high")
4. Immutability protection (locked parameters)
5. Interdependency validation (cusum_k < cusum_high, etc.)
```

**Mandatory Validation Rules:**

| Parameter | Type | Range | Constraint |
| --------- | ---- | ----- | --------- |
| `cusum_k` | float | [0.3, 1.5] | sensitivity factor (3-5 sigma rule) |
| `dgm_width_size` | int | [32, 256] | **must be power of 2** |
| `dgm_depth` | int | [3, 8] | hidden layer count |
| `stiffness_low` | float | [50, 500] | **must be < stiffness_high** |
| `stiffness_high` | float | [500, 5000] | **must be > stiffness_low** |
| `sinkhorn_epsilon` | float | [1e-4, 1e-1] | log-uniform, no underflow < 1e-4 |
| `signature_depth` | int | [3, 5] | **M > 2 (non-commutativity) & M ≤ 5 (curse of dim)** |
| `float_precision` | int | {64} | **locked: immutable** |
| `jax_platform` | str | {"cpu", "gpu", "tpu"} | **locked: immutable at deployment** |

**Audit Criteria:**

- ❌ **CRITICAL:** Any parameter outside specified range accepted without error
- ❌ **CRITICAL:** Power-of-2 constraint on `dgm_width_size` not enforced
- ❌ **CRITICAL:** `stiffness_low >= stiffness_high` allowed
- ❌ **CRITICAL:** `signature_depth` outside [3, 5] accepted
- ✅ **PASS:** Schema validation raises `ConfigValidationError` on constraint violation
- ✅ **PASS:** All mutation commits trigger validation
- ✅ **PASS:** Validation schema includes all 200+ parameters from FIELD_TO_SECTION_MAP

**Severity:** CRITICAL  
**Remediation:** Implement strict validation schema with `jsonschema` or Pydantic.

---

## 4. ATOMIC CONFIGURATION MUTATION PROTOCOL (Policy #4)

**Source:** `I/O.tex § Atomic TOML Update Algorithm`

**Policy Statement:**

```text
Configuration mutations must follow POSIX-compliant atomic protocol:

Phase 1: Validation (check ranges, types, constraints)
Phase 2: Immutable Backup (create timestamped archive)
Phase 3: Atomic Write via Temporary File (O_EXCL flag, fsync())
Phase 4: Atomic Replacement (os.replace() or atomic rename)
Phase 5: Audit Logging (append to io/mutations.log)
```

**Audit Criteria:**

- ❌ **CRITICAL:** Direct modification of `config.toml` without temporary file
- ❌ **CRITICAL:** No fsync() call after writing temp file
- ❌ **CRITICAL:** Missing O_EXCL flag (concurrent mutation vulnerability)
- ❌ **CRITICAL:** Backup not created before mutation
- ❌ **CRITICAL:** Mutations not logged to `io/mutations.log`
- ✅ **PASS:** Temporary file created with O_EXCL flag
- ✅ **PASS:** fsync() called before atomic replacement
- ✅ **PASS:** Timestamped backup created (`config.toml.bak.ISO8601`)
- ✅ **PASS:** Audit log includes timestamp, trigger, delta, validation status

**Severity:** CRITICAL  
**Remediation:** Implement atomic mutation per algorithm in I/O.tex.

---

## 5. MUTATION RATE LIMITING (Policy #5)

**Source:** `I/O.tex § Rate Limiting and Safety Guardrails`

**Policy Statement:**

```text
To prevent optimizer pathologies:

1. Maximum Mutation Rate: ≤ 10 mutations per hour
2. Minimum Stability Period: 1000 prediction steps before re-mutation
3. Delta Magnitude Limit: Δθ/θ_old ≤ 50% (relative change cap)
   Exception: Discrete params (dgm_depth) ±1 level allowed
4. Rollback on Degradation: If RMSE ↑ >30% within 500 steps → auto-rollback
```

**Audit Criteria:**

- ❌ **CRITICAL:** >10 mutations per hour observed
- ❌ **CRITICAL:** Mutation allowed <1000 steps after precious mutation
- ❌ **CRITICAL:** Parameter change >50% without discrete exception
- ❌ **CRITICAL:** Degradation >30% not triggering auto-rollback
- ✅ **PASS:** Rate limiter enforced at config_mutation.py
- ✅ **PASS:** Minimum stability counter incremented post-mutation
- ✅ **PASS:** Delta validator checks |(new-old)/old| ≤ 0.5
- ✅ **PASS:** RMSE monitor triggers rollback at 30% threshold

**Severity:** HIGH  
**Remediation:** Add rate limiter, stability counter, delta validator, degradation monitor.

---

## 6. WALK-FORWARD VALIDATION PROTOCOL (Policy #6)

**Source:** `Implementation.tex § Causal Cross-Validation (Walk-Forward Validation)`

**Policy Statement:**

```python
Static validation methods (K-Fold) are PROHIBITED. They violate the arrow 
of time and leak future information (look-ahead bias).

MANDATORY: Rolling walk-forward with sliding window:

for t in range(L_train, T - H):
    train_idx = max(1, t - W_max)  # Sliding window
    D_train = data[train_idx:t]
    D_test = data[t+1:t+H]
    fit_model(D_train)
    predict(D_test)
    evaluate_error()
    t += H  # Advance by horizon
```

**Audit Criteria:**

- ❌ **CRITICAL:** K-Fold or random stratified split used anywhere in validation
- ❌ **CRITICAL:** Test data comes from earlier time than training (look-ahead bias)
- ❌ **CRITICAL:** Overlapping train/test windows
- ❌ **CRITICAL:** Data leakage from future into training set
- ✅ **PASS:** Validation loop advances strictly forward in time
- ✅ **PASS:** Test data exclusively from future of training cutoff
- ✅ **PASS:** Rolling window size configurable (`W_max`)
- ✅ **PASS:** Horizon-based stepping maintains temporal integrity

**Severity:** CRITICAL  
**Remediation:** Replace any non-causal validation with walk-forward protocol.

---

## 7. CUSUM THRESHOLD DYNAMISM (Policy #7)

**Source:** `Implementation.tex § Decision Thresholds (Hard Boundaries)`

**Policy Statement:**

```text
CUSUM threshold h_t MUST be computed DYNAMICALLY with kurtosis adjustment:

h_t = k · σ_resid · (1 + ln(κ_t / 3))

where:
  k ∈ [3, 5]          base sensitivity factor (three-sigma rule)
  σ_resid             rolling std of residuals (window size = 252)
  κ_t                 kurtosis (4th standardized moment, rolling window)
  ln(κ_t/3)           adjustment for heavy-tail regimes

This adaptive threshold reduces false positives in non-Gaussian volatility.
```

**Audit Criteria:**

- ❌ **CRITICAL:** h_t hardcoded (e.g., `h = 3.0`)
- ❌ **CRITICAL:** Threshold computed without kurtosis term
- ❌ **CRITICAL:** k outside [3, 5]
- ❌ **CRITICAL:** Rolling window size != 252 (standard in finance)
- ✅ **PASS:** h_t computed from config-driven components
- ✅ **PASS:** Kurtosis adjustment term present: `1 + ln(κ_t/3)`
- ✅ **PASS:** k parameterized in config with range [3, 5]
- ✅ **PASS:** σ_resid and κ_t recomputed every step

**Severity:** CRITICAL  
**Remediation:** Implement dynamic CUSUM formula with all terms.

---

## 8. SIGNATURE DEPTH CONSTRAINT (Policy #8)

**Source:** `Implementation.tex § Regularization and Stability Parameters`

**Policy Statement:**

```text
Signature tensor algebra truncation depth M must satisfy:

M ∈ [3, 5]

Rationale:
  M < 3:  Loses non-commutativity (event ordering irrelevant)
  M > 5:  Curse of dimensionality: feature growth O(d^M) 
          saturates RAM without marginal prediction gain
          
Typical deployment: M = 4 (balanced feature space)
```

**Audit Criteria:**

- ❌ **CRITICAL:** M < 3 or M > 5 allowed in config
- ❌ **CRITICAL:** M not validated at initialization
- ❌ **CRITICAL:** Feature dimension explosion (d^M) not bounded
- ✅ **PASS:** Config schema enforces M ∈ [3, 5]
- ✅ **PASS:** Signature kernel raises AssertionError if M ∉ [3, 5]
- ✅ **PASS:** Memory allocation pre-computed from d^M term

**Severity:** CRITICAL  
**Remediation:** Add signature depth validator to config schema.

---

## 9. SINKHORN EPSILON BOUNDS (Policy #9)

**Source:** `Implementation.tex § Regularization and Stability Parameters`

**Policy Statement:**

```text
Entropic regularization ε (Sinkhorn) must satisfy:

ε ∈ [10^-4, 10^-1]

Lower Bound (ε ≥ 10^-4):  Prevents numerical underflow in K = e^(-C/ε)
                          float32 becomes denormalized < 10^-4
                          
Upper Bound (ε ≤ 10^-1):   ε >> 10^-1 yields uniform mixture (max entropy)
                           loses discriminative power

Recommended: ε ∼ 10^-2 (initialization), adaptively tuned per regime
```

**Audit Criteria:**

- ❌ **CRITICAL:** ε < 1e-4 or ε > 1e-1 allowed in config
- ❌ **CRITICAL:** No underflow protection in Sinkhorn kernel
- ❌ **CRITICAL:** ε not validated at initialization
- ✅ **PASS:** Config schema enforces ε ∈ [1e-4, 1e-1] (log-uniform scale)
- ✅ **PASS:** Sinkhorn kernel checks: `assert 1e-4 <= epsilon <= 1e-1`
- ✅ **PASS:** Kernel matrix K = exp(-C/eps) guarded against underflow

**Severity:** CRITICAL  
**Remediation:** Add epsilon bounds validation to Sinkhorn kernel.

---

## 10. CFL CONDITION FOR PIDE SCHEMES (Policy #10)

**Source:** `Implementation.tex § Discretization Fundamentals and Monte Carlo Simulations`

**Policy Statement:**

```text
Time step Δt for finite difference schemes (HJB/PIDE solvers) must satisfy 
the generalized CFL (Courant-Friedrichs-Lewy) condition for stochastic equations:

Δt ≤ (C_safe · (Δx)²) / (2 · sup|σ(x)|² + sup|b(x)| · Δx)

where:
  C_safe ≈ 0.9              safety factor (conservative margin)
  Δx                        spatial grid spacing
  σ(x)                      volatility (diffusion coefficient)
  b(x)                      drift (advection coefficient)
  
Violating CFL induces spurious oscillations in DGM/IMEX solver output.
```

**Audit Criteria:**

- ❌ **CRITICAL:** Δt fixed without CFL validation
- ❌ **CRITICAL:** DGM solver ignores spatial grid constraints
- ❌ **CRITICAL:** C_safe ≠ 0.9 (no tuning of safety margin)
- ✅ **PASS:** Δt computed dynamically from CFL formula
- ✅ **PASS:** Kernel B (DGM) pre-validates Δt ≤ CFL bound
- ✅ **PASS:** Spatial discretization respects diffusion/advection balance

**Severity:** HIGH  
**Remediation:** Implement CFL validator in DGM kernel.

---

## 11. MOLLIAVIN CALCULUS - 64-BIT PRECISION (Policy #11)

**Source:** `Copilot-Instructions.md § Coding Standards`

**Policy Statement:**

```python
All Malliavin derivative calculations must run with 64-bit floating-point 
precision (float64). Switching to float32 invalidates Malliavin operator 
and breaks signature immutability proofs.

This is enforced globally in __init__.py via JAX environment variable:
  jax.config.update("jax_enable_x64", True)
```

**Audit Criteria:**

- ❌ **CRITICAL:** Any Malliavin computation with float32
- ❌ **CRITICAL:** Kernel D (signature/rough paths) using 32-bit arrays
- ❌ **CRITICAL:** jax_enable_x64 not set in initialization
- ✅ **PASS:** `jax.config.update("jax_enable_x64", True)` in **init**.py
- ✅ **PASS:** All Malliavin kernels annotated with `@jax.jit` and float64
- ✅ **PASS:** Array precision validated: `assert arr.dtype == jnp.float64`

**Severity:** CRITICAL  
**Remediation:** Enable jax_enable_x64 globally and validate array dtypes.

---

## 12. JAX.LAX.STOP_GRADIENT ON DIAGNOSTICS (Policy #12)

**Source:** `Copilot-Instructions.md § Coding Standards`

**Policy Statement:**

```python
All diagnostic and telemetry modules must apply jax.lax.stop_gradient() 
to prevent gradients from flowing through analysis code into core kernels.

Pattern:
  def telemetry_metric(state):
      x_no_grad = jax.lax.stop_gradient(state.weights)
      # Compute diagnostics on gradient-stopped tensor
      entropy = -sum(x_no_grad * log(x_no_grad))
      return entropy
```

**Audit Criteria:**

- ❌ **CRITICAL:** Diagnostics code computes gradients through telemetry
- ❌ **CRITICAL:** Telemetry functions lack stop_gradient wrapper
- ❌ **CRITICAL:** Anomaly detection kernels gradient-enabled (should be pure reads)
- ✅ **PASS:** All diagnostic modules wrap state inputs with stop_gradient()
- ✅ **PASS:** Telemetry computations orthogonal to training gradients
- ✅ **PASS:** CUSUM alarm logic uses gradient-stopped signals

**Severity:** HIGH  
**Remediation:** Add stop_gradient() to all diagnostic code sections.

---

## 13. KERNEL PURITY & STATEESSNESS (Policy #13)

**Source:** `Copilot-Instructions.md § Coding Standards`

**Policy Statement:**

```python
All kernel functions (K_A, K_B, K_C, K_D) must be:

1. Pure functions: No side effects (no I/O, no state mutation, deterministic)
2. Stateless: All state passed as function arguments, not global variables
3. JIT-compatible: Functions must be compilable via @jax.jit

These guarantees enable GPU vectorization via vmap and deterministic 
compilation across hardware backends.
```

**Audit Criteria:**

- ❌ **CRITICAL:** Kernels access global variables
- ❌ **CRITICAL:** Kernels perform I/O (print, write to disk)
- ❌ **CRITICAL:** Random state not passed as seed argument
- ❌ **CRITICAL:** Kernels mutate external state (numpy array, list)
- ✅ **PASS:** All kernel signatures: `def kernel(state, params, signal) -> output`
- ✅ **PASS:** @jax.jit decorator present on all kernels
- ✅ **PASS:** Random number generation uses seed from signature
- ✅ **PASS:** No print statements or file I/O in kernel code

**Severity:** CRITICAL  
**Remediation:** Refactor kernels to pure, stateless implementations.

---

## 14. FROZEN SIGNAL DETECTION (Policy #14)

**Source:** `I/O.tex § Input Flow (Data Injection)`

**Policy Statement:**

```text
When observation stream injects identical value for N_freeze ≥ 5 consecutive 
steps, system must:

1. Detect variance Var([y_t-4, ..., y_t]) = 0
2. Identify as sensor failure/data corruption
3. Emit FrozenSignalAlarmEvent with timestamp
4. Freeze Kernel D (topological branch) at last valid value
5. NOT update orchestrator weights (maintain inertia)
6. Activate degraded inference mode
7. Continue predictions using branches A, B, C only
8. Recovery: Once Var > 0.1 · Var_historical for 2 consecutive steps, 
            release D lock and resume normal operation
```

**Audit Criteria:**

- ❌ **CRITICAL:** No frozen signal detection implemented
- ❌ **CRITICAL:** Var([y_t-4, ..., y_t]) threshold not hardcoded or configurable
- ❌ **CRITICAL:** Kernel D not frozen on signal freeze
- ❌ **CRITICAL:** Orchestrator weights updated during frozen signal
- ❌ **CRITICAL:** Alert not emitted
- ✅ **PASS:** Variance monitor tracks last 5 observations
- ✅ **PASS:** FrozenSignalAlarmEvent emitted (telemetry)
- ✅ **PASS:** Kernel D disabled, weights frozen during frozen signal
- ✅ **PASS:** Degraded inference flag set and broadcast to output
- ✅ **PASS:** Recovery logic re-enables D after 2 high-variance steps

**Severity:** CRITICAL  
**Remediation:** Implement frozen signal detection and mode freeze.

---

## 15. CATASTROPHIC OUTLIER DETECTION (Policy #15)

**Source:** `I/O.tex § Input Flow (Data Injection)`

**Policy Statement:**

```text
If observation |y_t| > 20σ (relative to historical normalization), system must:

1. Classify as catastrophic outlier
2. DISCARD the input (do not feed to kernels)
3. KEEP inertial state (don't update)
4. Emit critical validation alert
5. Protect kernels from numerical divergence

Justification: Extreme outliers can trigger NaN/Inf in nonlinear components 
(neural networks, WTMM singularity estimation)
```

**Audit Criteria:**

- ❌ **CRITICAL:** No outlier detection on input
- ❌ **CRITICAL:** Outliers propagated to kernels
- ❌ **CRITICAL:** 20σ threshold not configurable or validated
- ❌ **CRITICAL:** Alert not emitted on outlier rejection
- ✅ **PASS:** Input validation checks `|y_t| ≤ 20σ`
- ✅ **PASS:** Outlier rejected: kernel input remains unchanged
- ✅ **PASS:** System state (weights, buffers) not updated
- ✅ **PASS:** Critical alert emitted with outlier value and timestamp

**Severity:** CRITICAL  
**Remediation:** Add pre-kernel input validation for outliers.

---

## 16. MINIMUM INJECTION FREQUENCY (NYQUIST SOFT LIMIT) (Policy #16)

**Source:** `I/O.tex § Input Flow (Data Injection) § Inference grid`

**Policy Statement:**

```text
Data injection frequency must maintain sufficient density relative 
to finest wavelet scales in WTMM (Kernel D).

Minimum injection frequency enforced based on C_besov (Besov cone parameter).

If event density falls below this threshold:
  1. Multifractal spectrum collapses
  2. System must FREEZE topological branch update
  3. Continue with branches A, B, C

Configuration:
  [kernels]
  wtmm_num_scales = 8  # Number of dyadic scales
  besov_cone = 2.0     # Influence radius for maxima tracking
  
  Computed lower bound (pseudo-code):
  min_freq = sampling_rate / (2^num_scales * besov_cone)
```

**Audit Criteria:**

- ❌ **CRITICAL:** No minimum frequency check
- ❌ **CRITICAL:** WTMM forced to update with sparse data (Nyquist violated)
- ❌ **CRITICAL:** WTMM buffer undersized relative to scales
- ✅ **PASS:** Injection frequency monitor tracks Δt between events
- ✅ **PASS:** Comparison against computed Nyquist bound
- ✅ **PASS:** Kernel D disabled if data too sparse
- ✅ **PASS:** Alert emitted on Nyquist violation

**Severity:** HIGH  
**Remediation:** Implement frequency monitor and Kernel D freeze on violation.

---

## 17. STALE WEIGHTS DETECTION (Telemetry Recency) (Policy #17)

**Source:** `I/O.tex § Input Flow (Data Injection) § Staleness policy`

**Policy Statement:**

```text
Target validation delay (staleness) policy:

1. Parameter: Δ_max (time-to-live) configurable in [io] section
2. Violation: If delay of y_target exceeds Δ_max, JKO update canceled
3. Signal: System must emit persistent "degraded inference" flag
4. Semantic: Weights ρ are stale; prediction still produced but 
            risk not optimized geometrically

Configuration:
  [io]
  target_staleness_max_seconds = 60  # e.g., 60s for real-time systems
```

**Audit Criteria:**

- ❌ **CRITICAL:** No staleness monitoring
- ❌ **CRITICAL:** JKO update proceeds with stale targets (>Δ_max)
- ❌ **CRITICAL:** Degraded flag not emitted
- ❌ **CRITICAL:** Δ_max not configurable
- ✅ **PASS:** Timestamp on y_target compared to current time
- ✅ **PASS:** Delay > Δ_max triggers weight freeze + alert
- ✅ **PASS:** Degraded inference flag persists in output
- ✅ **PASS:** User/executor alerted to risk state

**Severity:** HIGH  
**Remediation:** Add timestamp validation and staleness flag to orchestrator.

---

## 18. SECRET INJECTION POLICY (Policy #18)

**Source:** `I/O.tex § Security Policies in the I/O Layer (Credentials)`

**Policy Statement:**

```python
STRICTLY FORBIDDEN: Hardcoding tokens, API keys, database secrets, 
or connection credentials in source code.

MANDATORY: Environment injection pattern at runtime from OS variables 
or local .env file (never .env committed to git).

Pattern:
  import os
  from dotenv import load_dotenv
  
  load_dotenv()  # Load from .env (gitignored)
  api_key = os.getenv("BROKER_API_KEY")
  assert api_key is not None, "Missing BROKER_API_KEY env var"
```

**Audit Criteria:**

- ❌ **CRITICAL:** Any hardcoded API key, token, password in source
- ❌ **CRITICAL:** .env file committed to git
- ❌ **CRITICAL:** Credentials in log files or telemetry
- ❌ **CRITICAL:** Plain-text credentials in config.toml
- ✅ **PASS:** All credentials read from environment variables
- ✅ **PASS:** .env in .gitignore with explicit rules
- ✅ **PASS:** Validation: if env var missing, raise AssertionError
- ✅ **PASS:** Credentials never logged or persisted

**Severity:** CRITICAL  
**Remediation:** Audit all files for hardcoded secrets; migrate to env vars.

---

## 19. STATE SERIALIZATION WITH INTEGRITY CHECKSUM (Policy #19)

**Source:** `I/O.tex § Persistence (Snapshotting) § Atomic and Verified Snapshotting`

**Policy Statement:**

```python
Full system state Σ_t MUST include robust validation hash (SHA-256 or CRC32c):

Σ_t = {
  ρ_t (orchestrator weights),
  G⁺_t, σ²_ema, κ_t (CUSUM state),
  H_DGM (differential entropy),
  Flags (DegradedInferenceMode, EmergencyMode, etc.),
  WTMMBuffer (singularity estimation history),
  KernelsState = {S_A, S_B, S_C, S_D},
  SHA256(Σ_t)  ← MANDATORY checksum
}

Restore validation:
  1. Deserialize Σ
  2. Recompute SHA256(Σ[:-32])
  3. Compare against stored hash
  4. If mismatch → discard snapshot, restart in cold-start mode
```

**Audit Criteria:**

- ❌ **CRITICAL:** No checksum in serialized state
- ❌ **CRITICAL:** Checksum not verified before injection
- ❌ **CRITICAL:** Corrupted snapshot injected without validation failure
- ❌ **CRITICAL:** Single-bit error in kernel matrices undetected
- ✅ **PASS:** SHA256 hash computed on full state at serialization
- ✅ **PASS:** Hash verified before state injection
- ✅ **PASS:** Mismatch triggers error and cold-start fallback
- ✅ **PASS:** Binary format (msgpack, protobuf) used, not JSON

**Severity:** CRITICAL  
**Remediation:** Add SHA256 validation to snapshot serialization/deserialization.

---

## 20. NON-BLOCKING TELEMETRY (Policy #20)

**Source:** `I/O.tex § Telemetry and Deterministic Logging`

**Policy Statement:**

```python
Telemetry emission MUST be decoupled from JAX/XLA execution.

Constraint: Compute threads MUST NEVER block on I/O.

Implementation:
  1. Telemetry buffers enqueued in non-blocking FIFO structure
  2. Separate background process/thread consumes telemetry
  3. Compute thread continues without waiting for I/O completion

Pattern:
  # In compute thread:
  telemetry_queue.put_nowait(metric_dict)  # Non-blocking
  
  # In background thread:
  while True:
      metric = telemetry_queue.get(timeout=1.0)  # Blocking OK here
      write_to_disk(metric)
```

**Audit Criteria:**

- ❌ **CRITICAL:** Compute thread blocks on telemetry I/O
- ❌ **CRITICAL:** Prediction latency increased by disk writes
- ❌ **CRITICAL:** No background telemetry consumer
- ❌ **CRITICAL:** Unbounded telemetry queue (memory leak)
- ✅ **PASS:** Telemetry buffered in queue.Queue or asyncio.Queue
- ✅ **PASS:** Background thread (separate Python thread) consumes queue
- ✅ **PASS:** Compute thread uses put_nowait() (never blocks)
- ✅ **PASS:** Queue size bounded (e.g., maxsize=10000)

**Severity:** HIGH  
**Remediation:** Implement background telemetry consumer thread.

---

## 21. HARDWARE PARITY AUDIT HASHES (Policy #21)

**Source:** `I/O.tex § Telemetry and Deterministic Logging`

**Policy Statement:**

```text
Log SHA-256 hashes of critical state at configurable intervals for 
CPU/GPU parity validation:

Hashable outputs:
  1. ρ_t (orchestrator weight vector)
  2. OT cost (Wasserstein transport objective)
  3. Prediction ŷ_t

Hash inputs MUST use canonical float64 serialization to guarantee 
deterministic parity across runs on different backends (CPU vs GPU).

Configuration:
  [security]
  telemetry_hash_interval_steps = 100  # Every 100 prediction steps
  
Logging format:
  [2026-02-20T14:32:05.123Z] PARITY_HASH
    iteration: 1000
    hash_weights: "a3d4e5f6c7b8a9d0e1f2a3b4c5d6e7f8..."
    hash_ot_cost: "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d..."
```

**Audit Criteria:**

- ❌ **CRITICAL:** No parity hashes logged
- ❌ **CRITICAL:** Hash interval not configurable
- ❌ **CRITICAL:** Hashes use float32 (non-deterministic across backends)
- ❌ **CRITICAL:** Weight vector hashes inconsistent CPU vs GPU
- ✅ **PASS:** SHA256 hashes logged at configurable interval
- ✅ **PASS:** Canonical float64 serialization (IEEE 754 standard)
- ✅ **PASS:** Hashes match across CPU and GPU runs
- ✅ **PASS:** Hashes logged to append-only telemetry file

**Severity:** HIGH  
**Remediation:** Implement parity hash logging with canonical serialization.

---

## 22. WALK-FORWARD VALIDATION TEST LEAKAGE (Policy #22)

**Source:** `Implementation.tex § Causal Cross-Validation`

**Policy Statement:**

```python
PROHIBITED: K-Fold, stratified split, or random shuffling of time series.

MANDATORY: Rolling window where:
  1. Train data: data[max(0, t-W_max) : t]
  2. Test data: data[t+1 : t+H]  (STRICTLY FUTURE only)
  3. No overlap between train and test
  4. Window slides forward by horizon H (not by 1 step)

This preserves temporal causality and prevents look-ahead bias.
```

**Audit Criteria:**

- ❌ **CRITICAL:** Any K-Fold validation in meta-optimizer
- ❌ **CRITICAL:** Random split applied to time series
- ❌ **CRITICAL:** Test data from past (t-lag to t)
- ❌ **CRITICAL:** Train/test overlap detected
- ✅ **PASS:** Walk-forward loop with start_idx = max(1, t - W_max)
- ✅ **PASS:** Test data strictly from [t+1, t+H]
- ✅ **PASS:** Temporal ordering preserved in evaluation
- ✅ **PASS:** No future information visible to training phase

**Severity:** CRITICAL  
**Remediation:** Replace all validation splits with walk-forward protocol.

---

## 23. ENCODER CAPACITY EXPANSION FOR HIGH-ENTROPY PROCESSES (Policy #23)

**Source:** `Theory.tex § Capacity Expansion Criterion`

**Policy Statement:**

```python
If process entropy H(X_t | ℱ_{t-1}) exceeds threshold (e.g., H > 4 nats),
neural architecture (Kernel B) must increase capacity to avoid bottleneck:

Minimum expansion:
  new_width ≥ width · sqrt(H / H_ref)
  
  where H_ref ∼ 2 nats (reference entropy)

Configuration:
  [kernels]
  dgm_entropy_adaptation_enabled = true
  entropy_threshold = 4.0  # nats
  
Upon entropy spike:
  1. Compute conditional entropy H_t
  2. Check: if H_t > entropy_threshold, trigger capacity increase
  3. Update width: width_new = ceil(width_old · sqrt(H_t / H_ref))
  4. Expand network in-place (if warm-restart supported) or flag restart
```

**Audit Criteria:**

- ❌ **CRITICAL:** Fixed network width regardless of process entropy
- ❌ **CRITICAL:** No entropy monitoring
- ❌ **CRITICAL:** High-entropy processes bottlenecked by undersized network
- ❌ **CRITICAL:** Capacity expansion formula not implemented
- ✅ **PASS:** Entropy monitor computes H(X_t | ℱ_t-1) periodically
- ✅ **PASS:** Capacity expansion trigger at configurable threshold
- ✅ **PASS:** Width scaling formula: `new_width = ceil(old_width * sqrt(H_t / H_ref))`
- ✅ **PASS:** System can restart with expanded network or warm-start injection

**Severity:** HIGH  
**Remediation:** Add entropy monitor and capacity expansion to Kernel B.

---

## Summary Table: Policy Violations

| Policy # | Name | Severity | Category | Auto-Fixable |
| --------- | ---- | --------- | --------- | ------------ |
| 1 | Zero-Heuristics | CRITICAL | Config | No |
| 2 | Config Immutability | CRITICAL | Config | No |
| 3 | Validation Schema | CRITICAL | Config | Yes |
| 4 | Atomic Mutation | CRITICAL | I/O | Partial |
| 5 | Mutation Rate Limit | HIGH | I/O | Yes |
| 6 | Walk-Forward Validation | CRITICAL | Validation | No |
| 7 | CUSUM Dynamism | CRITICAL | Numerical | Partial |
| 8 | Signature Depth Boundary | CRITICAL | Numerical | Yes |
| 9 | Sinkhorn Epsilon Bounds | CRITICAL | Numerical | Yes |
| 10 | CFL Condition | HIGH | Numerical | Partial |
| 11 | 64-Bit Precision (Malliavin) | CRITICAL | Precision | Yes |
| 12 | Stop Gradient (Diagnostics) | HIGH | JAX | Partial |
| 13 | Kernel Purity | CRITICAL | Architecture | No |
| 14 | Frozen Signal Detection | CRITICAL | Validation | No |
| 15 | Catastrophic Outlier | CRITICAL | Validation | Partial |
| 16 | Nyquist Soft Limit | HIGH | Validation | Partial |
| 17 | Stale Weights Detection | HIGH | Validation | Partial |
| 18 | Secret Injection | CRITICAL | Security | Yes |
| 19 | State Checksum Validation | CRITICAL | Persistence | Yes |
| 20 | Non-Blocking Telemetry | HIGH | I/O | Partial |
| 21 | Parity Audit Hashes | HIGH | Audit | Partial |
| 22 | Temporal Causality (Validation) | CRITICAL | Validation | No |
| 23 | Entropy-Driven Capacity | HIGH | Numerical | Partial |

---

## Audit Execution Workflow

### Phase 1: Automated Scanning (grep + static analysis)

```text
1. Search for hardcoded constants (Policy #1)
2. Search for .get() with defaults (Policy #1)
3. Search for magic numbers (Policy #7, #8, #9)
4. Search for K-Fold validators (Policy #6, #22)
5. Search for hardcoded credentials (Policy #18)
```

### Phase 2: Code Inspection (semantic review)

```text
1. Verify kernel purity (Policy #13)
2. Verify JIT compilation readiness (Policy #13)
3. Trace configuration dependencies (all policies)
4. Check mutation atomicity (Policy #4)
5. Validate state persistence (Policy #19)
```

### Phase 3: Integration Testing (runtime)

```text
1. Frozen signal test (Policy #14)
2. Outlier injection test (Policy #15)
3. Staleness test (Policy #17)
4. Rate limiter test (Policy #5)
5. Telemetry latency test (Policy #20)
```

### Phase 4: Reporting

```text
Generate AUDIT_REPORT with:
  - CRITICAL violations (fail entire audit)
  - HIGH violations (require remediation)
  - Recommendations for each policy
  - Traceability to specification source
```

---

## References

- **Theory.tex:** `doc/latex/specification/Stochastic_Predictor_Theory.tex`
- **Implementation.tex:** `doc/latex/specification/Stochastic_Predictor_Implementation.tex`
- **I/O.tex:** `doc/latex/specification/Stochastic_Predictor_IO.tex`
- **Copilot-Instructions.md:** `.github/copilot-instructions.md`

**Next Step for Auditor:** Await user code modification request, then invoke:

```bash
audita: [code-name]
```

to trigger 4-phase audit cycle.
