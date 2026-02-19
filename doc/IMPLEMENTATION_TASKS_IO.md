# Phase 4 IO - Implementation Tasks

## Objectives

- Implement IO layer primitives without stalling JAX/XLA compute.
- Enforce catastrophic outlier, frozen signal, and staleness policies.
- Persist snapshots atomically in binary format with integrity hashes.
- Provide deterministic telemetry/parity logging with SHA-256 hashes.
- Enforce credential injection and secret exclusion.

## Module Tasks

### 1) validators.py

- Implement `detect_catastrophic_outlier(y_t, sigma, threshold=20.0)`.
- Implement `detect_frozen_signal(buffer, n_freeze=5)` and recovery criteria.
- Implement `compute_staleness_ns(timestamp_ns, now_ns)` and `is_stale(delta_ns, max_delta_ns)`.
- Emit typed events: `FrozenSignalAlarmEvent`, `StaleSignalEvent`, `OutlierRejectedEvent`.
- Status: Implemented in io/validators.py.

### 2) loaders.py

- Define ingestion entry point that applies validators before state updates.
- On outlier: keep inertial state, emit critical alert, skip JKO update.
- On frozen signal: lock Kernel D, switch degraded mode, emit alarm.
- On staleness: suspend JKO update, set degraded inference flag.
- Status: Implemented in io/loaders.py (ingestion gate + decision flags).

### 3) snapshots.py

- Serialize state using MessagePack or Protocol Buffers (binary only).
- Append SHA-256 or CRC32c hash footer per snapshot.
- Implement `write_then_rename(temp_path, final_path)` for atomicity.
- Implement `load_snapshot(path)` with hash verification before injection.
- Status: Implemented in io/snapshots.py (msgpack + hash footer + atomic rename).

### 4) telemetry.py

- Implement `TelemetryBuffer` (non-blocking queue or ring buffer).
- Provide consumer worker to serialize and persist telemetry asynchronously.
- Add parity audit logging: SHA-256 of `rho` and OT cost.
- Ensure canonical float64 serialization before hashing.
- Status: Implemented in io/telemetry.py (buffer + parity hashes).

### 5) credentials.py

- Enforce environment-based credential injection.
- Provide helpers to read `.env` or OS variables safely.
- Reject missing credentials with explicit errors.
- Status: Implemented in io/credentials.py.

### 6) io/__init__.py

- Export public IO interfaces and document usage contracts.
- Status: Implemented in io/__init__.py.

## Testing Tasks

- Unit tests for validators (outlier, frozen signal, staleness).
- Snapshot round-trip test with hash verification and corruption detection.
- Telemetry non-blocking behavior under load (no compute stalls).
- Credential injection tests (missing/invalid env vars).

## Acceptance Criteria

- No blocking I/O on compute path.
- Snapshots are binary, atomic, and hash-verified.
- Parity hashes are deterministic across CPU/GPU.
- Secrets never appear in source or logs.
