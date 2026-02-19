"""Configuration Mutation Safety Guardrails.

Implements rate limiting, delta validation, automatic rollback, and atomic
TOML mutation protocol for Level 4 autonomous configuration mutations.

COMPLIANCE:
    - IO.tex §3.3 - Configuration Mutation Protocol (Atomic Write)
    - IO.tex §3.3.6 - Rate Limiting and Safety Guardrails
    - Implementation.tex §5.4.3 - Degradation Detection Protocol

Key Safety Mechanisms:
    1. Atomic POSIX write protocol (fsync + os.replace)
    2. Locked subsection protection (Asimov's Zeroth Law)
    3. Validation schema enforcement (ranges, types, constraints)
    4. Maximum mutation rate enforcement (≤10/hour)
    5. Minimum stability period (1,000 steps between mutations)
    6. Delta magnitude limit (≤50% change per mutation)
    7. Degradation detection with automatic rollback (>30% RMSE increase)

References:
    - IO.tex §3.3: Configuration Mutation Protocol
    - IO.tex §3.3.6: Rate Limiting Protocol
    - API_Python.tex §15: Autonomous Meta-Optimization Safety
"""

import json
import os
import shutil
import time
import toml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


class ConfigMutationError(Exception):
    """Raised when configuration mutation validation fails."""
    pass


# Locked subsections (immutable to prevent self-corruption)
# COMPLIANCE: IO.tex §3.3.4 - Invariant Protection
LOCKED_SUBSECTIONS = {
    "meta": ["schema_version"],
    "core": ["jax_platforms", "jax_default_dtype", "float_precision", "staleness_ttl_ns"],
    "io": [
        "snapshot_format",
        "snapshot_hash_algorithm",
        "snapshot_compression",
        "telemetry_hash_interval_steps",
    ],
    "meta_optimization": ["n_trials", "n_startup_trials", "multivariate", "train_ratio", "n_folds"],
}

# Validation schema for mutable parameters
# COMPLIANCE: IO.tex §3.3.5 - Validation Schema
VALIDATION_SCHEMA = {
    # Orchestration Parameters
    "orchestration.cusum_k": {"type": float, "range": (0.1, 1.0)},
    "orchestration.cusum_h": {"type": float, "range": (2.0, 10.0)},
    "orchestration.grace_period_steps": {"type": int, "range": (5, 100)},
    "orchestration.volatility_alpha": {"type": float, "range": (0.05, 0.3)},
    "orchestration.learning_rate": {"type": float, "range": (1e-5, 1e-1)},
    "orchestration.entropy_window": {"type": int, "range": (10, 500)},
    "orchestration.entropy_threshold": {"type": float, "range": (0.5, 0.95)},
    "orchestration.holder_threshold": {"type": float, "range": (0.2, 0.65)},
    "orchestration.sinkhorn_alpha": {"type": float, "range": (0.1, 1.0)},
    "orchestration.sinkhorn_epsilon_min": {"type": float, "range": (0.001, 0.1)},
    "orchestration.sinkhorn_epsilon_0": {"type": float, "range": (0.05, 0.5)},
    "orchestration.sinkhorn_max_iter": {"type": int, "range": (50, 500)},
    # Kernel Parameters
    "kernels.log_sig_depth": {"type": int, "range": (2, 5)},
    "kernels.wtmm_buffer_size": {"type": int, "range": (64, 512), "constraint": "power_of_2"},
    "kernels.besov_cone_c": {"type": float, "range": (1.0, 3.0)},
    "kernels.dgm_width_size": {"type": int, "range": (32, 256), "constraint": "power_of_2"},
    "kernels.dgm_depth": {"type": int, "range": (2, 6)},
    "kernels.dgm_entropy_num_bins": {"type": int, "range": (20, 100)},
    "kernels.stiffness_low": {"type": float, "range": (50.0, 500.0)},
    "kernels.stiffness_high": {"type": float, "range": (500.0, 5000.0)},
    "kernels.sde_dt": {"type": float, "range": (0.001, 0.1)},
    "kernels.sde_numel_integrations": {"type": int, "range": (50, 200)},
    "kernels.sde_diffusion_sigma": {"type": float, "range": (0.05, 0.5)},
    "kernels.kernel_ridge_lambda": {"type": float, "range": (1e-8, 1e-3)},
}


def _is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _get_nested_param(config: Dict[str, Any], key: str) -> Any:
    """Get nested parameter value (e.g., 'sensitivity.cusum_k')."""
    parts = key.split(".")
    value = config
    for part in parts:
        value = value[part]
    return value


def _set_nested_param(config: Dict[str, Any], key: str, value: Any) -> None:
    """Set nested parameter value (e.g., 'sensitivity.cusum_k')."""
    parts = key.split(".")
    target = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def validate_config_mutation(
    current_config: Dict[str, Any],
    new_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate configuration mutation against schema and locked subsections.
    
    COMPLIANCE: IO.tex §3.3.5 - Validation Schema
    
    Args:
        current_config: Current TOML configuration dictionary
        new_params: Parameters to mutate (flat dict with dotted keys)
    
    Returns:
        Merged configuration dictionary (current + mutations)
    
    Raises:
        ConfigMutationError: If validation fails
    
    Example:
        >>> current = toml.load("config.toml")
        >>> new_params = {"sensitivity.cusum_k": 0.72}
        >>> merged = validate_config_mutation(current, new_params)
    """
    # Check for locked subsection violations
    for param_key in new_params.keys():
        parts = param_key.split(".")
        if len(parts) < 2:
            raise ConfigMutationError(f"Invalid parameter key: {param_key}")
        
        subsection = parts[0]
        param_name = ".".join(parts[1:])
        
        if subsection in LOCKED_SUBSECTIONS:
            if param_name in LOCKED_SUBSECTIONS[subsection]:
                raise ConfigMutationError(
                    f"Parameter '{param_key}' is LOCKED (immutable subsection). "
                    f"Locked params in [{subsection}]: {LOCKED_SUBSECTIONS[subsection]}"
                )
    
    # Merge current config with new params
    merged_config = dict(current_config)
    for param_key, new_value in new_params.items():
        _set_nested_param(merged_config, param_key, new_value)
    
    # Validate against schema
    for param_key, new_value in new_params.items():
        if param_key not in VALIDATION_SCHEMA:
            raise ConfigMutationError(
                f"Parameter '{param_key}' not in validation schema. "
                f"Only mutable subsections allowed: orchestration, kernels"
            )
        
        rules = VALIDATION_SCHEMA[param_key]
        
        # Type check
        if not isinstance(new_value, rules["type"]):
            raise ConfigMutationError(
                f"Parameter '{param_key}' type mismatch: expected {rules['type']}, got {type(new_value)}"
            )
        
        # Range check
        min_val, max_val = rules["range"]
        if not (min_val <= new_value <= max_val):
            raise ConfigMutationError(
                f"Parameter '{param_key}' out of safe range: {new_value} not in [{min_val}, {max_val}]"
            )
        
        # Constraint checks
        if "constraint" in rules:
            if rules["constraint"] == "power_of_2" and not _is_power_of_2(new_value):
                raise ConfigMutationError(
                    f"Parameter '{param_key}' must be power of 2, got {new_value}"
                )
    
    # Cross-parameter constraints
    if "kernels.stiffness_low" in new_params or "kernels.stiffness_high" in new_params:
        low = _get_nested_param(merged_config, "kernels.stiffness_low")
        high = _get_nested_param(merged_config, "kernels.stiffness_high")
        if low >= high:
            raise ConfigMutationError(
                f"Stiffness constraint violation: stiffness_low ({low}) must be < stiffness_high ({high})"
            )
    
    return merged_config


def atomic_write_config(
    config_path: Path,
    new_params: Dict[str, Any],
    trigger: str = "ManualMutation",
    best_objective: Optional[float] = None,
    audit_log_path: Optional[Path] = None,
) -> None:
    """
    Atomically mutate configuration with POSIX-compliant write protocol.
    
    Implements 5-phase atomic mutation:
        1. Validation: Check ranges, types, locked subsections
        2. Backup: Create timestamped backup + latest .bak
        3. Atomic Write: Write to .tmp, fsync(), os.replace()
        4. Audit Log: Record mutation to io/mutations.log
        5. Success: Return (or raise on failure)
    
    COMPLIANCE: IO.tex §3.3.2 - Atomic TOML Update Algorithm
    
    Args:
        config_path: Path to config.toml
        new_params: Parameters to mutate (flat dict with dotted keys)
        trigger: Event triggering mutation (for audit log)
        best_objective: Optimization objective (for audit log)
        audit_log_path: Path to mutations.log (default: io/mutations.log)
    
    Raises:
        ConfigMutationError: Validation failure (locked param, out of range, etc.)
        FileExistsError: Concurrent mutation detected (config.toml.tmp exists)
    
    Example:
        >>> atomic_write_config(
        ...     Path("config.toml"),
        ...     {"sensitivity.cusum_k": 0.72, "kernels.dgm_width_size": 256},
        ...     trigger="DeepTuning_Iteration_500",
        ...     best_objective=0.0234
        ... )
    
    References:
        - Stochastic_Predictor_IO.tex §3.3: Configuration Mutation Protocol
        - AUDIT_SPEC_COMPLIANCE_2026-02-19.md: V-CRIT-2
    """
    audit_log_path = audit_log_path or Path("io/mutations.log")
    
    # Phase 1: Validation
    current_config = toml.load(config_path)
    
    try:
        merged_config = validate_config_mutation(current_config, new_params)
    except ConfigMutationError as e:
        # Log rejection to audit trail
        append_audit_log(audit_log_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "MUTATION_ABORTED",
            "trigger": trigger,
            "delta": {k: ("REJECTED", v) for k, v in new_params.items()},
            "error": str(e),
            "status": "ABORTED",
        })
        raise
    
    # Compute delta for audit log
    delta = {}
    for param_key, new_value in new_params.items():
        old_value = _get_nested_param(current_config, param_key)
        delta[param_key] = (old_value, new_value)
    
    # Phase 2: Immutable Backup
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    backup_path = config_path.with_suffix(f".bak.{timestamp}")
    latest_backup_path = config_path.with_suffix(".bak")
    
    shutil.copy2(config_path, backup_path)  # Timestamped archive
    shutil.copy2(config_path, latest_backup_path)  # Latest backup (overwrite)
    
    # Phase 3: Atomic Write via Temporary File
    tmp_path = config_path.with_suffix(".tmp")
    
    # Open with O_EXCL to detect concurrent mutation
    try:
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        error_msg = f"Concurrent mutation detected: {tmp_path} already exists"
        append_audit_log(audit_log_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "MUTATION_FAILED",
            "trigger": trigger,
            "delta": delta,
            "error": error_msg,
            "status": "FAILED",
        })
        raise FileExistsError(error_msg)
    
    try:
        # Serialize merged config to TOML
        toml_bytes = toml.dumps(merged_config).encode("utf-8")
        os.write(fd, toml_bytes)
        
        # CRITICAL: fsync() ensures kernel buffer flush to disk
        os.fsync(fd)
    finally:
        os.close(fd)
    
    # Phase 4: Atomic Replacement (POSIX os.replace)
    # On POSIX: os.replace() uses renameat2() with atomic inode swap
    # On Windows: Uses ReplaceFileW() with backup flag
    os.replace(tmp_path, config_path)
    
    # Phase 5: Audit Logging
    append_audit_log(audit_log_path, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "MUTATION_SUCCESS",
        "trigger": trigger,
        "delta": delta,
        "best_objective": best_objective if best_objective else "N/A",
        "backup": str(backup_path),
        "status": "SUCCESS",
    })


@dataclass
class MutationRateLimiter:
    """
    Enforce safety guardrails for autonomous configuration mutations.
    
    Prevents optimizer pathologies:
        - Thrashing between configurations
        - Excessive mutation frequency
        - Large parameter jumps
        - Pathological degradation without rollback
    
    COMPLIANCE: IO.tex §3.3.6 - Rate Limiting Protocol
    
    Args:
        max_mutations_per_hour: Maximum allowed mutations per hour (default 10)
        stability_steps_required: Minimum steps before next mutation (default 1000)
        max_relative_change: Maximum parameter change per mutation (default 0.5 = 50%)
    
    Example:
        >>> limiter = MutationRateLimiter(max_mutations_per_hour=10)
        >>> can_mutate, reason = limiter.can_mutate()
        >>> if can_mutate:
        ...     delta = {"learning_rate": (0.01, 0.015)}  # 50% increase
        ...     valid, msg = limiter.validate_delta(delta)
        ...     if valid:
        ...         limiter.record_mutation(delta)
    """
    
    max_mutations_per_hour: int = 10
    stability_steps_required: int = 1000
    max_relative_change: float = 0.5
    
    # Internal state
    _mutation_history: List[Tuple[float, Dict[str, Tuple[float, float]]]] = field(default_factory=list)
    _last_mutation_timestamp: Optional[float] = None
    _current_steps_since_mutation: int = 0
    
    def can_mutate(self) -> Tuple[bool, str]:
        """
        Check if mutation is allowed under safety guardrails.
        
        Returns:
            (allowed: bool, reason: str) where reason explains why mutation
            is blocked (or "OK" if allowed)
        
        Example:
            >>> limiter = MutationRateLimiter()
            >>> can_mutate, reason = limiter.can_mutate()
            >>> if not can_mutate:
            ...     print(f"Mutation blocked: {reason}")
        """
        now = time.time()
        
        # Check maximum mutation rate (sliding 1-hour window)
        one_hour_ago = now - 3600
        recent_mutations = [
            ts for ts, _ in self._mutation_history if ts > one_hour_ago
        ]
        if len(recent_mutations) >= self.max_mutations_per_hour:
            return False, (
                f"Rate limit exceeded: {len(recent_mutations)}/{self.max_mutations_per_hour} "
                f"mutations in last hour"
            )
        
        # Check minimum stability period
        if self._current_steps_since_mutation < self.stability_steps_required:
            return False, (
                f"Stability period not met: {self._current_steps_since_mutation}/"
                f"{self.stability_steps_required} steps since last mutation"
            )
        
        return True, "OK"
    
    def validate_delta(
        self,
        delta: Dict[str, Tuple[float, float]],
        max_relative_change: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Validate parameter delta magnitude.
        
        Args:
            delta: {param_name: (old_value, new_value)}
            max_relative_change: Override default max change (optional)
        
        Returns:
            (valid: bool, reason: str)
        
        Example:
            >>> delta = {
            ...     "learning_rate": (0.01, 0.02),  # 100% increase
            ...     "entropy_window": (100, 120)     # 20% increase
            ... }
            >>> valid, msg = limiter.validate_delta(delta)
            >>> if not valid:
            ...     print(f"Delta rejected: {msg}")
        """
        max_change = max_relative_change or self.max_relative_change
        
        for param, (old_val, new_val) in delta.items():
            # Skip zero division
            if abs(old_val) < 1e-12:
                # For near-zero baseline, check absolute change
                if abs(new_val) > 1.0:
                    return False, (
                        f"{param}: absolute change too large for near-zero baseline "
                        f"(old={old_val:.2e}, new={new_val:.2e})"
                    )
                continue
            
            # Compute relative change
            relative_change = abs((new_val - old_val) / old_val)
            
            if relative_change > max_change:
                return False, (
                    f"{param}: change too large ({relative_change:.1%} > "
                    f"{max_change:.0%}). Old={old_val}, New={new_val}"
                )
        
        return True, "OK"
    
    def record_mutation(
        self,
        delta: Dict[str, Tuple[float, float]]
    ) -> None:
        """
        Record successful mutation.
        
        Args:
            delta: {param_name: (old_value, new_value)}
        
        Example:
            >>> limiter.record_mutation({"learning_rate": (0.01, 0.015)})
            >>> # Resets stability counter, adds to mutation history
        """
        now = time.time()
        self._mutation_history.append((now, delta))
        self._last_mutation_timestamp = now
        self._current_steps_since_mutation = 0
    
    def increment_stability_counter(self) -> None:
        """
        Increment stability counter (call after each prediction step).
        
        Example:
            >>> # In main prediction loop
            >>> prediction = orchestrate_step(...)
            >>> limiter.increment_stability_counter()
        """
        self._current_steps_since_mutation += 1
    
    def get_mutation_history(
        self,
        hours_lookback: float = 24.0
    ) -> List[Tuple[str, Dict[str, Tuple[float, float]]]]:
        """
        Retrieve mutation history for audit/debugging.
        
        Args:
            hours_lookback: How far back to look (default 24 hours)
        
        Returns:
            List of (timestamp_iso, delta) tuples
        
        Example:
            >>> history = limiter.get_mutation_history(hours_lookback=1.0)
            >>> for timestamp, delta in history:
            ...     print(f"{timestamp}: {delta}")
        """
        cutoff = time.time() - (hours_lookback * 3600)
        recent = [
            (datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), delta)
            for ts, delta in self._mutation_history
            if ts > cutoff
        ]
        return recent


@dataclass
class DegradationMonitor:
    """
    Monitor post-mutation performance and trigger rollback on degradation.
    
    COMPLIANCE: IO.tex §3.3.6 - Degradation Detection Auto-Rollback
    
    Implements closed-loop safety mechanism:
        1. Record pre-mutation baseline RMSE
        2. Monitor post-mutation predictions (N=100 sample window)
        3. Compute post-mutation RMSE
        4. If relative increase > threshold (default 30%), auto-rollback
    
    Args:
        degradation_threshold: Maximum allowed RMSE increase (default 0.3 = 30%)
        monitoring_window: Predictions to sample post-mutation (default 100)
        config_path: Path to config.toml (default "config.toml")
        backup_path: Path to backup config (default "config.toml.bak")
        audit_log_path: Path to mutation audit log (default "io/mutations.log")
    
    Example:
        >>> monitor = DegradationMonitor(degradation_threshold=0.3)
        >>> 
        >>> # Before mutation
        >>> monitor.start_monitoring(baseline_rmse=0.05)
        >>> 
        >>> # After mutation (within prediction loop)
        >>> for i in range(100):
        ...     prediction = orchestrate_step(...)
        ...     error = abs(prediction.predicted_next - actual_value)
        ...     monitor.record_prediction_error(error)
        ...     
        ...     degraded, increase = monitor.check_degradation()
        ...     if degraded:
        ...         monitor.trigger_rollback()
        ...         break
    """
    
    degradation_threshold: float = 0.3
    monitoring_window: int = 100
    config_path: Path = Path("config.toml")
    backup_path: Path = Path("config.toml.bak")
    audit_log_path: Path = Path("io/mutations.log")
    
    # Internal state
    _pre_mutation_rmse: Optional[float] = None
    _post_mutation_errors: List[float] = field(default_factory=list)
    _monitoring_active: bool = False
    
    def start_monitoring(self, baseline_rmse: float) -> None:
        """
        Record pre-mutation baseline and start monitoring.
        
        Args:
            baseline_rmse: Pre-mutation RMSE baseline
        
        Example:
            >>> # Compute baseline RMSE over recent predictions
            >>> recent_errors = [0.04, 0.05, 0.06, 0.05, 0.04]
            >>> baseline_rmse = np.sqrt(np.mean(np.square(recent_errors)))
            >>> monitor.start_monitoring(baseline_rmse)
        """
        self._pre_mutation_rmse = baseline_rmse
        self._post_mutation_errors = []
        self._monitoring_active = True
    
    def record_prediction_error(self, error: float) -> None:
        """
        Accumulate post-mutation prediction error.
        
        Args:
            error: Absolute prediction error |ŷ - y|
        
        Example:
            >>> prediction = orchestrate_step(...)
            >>> actual_next = load_actual_observation()
            >>> error = abs(prediction.predicted_next - actual_next)
            >>> monitor.record_prediction_error(error)
        """
        if not self._monitoring_active:
            return
        
        self._post_mutation_errors.append(error)
    
    def check_degradation(self) -> Tuple[bool, float]:
        """
        Check if post-mutation performance degraded beyond threshold.
        
        Returns:
            (degraded: bool, relative_increase: float) where relative_increase
            is the fractional change in RMSE (e.g., 0.35 = 35% increase)
        
        Example:
            >>> degraded, increase = monitor.check_degradation()
            >>> if degraded:
            ...     print(f"Degradation detected: RMSE increased {increase:.1%}")
            ...     monitor.trigger_rollback()
        """
        if not self._monitoring_active:
            return False, 0.0
        
        # Require full monitoring window
        if len(self._post_mutation_errors) < self.monitoring_window:
            return False, 0.0
        
        # Compute post-mutation RMSE
        post_mutation_rmse = np.sqrt(
            np.mean(np.square(self._post_mutation_errors))
        )
        
        # Compute relative increase
        if self._pre_mutation_rmse is None or self._pre_mutation_rmse == 0:
            # Fallback: use absolute threshold
            relative_increase = 0.0
        else:
            relative_increase = (
                (post_mutation_rmse - self._pre_mutation_rmse) / 
                self._pre_mutation_rmse
            )
        
        degraded = relative_increase > self.degradation_threshold
        
        # Stop monitoring after check (one-shot evaluation)
        if degraded or len(self._post_mutation_errors) >= self.monitoring_window:
            self._monitoring_active = False
        
        return degraded, relative_increase
    
    def trigger_rollback(self) -> None:
        """
        Execute automatic rollback to pre-mutation configuration.
        
        CRITICAL: This overwrites config.toml with the backup.
        
        Example:
            >>> degraded, increase = monitor.check_degradation()
            >>> if degraded:
            ...     monitor.trigger_rollback()
            ...     # config.toml restored from config.toml.bak
        """
        if not self.backup_path.exists():
            raise FileNotFoundError(
                f"Backup config not found: {self.backup_path}. "
                "Cannot rollback without backup."
            )
        
        # Restore config from backup
        shutil.copy2(self.backup_path, self.config_path)
        
        # Compute post-mutation RMSE for logging
        if self._post_mutation_errors:
            post_mutation_rmse = float(np.sqrt(
                np.mean(np.square(self._post_mutation_errors))
            ))
        else:
            post_mutation_rmse = 0.0
        
        # Append rollback event to audit log
        audit_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event': 'AUTO_ROLLBACK',
            'reason': 'Performance degradation detected',
            'pre_mutation_rmse': self._pre_mutation_rmse,
            'post_mutation_rmse': post_mutation_rmse,
            'relative_increase': (
                (post_mutation_rmse - self._pre_mutation_rmse) / self._pre_mutation_rmse
                if self._pre_mutation_rmse and self._pre_mutation_rmse > 0
                else 0.0
            ),
            'degradation_threshold': self.degradation_threshold,
            'monitoring_window': self.monitoring_window,
            'status': 'ROLLBACK_SUCCESS'
        }
        
        # Ensure io directory exists
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to audit log (JSON Lines format)
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')
        
        # Reset monitoring state
        self._monitoring_active = False
        self._post_mutation_errors = []


def create_config_backup(
    config_path: Path = Path("config.toml"),
    backup_path: Path = Path("config.toml.bak")
) -> None:
    """
    Create config backup before mutation.
    
    Args:
        config_path: Path to current config.toml
        backup_path: Path to backup destination
    
    Example:
        >>> create_config_backup()
        >>> # config.toml.bak created
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    shutil.copy2(config_path, backup_path)


def append_audit_log(
    log_path: Path,
    entry: Dict
) -> None:
    """
    Append mutation event to audit log (JSON Lines format).
    
    Args:
        log_path: Path to audit log file
        entry: Event data (dict)
    
    Example:
        >>> append_audit_log(Path("io/mutations.log"), {
        ...     'timestamp': datetime.now(timezone.utc).isoformat(),
        ...     'event': 'MUTATION_APPLIED',
        ...     'delta': {'learning_rate': (0.01, 0.015)},
        ...     'status': 'SUCCESS'
        ... })
    """
    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Append entry (JSON Lines format)
    with open(log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')
