#!/usr/bin/env python3
"""Policy compliance checker - policies defined in code.

Validation Scope: Entire repository (policies apply repo-wide)

NOTE: CODE_AUDIT_POLICIES_SPECIFICATION.md is documentation only.
All policies are defined in code (policy_checks() function).
If policies change, update policy_checks() directly in this script.

Outputs:
- Console summary (PASS/FAIL per policy)
- JSON report: tests/results/code_alignement_YYYY-MM-DD_HH-MM-SS.ffffff.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
REPORT_DIR = os.path.join(ROOT, "tests", "results")


@dataclass(frozen=True)
class PolicyResult:
    policy_id: int
    name: str
    passed: bool
    details: str


def read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def file_exists(path: str) -> bool:
    return os.path.exists(path)


def find_in_file(pattern: str, path: str) -> bool:
    return re.search(pattern, read_text(path), re.MULTILINE) is not None


def find_in_dir(pattern: str, dir_path: str, extensions: Tuple[str, ...] = (".py",)) -> Tuple[bool, str]:
    regex = re.compile(pattern, re.MULTILINE)
    for root, _, files in os.walk(dir_path):
        for name in files:
            if not name.endswith(extensions):
                continue
            full_path = os.path.join(root, name)
            if regex.search(read_text(full_path)):
                return True, "OK"
    return False, f"Pattern '{pattern}' not found in {dir_path}"


def check_dir_exists(dir_path: str) -> Tuple[bool, str]:
    """Check if directory exists."""
    if os.path.isdir(dir_path):
        return True, "OK"
    return False, f"Directory not found: {dir_path}"


def require_all(patterns: List[str], path: str) -> Tuple[bool, str]:
    missing = [p for p in patterns if not find_in_file(p, path)]
    if missing:
        return False, f"Missing patterns in {path}: {missing}"
    return True, "OK"


def require_any(patterns: List[str], path: str) -> Tuple[bool, str]:
    for p in patterns:
        if find_in_file(p, path):
            return True, "OK"
    return False, f"None of patterns found in {path}: {patterns}"


def policy_checks() -> List[Tuple[int, str, Callable[[], Tuple[bool, str]]]]:
    return [
        (
            1,
            "Configuration Sourcing (Zero-Heuristics)",
            lambda: require_all(
                [
                    r"FIELD_TO_SECTION_MAP",
                    r"Missing required config\.toml entry",
                ],
                os.path.join(ROOT, "Python", "api", "config.py"),
            ),
        ),
        (
            2,
            "Configuration Immutability (Locked Subsections)",
            lambda: require_all(
                [
                    r"float_precision",
                    r"jax_platform",
                    r"snapshot_path",
                    r"telemetry_buffer_maxlen",
                    r"credentials_vault_path",
                    r"telemetry_hash_interval_steps",
                    r"snapshot_integrity_hash_algorithm",
                    r"allowed_mutation_rate_per_hour",
                    r"max_deep_tuning_iterations",
                    r"checkpoint_path",
                    r"mutation_protocol_version",
                ],
                os.path.join(ROOT, "Python", "io", "config_mutation.py"),
            ),
        ),
        (
            3,
            "Validation Schema Enforcement",
            lambda: require_any(
                [r"validation_schema", r"ConfigMutationError", r"_validate_config"],
                os.path.join(ROOT, "Python", "io", "config_mutation.py"),
            ),
        ),
        (
            4,
            "Atomic Configuration Mutation Protocol",
            lambda: require_all(
                [r"O_EXCL", r"fsync", r"os\.replace", r"mutations\.log"],
                os.path.join(ROOT, "Python", "io", "config_mutation.py"),
            ),
        ),
        (
            5,
            "Mutation Rate Limiting and Rollback",
            lambda: require_any(
                [r"allowed_mutation_rate_per_hour", r"rollback", r"rmse"],
                os.path.join(ROOT, "Python", "io", "config_mutation.py"),
            ),
        ),
        (
            6,
            "Walk-Forward Validation (Causal)",
            lambda: require_any(
                [r"walk_forward", r"WalkForward"],
                os.path.join(ROOT, "Python", "core", "meta_optimizer.py"),
            ),
        ),
        (
            7,
            "CUSUM Dynamic Threshold with Kurtosis",
            lambda: find_in_dir(r"kurtosis|kappa|ln.*kappa", os.path.join(ROOT, "Python")),
        ),
        (
            8,
            "Signature Depth Constraint (M in [3,5])",
            lambda: find_in_dir(r"log_sig_depth|kernel_d_depth", os.path.join(ROOT, "Python")),
        ),
        (
            9,
            "Sinkhorn Epsilon Bounds",
            lambda: require_any(
                [r"sinkhorn_epsilon_min", r"epsilon_min", r"1e-4"],
                os.path.join(ROOT, "Python", "core", "sinkhorn.py"),
            ),
        ),
        (
            10,
            "CFL Condition for PIDE Schemes",
            lambda: find_in_dir(r"CFL|courant|c_safe", os.path.join(ROOT, "Python")),
        ),
        (
            11,
            "64-bit Precision Enablement",
            lambda: require_any(
                [r"jax_enable_x64", r"float64"],
                os.path.join(ROOT, "Python", "api", "config.py"),
            ),
        ),
        (
            12,
            "Stop-Gradient for Diagnostics",
            lambda: find_in_dir(r"stop_gradient", os.path.join(ROOT, "Python")),
        ),
        (
            13,
            "Kernel Purity and Statelessness",
            lambda: (
                not find_in_dir(r"\bprint\(|\bopen\(", os.path.join(ROOT, "Python", "kernels"))[0],
                "No I/O in kernels" if not find_in_dir(r"\bprint\(|\bopen\(", os.path.join(ROOT, "Python", "kernels"))[0] else "I/O found in kernels",
            ),
        ),
        (
            14,
            "Frozen Signal Detection and Recovery",
            lambda: require_all(
                [r"detect_frozen_signal", r"FrozenSignal"],
                os.path.join(ROOT, "Python", "io", "validators.py"),
            ),
        ),
        (
            15,
            "Catastrophic Outlier Rejection (20 sigma)",
            lambda: require_all(
                [r"detect_catastrophic_outlier", r"sigma_bound"],
                os.path.join(ROOT, "Python", "io", "validators.py"),
            ),
        ),
        (
            16,
            "Minimum Injection Frequency (Nyquist Soft Limit)",
            lambda: require_any(
                [r"besov_nyquist_interval", r"nyquist"],
                os.path.join(ROOT, "Python", "io", "validators.py"),
            ),
        ),
        (
            17,
            "Staleness Policy and Degraded Mode Recovery (TTL)",
            lambda: require_any(
                [r"staleness_ttl", r"degraded", r"TTL"],
                os.path.join(ROOT, "Python", "core", "orchestrator.py"),
            ),
        ),
        (
            18,
            "Secret Injection via Environment Variables",
            lambda: require_any(
                [r"getenv", r"MissingCredentialError"],
                os.path.join(ROOT, "Python", "io", "credentials.py"),
            ),
        ),
        (
            19,
            "Snapshot Integrity (SHA-256) and Validation",
            lambda: require_any(
                [r"sha256", r"SHA256"],
                os.path.join(ROOT, "Python", "io", "snapshots.py"),
            ),
        ),
        (
            20,
            "Non-Blocking Telemetry and I/O",
            lambda: require_any(
                [r"Thread", r"queue", r"deque"],
                os.path.join(ROOT, "Python", "io", "telemetry.py"),
            ),
        ),
        (
            21,
            "Hardware Parity Audit Hashes",
            lambda: require_any(
                [r"telemetry_hash_interval", r"parity"],
                os.path.join(ROOT, "Python", "io", "telemetry.py"),
            ),
        ),
        (
            22,
            "Emergency Mode on Singularities (Holder Threshold)",
            lambda: require_any(
                [r"holder_threshold", r"Huber", r"robust"],
                os.path.join(ROOT, "Python", "core", "orchestrator.py"),
            ),
        ),
        (
            23,
            "Entropy-Driven Capacity Expansion (DGM)",
            lambda: require_any(
                [r"entropy", r"capacity", r"dgm_max_capacity"],
                os.path.join(ROOT, "Python", "core", "orchestrator.py"),
            ),
        ),
        (
            24,
            "Dynamic Sinkhorn Regularization Coupling",
            lambda: require_any(
                [r"epsilon_t", r"sigma_t", r"sinkhorn"],
                os.path.join(ROOT, "Python", "core", "sinkhorn.py"),
            ),
        ),
        (
            25,
            "Entropy Window and Learning Rate Scaling (JKO)",
            lambda: require_any(
                [r"entropy_window", r"learning_rate"],
                os.path.join(ROOT, "Python", "core", "orchestrator.py"),
            ),
        ),
        (
            26,
            "Load Shedding (Kernel D Depth Set)",
            lambda: require_any(
                [r"warmup_kernel_d_load_shedding", r"kernel_d_load_shedding_depths"],
                os.path.join(ROOT, "Python", "api", "warmup.py"),
            ),
        ),
        (
            27,
            "Deterministic Execution and PRNG Configuration",
            lambda: find_in_dir(r"JAX_DETERMINISTIC_REDUCTIONS|jax_default_prng_impl|XLA_FLAGS", ROOT),
        ),
        (
            28,
            "Dependency Pinning (Exact Versions)",
            lambda: require_any(
                [r"=="],
                os.path.join(ROOT, "requirements.txt"),
            ),
        ),
        (
            29,
            "Five-Layer Architecture Enforcement",
            lambda: (
                all(
                    os.path.isdir(os.path.join(ROOT, "Python", name))
                    for name in ("api", "core", "kernels", "io")
                )
                and os.path.isdir(os.path.join(ROOT, "tests")),
                "OK" if all(
                    os.path.isdir(os.path.join(ROOT, "Python", name))
                    for name in ("api", "core", "kernels", "io")
                ) else "Missing required layer directories",
            ),
        ),
        (
            30,
            "Snapshot Atomicity and Recovery (I/O)",
            lambda: require_any(
                [r"\.tmp", r"os\.replace", r"fsync"],
                os.path.join(ROOT, "Python", "io", "snapshots.py"),
            ),
        ),
        (
            31,
            "Meta-Optimization Checkpoint Integrity",
            lambda: require_any(
                [r"sha256", r"\.sha256"],
                os.path.join(ROOT, "Python", "core", "meta_optimizer.py"),
            ),
        ),
        (
            32,
            "TPE Resume Determinism",
            lambda: require_any(
                [r"rng_state", r"resume", r"trial_history"],
                os.path.join(ROOT, "Python", "core", "meta_optimizer.py"),
            ),
        ),
        (
            33,
            "Telemetry Flags and Alerts (Required Fields)",
            lambda: require_any(
                [r"degraded", r"emergency", r"regime_change", r"mode_collapse"],
                os.path.join(ROOT, "Python", "api", "schemas.py"),
            ),
        ),

        (
            36,
            "XLA No Host-Device Sync in Orchestrator",
            lambda: (
                not find_in_file(r"\.item\(\)", os.path.join(ROOT, "Python", "core", "orchestrator.py")),
                "OK" if not find_in_file(r"\.item\(\)", os.path.join(ROOT, "Python", "core", "orchestrator.py")) else "Host sync found in orchestrator",
            ),
        ),
        (
            37,
            "Vectorized vmap Parity",
            lambda: find_in_dir(r"vmap", os.path.join(ROOT, "Python")),
        ),
        (
            38,
            "JIT Cache Warmup Guarantees",
            lambda: require_any(
                [r"warmup_kernel_d_load_shedding"],
                os.path.join(ROOT, "Python", "api", "warmup.py"),
            ),
        ),

    ]


def run_checks() -> List[PolicyResult]:
    results: List[PolicyResult] = []
    for policy_id, name, checker in policy_checks():
        try:
            passed, details = checker()
        except Exception as exc:
            passed = False
            details = f"Exception: {exc}"
        results.append(PolicyResult(policy_id=policy_id, name=name, passed=passed, details=details))
    return results


def write_report(results: List[PolicyResult]) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S.%f")
    report_path = os.path.join(REPORT_DIR, f"code_alignement_{timestamp}.json")
    payload = {
        "timestamp_utc": timestamp,
        "policy_count": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "results": [
            {
                "id": r.policy_id,
                "name": r.name,
                "passed": r.passed,
                "details": r.details,
            }
            for r in results
        ],
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return report_path


def main() -> int:
    """Run all policy compliance checks. Policies are defined in code, not loaded from file."""
    results = run_checks()
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}: Policy #{result.policy_id} - {result.name}")
        if not result.passed:
            print(f"  - {result.details}")

    report_path = write_report(results)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    print("")
    print("SUMMARY")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"Report: {report_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
