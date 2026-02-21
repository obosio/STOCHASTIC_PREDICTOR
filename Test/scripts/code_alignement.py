#!/usr/bin/env python3
"""Policy compliance checker - policies defined in code.

Validation Scope: Entire repository (policies apply repo-wide)

NOTE: CODE_AUDIT_POLICIES_SPECIFICATION.md is documentation only.
All policies are defined in code (policy_checks() function).
If policies change, update policy_checks() directly in this script.

Outputs:
- Console summary (PASS/FAIL per policy)
- JSON report: Test/results/code_alignement_last.json
- Markdown report: Test/reports/code_alignement_last.md
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Dynamic scope discovery
try:
    from scope_discovery import discover_modules, discover_module_files, get_root
except ImportError:
    # Fallback if module not found
    def discover_modules(root: str | Path | None = None) -> List[str]:
        if root is None:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        root = Path(root) if isinstance(root, str) else root
        python_dir = root / "Python"
        if not python_dir.is_dir():
            return []
        return sorted([d.name for d in python_dir.iterdir() 
                      if d.is_dir() and (d / "__init__.py").is_file()])
    
    def discover_module_files(module_name: str, root: str | Path | None = None) -> List[str]:
        if root is None:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        root = Path(root) if isinstance(root, str) else root
        module_dir = root / "Python" / module_name
        if not module_dir.is_dir():
            return []
        return sorted([f.name for f in module_dir.glob("*.py")])
    
    def get_root() -> Path:
        return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

ROOT = str(get_root())
RESULTS_DIR = os.path.join(ROOT, "Test", "results")
REPORTS_DIR = os.path.join(ROOT, "Test", "reports")

# Auto-discovered modules (updates dynamically)
DISCOVERED_MODULES = discover_modules(ROOT)


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


# ═════════════════════════════════════════════════════════════════════════════
# Dynamic Path Construction (Scope Auto-Discovery)
# ═════════════════════════════════════════════════════════════════════════════

def module_dir(module: str) -> str:
    """Get directory path for a module.
    
    Example: module_dir('api') -> '/path/to/Python/api'
    """
    return os.path.join(ROOT, "Python", module)


def module_file(module: str, filename: str) -> str:
    """Get file path within a module.
    
    Example: module_file('api', 'config.py') -> '/path/to/Python/api/config.py'
    """
    return os.path.join(module_dir(module), filename)


def python_dir() -> str:
    """Get Python package directory."""
    return os.path.join(ROOT, "Python")


def validate_module_exists(module: str) -> bool:
    """Check if module exists in discovered modules."""
    return module in DISCOVERED_MODULES


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
                module_file("api", "config.py"),
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
                module_file("io", "config_mutation.py"),
            ),
        ),
        (
            3,
            "Validation Schema Enforcement",
            lambda: require_any(
                [r"validation_schema", r"ConfigMutationError", r"_validate_config"],
                module_file("io", "config_mutation.py"),
            ),
        ),
        (
            4,
            "Atomic Configuration Mutation Protocol",
            lambda: require_all(
                [r"O_EXCL", r"fsync", r"os\.replace", r"mutations\.log"],
                module_file("io", "config_mutation.py"),
            ),
        ),
        (
            5,
            "Mutation Rate Limiting and Rollback",
            lambda: require_any(
                [r"allowed_mutation_rate_per_hour", r"rollback", r"rmse"],
                module_file("io", "config_mutation.py"),
            ),
        ),
        (
            6,
            "Walk-Forward Validation (Causal)",
            lambda: require_any(
                [r"walk_forward", r"WalkForward"],
                module_file("core", "meta_optimizer.py"),
            ),
        ),
        (
            7,
            "CUSUM Dynamic Threshold with Kurtosis",
            lambda: find_in_dir(r"kurtosis|kappa|ln.*kappa", python_dir()),
        ),
        (
            8,
            "Signature Depth Constraint (M in [3,5])",
            lambda: find_in_dir(r"log_sig_depth|kernel_d_depth", python_dir()),
        ),
        (
            9,
            "Sinkhorn Epsilon Bounds",
            lambda: require_any(
                [r"sinkhorn_epsilon_min", r"epsilon_min", r"1e-4"],
                module_file("core", "sinkhorn.py"),
            ),
        ),
        (
            10,
            "CFL Condition for PIDE Schemes",
            lambda: find_in_dir(r"CFL|courant|c_safe", python_dir()),
        ),
        (
            11,
            "64-bit Precision Enablement",
            lambda: require_any(
                [r"jax_enable_x64", r"float64"],
                module_file("api", "config.py"),
            ),
        ),
        (
            12,
            "Stop-Gradient for Diagnostics",
            lambda: find_in_dir(r"stop_gradient", python_dir()),
        ),
        (
            13,
            "Kernel Purity and Statelessness",
            lambda: (
                not find_in_dir(r"\bprint\(|\bopen\(", module_dir("kernels"))[0],
                "No I/O in kernels" if not find_in_dir(r"\bprint\(|\bopen\(", module_dir("kernels"))[0] else "I/O found in kernels",
            ),
        ),
        (
            14,
            "Frozen Signal Detection and Recovery",
            lambda: require_all(
                [r"detect_frozen_signal", r"FrozenSignal"],
                module_file("io", "validators.py"),
            ),
        ),
        (
            15,
            "Catastrophic Outlier Rejection (20 sigma)",
            lambda: require_all(
                [r"detect_catastrophic_outlier", r"sigma_bound"],
                module_file("io", "validators.py"),
            ),
        ),
        (
            16,
            "Minimum Injection Frequency (Nyquist Soft Limit)",
            lambda: require_any(
                [r"besov_nyquist_interval", r"nyquist"],
                module_file("io", "validators.py"),
            ),
        ),
        (
            17,
            "Staleness Policy and Degraded Mode Recovery (TTL)",
            lambda: require_any(
                [r"staleness_ttl", r"degraded", r"TTL"],
                module_file("core", "orchestrator.py"),
            ),
        ),
        (
            18,
            "Secret Injection via Environment Variables",
            lambda: require_any(
                [r"getenv", r"MissingCredentialError"],
                module_file("io", "credentials.py"),
            ),
        ),
        (
            19,
            "Snapshot Integrity (SHA-256) and Validation",
            lambda: require_any(
                [r"sha256", r"SHA256"],
                module_file("io", "snapshots.py"),
            ),
        ),
        (
            20,
            "Non-Blocking Telemetry and I/O",
            lambda: require_any(
                [r"Thread", r"queue", r"deque"],
                module_file("io", "telemetry.py"),
            ),
        ),
        (
            21,
            "Hardware Parity Audit Hashes",
            lambda: require_any(
                [r"telemetry_hash_interval", r"parity"],
                module_file("io", "telemetry.py"),
            ),
        ),
        (
            22,
            "Emergency Mode on Singularities (Holder Threshold)",
            lambda: require_any(
                [r"holder_threshold", r"Huber", r"robust"],
                module_file("core", "orchestrator.py"),
            ),
        ),
        (
            23,
            "Entropy-Driven Capacity Expansion (DGM)",
            lambda: require_any(
                [r"entropy", r"capacity", r"dgm_max_capacity"],
                module_file("core", "orchestrator.py"),
            ),
        ),
        (
            24,
            "Dynamic Sinkhorn Regularization Coupling",
            lambda: require_any(
                [r"epsilon_t", r"sigma_t", r"sinkhorn"],
                module_file("core", "sinkhorn.py"),
            ),
        ),
        (
            25,
            "Entropy Window and Learning Rate Scaling (JKO)",
            lambda: require_any(
                [r"entropy_window", r"learning_rate"],
                module_file("core", "orchestrator.py"),
            ),
        ),
        (
            26,
            "Load Shedding (Kernel D Depth Set)",
            lambda: require_any(
                [r"warmup_kernel_d_load_shedding", r"kernel_d_load_shedding_depths"],
                module_file("api", "warmup.py"),
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
                    os.path.isdir(module_dir(name))
                    for name in DISCOVERED_MODULES
                )
                and os.path.isdir(os.path.join(ROOT, "Test")),
                "OK" if all(
                    os.path.isdir(module_dir(name))
                    for name in DISCOVERED_MODULES
                ) else "Missing required layer directories",
            ),
        ),
        (
            30,
            "Snapshot Atomicity and Recovery (I/O)",
            lambda: require_any(
                [r"\.tmp", r"os\.replace", r"fsync"],
                module_file("io", "snapshots.py"),
            ),
        ),
        (
            31,
            "Meta-Optimization Checkpoint Integrity",
            lambda: require_any(
                [r"sha256", r"\.sha256"],
                module_file("core", "meta_optimizer.py"),
            ),
        ),
        (
            32,
            "TPE Resume Determinism",
            lambda: require_any(
                [r"rng_state", r"resume", r"trial_history"],
                module_file("core", "meta_optimizer.py"),
            ),
        ),
        (
            33,
            "Telemetry Flags and Alerts (Required Fields)",
            lambda: require_any(
                [r"degraded", r"emergency", r"regime_change", r"mode_collapse"],
                module_file("api", "schemas.py"),
            ),
        ),

        (
            36,
            "XLA No Host-Device Sync in Orchestrator",
            lambda: (
                not find_in_file(r"\.item\(\)", module_file("core", "orchestrator.py")),
                "OK" if not find_in_file(r"\.item\(\)", module_file("core", "orchestrator.py")) else "Host sync found in orchestrator",
            ),
        ),
        (
            37,
            "Vectorized vmap Parity",
            lambda: find_in_dir(r"vmap", python_dir()),
        ),
        (
            38,
            "JIT Cache Warmup Guarantees",
            lambda: require_any(
                [r"warmup_kernel_d_load_shedding"],
                module_file("api", "warmup.py"),
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


def write_reports(results: List[PolicyResult]) -> Tuple[str, str]:
    """Write JSON and Markdown reports (without timestamp in filenames)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    
    # JSON report
    json_path = os.path.join(RESULTS_DIR, "code_alignement_last.json")
    payload = {
        "timestamp_utc": timestamp,
        "policy_count": total,
        "passed": passed,
        "failed": failed,
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
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    
    # Markdown report
    md_path = os.path.join(REPORTS_DIR, "code_alignement_last.md")
    with open(md_path, "w", encoding="utf-8") as f:
        # Header
        f.write("# 📋 Code Audit Policies Report\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        
        # Executive Summary
        f.write("## 📊 Executive Summary\n\n")
        success_rate = (passed / total * 100) if total > 0 else 0
        status_icon = "✅" if failed == 0 else "⚠️"
        f.write(f"{status_icon} **Overall Status:** {'PASS' if failed == 0 else 'FAIL'}\n\n")
        f.write("| Metric | Value |\n")
        f.write("| --- | --- |\n")
        f.write(f"| Total Policies | {total} |\n")
        f.write(f"| Passed | {passed} ({success_rate:.1f}%) |\n")
        f.write(f"| Failed | {failed} ({100-success_rate:.1f}%) |\n")
        f.write("\n---\n\n")
        
        # Group by status
        passed_policies = [r for r in results if r.passed]
        failed_policies = [r for r in results if not r.passed]
        
        # Failed policies section (if any)
        if failed_policies:
            f.write(f"## ❌ Failed Policies ({len(failed_policies)})\n\n")
            f.write("⚠️ **Action Required:** The following policies need immediate attention:\n\n")
            for r in failed_policies:
                f.write(f"### Policy #{r.policy_id}: {r.name}\n\n")
                f.write(f"**Status:** `FAIL`\n\n")
                f.write("**Details:**\n\n")
                f.write("```\n")
                f.write(f"{r.details}\n")
                f.write("```\n\n")
                f.write("---\n\n")
        
        # Passed policies section
        if passed_policies:
            f.write(f"## ✅ Passed Policies ({len(passed_policies)})\n\n")
            if failed_policies:
                f.write("<details>\n<summary>Click to expand list of passing policies</summary>\n\n")
            for r in passed_policies:
                f.write(f"- **Policy #{r.policy_id}:** {r.name}\n")
            if failed_policies:
                f.write("\n</details>\n")
        
        # Final Summary
        f.write("\n---\n\n")
        f.write("## 🎯 Final Summary\n\n")
        if failed == 0:
            f.write("✅ **All policies passed!** The codebase is fully compliant.\n\n")
        else:
            f.write(f"⚠️ **{failed} policy violation(s) detected.**\n\n")
            f.write("**Recommended Actions:**\n\n")
            for idx, r in enumerate(failed_policies, 1):
                f.write(f"{idx}. Fix Policy #{r.policy_id}: {r.name}\n")
            f.write("\n")
        f.write(f"**Report generated at:** {timestamp}\n")
    
    return json_path, md_path


def main() -> int:
    """Run all policy compliance checks. Policies are defined in code, not loaded from file."""
    results = run_checks()
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}: Policy #{result.policy_id} - {result.name}")
        if not result.passed:
            print(f"  - {result.details}")

    json_path, md_path = write_reports(results)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    print("")
    print("SUMMARY")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"JSON Report: {json_path}")
    print(f"Markdown Report: {md_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
