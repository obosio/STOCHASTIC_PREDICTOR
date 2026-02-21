#!/usr/bin/env python3
"""Dependency Version Validation - Golden Master Compliance.

Validates that all installed dependencies match the exact versions specified
in requirements.txt. Handles PEP 508 environment markers for platform-specific
dependencies (JAX/JAXlib).

Validation Scope: All packages in requirements.txt

Philosophy:
    - Fail-fast: Stop at first version mismatch
    - Platform-aware: Detect current platform and parse environment markers
    - Zero tolerance: Exact version matching required (Golden Master policy)

Output:
    - Console summary (PASS/FAIL per package)
    - JSON report: Test/results/dependency_check_last.json
    - Markdown report: Test/reports/dependency_check_last.md
    - Exit code: 0 if all match, 1 if any mismatch or error
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


def get_root() -> Path:
    """Get project root directory."""
    return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def get_platform_info() -> Dict[str, str]:
    """Get current platform information for environment marker evaluation.
    
    Returns:
        Dict with sys_platform and platform_machine
    """
    return {
        "sys_platform": sys.platform,
        "platform_machine": platform.machine(),
    }


def parse_requirement_line(line: str, platform_info: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """Parse a requirements.txt line with optional environment markers.
    
    Args:
        line: Line from requirements.txt (e.g., "jax==0.4.38; sys_platform == 'darwin'")
        platform_info: Current platform information
    
    Returns:
        Tuple of (package_name, version) if line matches current platform, None otherwise
    """
    line = line.strip()
    
    # Skip empty lines and comments
    if not line or line.startswith("#"):
        return None
    
    # Check if line has environment marker
    if ";" in line:
        requirement_part, marker_part = line.split(";", 1)
        requirement_part = requirement_part.strip()
        marker_part = marker_part.strip()
        
        # Evaluate environment marker
        if not evaluate_marker(marker_part, platform_info):
            return None  # This requirement doesn't apply to current platform
    else:
        requirement_part = line
    
    # Parse package==version
    match = re.match(r"^([a-zA-Z0-9_-]+)==([0-9.]+)$", requirement_part)
    if match:
        package_name = match.group(1)
        version = match.group(2)
        return (package_name, version)
    
    return None


def evaluate_marker(marker: str, platform_info: Dict[str, str]) -> bool:
    """Evaluate PEP 508 environment marker.
    
    Args:
        marker: Environment marker string (e.g., "sys_platform == 'darwin' and platform_machine == 'x86_64'")
        platform_info: Current platform information
    
    Returns:
        True if marker matches current platform
    """
    # Simple marker evaluation for sys_platform and platform_machine
    # Replace marker variables with actual values
    marker_eval = marker
    
    # Replace sys_platform
    if "sys_platform" in marker:
        marker_eval = marker_eval.replace("sys_platform", f"'{platform_info['sys_platform']}'")
    
    # Replace platform_machine
    if "platform_machine" in marker:
        marker_eval = marker_eval.replace("platform_machine", f"'{platform_info['platform_machine']}'")
    
    try:
        # Evaluate the expression (safe because we control the inputs)
        return eval(marker_eval, {"__builtins__": {}})
    except Exception:
        # If evaluation fails, assume marker doesn't match
        return False


def get_installed_version(package_name: str) -> Optional[str]:
    """Get installed version of a package.
    
    Args:
        package_name: Package name (e.g., "jax", "numpy")
    
    Returns:
        Installed version string or None if not installed
    """
    # Handle package name variations
    package_map = {
        "jaxtyping": "jaxtyping",
        "ott-jax": "ott_jax",
        "pywavelets": "PyWavelets",
    }
    
    lookup_name = package_map.get(package_name.lower(), package_name)
    
    try:
        return importlib.metadata.version(lookup_name)
    except importlib.metadata.PackageNotFoundError:
        # Try the original name if mapped name fails
        if lookup_name != package_name:
            try:
                return importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                return None
        return None


def load_requirements(root: Path, platform_info: Dict[str, str], req_file: str = "requirements.txt") -> Dict[str, str]:
    """Load requirements.txt and filter for current platform (recursive for -r includes).
    
    Args:
        root: Project root directory
        platform_info: Current platform information
        req_file: Requirements file path (relative to root or absolute)
    
    Returns:
        Dict mapping package names to expected versions
    """
    # Resolve path relative to root
    if Path(req_file).is_absolute():
        requirements_path = Path(req_file)
    else:
        requirements_path = root / req_file
    
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found at {requirements_path}")
    
    requirements = {}
    
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            
            # Handle -r includes (recursive)
            if line_stripped.startswith("-r"):
                included_file = line_stripped[2:].strip()
                # Resolve included file relative to current file's directory
                included_path = requirements_path.parent / included_file
                # Recursively load included requirements
                included_reqs = load_requirements(root, platform_info, str(included_path))
                # Merge (later entries override earlier ones if duplicates exist)
                requirements.update(included_reqs)
                continue
            
            # Parse normal requirement line
            result = parse_requirement_line(line, platform_info)
            if result:
                package_name, version = result
                requirements[package_name] = version
    
    return requirements


def check_dependencies(root: Path) -> Dict[str, Any]:
    """Check all dependencies against requirements.txt.
    
    Args:
        root: Project root directory
    
    Returns:
        Dict with check results
    """
    platform_info = get_platform_info()
    
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform_info,
        "total_packages": 0,
        "passed": 0,
        "failed": 0,
        "missing": 0,
        "packages": [],
    }
    
    try:
        requirements = load_requirements(root, platform_info)
        results["total_packages"] = len(requirements)
        
        for package_name, expected_version in sorted(requirements.items()):
            installed_version = get_installed_version(package_name)
            
            if installed_version is None:
                status = "MISSING"
                results["missing"] += 1
            elif installed_version == expected_version:
                status = "PASS"
                results["passed"] += 1
            else:
                status = "FAIL"
                results["failed"] += 1
            
            package_result = {
                "package": package_name,
                "expected": expected_version,
                "installed": installed_version or "NOT INSTALLED",
                "status": status,
            }
            
            results["packages"].append(package_result)
            
            # Print to console
            if status == "PASS":
                print(f"✓ {package_name:20s} {expected_version:10s} [OK]")
            elif status == "MISSING":
                print(f"✗ {package_name:20s} {expected_version:10s} [MISSING]")
            else:
                print(f"✗ {package_name:20s} expected: {expected_version}, installed: {installed_version} [MISMATCH]")
    
    except Exception as e:
        results["error"] = str(e)
    
    return results


def save_json_report(results: Dict, output_path: Path) -> None:
    """Save results to JSON file.
    
    Args:
        results: Check results
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_markdown_report(results: Dict, output_path: Path) -> None:
    """Save results to Markdown file.
    
    Args:
        results: Check results
        output_path: Path to output Markdown file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 🔧 Dependency Version Check Report\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write(f"**Platform:** {results['platform']['sys_platform']} ({results['platform']['platform_machine']})\n\n")
        
        # Executive Summary
        f.write("## 📊 Executive Summary\n\n")
        status_icon = "✅" if (results["failed"] == 0 and results["missing"] == 0) else "❌"
        status_text = "PASS" if (results["failed"] == 0 and results["missing"] == 0) else "FAIL"
        success_rate = (results['passed'] / results['total_packages'] * 100) if results['total_packages'] > 0 else 0
        
        f.write(f"{status_icon} **Overall Status:** {status_text}\n\n")
        f.write("| Metric | Value |\n")
        f.write("| --- | --- |\n")
        f.write(f"| Total Packages | {results['total_packages']} |\n")
        f.write(f"| Passed | {results['passed']} ({success_rate:.1f}%) |\n")
        f.write(f"| Failed | {results['failed']} |\n")
        f.write(f"| Missing | {results['missing']} |\n")
        f.write("\n---\n\n")
        
        f.write("## Package Details\n\n")
        f.write("| Package | Expected | Installed | Status |\n")
        f.write("| --- | --- | --- | --- |\n")
        
        for pkg in results["packages"]:
            status_icon = "✅" if pkg["status"] == "PASS" else "❌"
            f.write(f"| {pkg['package']} | {pkg['expected']} | {pkg['installed']} | {status_icon} {pkg['status']} |\n")
        
        # Final Summary
        f.write("\n---\n\n")
        f.write("## 🎯 Final Summary\n\n")
        if results["failed"] == 0 and results["missing"] == 0:
            f.write("✅ **All dependencies match Golden Master specification!**\n\n")
        else:
            f.write(f"❌ **{results['failed'] + results['missing']} package(s) do not match requirements.txt**\n\n")
            f.write("**Recommended Actions:**\n\n")
            f.write("1. Run: `pip install -r requirements.txt`\n")
            f.write("2. Verify installed versions match expected versions\n")
            f.write("3. Re-run dependency check to confirm\n\n")
        
        f.write(f"**Report generated at:** {timestamp}\n")


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code: 0 if all dependencies match, 1 otherwise
    """
    root = get_root()
    results_dir = root / "Test" / "results"
    reports_dir = root / "Test" / "reports"
    
    print("=" * 80)
    print("DEPENDENCY VERSION CHECK - Golden Master Validation")
    print("=" * 80)
    print(f"Platform: {sys.platform} ({platform.machine()})")
    print(f"Requirements: {root / 'requirements.txt'}")
    print("=" * 80)
    print()
    
    results = check_dependencies(root)
    
    # Save reports
    save_json_report(results, results_dir / "dependency_check_last.json")
    save_markdown_report(results, reports_dir / "dependency_check_last.md")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Packages:  {results['total_packages']}")
    print(f"Passed:          {results['passed']}")
    print(f"Failed:          {results['failed']}")
    print(f"Missing:         {results['missing']}")
    print("=" * 80)
    
    if results["failed"] > 0 or results["missing"] > 0:
        print("\n❌ DEPENDENCY CHECK FAILED")
        print("\nInstalled versions do not match requirements.txt.")
        print("Please run: pip install -r requirements.txt")
        return 1
    else:
        print("\n✅ DEPENDENCY CHECK PASSED")
        print("\nAll dependencies match Golden Master specification.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
