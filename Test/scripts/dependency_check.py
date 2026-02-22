#!/usr/bin/env python3
"""Analyze dependencies and generate JSON report.

Scope: Python/ and Test/ imports vs requirements.
Output: Test/results/dependency_check_last.json
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from framework.discovery import get_project_root
from framework.reports import render_report

FRAMEWORK_VERSION = "2.1.0"


def parse_requirements(req_file: Path, seen: set[Path]) -> set[str]:
    """Parse requirements.txt and return set of package names."""
    if req_file in seen or not req_file.exists():
        return set()
    seen.add(req_file)
    requirements = set()
    base_dir = req_file.parent
    for raw_line in req_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r"):
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                nested_path = (base_dir / parts[1]).resolve()
                requirements |= parse_requirements(nested_path, seen)
            continue
        if line.startswith("-e") or line.startswith("--"):
            continue
        if "@" in line:
            line = line.split("@", 1)[0].strip()
        line = line.split(";", 1)[0].strip()
        line = line.split("[", 1)[0].strip()
        name = re.split(r"[<>=!~]", line, maxsplit=1)[0].strip()
        if name:
            requirements.add(name)
    return requirements


def parse_requirements_with_versions(req_file: Path, seen: set[Path]) -> dict[str, str]:
    """Parse requirements.txt and return dict of {package_name: expected_version}."""
    if req_file in seen or not req_file.exists():
        return {}
    seen.add(req_file)
    requirements: dict[str, str] = {}
    base_dir = req_file.parent
    for raw_line in req_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r"):
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                nested_path = (base_dir / parts[1]).resolve()
                requirements.update(parse_requirements_with_versions(nested_path, seen))
            continue
        if line.startswith("-e") or line.startswith("--"):
            continue
        if "@" in line:
            line = line.split("@", 1)[0].strip()
        line = line.split(";", 1)[0].strip()
        line = line.split("[", 1)[0].strip()

        # Extract version if specified with ==
        if "==" in line:
            name, version = line.split("==", 1)
            name = name.strip()
            version = version.strip()
            if name:
                requirements[name] = version
        else:
            # No exact version pin, skip version check
            name = re.split(r"[<>=!~]", line, maxsplit=1)[0].strip()
            if name:
                requirements[name] = ""  # Empty string means no version check
    return requirements
    seen.add(req_file)
    requirements = {}
    base_dir = req_file.parent
    for raw_line in req_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r"):
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                nested_path = (base_dir / parts[1]).resolve()
                requirements.update(parse_requirements_with_versions(nested_path, seen))
            continue
        if line.startswith("-e") or line.startswith("--"):
            continue
        if "@" in line:
            continue
        # Remove platform markers
        line = line.split(";", 1)[0].strip()
        # Remove extras
        line = line.split("[", 1)[0].strip()
        # Parse name and version
        match = re.match(r"^([a-zA-Z0-9_-]+)([<>=!~]+.*)$", line)
        if match:
            name = match.group(1).strip()
            version_spec = match.group(2).strip()
            requirements[name.lower()] = version_spec
    return requirements


def collect_imports(root_dir: Path) -> set[str]:
    imports = set()
    for py_file in root_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                if not node.module:
                    continue
                imports.add(node.module.split(".")[0])
    return imports


def normalize_set(items: set[str]) -> set[str]:
    return {item.strip().lower() for item in items if item.strip()}


def resolve_imports(
    modules: set[str],
    mapping: Mapping[str, list[str]],
) -> tuple[set[str], set[str], dict[str, str]]:
    resolved = set()
    unresolved = set()
    module_to_dist = {}
    for module in modules:
        dists = mapping.get(module)
        if not dists:
            unresolved.add(module)
            continue
        dist_name = dists[0]
        module_to_dist[module] = dist_name
        resolved.add(dist_name)
    return resolved, unresolved, module_to_dist


def main(block_on_warnings: bool = False) -> int:
    """Run dependency check.

    Args:
        block_on_warnings: If True, warnings are treated as errors (strict mode)

    Returns:
        0 if pass, 1 if critical issues or (warnings and block_on_warnings=True)
    """
    root = get_project_root()
    results_dir = root / "Test" / "results"
    results_dir.mkdir(exist_ok=True)

    dep_data: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files": {},
        "summary": {"total_packages": 0, "layers": []},
        "checks": {},
    }

    req_files = [
        (root / "Python" / "requirements.txt", "Production"),
        (root / "Test" / "requirements.txt", "Testing"),
    ]

    local_modules = {"Test", "Python"}
    for base_dir in [root / "Python", root / "Test"]:
        if not base_dir.exists():
            continue
        for entry in base_dir.iterdir():
            if entry.is_dir() and (entry / "__init__.py").exists():
                local_modules.add(entry.name)

    stdlib_modules: set[str] = getattr(sys, "stdlib_module_names", set())
    packages_map = importlib.metadata.packages_distributions()

    python_imports = collect_imports(root / "Python")
    test_imports = collect_imports(root / "Test")

    python_imports = {mod for mod in python_imports if mod not in local_modules and mod not in stdlib_modules}
    test_imports = {mod for mod in test_imports if mod not in local_modules and mod not in stdlib_modules}

    python_dists, python_unresolved, _ = resolve_imports(python_imports, packages_map)
    test_dists, test_unresolved, _ = resolve_imports(test_imports, packages_map)

    installed_dists = {
        dist.metadata["Name"].strip() if "Name" in dist.metadata else "" for dist in importlib.metadata.distributions()
    }

    # Get installed versions
    installed_versions: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"].strip() if "Name" in dist.metadata else ""
        version = dist.metadata["Version"].strip() if "Version" in dist.metadata else ""
        if name:
            installed_versions[name.lower()] = version

    all_packages = set()
    layer_data_map: dict[str, list[str]] = {}

    for req_file, layer_name in req_files:
        if not req_file.exists():
            continue
        packages = parse_requirements(req_file, seen=set())
        for pkg in packages:
            all_packages.add(pkg)
        layer_data_map[layer_name] = sorted(list(packages))
        packages_list = sorted(list(packages))[:10]
        dep_data["files"][layer_name] = {
            "file": str(req_file.relative_to(root)),
            "packages": len(packages),
            "packages_list": packages_list,
        }
        dep_data["summary"]["layers"].append({"name": layer_name, "count": len(packages)})

    dep_data["summary"]["total_packages"] = len(all_packages)

    prod_requirements: list[str] = layer_data_map.get("Production") or []
    test_requirements: list[str] = layer_data_map.get("Testing") or []

    prod_requirements_norm = normalize_set(set(prod_requirements))
    test_requirements_norm = normalize_set(set(test_requirements))
    installed_norm = normalize_set(installed_dists)

    python_dists_norm = normalize_set(python_dists)
    test_dists_norm = normalize_set(test_dists)
    used_in_tests_norm = python_dists_norm | test_dists_norm

    missing_in_prod = sorted(python_dists_norm - prod_requirements_norm)
    missing_in_test = sorted(used_in_tests_norm - test_requirements_norm)
    extra_in_prod = sorted(prod_requirements_norm - python_dists_norm)
    extra_in_test = sorted(test_requirements_norm - used_in_tests_norm)
    not_installed_prod = sorted(prod_requirements_norm - installed_norm)
    not_installed_test = sorted(test_requirements_norm - installed_norm)

    # Validate installed versions match requirements (Golden Master check)
    prod_req_versions = parse_requirements_with_versions(root / "Python" / "requirements.txt", seen=set())
    test_req_versions = parse_requirements_with_versions(root / "Test" / "requirements.txt", seen=set())

    version_mismatches = []
    for pkg_name, req_version_spec in {**prod_req_versions, **test_req_versions}.items():
        installed_version = installed_versions.get(pkg_name)
        if installed_version:
            # Check if it's a pinned version (==)
            if req_version_spec.startswith("=="):
                expected_version = req_version_spec[2:].strip()
                if installed_version != expected_version:
                    version_mismatches.append(f"{pkg_name}: expected {expected_version}, found {installed_version}")

    dep_data["checks"] = {
        "missing_in_production_requirements": missing_in_prod,
        "missing_in_testing_requirements": missing_in_test,
        "extra_in_production_requirements": extra_in_prod,
        "extra_in_testing_requirements": extra_in_test,
        "not_installed_production": not_installed_prod,
        "not_installed_testing": not_installed_test,
        "unresolved_production_imports": sorted(python_unresolved),
        "unresolved_testing_imports": sorted(test_unresolved),
        "version_mismatches": version_mismatches,
    }

    # Classify issues into 3 levels with [dependency] prefix
    # BLOCKING: Imports not resolved, missing critical dependencies (code won't execute)
    # ERROR: Version mismatches, packages in requirements but not installed
    # WARNING: Extra packages listed but not used

    blocking_checks = [
        "missing_in_production_requirements",
        "missing_in_testing_requirements",
        "unresolved_production_imports",
        "unresolved_testing_imports",
    ]

    error_checks = [
        "version_mismatches",
        "not_installed_production",
        "not_installed_testing",
    ]

    warning_checks = [
        "extra_in_production_requirements",
        "extra_in_testing_requirements",
    ]

    blocking_issues = []
    error_issues = []
    warning_issues = []

    for name, items in dep_data["checks"].items():
        if isinstance(items, list) and items:
            prefix = f"[dependency] {name}"
            if name in blocking_checks:
                blocking_issues.append(f"{prefix}: {len(items)}")
            elif name in error_checks:
                error_issues.append(f"{prefix}: {len(items)}")
            elif name in warning_checks:
                warning_issues.append(f"{prefix}: {len(items)}")

    # Status: FAIL if blocking or errors, PASS otherwise
    has_blocking = len(blocking_issues) > 0
    has_errors = len(error_issues) > 0
    status = "FAIL" if (has_blocking or has_errors) else "PASS"

    details_rows = []
    for layer in dep_data["summary"].get("layers", []):
        details_rows.append([layer.get("name", "Layer"), str(layer.get("count", 0))])

    extras = []
    for name, items in dep_data["checks"].items():
        if not isinstance(items, list):
            continue
        extras.append(
            {
                "id": name,
                "title": name.replace("_", " ").title(),
                "type": "list",
                "items": items if items else ["No findings"],
            }
        )

    payload = {
        "metadata": {
            "report_id": "dependency_check",
            "timestamp_utc": dep_data.get("timestamp", ""),
            "status": status,
            "source": "Test/scripts/dependency_check.py",
            "framework_version": FRAMEWORK_VERSION,
            "notes": "Requirements vs imports and installed packages (3-level: BLOCKING/ERROR/WARNING)",
            "blocking_count": len(blocking_issues),
            "error_count": len(error_issues),
            "warning_count": len(warning_issues),
        },
        "summary": {
            "title": "Execution Summary",
            "metrics": [
                {"label": "Total Unique Packages", "value": dep_data["summary"].get("total_packages", 0)},
                {"label": "Blocking Issues", "value": len(blocking_issues)},
                {"label": "Error Issues", "value": len(error_issues)},
                {"label": "Warning Issues", "value": len(warning_issues)},
                {"label": "Missing in Production", "value": len(missing_in_prod)},
                {"label": "Missing in Testing", "value": len(missing_in_test)},
                {"label": "Version Mismatches", "value": len(version_mismatches)},
                {"label": "Unresolved Imports", "value": len(python_unresolved) + len(test_unresolved)},
            ],
        },
        "scope": {
            "targets": {
                "folders": ["Python", "Test"],
                "files": [str(path.relative_to(root)) for path, _ in req_files if path.exists()],
                "modules": sorted(python_imports | test_imports),
                "functions": [],
                "classes": [],
            }
        },
        "details": {
            "type": "table",
            "columns": ["Layer", "Package Count"],
            "rows": details_rows,
        },
        "issues_blocking": {
            "type": "list",
            "items": blocking_issues if blocking_issues else ["No blocking issues"],
        },
        "issues_errors": {
            "type": "list",
            "items": error_issues if error_issues else ["No error issues"],
        },
        "issues_warnings": {
            "type": "list",
            "items": warning_issues if warning_issues else ["No warnings"],
        },
        "extras": extras,
    }

    json_file = results_dir / "dependency_check_last.json"
    json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Dependency report JSON: {json_file.name}")

    # Generate Markdown report
    reports_dir = root / "Test" / "reports"
    reports_dir.mkdir(exist_ok=True)
    md_file = reports_dir / "DEPENDENCY_CHECK_LAST.md"
    md_content = render_report(payload)
    md_file.write_text(md_content, encoding="utf-8")
    print(f"Dependency report MD: {md_file.name}")

    # Determine exit code (3-level classification)
    # BLOCKING issues always fail
    if has_blocking:
        return 1

    # ERROR issues fail (version mismatches, not installed packages are serious)
    if has_errors:
        return 1

    # WARNINGS fail only in strict mode
    if len(warning_issues) > 0 and block_on_warnings:
        return 1

    # All checks passed
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block-on-warnings",
        action="store_true",
        help="Treat warnings as errors (strict mode for CI/CD)",
    )
    args = parser.parse_args()
    raise SystemExit(main(block_on_warnings=args.block_on_warnings))
