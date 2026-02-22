#!/usr/bin/env python3
"""Run unified code quality checks (black, isort, flake8, mypy) and generate JSON report.

Scope: Python/ and Test/ directories.
Output: Test/results/code_lint_last.json

Classification (3-level):
- BLOCKING: Code won't execute (E999 syntax, F821 undefined, F823 __all__)
- ERROR: Code executes but has bugs (F8xx unused/redefined, E4xx imports, mypy)
- WARNING: Style/formatting (black/isort, W-class whitespace)

Usage:
    python code_lint.py           # Check-only mode
    python code_lint.py --autofix # Auto-fix formatting before checking
    python code_lint.py --block-on-lint-errors --block-on-type-errors  # Recommended CI/CD
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from framework.discovery import get_project_root
from framework.reports import render_report

FRAMEWORK_VERSION = "2.1.0"


def _collect_files(roots: list[Path]) -> list[str]:
    """Collect all Python files from given root directories."""
    files: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(str(path.relative_to(get_project_root())))
    return sorted(set(files))


def _run_black_check(target: Path, root: Path) -> tuple[int, list[str]]:
    """Run black --check and return (exit_code, files_needing_format)."""
    cmd = [sys.executable, "-m", "black", "--check", "--quiet", str(target)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    files = []
    for line in result.stdout.split("\n"):
        if "would reformat" in line:
            # Extract filename from "would reformat <filename>"
            parts = line.strip().split()
            if len(parts) >= 3:
                files.append(parts[2])
    return result.returncode, files


def _run_black_fix(target: Path, root: Path) -> int:
    """Run black to auto-format code."""
    cmd = [sys.executable, "-m", "black", "--quiet", str(target)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    return result.returncode


def _run_isort_check(target: Path, root: Path) -> tuple[int, list[str]]:
    """Run isort --check-only and return (exit_code, files_needing_sort)."""
    cmd = [sys.executable, "-m", "isort", "--check-only", "--quiet", str(target)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    files = []
    for line in result.stderr.split("\n"):
        if "would be reformatted" in line.lower():
            files.append(line.strip())
    return result.returncode, files


def _run_isort_fix(target: Path, root: Path) -> int:
    """Run isort to auto-sort imports."""
    cmd = [sys.executable, "-m", "isort", "--quiet", str(target)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    return result.returncode


def _run_flake8(target: Path, root: Path) -> list[dict]:
    """Run flake8 and return parsed issues."""
    cmd = [sys.executable, "-m", "flake8", str(target), "--format=default"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=root)

    issues = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip() or ":" not in line:
            continue
        parts = line.split(":", 3)
        if len(parts) >= 4:
            code_msg = parts[3].strip()
            code = code_msg.split()[0] if code_msg else "UNKNOWN"
            issues.append(
                {
                    "tool": "flake8",
                    "filename": parts[0].strip(),
                    "line": parts[1].strip(),
                    "column": parts[2].strip(),
                    "code": code,
                    "text": code_msg,
                }
            )
    return issues


def _run_mypy(target: Path, root: Path) -> list[dict]:
    """Run mypy and return parsed issues."""
    cmd = [sys.executable, "-m", "mypy", str(target), "--no-error-summary"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=root)

    issues = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip() or ":" not in line or "error:" not in line.lower():
            continue
        # Format: filename.py:line: error: message [code]
        parts = line.split(":", 3)
        if len(parts) >= 3:
            msg_part = parts[2].strip() if len(parts) > 2 else ""
            # Extract code from [code] if present
            code = "mypy"
            if "[" in msg_part and "]" in msg_part:
                code = msg_part[msg_part.rfind("[") + 1 : msg_part.rfind("]")]

            issues.append(
                {
                    "tool": "mypy",
                    "filename": parts[0].strip(),
                    "line": parts[1].strip(),
                    "column": "0",
                    "code": code,
                    "text": msg_part,
                }
            )
    return issues


def _classify_issue(issue: dict) -> str:
    """Classify an issue into BLOCKING, ERROR, or WARNING.

    Returns:
        "BLOCKING" - Code that won't execute (syntax errors, undefined names)
        "ERROR" - Logic errors but code executes (unused vars, type errors)
        "WARNING" - Style/formatting (whitespace, W-class)
    """
    tool = issue.get("tool", "")
    code = issue.get("code", "")
    filename = issue.get("filename", "")

    # ═══════════════════════════════════════════════════════════════
    # FLAKE8 CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════
    if tool == "flake8":
        # ───────────────────────────────────────────────────────────
        # BLOCKING: Code that won't execute
        # ───────────────────────────────────────────────────────────
        if code == "E999":  # SyntaxError
            return "BLOCKING"
        if code == "F821":  # Undefined name
            return "BLOCKING"
        if code == "F823":  # Undefined in __all__
            return "BLOCKING"

        # ───────────────────────────────────────────────────────────
        # WARNING: Style/formatting (all W-class)
        # ───────────────────────────────────────────────────────────
        if code.startswith("W"):
            return "WARNING"
        if code == "E202":  # whitespace before ']'
            return "WARNING"

        # ───────────────────────────────────────────────────────────
        # ERROR: Logic errors but code executes
        # ───────────────────────────────────────────────────────────
        # Exceptions: Expected issues in specific scripts
        if code == "F401" and "code_structure.py" in filename:  # Unused imports (introspection)
            return "WARNING"
        if code == "E402" and "code_structure.py" in filename:  # Late imports (dynamic)
            return "WARNING"
        if code == "F824" and "code_alignment.py" in filename:  # Unused global (technical debt)
            return "WARNING"

        # All other E/F codes are ERROR
        if code.startswith(("E", "F")):
            return "ERROR"

        # Unknown → ERROR (conservative)
        return "ERROR"

    # ═══════════════════════════════════════════════════════════════
    # MYPY CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════
    if tool == "mypy":
        # All mypy errors are ERROR (type hints don't affect runtime)
        # No BLOCKING category for mypy (Python ignores type hints)
        return "ERROR"

    # ═══════════════════════════════════════════════════════════════
    # UNKNOWN TOOL → ERROR (conservative)
    # ═══════════════════════════════════════════════════════════════
    return "ERROR"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--autofix",
        action="store_true",
        help="Auto-fix formatting issues (black/isort) before checking",
    )
    parser.add_argument(
        "--block-on-format-warnings",
        action="store_true",
        help="Treat formatting warnings (black/isort) as errors (strict CI/CD mode)",
    )
    parser.add_argument(
        "--block-on-format-errors",
        action="store_true",
        help="Treat formatting errors as blocking (currently no format errors exist)",
    )
    parser.add_argument(
        "--block-on-lint-warnings",
        action="store_true",
        help="Treat lint style warnings (flake8 W-class) as errors (strict CI/CD mode)",
    )
    parser.add_argument(
        "--block-on-lint-errors",
        action="store_true",
        help="Treat lint logic errors (flake8 E/F-class) as blocking (recommended for CI/CD)",
    )
    parser.add_argument(
        "--block-on-type-warnings",
        action="store_true",
        help="Treat type warnings (mypy) as errors (strict CI/CD mode)",
    )
    parser.add_argument(
        "--block-on-type-errors",
        action="store_true",
        help="Treat type errors (mypy) as blocking (recommended for type-safe CI/CD)",
    )
    args = parser.parse_args()

    root = get_project_root()
    results_dir = root / "Test" / "results"
    results_dir.mkdir(exist_ok=True)

    lint_dirs = [root / "Python", root / "Test"]
    timestamp = datetime.now(timezone.utc).isoformat()

    # Step 1: Auto-fix formatting if requested
    if args.autofix:
        print("🔧 Auto-fixing formatting issues...")
        for lint_dir in lint_dirs:
            if not lint_dir.exists():
                continue
            _run_black_fix(lint_dir, root)
            _run_isort_fix(lint_dir, root)
        print("✅ Auto-fix complete\n")

    # Step 2: Check formatting (black + isort)
    format_issues = []
    for lint_dir in lint_dirs:
        if not lint_dir.exists():
            continue

        # Black check
        exit_code, black_files = _run_black_check(lint_dir, root)
        for file in black_files:
            format_issues.append(f"{file}: needs black formatting")

        # Isort check
        exit_code, isort_files = _run_isort_check(lint_dir, root)
        for file in isort_files:
            format_issues.append(f"{file}: needs isort")

    # Step 3: Run flake8 + mypy
    all_issues = []
    for lint_dir in lint_dirs:
        if not lint_dir.exists():
            continue

        # Flake8
        flake8_issues = _run_flake8(lint_dir, root)
        all_issues.extend(flake8_issues)

        # Mypy
        mypy_issues = _run_mypy(lint_dir, root)
        all_issues.extend(mypy_issues)

    # Step 4: Classify issues into categories with tool prefixes
    blocking_issues = []  # E999, F821, F823 - code won't execute
    error_issues = []  # F8xx, E4xx, mypy - logic errors but code runs
    warning_issues = []  # black/isort, W-class, style issues

    # Format issues are always WARNING (with [format] prefix)
    for issue_text in format_issues:
        warning_issues.append(f"[format] {issue_text}")

    # Classify flake8/mypy issues (with [lint] or [type] prefix)
    for issue in all_issues:
        filename = issue.get("filename", "unknown")
        # Convert absolute path to relative path
        try:
            rel_path = Path(filename).relative_to(root)
            filename = str(rel_path)
        except (ValueError, TypeError):
            pass  # Keep absolute if can't convert

        formatted = f"{filename}:{issue.get('line', '?')} " f"{issue.get('code', 'UNKNOWN')} {issue.get('text', '')}"
        severity = _classify_issue(issue)
        tool = issue.get("tool", "")

        if severity == "BLOCKING":
            if tool == "flake8":
                blocking_issues.append(f"[lint] {formatted}")
            elif tool == "mypy":
                blocking_issues.append(f"[type] {formatted}")
            else:
                blocking_issues.append(f"[{tool}] {formatted}")
        elif severity == "ERROR":
            if tool == "flake8":
                error_issues.append(f"[lint] {formatted}")
            elif tool == "mypy":
                error_issues.append(f"[type] {formatted}")
            else:
                error_issues.append(f"[{tool}] {formatted}")
        elif severity == "WARNING":
            if tool == "flake8":
                warning_issues.append(f"[lint] {formatted}")
            elif tool == "mypy":
                warning_issues.append(f"[type] {formatted}")
            else:
                warning_issues.append(f"[{tool}] {formatted}")

    blocking_count = len(blocking_issues)
    error_count = len(error_issues)
    warning_count = len(warning_issues)
    total_count = blocking_count + error_count + warning_count

    # Step 5: Generate JSON report
    payload = {
        "metadata": {
            "report_id": "code_quality",
            "timestamp_utc": timestamp,
            "status": "FAIL" if (blocking_count > 0 or error_count > 0) else "PASS",
            "source": "Test/scripts/code_lint.py",
            "framework_version": FRAMEWORK_VERSION,
            "notes": "Unified code quality: black, isort, flake8, mypy (3-level: BLOCKING/ERROR/WARNING)",
            "blocking_count": blocking_count,
            "error_count": error_count,
            "warning_count": warning_count,
            "autofix_enabled": args.autofix,
            "block_on_format_warnings": args.block_on_format_warnings,
            "block_on_format_errors": args.block_on_format_errors,
            "block_on_lint_warnings": args.block_on_lint_warnings,
            "block_on_lint_errors": args.block_on_lint_errors,
            "block_on_type_warnings": args.block_on_type_warnings,
            "block_on_type_errors": args.block_on_type_errors,
        },
        "summary": {
            "title": "Execution Summary",
            "metrics": [
                {"label": "Blocking Issues", "value": blocking_count},
                {"label": "Error Issues", "value": error_count},
                {"label": "Warning Issues", "value": warning_count},
                {"label": "Total Issues", "value": total_count},
                {"label": "Files Scanned", "value": len(_collect_files(lint_dirs))},
            ],
        },
        "scope": {
            "targets": {
                "folders": [d.name for d in lint_dirs if d.exists()],
                "files": _collect_files(lint_dirs),
                "modules": [],
                "functions": [],
                "classes": [],
            }
        },
        "details": {
            "type": "breakdown",
            "tools": ["black", "isort", "flake8", "mypy"],
            "classification": "BLOCKING (won't execute) / ERROR (executes but buggy) / WARNING (style)",
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
        "extras": [],
    }

    json_file = results_dir / "code_lint_last.json"
    json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Code quality report: {json_file.name}")

    # Generate Markdown report
    reports_dir = root / "Test" / "reports"
    reports_dir.mkdir(exist_ok=True)
    md_file = reports_dir / "CODE_LINT_LAST.md"
    md_content = render_report(payload)
    md_file.write_text(md_content, encoding="utf-8")
    print(f"Code quality report MD: {md_file.name}")

    # ═══════════════════════════════════════════════════════════════
    # EXIT CODE LOGIC (3-level classification)
    # ═══════════════════════════════════════════════════════════════
    # BLOCKING issues always fail (code won't execute)
    if blocking_count > 0:
        return 1

    # ERROR issues fail based on tool-specific flags
    # Check error_issues for [format], [lint], [type] prefixes
    has_format_errors = any(e.startswith("[format]") for e in error_issues)
    has_lint_errors = any(e.startswith("[lint]") for e in error_issues)
    has_type_errors = any(e.startswith("[type]") for e in error_issues)

    if has_format_errors and args.block_on_format_errors:
        return 1
    if has_lint_errors and args.block_on_lint_errors:
        return 1
    if has_type_errors and args.block_on_type_errors:
        return 1

    # WARNING issues fail based on tool-specific flags
    # Check warning_issues for [format], [lint], [type] prefixes
    has_format_warnings = any(w.startswith("[format]") for w in warning_issues)
    has_lint_warnings = any(w.startswith("[lint]") for w in warning_issues)
    has_type_warnings = any(w.startswith("[type]") for w in warning_issues)

    if has_format_warnings and args.block_on_format_warnings:
        return 1
    if has_lint_warnings and args.block_on_lint_warnings:
        return 1
    if has_type_warnings and args.block_on_type_warnings:
        return 1

    # All checks passed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
