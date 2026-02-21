#!/usr/bin/env python3
"""Code linting and style enforcement.

Linting Scope: Python/ directory (all modules: api, core, kernels, io)

Tools:
- flake8: Style violations, complexity
- mypy: Static type checking
- isort: Import organization
- black: Code formatting (check mode)

Outputs:
- Console summary (PASS/FAIL per linter)
- JSON report: tests/results/code_lint_last.json
- Markdown report: tests/reports/code_lint_last.md

EXIT CODES:
- 0: All linters passed
- 1: One or more linters failed
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from scope_discovery import discover_modules, get_root
except ImportError:
    def _discover_modules_fallback(root: str | Path | None = None) -> List[str]:
        if root is None:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        root = Path(root) if isinstance(root, str) else root
        python_dir = root / "Python"
        if not python_dir.is_dir():
            return []
        return sorted([d.name for d in python_dir.iterdir() 
                      if d.is_dir() and (d / "__init__.py").is_file()])
    
    def _get_root_fallback() -> Path:
        return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    
    discover_modules = _discover_modules_fallback
    get_root = _get_root_fallback


ROOT = str(get_root())
RESULTS_DIR = os.path.join(ROOT, "tests", "results")
REPORTS_DIR = os.path.join(ROOT, "tests", "reports")
PYTHON_DIR = os.path.join(ROOT, "Python")
VENV_BIN = os.path.join(ROOT, ".venv", "bin")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


@dataclass(frozen=True)
class LinterResult:
    """Single linter check result."""
    linter: str
    passed: bool
    violations: int
    details: str
    output: str = ""


def run_flake8() -> LinterResult:
    """Run flake8 for PEP8 style violations."""
    flake8_path = os.path.join(VENV_BIN, "flake8")
    try:
        result = subprocess.run(
            [
                flake8_path,
                PYTHON_DIR,
                "--max-line-length=120",  # Increased for black compatibility
                "--show-source",
                "--ignore=F722,F821,E203,W503",  # F722/F821: jaxtyping syntax, E203/W503: black compat
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Count violations (lines of output)
        output = result.stdout
        violations = len([l for l in output.split('\n') if l.strip()])
        
        if result.returncode == 0:
            return LinterResult(
                linter="flake8",
                passed=True,
                violations=0,
                details="No style violations found",
                output=output
            )
        else:
            return LinterResult(
                linter="flake8",
                passed=False,
                violations=violations,
                details=f"Found {violations} style violations",
                output=output
            )
    except FileNotFoundError:
        return LinterResult(
            linter="flake8",
            passed=False,
            violations=0,
            details="flake8 not installed",
            output="ERROR: flake8 not found"
        )
    except subprocess.TimeoutExpired:
        return LinterResult(
            linter="flake8",
            passed=False,
            violations=0,
            details="flake8 timed out",
            output="ERROR: Timeout"
        )
    except Exception as e:
        return LinterResult(
            linter="flake8",
            passed=False,
            violations=0,
            details=f"flake8 error: {str(e)}",
            output=str(e)
        )


def run_mypy() -> LinterResult:
    """Run mypy for static type checking."""
    mypy_path = os.path.join(VENV_BIN, "mypy")
    try:
        result = subprocess.run(
            [mypy_path, PYTHON_DIR, "--ignore-missing-imports", "--follow-imports=silent"],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        output = result.stdout + result.stderr
        
        # Count type errors (lines with error keyword)
        violations = len([l for l in output.split('\n') 
                         if 'error:' in l.lower()])
        
        if result.returncode == 0:
            return LinterResult(
                linter="mypy",
                passed=True,
                violations=0,
                details="No type checking errors",
                output=output[:500]  # Truncate long output
            )
        else:
            return LinterResult(
                linter="mypy",
                passed=False,
                violations=violations,
                details=f"Found {violations} type errors",
                output=output[:500]
            )
    except FileNotFoundError:
        return LinterResult(
            linter="mypy",
            passed=False,
            violations=0,
            details="mypy not installed",
            output="ERROR: mypy not found"
        )
    except subprocess.TimeoutExpired:
        return LinterResult(
            linter="mypy",
            passed=False,
            violations=0,
            details="mypy timed out",
            output="ERROR: Timeout"
        )
    except Exception as e:
        return LinterResult(
            linter="mypy",
            passed=False,
            violations=0,
            details=f"mypy error: {str(e)}",
            output=str(e)
        )


def run_isort() -> LinterResult:
    """Run isort to check import organization."""
    isort_path = os.path.join(VENV_BIN, "isort")
    try:
        result = subprocess.run(
            [isort_path, PYTHON_DIR, "--check-only", "--diff", "--profile", "black", "--line-length", "120"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout + result.stderr
        
        if result.returncode == 0:
            return LinterResult(
                linter="isort",
                passed=True,
                violations=0,
                details="Import organization is correct",
                output=output
            )
        else:
            # Count files with issues
            violations = len([l for l in output.split('\n') if l.startswith('---')])
            return LinterResult(
                linter="isort",
                passed=False,
                violations=violations,
                details=f"Import organization issues in {violations} file(s)",
                output=output[:500]
            )
    except FileNotFoundError:
        return LinterResult(
            linter="isort",
            passed=False,
            violations=0,
            details="isort not installed",
            output="ERROR: isort not found"
        )
    except subprocess.TimeoutExpired:
        return LinterResult(
            linter="isort",
            passed=False,
            violations=0,
            details="isort timed out",
            output="ERROR: Timeout"
        )
    except Exception as e:
        return LinterResult(
            linter="isort",
            passed=False,
            violations=0,
            details=f"isort error: {str(e)}",
            output=str(e)
        )


def run_black() -> LinterResult:
    """Run black to check code formatting."""
    black_path = os.path.join(VENV_BIN, "black")
    try:
        result = subprocess.run(
            [black_path, PYTHON_DIR, "--check", "--line-length=120"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout + result.stderr
        
        if result.returncode == 0:
            return LinterResult(
                linter="black",
                passed=True,
                violations=0,
                details="Code formatting is correct",
                output=output
            )
        else:
            # Count files that need reformatting
            violations = len([l for l in output.split('\n') 
                            if 'would be reformatted' in l or 'would reformat' in l])
            return LinterResult(
                linter="black",
                passed=False,
                violations=violations,
                details=f"Code formatting issues in {violations} file(s)",
                output=output[:500]
            )
    except FileNotFoundError:
        return LinterResult(
            linter="black",
            passed=False,
            violations=0,
            details="black not installed",
            output="ERROR: black not found"
        )
    except subprocess.TimeoutExpired:
        return LinterResult(
            linter="black",
            passed=False,
            violations=0,
            details="black timed out",
            output="ERROR: Timeout"
        )
    except Exception as e:
        return LinterResult(
            linter="black",
            passed=False,
            violations=0,
            details=f"black error: {str(e)}",
            output=str(e)
        )


def run_all_linters() -> List[LinterResult]:
    """Run all configured linters."""
    return [
        run_flake8(),
        run_mypy(),
        run_isort(),
        run_black(),
    ]


def generate_json_report(results: List[LinterResult], filepath: str) -> None:
    """Generate JSON report."""
    summary = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_linters": len(results),
        "passed": len([r for r in results if r.passed]),
        "failed": len([r for r in results if not r.passed]),
        "linters": [asdict(r) for r in results]
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def generate_markdown_report(results: List[LinterResult], filepath: str) -> None:
    """Generate Markdown report."""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    total = len(results)
    passed = len([r for r in results if r.passed])
    failed = len([r for r in results if not r.passed])
    success_rate = (passed / total * 100) if total > 0 else 0
    status_icon = "✅" if failed == 0 else "❌"
    status_text = "PASS" if failed == 0 else "FAIL"
    
    md_lines = [
        "# 🔍 Code Linting Report",
        "",
        f"**Generated:** {timestamp}",
        "",
        f"**Scope:** {PYTHON_DIR}",
        "",
        "## 📊 Executive Summary",
        "",
        f"{status_icon} **Overall Status:** {status_text}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total Linters | {total} |",
        f"| Passed | {passed} ({success_rate:.1f}%) |",
        f"| Failed | {failed} ({100-success_rate:.1f}%) |",
        "",
        "---",
        "",
        "## 📝 Detailed Results",
        "",
    ]
    
    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        md_lines.append(f"### {result.linter} {status}")
        md_lines.append("")
        md_lines.append(f"**Status:** {result.details}")
        md_lines.append("")
        if result.violations > 0:
            md_lines.append(f"**Violations:** {result.violations}")
            md_lines.append("")
        if result.output:
            md_lines.append("```text")
            md_lines.append(result.output[:400])
            md_lines.append("```")
            md_lines.append("")
    
    # Final Summary
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 🎯 Final Summary")
    md_lines.append("")
    if failed == 0:
        md_lines.append(f"✅ **All {total} linters passed!** Code quality standards met.")
    else:
        md_lines.append(f"❌ **{failed} linter(s) failed out of {total}.**")
        md_lines.append("")
        md_lines.append("**Recommended Actions:**")
        md_lines.append("")
        failed_linters = [r.linter for r in results if not r.passed]
        for idx, linter in enumerate(failed_linters, 1):
            md_lines.append(f"{idx}. Fix {linter} violations")
        md_lines.append("")
    md_lines.append(f"**Report generated at:** {timestamp}")
    
    # Remove trailing blank line and add single newline at end
    while md_lines and md_lines[-1] == "":
        md_lines.pop()
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def print_console_summary(results: List[LinterResult]) -> None:
    """Print console summary."""
    print("\n" + "=" * 80)
    print("CODE LINTING SUMMARY")
    print("=" * 80)
    
    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status}: {result.linter} - {result.details}")
    
    total = len(results)
    passed = len([r for r in results if r.passed])
    failed = len([r for r in results if not r.passed])
    
    print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed}")
    
    json_report = os.path.join(RESULTS_DIR, "code_lint_last.json")
    md_report = os.path.join(REPORTS_DIR, "code_lint_last.md")
    print(f"JSON Report: {json_report}")
    print(f"Markdown Report: {md_report}")
    
    print("=" * 80 + "\n")


def main() -> int:
    """Main entry point."""
    print("Discovering Python modules...")
    modules = discover_modules(ROOT)
    print(f"Found modules: {', '.join(modules)}")
    print(f"Linting scope: {PYTHON_DIR}\n")
    
    print("Running linters...\n")
    results = run_all_linters()
    
    # Generate reports
    json_file = os.path.join(RESULTS_DIR, "code_lint_last.json")
    md_file = os.path.join(REPORTS_DIR, "code_lint_last.md")
    
    generate_json_report(results, json_file)
    generate_markdown_report(results, md_file)
    
    # Print console summary
    print_console_summary(results)
    
    # Return exit code
    failed_count = len([r for r in results if not r.passed])
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
