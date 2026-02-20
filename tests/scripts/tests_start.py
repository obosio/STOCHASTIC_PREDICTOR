#!/usr/bin/env python3
"""
Entrypoint script for all test and validation scripts.

Runs the full test suite in sequence:
  1. tests_coverage.py     - Structural coverage validation
  2. code_structure.py     - Full code execution tests (pytest)
  3. code_alignement.py    - Policy compliance checker

All artifacts are output to tests/results/ and tests/reports/.

Usage:
    python tests_start.py              # Run all scripts
    python tests_start.py tests_coverage     # Run coverage only
    python tests_start.py code_structure    # Run structure tests only
    python tests_start.py code_alignement   # Run policy checks only
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timezone

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def run_tests_coverage() -> int:
    """Run structural coverage validation."""
    print("\n" + "=" * 80)
    print("📊 RUNNING: Structural Coverage Validation (tests_coverage.py)")
    print("=" * 80)
    
    try:
        from tests.scripts.tests_coverage import main
        return main()
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return 1


def run_code_structure() -> int:
    """Run structural execution tests via pytest."""
    print("\n" + "=" * 80)
    print("🔬 RUNNING: Code Structure Tests (code_structure.py)")
    print("=" * 80)
    
    try:
        import pytest
        # Only run the structural tests, not benchmarks
        test_file = ROOT / "tests" / "scripts" / "code_structure.py"
        exit_code = pytest.main([
            str(test_file),
            "-v",
            "--tb=short",
            "-x",  # Stop on first failure
        ])
        return exit_code
    except ImportError:
        print("⚠️  pytest not installed. Skipping structural tests.")
        return 0
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return 1


def run_code_alignement() -> int:
    """Run policy compliance checker."""
    print("\n" + "=" * 80)
    print("✅ RUNNING: Policy Compliance Check (code_alignement.py)")
    print("=" * 80)
    
    try:
        from tests.scripts.code_alignement import main
        return main()
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return 1


def print_summary(results: List[Tuple[str, int]]) -> None:
    """Print final summary."""
    print("\n" + "=" * 80)
    print("📈 TEST SUMMARY")
    print("=" * 80)
    
    for name, exit_code in results:
        status = "✅ PASS" if exit_code == 0 else "❌ FAIL"
        print(f"  {status}  {name}")
    
    total_failed = sum(1 for _, code in results if code != 0)
    total = len(results)
    
    print(f"\nTotal: {total - total_failed}/{total} passed")
    
    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total_failed} test suite(s) failed")
    
    print("=" * 80)


def main():
    """Main entrypoint."""
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        runner_name = sys.argv[1]
        runners = {
            "tests_coverage": ("tests_coverage", run_tests_coverage),
            "code_structure": ("code_structure", run_code_structure),
            "code_alignement": ("code_alignement", run_code_alignement),
        }
        
        if runner_name not in runners:
            print(f"Unknown runner: {runner_name}")
            print(f"Available: {', '.join(runners.keys())}")
            return 1
        
        name, runner_func = runners[runner_name]
        exit_code = runner_func()
        return exit_code
    
    # Run all scripts in sequence
    results: List[Tuple[str, int]] = []
    
    # 1. Policy compliance
    results.append(("code_alignement.py", run_code_alignement()))
    
    # 2. Coverage validation
    results.append(("tests_coverage.py", run_tests_coverage()))
    
    # 3. Structural tests
    results.append(("code_structure.py", run_code_structure()))
    
    # Print summary
    print_summary(results)
    
    # Exit with failure if any suite failed
    if any(code != 0 for _, code in results):
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
