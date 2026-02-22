#!/usr/bin/env python3
"""
Test Framework Orchestrator - Python Entry Point

Replaces tests_start.sh with a more powerful Python-based orchestrator.

This script:
1. Regenerates auto-generated tests (optional)
2. Runs pytest with configured settings
3. Orchestrates report scripts (JSON format)
4. Provides feedback and exit codes

Usage:
    python Test/run_tests.py                    # Run all with defaults
    python Test/run_tests.py --help             # Show help
    python Test/run_tests.py --marker api       # Only API tests
    python Test/run_tests.py --dry-run          # Plan without executing
    python Test/run_tests.py --regenerate       # Force regenerate tests
    python Test/run_tests.py --coverage         # With coverage report

Environment:
    - Auto-discovers project root from config.toml
    - Loads configuration from test_config.yaml
    - Respects PYTEST_ARGS environment variable for extra args
"""

import json
import os
import subprocess
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class TestOrchestrator:
    """Orchestrates test execution with framework integration."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize orchestrator."""
        if project_root is None:
            # Try multiple strategies to find project root
            project_root = self._find_project_root()

        self.project_root = project_root
        self.test_dir = self.project_root / "Test"
        self.config = self._load_config()

    @staticmethod
    def _find_project_root() -> Path:
        """Find project root by looking for marker files."""
        # Start from script location: Test/run_tests.py
        script = Path(__file__).resolve()
        test_dir = script.parent

        # Project root is one level up from Test/
        candidate = test_dir.parent

        # Verify it looks like a project root (has Python/ and Test/ directories)
        if (candidate / "Python").exists() and (candidate / "Test").exists():
            return candidate

        # Fallback: search upwards for config.toml or other markers
        for parent in [test_dir] + list(test_dir.parents):
            if (parent / "config.toml").exists():
                return parent
            if (parent / "Python").exists() and (parent / "Test").exists():
                return parent

        raise FileNotFoundError(
            f"Could not find project root. Looked in: {test_dir} and parents. "
            "Expected Python/ and Test/ directories."
        )

    def _load_config(self) -> dict:
        """Load test_config.yaml if exists."""
        config_file = self.test_dir / "test_config.yaml"
        if config_file.exists():
            try:
                import yaml

                with open(config_file) as f:
                    return yaml.safe_load(f) or {}
            except ImportError:
                print("⚠️  PyYAML not installed, using defaults")
            except Exception as e:
                print(f"⚠️  Could not load {config_file}: {e}")
        return {}

    def _validate_config_parameter(self, path: str, expected_type: type | None = None) -> bool:
        """Validate that a required config parameter exists.

        Args:
            path: Dot-separated path (e.g., "quality_gates.autofix")
            expected_type: Optional type to validate against

        Returns:
            True if valid, False otherwise (prints error message)
        """
        parts = path.split(".")
        current = self.config

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                print(f"❌ Missing required parameter: {path}", file=sys.stderr)
                print(f"   Add '{parts[-1]}: <value>' to test_config.yaml", file=sys.stderr)
                return False
            current = current[part]

        if expected_type and not isinstance(current, expected_type):
            print(f"❌ Invalid type for parameter: {path}", file=sys.stderr)
            print(f"   Expected {expected_type.__name__}, got {type(current).__name__}", file=sys.stderr)
            return False

        return True

    def _run_script(self, script_name: str, args: Optional[List[str]] = None) -> int:
        script_path = self.test_dir / "scripts" / script_name
        if not script_path.exists():
            print(f"⚠️  Missing script: {script_path}")
            return 1
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)

        # Add Test/ to PYTHONPATH so scripts can import framework modules
        env = os.environ.copy()
        test_dir_str = str(self.test_dir)
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{test_dir_str}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = test_dir_str

        result = subprocess.run(cmd, cwd=self.project_root, env=env)
        return result.returncode

    def _load_report(self, filename: str) -> Optional[dict]:
        report_path = self.test_dir / "results" / filename
        if not report_path.exists():
            return None
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _clean_output_directories(self) -> None:
        """Clean Test/results and Test/reports directories before execution."""
        results_dir = self.test_dir / "results"
        reports_dir = self.test_dir / "reports"

        # Remove all files in results/
        if results_dir.exists():
            for file in results_dir.glob("*.json"):
                file.unlink()

        # Remove all files in reports/
        if reports_dir.exists():
            for file in reports_dir.glob("*.md"):
                file.unlink()

    def run_quality_gates(self, verbose: bool = False, autofix: bool = False) -> int:
        """Run quality gates with fail-fast.

        Args:
            verbose: Show detailed output
            autofix: Auto-fix formatting issues (black/isort) before linting

        Returns:
            0 if all gates pass
            1 if GATE 1 (dependencies) fails
            2 if GATE 2 (linting) fails
        """
        # Read configuration (already validated at startup)
        qg_config = self.config["quality_gates"]
        dep_block_warnings = qg_config["dependencies"]["block_on_warnings"]
        format_block_warnings = qg_config["format"]["block_on_warnings"]
        format_block_errors = qg_config["format"]["block_on_errors"]
        lint_block_warnings = qg_config["lint"]["block_on_warnings"]
        lint_block_errors = qg_config["lint"]["block_on_errors"]
        types_block_warnings = qg_config["types"]["block_on_warnings"]
        types_block_errors = qg_config["types"]["block_on_errors"]

        print("\n" + "=" * 50)
        print("🚦 RUNNING QUALITY GATES")
        print("=" * 50 + "\n")

        # GATE 1: Dependency Check (most fundamental)
        print("[GATE 1] Dependency Check...")
        dep_args = ["--block-on-warnings"] if dep_block_warnings else []
        exit_code = self._run_script("dependency_check.py", dep_args)
        if exit_code != 0:
            print("❌ GATE 1 FAILED: Missing dependencies or import errors")
            print("   Cannot proceed without valid dependencies.")
            return 1

        # Show warnings (non-blocking by default, unless strict mode)
        dep_report = self._load_report("dependency_check_last.json")
        if dep_report and dep_report.get("metadata", {}).get("warnings_count", 0) > 0:
            warnings = dep_report.get("warnings", {}).get("items", [])
            if warnings and warnings != ["No warnings"]:
                mode = "BLOCKING" if dep_block_warnings else "non-blocking"
                print(f"⚠️  GATE 1 WARNINGS ({mode}):")
                for warning in warnings[:5]:  # Show first 5
                    print(f"   - {warning}")
                if len(warnings) > 5:
                    print(f"   ... and {len(warnings) - 5} more warnings")
                print()

        print("✅ GATE 1 PASSED: All dependencies satisfied\n")

        # GATE 2: Code Quality Check (formatting + linting + types)
        print("[GATE 2] Code Quality Check...")
        if autofix:
            print("   🔧 Auto-fix mode enabled (black/isort)")

        lint_args = []
        if autofix:
            lint_args.append("--autofix")
        if format_block_warnings:
            lint_args.append("--block-on-format-warnings")
        if format_block_errors:
            lint_args.append("--block-on-format-errors")
        if lint_block_warnings:
            lint_args.append("--block-on-lint-warnings")
        if lint_block_errors:
            lint_args.append("--block-on-lint-errors")
        if types_block_warnings:
            lint_args.append("--block-on-type-warnings")
        if types_block_errors:
            lint_args.append("--block-on-type-errors")

        exit_code = self._run_script("code_lint.py", lint_args)
        if exit_code != 0:
            print("❌ GATE 2 FAILED: Code quality issues detected")
            if autofix:
                print("   Remaining issues cannot be auto-fixed - manual intervention required.")
            else:
                print("   Run with --autofix to auto-format, or fix issues manually.")
            return 2
        print("✅ GATE 2 PASSED: Code style compliant\n")

        print("=" * 50)
        print("✅ ALL QUALITY GATES PASSED")
        print("=" * 50 + "\n")
        return 0

    def regenerate_tests(self, verbose: bool = False):
        """Regenerate auto-generated tests."""
        print("🔧 Regenerating auto-generated tests...")

        try:
            from framework.generator import TestGenerator

            generator = TestGenerator(project_root=self.project_root, source_dir="Python")
            output_dir = self.test_dir / "tests" / "unit"
            generated_count = generator.generate_all_tests(output_dir)
            success = generated_count >= 0

            if not success:
                print("❌ Test regeneration failed")
                return False

            print("✅ Tests regenerated")
            return True
        except Exception as e:
            print(f"❌ Test regeneration error: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            return False

    def run_pytest(
        self,
        markers: Optional[List[str]] = None,
        coverage: bool = False,
        verbose: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> int:
        """Run pytest with configured settings."""

        print("▶️  Running pytest...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.test_dir / "tests"),
            "-v",
            "--tb=short",
        ]

        # Add markers
        if markers:
            marker_expr = " or ".join(markers)
            cmd.extend(["-m", marker_expr])
            print(f"   Marker filter: {marker_expr}")

        # Add coverage
        if coverage:
            cmd.extend(["--cov=Python", "--cov-report=term-missing", "--cov-report=html"])
            print("   Coverage: enabled")

        # Add extra pytest args
        if extra_args:
            cmd.extend(extra_args)

        # Add environment args
        if "PYTEST_ARGS" in os.environ:
            cmd.extend(os.environ["PYTEST_ARGS"].split())

        if verbose:
            print(f"   Command: {' '.join(cmd)}")
            print()

        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode

    def generate_summary_report(
        self,
        lint_data=None,
        dep_data=None,
        fixtures_data=None,
        struct_data=None,
        test_data=None,
    ):
        """Generate aggregated summary report consolidating all individual reports."""
        results_dir = self.test_dir / "results"
        results_dir.mkdir(exist_ok=True)

        lint_pass = bool(lint_data) and lint_data.get("metadata", {}).get("status") == "PASS"
        test_pass = bool(test_data) and test_data.get("metadata", {}).get("status") == "PASS"
        fixture_pass = bool(fixtures_data) and fixtures_data.get("metadata", {}).get("status") == "PASS"

        fixtures_status = fixtures_data.get("metadata", {}).get("status", "UNKNOWN") if fixtures_data else "UNKNOWN"

        if lint_pass and test_pass and fixture_pass:
            overall_status = "PASS"
        elif lint_pass:
            overall_status = "PARTIAL"
        else:
            overall_status = "NEEDS_REVIEW"

        issues = []
        if not lint_pass:
            issues.append("Lint issues detected")
        if not fixture_pass:
            issues.append("Fixture coverage issues detected")
        if not test_pass:
            issues.append("Test execution failures detected")

        payload = {
            "metadata": {
                "report_id": "summary",
                "timestamp_utc": datetime.now().isoformat(),
                "status": overall_status,
                "source": "Test/run_tests.py",
                "framework_version": "2.1.0",
                "notes": "Consolidated QA summary",
            },
            "summary": {
                "title": "Execution Summary",
                "metrics": [
                    {"label": "Overall Status", "value": overall_status},
                    {"label": "Lint Status", "value": "PASS" if lint_pass else "FAIL"},
                    {"label": "Fixtures Status", "value": fixtures_status},
                    {
                        "label": "Tests Status",
                        "value": test_data.get("metadata", {}).get("status", "UNKNOWN") if test_data else "UNKNOWN",
                    },
                ],
            },
            "scope": {
                "targets": {
                    "folders": ["Python", "Test"],
                    "files": [
                        str((self.test_dir / "results" / "code_lint_last.json").relative_to(self.project_root)),
                        str((self.test_dir / "results" / "dependency_check_last.json").relative_to(self.project_root)),
                        str((self.test_dir / "results" / "code_fixtures_last.json").relative_to(self.project_root)),
                        str((self.test_dir / "results" / "code_structure_last.json").relative_to(self.project_root)),
                        str((self.test_dir / "results" / "tests_generation_last.json").relative_to(self.project_root)),
                    ],
                    "modules": [],
                    "functions": [],
                    "classes": [],
                }
            },
            "details": {
                "type": "table",
                "columns": ["Check", "Status"],
                "rows": [
                    ["Lint", "PASS" if lint_pass else "FAIL"],
                    ["Fixtures", fixtures_status],
                    [
                        "Tests",
                        test_data.get("metadata", {}).get("status", "UNKNOWN") if test_data else "UNKNOWN",
                    ],
                ],
            },
            "issues": {
                "type": "list",
                "items": issues if issues else ["No findings"],
            },
            "extras": [],
        }

        summary_file = results_dir / "summary_last.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Summary report JSON: {summary_file.name}")
        return summary_file

    def generate_all_reports(self, test_results: int):
        """Generate all reports."""
        print("\n" + "=" * 50)
        print("📊 GENERATING ALL REPORTS")
        print("=" * 50 + "\n")

        script_calls = [
            ("code_lint.py", []),
            ("dependency_check.py", []),
            ("code_fixtures.py", []),
            ("code_structure.py", []),
            ("tests_generation.py", ["--exit-code", str(test_results)]),
        ]

        for script_name, args in script_calls:
            self._run_script(script_name, args)

        lint_data = self._load_report("code_lint_last.json")
        dep_data = self._load_report("dependency_check_last.json")
        fixtures_data = self._load_report("code_fixtures_last.json")
        struct_data = self._load_report("code_structure_last.json")
        test_data = self._load_report("tests_generation_last.json")

        self.generate_summary_report(
            lint_data=lint_data,
            dep_data=dep_data,
            fixtures_data=fixtures_data,
            struct_data=struct_data,
            test_data=test_data,
        )

        print("\n" + "=" * 50)
        print("ALL JSON REPORTS GENERATED")
        print("=" * 50)


def main():
    """Main entry point."""

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate auto-generated tests before running",
    )

    parser.add_argument(
        "--marker",
        "-m",
        action="append",
        dest="markers",
        help="Run only tests with this marker (can be repeated)",
    )

    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output (show all commands)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )

    parser.add_argument(
        "--no-autofix",
        action="store_true",
        help="Disable auto-fix of formatting issues (reads from test_config.yaml by default)",
    )

    parser.add_argument(
        "--keep-going",
        "-k",
        action="store_true",
        help="Don't stop on first failure (opposite of --ff)",
    )

    parser.add_argument("pytest_args", nargs="*", help="Additional pytest arguments")

    args = parser.parse_args()

    try:
        orchestrator = TestOrchestrator()
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    # Validate all required configuration parameters (NO FALLBACKS)
    required_params = [
        ("quality_gates.dependencies.block_on_warnings", bool),
        ("quality_gates.autofix", bool),
        ("quality_gates.format.block_on_warnings", bool),
        ("quality_gates.format.block_on_errors", bool),
        ("quality_gates.lint.block_on_warnings", bool),
        ("quality_gates.lint.block_on_errors", bool),
        ("quality_gates.types.block_on_warnings", bool),
        ("quality_gates.types.block_on_errors", bool),
    ]

    for param_path, param_type in required_params:
        if not orchestrator._validate_config_parameter(param_path, param_type):
            print("\n💡 Tip: Check Test/test_config.yaml for the required structure", file=sys.stderr)
            return 1

    # Clean output directories before execution
    orchestrator._clean_output_directories()

    # Read configuration (after validation)
    autofix_from_config = orchestrator.config["quality_gates"]["autofix"]
    autofix = not args.no_autofix and autofix_from_config

    # Run quality gates first (fail-fast)
    gate_result = orchestrator.run_quality_gates(verbose=args.verbose, autofix=autofix)
    if gate_result != 0:
        print(f"\n❌ Quality gate failed with exit code {gate_result}")
        return gate_result

    # TEMPORAL: Exit after GATE 2 - following stages not yet audited
    print("\n⚠️  TEMPORAL EXIT: Stopping after GATE 2")
    print("   Following stages (test generation, pytest execution) not yet audited.")
    print("   Remove this block after auditing remaining stages.\n")
    return 0

    # Regenerate if requested via CLI flag
    if args.regenerate and not args.dry_run:
        if not orchestrator.regenerate_tests(verbose=args.verbose):
            return 1

    if args.dry_run:
        print("📋 Dry run mode - would execute:")
        print(f"   pytest {orchestrator.test_dir / 'tests'}")
        if args.markers:
            print(f"   Markers: {' or '.join(args.markers)}")
        if args.coverage:
            print("   Coverage: enabled")
        print("\n✅ Dry run complete (no tests executed)")
        return 0

    # Run pytest
    pytest_args = list(args.pytest_args) if args.pytest_args else []
    if args.keep_going:
        pytest_args.append("--maxfail=999")  # Don't stop on first failure

    exit_code = orchestrator.run_pytest(
        markers=args.markers,
        coverage=args.coverage,
        verbose=args.verbose,
        extra_args=pytest_args,
    )

    # NOTE: MD reports are now generated by each script after creating JSON
    # orchestrator.generate_all_reports(exit_code)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
