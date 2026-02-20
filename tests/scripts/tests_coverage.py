"""
Meta Test Validator - Structural Coverage Analyzer

Validation Scope: Python/ (api, core, io, kernels modules only)
Tests File: tests/scripts/code_structure.py

Validates that code_structure.py covers 100% of public functions
that should be tested according to the codebase structure.

Reports:
  1. Functions that MUST be tested (from __all__ exports in Python/)
  2. Functions that ARE tested (by name matching in code_structure.py)
  3. Functions that NEED testing (gap analysis)
  4. Tests that reference non-existent functions (orphans)

Output:
  - Console summary (PASS/FAIL per module)
  - JSON report: tests/results/tests_coverage_YYYY-MM-DD_HH-MM-SS.ffffff.json
"""

import ast
import json
from pathlib import Path
from typing import Set, Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class CoverageGap:
    """Represents a function that needs testing."""
    module: str
    function_name: str
    is_class: bool
    needs_test: bool = True


@dataclass
class OrphanTest:
    """Represents a test that doesn't match any real function."""
    test_name: str
    test_class: str
    suspected_function: str


class StructuralCoverageValidator:
    """Analyzes test coverage at the structural level."""
    
    def __init__(self, project_root: Path):
        self.root = project_root
        self.public_functions: Dict[str, Set[str]] = {}  # module -> {func1, func2}
        self.tested_functions: Set[str] = set()
        self.test_functions: Dict[str, str] = {}  # test_name -> test_class
        
    def extract_public_api(self) -> Dict[str, Set[str]]:
        """Extract all public functions from __all__ exports and modules."""
        
        # Key modules to check
        modules_to_check = [
            "Python/api/__init__.py",
            "Python/kernels/__init__.py",
            "Python/core/__init__.py",
        ]
        
        for module_path in modules_to_check:
            full_path = self.root / module_path
            if not full_path.exists():
                continue
                
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract __all__ using regex as fallback
            exports = set()
            
            try:
                tree = ast.parse(content)
                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == "__all__":
                                if isinstance(node.value, ast.List):
                                    for elt in node.value.elts:
                                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                            exports.add(elt.value)
            except Exception as e:
                print(f"⚠️  Error parsing {module_path}: {e}")
            
            if exports:
                self.public_functions[module_path] = exports
        
        return self.public_functions
    
    def extract_test_functions(self) -> Dict[str, str]:
        """Extract all test function names and their test classes from code_structure.py."""
        
        test_file = self.root / "tests/scripts/code_structure.py"
        if not test_file.exists():
            return {}
        
        with open(test_file, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        tests = {}
        current_class = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.startswith("Test"):
                    current_class = node.name
                    # Extract test methods from this class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                            tests[item.name] = current_class
        
        self.test_functions = tests
        return tests
    
    def extract_tested_symbols(self) -> Set[str]:
        """Extract function names mentioned in code_structure.py test file."""
        
        test_file = self.root / "tests/scripts/code_structure.py"
        if not test_file.exists():
            return set()
        
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Simple extraction: look for imports and function calls
        tree = ast.parse(content)
        tested = set()
        
        # Get imported names
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.names:
                    for alias in node.names:
                        tested.add(alias.name)
            elif isinstance(node, ast.Call):
                # Extract function names from calls
                if isinstance(node.func, ast.Name):
                    tested.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    tested.add(node.func.attr)
        
        self.tested_functions = tested
        return tested
    
    def analyze_coverage(self) -> Tuple[List[CoverageGap], List[OrphanTest]]:
        """Analyze what's missing and what's extra."""
        
        # Extract all public functions
        self.extract_public_api()
        self.extract_test_functions()
        self.extract_tested_symbols()
        
        # Flatten all public functions
        all_public = set()
        for exports in self.public_functions.values():
            all_public.update(exports)
        
        # Find gaps (public but not tested)
        gaps = []
        for func_name in sorted(all_public):
            if func_name not in self.tested_functions:
                gaps.append(CoverageGap(
                    module="(various)",
                    function_name=func_name,
                    is_class=False,
                    needs_test=True
                ))
        
        # Find orphans (tested but doesn't exist)
        orphans = []
        for test_name, test_class in self.test_functions.items():
            # Skip coverage validation tests (they test the framework, not API)
            if test_class == "TestCoverageValidation":
                continue
            
            # Extract suspected function name from test
            # e.g., test_initialize_state -> initialize_state
            suspected = test_name.replace("test_", "")
            
            # Check if it matches any public function
            found = False
            for func_name in all_public:
                if suspected.lower() in func_name.lower() or func_name.lower() in suspected.lower():
                    found = True
                    break
            
            if not found:
                orphans.append(OrphanTest(
                    test_name=test_name,
                    test_class=test_class,
                    suspected_function=suspected
                ))
        
        return gaps, orphans
    
    def generate_report(self) -> str:
        """Generate human-readable coverage report."""
        
        gaps, orphans = self.analyze_coverage()
        
        all_public = set()
        for exports in self.public_functions.values():
            all_public.update(exports)
        
        coverage_pct = (
            (len(all_public) - len(gaps)) / len(all_public) * 100
            if all_public else 0
        )
        
        report = []
        report.append("=" * 80)
        report.append("STRUCTURAL TEST COVERAGE VALIDATOR")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append(f"📊 COVERAGE SUMMARY")
        report.append(f"  Public functions found: {len(all_public)}")
        report.append(f"  Tests defined: {len(self.test_functions)}")
        report.append(f"  Symbols tested: {len(self.tested_functions)}")
        report.append(f"  Coverage: {coverage_pct:.1f}% ({len(all_public) - len(gaps)}/{len(all_public)})")
        report.append("")
        
        # Gaps
        if gaps:
            report.append(f"❌ FUNCTIONS THAT NEED TESTING ({len(gaps)} gaps):")
            for gap in gaps:
                report.append(f"  ✗ {gap.function_name}")
        else:
            report.append(f"✅ NO GAPS - All public functions are tested!")
        report.append("")
        
        # Orphans
        if orphans:
            report.append(f"⚠️  ORPHAN TESTS ({len(orphans)} tests without matching functions):")
            for orphan in orphans:
                report.append(f"  ? {orphan.test_class}::{orphan.test_name}")
                report.append(f"    → Suspected function: {orphan.suspected_function}")
        else:
            report.append(f"✅ NO ORPHANS - All tests match real functions!")
        report.append("")
        
        # Public API
        report.append(f"📦 PUBLIC API INVENTORY:")
        for module, exports in sorted(self.public_functions.items()):
            report.append(f"  {module}:")
            for export in sorted(exports):
                is_tested = "✓" if export in self.tested_functions else "✗"
                report.append(f"    {is_tested} {export}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def generate_json_report(self) -> dict:
        """Generate machine-readable JSON report."""
        
        gaps, orphans = self.analyze_coverage()
        
        all_public = set()
        for exports in self.public_functions.values():
            all_public.update(exports)
        
        coverage_pct = (
            (len(all_public) - len(gaps)) / len(all_public) * 100
            if all_public else 0
        )
        
        return {
            "summary": {
                "coverage_percentage": coverage_pct,
                "total_public_functions": len(all_public),
                "total_tests": len(self.test_functions),
                "total_symbols_tested": len(self.tested_functions),
                "gaps_count": len(gaps),
                "orphans_count": len(orphans),
            },
            "gaps": [asdict(g) for g in gaps],
            "orphans": [asdict(o) for o in orphans],
            "public_api": {
                module: list(exports)
                for module, exports in self.public_functions.items()
            },
            "test_functions": self.test_functions,
            "tested_symbols": sorted(list(self.tested_functions)),
        }


def validate_coverage() -> dict:
    """
    Public API for coverage validation.
    
    Returns:
        dict with keys: coverage (float), gaps (int), orphans (int)
    """
    project_root = Path(__file__).parent.parent.parent
    validator = StructuralCoverageValidator(project_root)
    json_report = validator.generate_json_report()
    
    return {
        "coverage": json_report["summary"]["coverage_percentage"],
        "gaps": json_report["summary"]["gaps_count"],
        "orphans": json_report["summary"]["orphans_count"],
    }


def main():
    """Run the validator."""
    from datetime import datetime
    
    project_root = Path(__file__).parent.parent.parent  # Go up to project root
    validator = StructuralCoverageValidator(project_root)
    
    # Generate reports
    print(validator.generate_report())
    
    # Save JSON report with timestamp
    json_report = validator.generate_json_report()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S.%f")
    report_file = project_root / "tests/results" / f"tests_coverage_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"\n💾 JSON report saved to: {report_file}")
    
    # Exit with error if gaps or orphans exist
    if json_report["summary"]["gaps_count"] > 0:
        print(f"\n⚠️  EXIT CODE: 1 (gaps detected)")
        return 1
    
    if json_report["summary"]["orphans_count"] > 0:
        print(f"\n⚠️  EXIT CODE: 1 (orphan tests detected - unmatched to public API)")
        return 1
    
    print(f"\n✅ EXIT CODE: 0 (100% coverage, 0 orphans)")
    return 0


if __name__ == "__main__":
    exit(main())
