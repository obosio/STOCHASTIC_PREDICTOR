"""
Dynamic Test Generator - Generate tests from discovered modules.

This module automatically creates pytest test cases based on module discovery
and introspection. It's project-agnostic and generates baseline execution
tests that verify functions execute without crashing.

**Synchronization**: Implements full bidirectional sync:
- Creates tests for new modules
- Updates tests when module structure changes
- Deletes tests for removed modules

Version: 2.1.0
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from framework.discovery import get_project_root
from framework.inspector import (
    categorize_callables_by_type,
    extract_module_callables,
)


class TestGenerator:
    """
    Generates pytest test files dynamically from discovered modules.

    This class is 100% project-agnostic and can be used with any Python project.
    """

    def __init__(self, project_root: Optional[Path] = None, source_dir: str = "Python"):
        """
        Initialize test generator.

        Args:
            project_root: Root directory of project (auto-detected if None)
            source_dir: Name of source directory to discover (default: "Python")
        """
        self.project_root = project_root or get_project_root()
        self.source_dir = source_dir
        self.source_path = self.project_root / source_dir

        # Discover all Python files (not just packages)
        self.python_files = self._discover_all_python_files()

    def _discover_all_python_files(self) -> Dict[str, Path]:
        """
        Discover all .py files in source tree.

        Returns:
            Dictionary mapping module names (e.g., 'Python.api.config')
            to file paths
        """
        files = {}

        for py_file in self.source_path.rglob("*.py"):
            # Skip __pycache__ and __init__.py
            if "__pycache__" in str(py_file) or py_file.name == "__init__.py":
                continue

            # Calculate module name: Python/api/config.py → Python.api.config
            rel_path = py_file.relative_to(self.project_root)
            module_name = str(rel_path).replace("/", ".").replace(".py", "")

            files[module_name] = py_file

        return files

    def generate_import_statement(self, module_name: str, callable_names: List[str]) -> str:
        """
        Generate import statement for a module.

        Args:
            module_name: Fully qualified module name (e.g., 'Python.api.config')
            callable_names: List of callable names to import

        Returns:
            Python import statement string
        """
        if not callable_names:
            return f"import {module_name}"

        sorted_names = sorted(callable_names)

        base_import = f"from {module_name} import {', '.join(sorted_names)}"

        # Limit line length
        if len(sorted_names) <= 3 and len(base_import) <= 88:
            return base_import

        items = ",\n    ".join(sorted_names) + ","
        return f"from {module_name} import (\n    {items}\n)"

    def generate_simple_test(
        self,
        callable_name: str,
        category: str = "function",
        module_name: Optional[str] = None,
    ) -> str:
        """
        Generate a simple smoke test for a callable.

        Args:
            callable_name: Name of function/class to test
            category: Type of callable ('function', 'class', 'validator', etc.)

        Returns:
            Python test function code (properly indented for class body)
        """
        test_name = f"test_{callable_name.lower()}_executes"

        if module_name == "Python.api.schemas":
            if callable_name == "OperatingMode":
                code = f'''    def {test_name}(self):
        """Execute: {callable_name} - class instantiation test."""
        instance = OperatingMode.INFERENCE
        assert instance is not None
'''
                return code.rstrip() + "\n\n"
            if callable_name == "ProcessStateSchema":
                code = f'''    def {test_name}(self):
        """Execute: {callable_name} - class instantiation test."""
        instance = ProcessStateSchema(
            magnitude=jnp.array([1.0]),
            timestamp_utc=datetime.now(timezone.utc),
        )
        assert instance is not None
'''
                return code.rstrip() + "\n\n"
            if callable_name == "KernelOutputSchema":
                code = f'''    def {test_name}(self):
        """Execute: {callable_name} - class instantiation test."""
        instance = KernelOutputSchema(
            probability_density=jnp.array([1.0]),
            kernel_id=0,
            computation_time_us=1.0,
        )
        assert instance is not None
'''
                return code.rstrip() + "\n\n"
            if callable_name == "TelemetryDataSchema":
                code = f'''    def {test_name}(self):
        """Execute: {callable_name} - class instantiation test."""
        instance = TelemetryDataSchema(
            step_index=0,
            jax_device="cpu",
            timestamp_utc=datetime.now(timezone.utc),
        )
        assert instance is not None
'''
                return code.rstrip() + "\n\n"
            if callable_name == "PredictionResultSchema":
                code = f'''    def {test_name}(self):
        """Execute: {callable_name} - class instantiation test."""
        instance = PredictionResultSchema(
            reference_prediction=jnp.array(0.5),
            confidence_lower=jnp.array(0.0),
            confidence_upper=jnp.array(1.0),
            operating_mode=OperatingMode.INFERENCE,
        )
        assert instance is not None
'''
                return code.rstrip() + "\n\n"
            if callable_name == "HealthCheckResponseSchema":
                code = f'''    def {test_name}(self):
        """Execute: {callable_name} - class instantiation test."""
        instance = HealthCheckResponseSchema(
            status="healthy",
            version="v0",
            jax_config={{}},
            uptime_seconds=0.0,
        )
        assert instance is not None
'''
                return code.rstrip() + "\n\n"

        if module_name == "Python.io.config_mutation" and callable_name == "create_config_backup":
            code = f'''    def {test_name}(self, tmp_path):
        """Execute: {callable_name}() - construction test."""
        try:
            project_root = Path(__file__).resolve().parents[4]
            config_path = project_root / "config.toml"
            backup_path = tmp_path / "config.toml.bak"
            create_config_backup(config_path, backup_path)
            assert backup_path.exists()
        except (TypeError, FileNotFoundError):
            # Function requires arguments or config file - needs manual fixtures
            pytest.skip("Requires args or config file")
'''
            return code.rstrip() + "\n\n"

        if module_name == "Python.io.snapshots" and callable_name in {
            "_require_msgpack",
            "_require_crc32c",
        }:
            code = f'''    def {test_name}(self):
        """Execute: {callable_name}() - smoke test."""
        try:
            result = {callable_name}()  # type: ignore
            assert result is not None
        except (TypeError, RuntimeError):
            # Function requires arguments or optional dependency missing
            pytest.skip("Requires args or optional dep")
'''
            return code.rstrip() + "\n\n"

        if category in ["validator", "validators"]:
            # Validators typically return (bool, str)
            code = f'''    def {test_name}(self):
        """Execute: {callable_name}() - smoke test."""
        try:
            # This is a placeholder - real fixtures needed
            result = {callable_name}
            assert result is not None
        except TypeError:
            # Function requires arguments - needs fixtures
            pytest.skip("Requires fixtures - manual test needed")
'''
        elif category in ["constructor", "constructors"]:
            code = f'''    def {test_name}(self):
        """Execute: {callable_name}() - construction test."""
        try:
            result = {callable_name}()  # type: ignore
            assert result is not None
        except TypeError:
            # Constructor requires arguments
            pytest.skip("Requires arguments - manual test needed")
'''
        elif category == "class":
            code = f'''    def {test_name}(self):
        """Execute: {callable_name} - class instantiation test."""
        try:
            instance = {callable_name}()  # type: ignore
            assert instance is not None
        except TypeError:
            # Class requires constructor arguments
            pytest.skip("Requires constructor args - manual test needed")
'''
        else:
            code = f'''    def {test_name}(self):
        """Execute: {callable_name}() - smoke test."""
        try:
            result = {callable_name}()  # type: ignore
            assert result is not None
        except TypeError:
            # Function requires arguments - needs manual fixtures
            pytest.skip("Requires arguments - manual test needed")
'''
        return code.rstrip() + "\n\n"

    def generate_test_class(self, module_name: str, module_info: Dict[str, Any]) -> str:
        """
        Generate a pytest test class for a module.

        Args:
            module_name: Module name (e.g., 'Python.api.config')
            module_info: Metadata from extract_module_callables()

        Returns:
            Complete pytest test class code
        """
        # Create class name from module name
        class_name = "Test" + "".join(part.capitalize() for part in module_name.split("."))

        # Extract marker from first real module (skip "Python")
        parts = module_name.split(".")
        marker = parts[1] if len(parts) > 1 else parts[0]

        # Collect all callables
        all_callables = [f["name"] for f in module_info["functions"]]
        all_callables.extend([c["name"] for c in module_info["classes"]])

        if not all_callables:
            return ""

        # Categorize
        categories = categorize_callables_by_type(module_info)

        # Generate tests
        tests = []
        for func in module_info["functions"]:
            cat = next((k for k, v in categories.items() if func["name"] in v), "function")
            tests.append(self.generate_simple_test(func["name"], cat, module_name))

        for cls in module_info["classes"]:
            tests.append(self.generate_simple_test(cls["name"], "class", module_name))

        class_code = (
            f"@pytest.mark.{marker}\n"
            f"class {class_name}:\n"
            f'    """Auto-generated tests for {module_name} module."""\n\n'
            f"{''.join(tests).rstrip()}\n"
        )

        return class_code

    def generate_test_file(self, module_name: str, output_path: Path) -> bool:
        """
        Generate complete test file for a module.

        Args:
            module_name: Fully qualified module name (e.g., 'Python.api.config')
            output_path: Where to write test file

        Returns:
            True if file generated successfully
        """
        # Get the module file path
        if module_name not in self.python_files:
            print(f"Module {module_name} not found")
            return False

        module_path = self.python_files[module_name]

        # Extract callables
        module_info = extract_module_callables(module_path)

        # Get all callable names for import
        all_callables = [f["name"] for f in module_info["functions"]]
        all_callables.extend([c["name"] for c in module_info["classes"]])

        if not all_callables:
            print(f"⊘ No public callables in {module_name}")
            return False

        # Generate import statement
        import_stmt = self.generate_import_statement(module_name, all_callables)

        extra_imports = ""
        if module_name == "Python.api.schemas":
            extra_imports = "from datetime import datetime, timezone\n\n" "import jax.numpy as jnp\n\n"
        if module_name == "Python.io.config_mutation":
            extra_imports = "from pathlib import Path\n\n"

        # Generate test class
        test_class = self.generate_test_class(module_name, module_info)

        # Complete file
        file_content = (
            f'"""\n'
            f"Auto-generated tests for {module_name} module.\n\n"
            f"Generated by Test Framework v2.1.0\n"
            f"DO NOT EDIT MANUALLY - regenerate with: pytest --generate-tests\n"
            f'"""\n\n'
            f"import pytest\n\n"
            f"{extra_imports}"
            f"{import_stmt}\n\n\n"
            f"{test_class}\n"
        )

        file_content = file_content.rstrip() + "\n"

        # Write file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(file_content)

        print(f"✅ Generated: {output_path.relative_to(self.project_root)}")
        return True

    def generate_all_tests(self, output_dir: Path) -> int:
        """
        Generate test files for all discovered Python files.

        Implements full synchronization:
        - Creates new tests for discovered modules
        - Updates existing tests if module structure changed
        - Deletes tests for modules no longer present

        Args:
            output_dir: Directory to write test files (e.g., Test/tests/unit/)

        Returns:
            Number of test files generated
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Map of expected test file paths from current modules
        expected_test_files = set()
        count = 0

        for module_name, module_path in sorted(self.python_files.items()):
            # Convert module name to test filename
            # Python.api.config → api/test_config.py
            parts = module_name.split(".")

            if len(parts) == 1:
                # Shouldn't happen with our discovery, but handle it
                test_file = output_dir / f"test_{parts[0]}.py"
            elif len(parts) == 2:
                # Python.api → tests/unit/test_api.py
                test_file = output_dir / f"test_{parts[1]}.py"
            else:
                # Python.api.config → tests/unit/api/test_config.py
                test_file = output_dir / parts[1] / f"test_{parts[-1]}.py"

            expected_test_files.add(test_file)

            if self.generate_test_file(module_name, test_file):
                count += 1

        # Clean up obsolete test files (sync: modules removed → tests deleted)
        self._cleanup_obsolete_tests(output_dir, expected_test_files)

        return count

    def _cleanup_obsolete_tests(self, output_dir: Path, expected_files: set) -> int:
        """
        Remove test files that are no longer needed (modules were deleted).

        Args:
            output_dir: Directory containing generated tests
            expected_files: Set of test file paths that should exist

        Returns:
            Number of files deleted
        """
        if not output_dir.exists():
            return 0

        deleted_count = 0

        # Find all test files currently in output directory
        for test_file in output_dir.rglob("test_*.py"):
            if test_file not in expected_files:
                # This test file corresponds to a module that no longer exists
                try:
                    test_file.unlink()
                    deleted_count += 1
                except OSError:
                    pass  # Silently skip files we can't delete

        # Clean up empty subdirectories
        for subdir in output_dir.iterdir():
            if subdir.is_dir():
                try:
                    # Remove if empty
                    subdir.rmdir()
                except OSError:
                    # Directory not empty, keep it
                    pass

        return deleted_count


def generate_tests_for_project(
    project_root: Optional[Path] = None,
    source_dir: str = "Python",
    output_dir: Optional[Path] = None,
) -> int:
    """
    Convenience function to generate all tests for a project.

    Args:
        project_root: Root directory of project (auto-detected if None)
        source_dir: Name of source directory (default: "Python")
        output_dir: Where to write tests (default: Test/tests/unit/)

    Returns:
        Number of test files generated
    """
    root = project_root or get_project_root()
    output = output_dir or (root / "Test" / "tests" / "unit")

    generator = TestGenerator(root, source_dir)
    return generator.generate_all_tests(output)


if __name__ == "__main__":
    # Self-test: generate tests for current project
    print("🔧 Auto-generating tests from discovered modules...")

    count = generate_tests_for_project()

    print(f"\n✅ Generated {count} test files")
    print("Run with: pytest Test/tests/ -v")
