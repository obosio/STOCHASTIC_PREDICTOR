"""
Module Inspector - Extract callable signatures and metadata.

This module provides utilities to introspect Python modules and extract
information about functions, classes, and methods without importing them.
Part of the project-agnostic test framework.

Version: 2.1.0
"""

import ast
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def analyze_function_signature(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> Dict[str, Any]:
    """
    Extract function signature metadata from AST node.

    Args:
        func_node: AST FunctionDef node

    Returns:
        Dictionary with signature metadata:
        - name: Function name
        - args: List of argument names
        - defaults: Number of arguments with defaults
        - has_varargs: Whether function accepts *args
        - has_kwargs: Whether function accepts **kwargs
        - is_async: Whether function is async
        - docstring: First line of docstring (if any)
    """
    signature = {
        "name": func_node.name,
        "args": [],
        "defaults": 0,
        "has_varargs": False,
        "has_kwargs": False,
        "is_async": isinstance(func_node, ast.AsyncFunctionDef),
        "docstring": ast.get_docstring(func_node) or "",
    }

    # Extract arguments
    if func_node.args:
        # Regular args
        signature["args"] = [
            arg.arg for arg in func_node.args.args if arg.arg != "self"
        ]
        signature["defaults"] = len(func_node.args.defaults)
        signature["has_varargs"] = func_node.args.vararg is not None
        signature["has_kwargs"] = func_node.args.kwarg is not None

    return signature


def analyze_class(class_node: ast.ClassDef) -> Dict[str, Any]:
    """
    Extract class metadata from AST node.

    Args:
        class_node: AST ClassDef node

    Returns:
        Dictionary with class metadata:
        - name: Class name
        - methods: List of method signatures
        - bases: List of base class names
        - docstring: First line of docstring (if any)
    """
    class_info = {
        "name": class_node.name,
        "methods": [],
        "bases": [base.id for base in class_node.bases if isinstance(base, ast.Name)],
        "docstring": ast.get_docstring(class_node) or "",
    }

    # Extract methods
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            class_info["methods"].append(analyze_function_signature(node))

    return class_info


def extract_module_callables(module_path: Path) -> Dict[str, Any]:
    """
    Extract all callable entities (functions, classes) from a Python module.

    Uses AST parsing to avoid importing the module (safer for discovery).

    Args:
        module_path: Path to Python module file

    Returns:
        Dictionary with:
        - functions: List of function signatures
        - classes: List of class metadata
        - imports: List of imported modules (for dependency tracking)
    """
    with open(module_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=str(module_path))

    module_info = {"functions": [], "classes": [], "imports": []}

    # Extract top-level functions and classes
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_info["functions"].append(analyze_function_signature(node))
        elif isinstance(node, ast.ClassDef):
            module_info["classes"].append(analyze_class(node))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                module_info["imports"].extend([alias.name for alias in node.names])
            else:
                if node.module:
                    module_info["imports"].append(node.module)

    return module_info


def load_module_dynamically(module_path: Path, module_name: str) -> Optional[Any]:
    """
    Dynamically load a Python module from file path.

    Args:
        module_path: Path to .py file
        module_name: Desired module name (e.g., 'Python.api.config')

    Returns:
        Loaded module object or None if loading fails
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    except Exception as e:
        print(f"Failed to load {module_name}: {e}")
        return None


def get_callable_object(module, callable_name: str) -> Optional[Any]:
    """
    Get callable object from loaded module.

    Args:
        module: Loaded module object
        callable_name: Name of function/class to retrieve

    Returns:
        Callable object or None if not found
    """
    return getattr(module, callable_name, None)


def inspect_callable_signature_runtime(callable_obj) -> Dict[str, Any]:
    """
    Inspect callable signature at runtime using Python's inspect module.

    More accurate than AST parsing but requires module to be importable.

    Args:
        callable_obj: Function or class object

    Returns:
        Dictionary with runtime signature metadata
    """
    try:
        sig = inspect.signature(callable_obj)
        return {
            "name": callable_obj.__name__,
            "params": list(sig.parameters.keys()),
            "required_params": [
                name
                for name, param in sig.parameters.items()
                if param.default == inspect.Parameter.empty
                and param.kind
                not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ],
            "has_varargs": any(
                p.kind == inspect.Parameter.VAR_POSITIONAL
                for p in sig.parameters.values()
            ),
            "has_kwargs": any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ),
            "return_annotation": sig.return_annotation,
            "docstring": inspect.getdoc(callable_obj) or "",
        }
    except Exception as e:
        return {"name": getattr(callable_obj, "__name__", "unknown"), "error": str(e)}


def categorize_callables_by_type(module_info: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Categorize discovered callables by their type/purpose.

    Args:
        module_info: Output from extract_module_callables()

    Returns:
        Dictionary with categorized callable names:
        - validators: Functions with 'validate', 'check', 'verify' in name
        - constructors: Functions with 'create', 'initialize', 'build' in name
        - computations: Other functions
        - dataclasses: Classes with no methods or only __init__
        - utilities: Classes with helper methods
    """
    categories = {
        "validators": [],
        "constructors": [],
        "computations": [],
        "dataclasses": [],
        "utilities": [],
    }

    # Categorize functions
    for func in module_info["functions"]:
        name_lower = func["name"].lower()
        if any(kw in name_lower for kw in ["validate", "check", "verify", "ensure"]):
            categories["validators"].append(func["name"])
        elif any(kw in name_lower for kw in ["create", "initialize", "build", "init"]):
            categories["constructors"].append(func["name"])
        else:
            categories["computations"].append(func["name"])

    # Categorize classes
    for cls in module_info["classes"]:
        # Simple heuristic: classes with few methods are likely dataclasses
        if len(cls["methods"]) <= 1:
            categories["dataclasses"].append(cls["name"])
        else:
            categories["utilities"].append(cls["name"])

    return categories


if __name__ == "__main__":
    # Self-test
    from pathlib import Path

    # Test on this file itself
    current_file = Path(__file__)
    print(f"Analyzing: {current_file.name}")

    info = extract_module_callables(current_file)
    print(f"\nFunctions: {len(info['functions'])}")
    for func in info["functions"]:
        print(f"  - {func['name']}({', '.join(func['args'])})")

    print(f"\nClasses: {len(info['classes'])}")
    for cls in info["classes"]:
        print(f"  - {cls['name']}")

    categories = categorize_callables_by_type(info)
    print(f"\nCategories:")
    for cat, items in categories.items():
        if items:
            print(f"  {cat}: {items}")
