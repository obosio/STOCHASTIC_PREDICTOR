"""Dynamic scope discovery for test scripts.

Provides reusable functions to automatically discover Python modules
and their public API without hardcoding module names.

Usage:
    from scope_discovery import discover_modules, discover_public_api, get_root
    
    modules = discover_modules()  # ['api', 'core', 'kernels', 'io']
    api = discover_public_api(modules)  # {'api': [func1, func2, ...], ...}
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set


def get_root() -> Path:
    """Get project root directory."""
    return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def discover_modules(root: Path | str | None = None) -> List[str]:
    """Auto-discover all submodules in Python/ directory.
    
    Returns sorted list of module names (subdirectories containing __init__.py).
    
    Example:
        ['api', 'core', 'io', 'kernels']
    """
    if root is None:
        root = get_root()
    
    root = Path(root) if isinstance(root, str) else root
    python_dir = root / "Python"
    if not python_dir.exists():
        return []
    
    init_files = sorted(python_dir.glob("*/__init__.py"))
    return [init_file.parent.name for init_file in init_files]


def discover_module_files(module_name: str, root: Path | str | None = None) -> List[str]:
    """Discover all .py files in a module (excluding __pycache__)."""
    if root is None:
        root = get_root()
    
    root = Path(root) if isinstance(root, str) else root
    module_dir = root / "Python" / module_name
    if not module_dir.exists():
        return []
    
    py_files = sorted(module_dir.glob("*.py"))
    return [f.name for f in py_files if f.name != "__pycache__"]


def extract_public_api(module_name: str, root: Path | str | None = None) -> Set[str]:
    """Extract public API symbols from module's __init__.py.
    
    Reads __all__ or falls back to non-underscore top-level names.
    
    Returns:
        Set of public symbol names
    """
    if root is None:
        root = get_root()
    
    root = Path(root) if isinstance(root, str) else root
    init_file = root / "Python" / module_name / "__init__.py"
    if not init_file.exists():
        return set()
    
    try:
        tree = ast.parse(init_file.read_text())
    except SyntaxError:
        return set()
    
    # First check for __all__
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        return {
                            str(elt.value)
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        }
    
    # Fallback: non-underscore top-level names
    public = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    public.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name
                        if not name.startswith("_"):
                            public.add(name)
    
    return public


def discover_all_public_api(root: Path | str | None = None) -> Dict[str, Set[str]]:
    """Discover all public API symbols across all modules.
    
    Returns:
        Dict mapping module name to set of public symbols
        
    Example:
        {
            'api': {'PredictorConfig', 'initialize_jax_prng', ...},
            'core': {'OrchestrationResult', ...},
            'io': {'ConfigMutationError', ...},
            'kernels': {'PredictionKernel', ...}
        }
    """
    if root is None:
        root = get_root()
    
    result = {}
    for module_name in discover_modules(root):
        result[module_name] = extract_public_api(module_name, root)
    
    return result


def get_module_paths(root: Path | str | None = None) -> Dict[str, Path]:
    """Get file system paths for each discovered module.
    
    Returns:
        Dict mapping module name to module directory path
    """
    if root is None:
        root = get_root()
    
    root = Path(root) if isinstance(root, str) else root
    result = {}
    for module_name in discover_modules(root):
        result[module_name] = root / "Python" / module_name
    
    return result
