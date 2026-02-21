"""Dynamic scope discovery for test scripts.

Provides reusable functions to automatically discover Python modules
and their public API without hardcoding module names.

Includes change tracking to identify only modified files since last run.

Usage:
    from scope_discovery import discover_modules, discover_public_api, get_root
    from scope_discovery import discover_changed_files, discover_module_files_changed
    
    modules = discover_modules()  # ['api', 'core', 'kernels', 'io']
    api = discover_public_api(modules)  # {'api': [func1, func2, ...], ...}
    
    # Only get changed files (default, faster)
    changed = discover_changed_files()  # ['/path/to/api/config.py', ...]
    
    # Force all files (full scope)
    all_files = discover_changed_files(force_all=True)
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set


def get_root() -> Path:
    """Get project root directory."""
    return Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def get_cache_file() -> Path:
    """Get path to file timestamp cache.
    
    Cache stores modification times of Python files to detect changes.
    Located in: tests/.scope_cache.json
    """
    root = get_root()
    return root / "tests" / ".scope_cache.json"


def load_file_timestamps() -> Dict[str, float]:
    """Load cached file modification times from previous run.
    
    Returns:
        Dict mapping file paths to modification timestamps
        Empty dict if cache doesn't exist
    """
    cache_file = get_cache_file()
    if not cache_file.exists():
        return {}
    
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_file_timestamps(file_dict: Dict[str, float]) -> None:
    """Save file modification times to cache for next run.
    
    Args:
        file_dict: Dict mapping file paths to modification timestamps
    """
    cache_file = get_cache_file()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(file_dict, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save cache file: {e}", file=sys.stderr)


def get_all_python_files(root: Path | str | None = None) -> Dict[str, float]:
    """Get all .py files in Python/ directory with their modification times.
    
    Args:
        root: Project root directory
    
    Returns:
        Dict mapping normalized file paths to modification times
    """
    if root is None:
        root = get_root()
    
    root = Path(root) if isinstance(root, str) else root
    python_dir = root / "Python"
    
    if not python_dir.exists():
        return {}
    
    file_dict = {}
    for py_file in sorted(python_dir.rglob("*.py")):
        # Skip __pycache__ and .pyc files
        if "__pycache__" in py_file.parts:
            continue
        
        # Normalize path for caching (relative to root)
        rel_path = str(py_file.relative_to(root))
        mtime = py_file.stat().st_mtime
        file_dict[rel_path] = mtime
    
    return file_dict


def discover_changed_files(root: Path | str | None = None, force_all: bool = False) -> List[str]:
    """Discover Python files that have changed since last scan.
    
    By default, returns only files modified since the last cache update.
    Use force_all=True to get all files (for full audit/test runs).
    
    Args:
        root: Project root directory
        force_all: If True, return all files regardless of timestamps
    
    Returns:
        List of absolute paths to changed files
    """
    if root is None:
        root = get_root()
    
    root = Path(root) if isinstance(root, str) else root
    
    # Get current state
    current_files = get_all_python_files(root)
    
    if force_all:
        # Return all files
        changed = list(current_files.keys())
    else:
        # Get previous state
        previous_files = load_file_timestamps()
        
        # Find changed files
        changed = []
        for path, mtime in current_files.items():
            prev_mtime = previous_files.get(path)
            
            # Include if: new file, or modification time changed
            if prev_mtime is None or abs(mtime - prev_mtime) > 0.001:
                changed.append(path)
    
    # Update cache for next run
    save_file_timestamps(current_files)
    
    # Convert to absolute paths
    return [str(root / path) for path in sorted(changed)]


def discover_module_files_changed(module_name: str, root: Path | str | None = None, 
                                 force_all: bool = False) -> List[str]:
    """Discover .py files changed in a specific module.
    
    Args:
        module_name: Module to scan (e.g., 'api', 'core', 'kernels', 'io')
        root: Project root directory
        force_all: If True, return all files in module
    
    Returns:
        List of filenames (relative to module directory)
    """
    if root is None:
        root = get_root()
    
    root = Path(root) if isinstance(root, str) else root
    
    all_changed = discover_changed_files(root, force_all=force_all)
    module_path = f"/Python/{module_name}/"
    
    # Filter to files in this module and extract just the filename
    module_changed = [
        Path(f).name 
        for f in all_changed 
        if module_path in f
    ]
    
    return sorted(set(module_changed))  # Remove duplicates, sort


def reset_cache(root: Path | str | None = None) -> None:
    """Reset change detection cache.
    
    Removes the cached file timestamps. Next scan will treat all files
    as "changed" (useful for full audit runs after major refactorings).
    
    Args:
        root: Project root directory
    """
    if root is None:
        root = get_root()
    
    cache_file = get_cache_file()
    if cache_file.exists():
        try:
            cache_file.unlink()
        except OSError as e:
            print(f"Warning: Could not delete cache: {e}", file=sys.stderr)


def get_cache_info() -> Dict[str, int]:
    """Get information about current cache state.
    
    Returns:
        Dict with cache info: file_count, cache_size_bytes
    """
    cache_file = get_cache_file()
    
    if not cache_file.exists():
        return {"cached_files": 0, "cache_size_bytes": 0, "exists": False}
    
    cached_files = load_file_timestamps()
    size = cache_file.stat().st_size
    
    return {
        "cached_files": len(cached_files),
        "cache_size_bytes": size,
        "exists": True,
    }


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
