"""Auto-discovery utilities for dynamic test generation.

This module provides reusable functions for discovering:
- Python modules in a project
- Source files and their modifications
- Public API symbols for test coverage

Designed to be project-agnostic - works with any Python project structure.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def get_project_root(marker_file: str = "config.toml") -> Path:
    """Find project root by looking for a marker file.

    Searches in this order:
    1. Python/{marker_file} (source-relative marker)
    2. Parent directories of this file

    Args:
        marker_file: Filename to search for (e.g., "setup.py", "pyproject.toml")

    Returns:
        Path: Absolute path to project root

    Raises:
        FileNotFoundError: If marker file not found in parent directories
    """
    current = Path(__file__).resolve()

    # First, check if marker is in Python/ subdirectory
    # (common pattern: config.toml in source dir)
    for parent in current.parents:
        python_marker = parent / "Python" / marker_file
        if python_marker.exists():
            return parent  # Return the parent of Python/, not Python/ itself

    # Walk up directory tree
    for parent in [current] + list(current.parents):
        if (parent / marker_file).exists():
            return parent

    raise FileNotFoundError(
        f"Could not find project root (no {marker_file} found in parent directories or Python/{marker_file})"
    )


def discover_python_modules(source_path: Path | str | None = None) -> List[str]:
    """Discover all Python modules (packages) in a source directory.

    A module is identified by the presence of __init__.py file.
    Returns fully-qualified module names for both top-level and nested packages.

    Args:
        source_path: Path to source directory (e.g., /project/Python)
                     If None, auto-detected from project root

    Returns:
        List of module names (e.g., ['Python.api', 'Python.core', 'Python.api.validators'])

    Example:
        >>> modules = discover_python_modules('/path/to/Python')
        >>> print(modules)
        ['Python.api', 'Python.core', 'Python.io', 'Python.kernels']
    """
    if source_path is None:
        root = get_project_root()
        source_path = root / "Python"

    source_path = Path(source_path) if isinstance(source_path, str) else source_path

    if not source_path.is_dir():
        return []

    modules = []
    source_name = source_path.name

    def walk_packages(directory: Path, prefix: str = ""):
        """Recursively discover packages."""
        for item in sorted(directory.iterdir()):
            if item.is_dir() and (item / "__init__.py").exists():
                module_name = f"{prefix}.{item.name}" if prefix else item.name
                full_name = f"{source_name}.{module_name}"
                modules.append(full_name)
                # Recurse into subdirectories
                walk_packages(item, module_name)

    walk_packages(source_path)
    return sorted(modules)


def discover_module_files(module_name: str, root: Path | str | None = None, source_dir: str = "Python") -> List[str]:
    """Get all .py files in a specific module.

    Args:
        module_name: Name of module (e.g., 'api')
        root: Project root directory (auto-detected if None)
        source_dir: Name of source code directory

    Returns:
        List of filenames (e.g., ['config.py', 'prng.py'])
    """
    if root is None:
        root = get_project_root()
    root = Path(root) if isinstance(root, str) else root

    module_dir = root / source_dir / module_name
    if not module_dir.is_dir():
        return []

    return sorted([f.name for f in module_dir.glob("*.py")])


def get_file_mtime(filepath: Path) -> float:
    """Get file modification time with high precision.

    Args:
        filepath: Path to file

    Returns:
        Unix timestamp with microsecond precision
    """
    try:
        return filepath.stat().st_mtime
    except OSError:
        return 0.0


def load_cache(cache_file: Path) -> Dict[str, float]:
    """Load file modification times from cache.

    Args:
        cache_file: Path to JSON cache file

    Returns:
        Dict mapping filepath → mtime
    """
    if not cache_file.exists():
        return {}

    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache_file: Path, timestamps: Dict[str, float]) -> None:
    """Save file modification times to cache.

    Args:
        cache_file: Path to JSON cache file
        timestamps: Dict mapping filepath → mtime
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_file, "w") as f:
        json.dump(timestamps, f, indent=2, sort_keys=True)


def discover_changed_files(
    root: Path | str | None = None,
    source_dir: str = "Python",
    cache_file: Path | str | None = None,
    force_all: bool = False,
) -> Tuple[List[str], Dict[str, float]]:
    """Discover files that changed since last run.

    Args:
        root: Project root directory (auto-detected if None)
        source_dir: Source code directory name
        cache_file: Path to cache file (default: Test/.scope_cache.json)
        force_all: If True, return all files regardless of cache

    Returns:
        Tuple of (changed_files, current_timestamps)
        - changed_files: List of relative paths to changed files
        - current_timestamps: Dict of all current file mtimes (for cache update)

    Example:
        >>> changed, current = discover_changed_files()
        >>> print(f"Changed: {len(changed)} files")
        Changed: 3 files
    """
    if root is None:
        root = get_project_root()
    root = Path(root) if isinstance(root, str) else root

    if cache_file is None:
        cache_file = root / "Test" / ".scope_cache.json"
    cache_file = Path(cache_file) if isinstance(cache_file, str) else cache_file

    # Get all current Python files with mtimes
    python_dir = root / source_dir
    current_timestamps = {}

    if python_dir.exists():
        for py_file in python_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            rel_path = str(py_file.relative_to(root))
            current_timestamps[rel_path] = get_file_mtime(py_file)

    # If force_all, return everything
    if force_all:
        changed_files = list(current_timestamps.keys())
        save_cache(cache_file, current_timestamps)
        return changed_files, current_timestamps

    # Load previous timestamps
    previous_timestamps = load_cache(cache_file)

    # Detect changes
    changed_files = []
    threshold = 0.001  # 1ms threshold to avoid floating point issues

    for filepath, current_mtime in current_timestamps.items():
        if filepath not in previous_timestamps:
            # New file
            changed_files.append(filepath)
        elif abs(current_mtime - previous_timestamps[filepath]) > threshold:
            # Modified file
            changed_files.append(filepath)

    # Save current state for next run
    save_cache(cache_file, current_timestamps)

    return changed_files, current_timestamps


def extract_public_symbols(module_file: Path) -> Set[str]:
    """Extract public symbols from a Python module file.

    Looks for __all__ definition or extracts non-private names.

    Args:
        module_file: Path to Python file

    Returns:
        Set of public symbol names
    """
    try:
        with open(module_file, "r") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return set()

    # Look for __all__ definition
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        result: Set[str] = set()
                        for elt in node.value.elts:
                            value: object | None = None
                            # Python 3.8+: strings are ast.Constant
                            if isinstance(elt, ast.Constant):
                                value = elt.value
                            # Python <3.8: strings are ast.Str (deprecated)
                            elif isinstance(elt, ast.Str):
                                value = elt.s
                            if isinstance(value, str):
                                result.add(value)
                        return result

    # Fallback: extract all non-private top-level names
    public_names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            public_names.add(node.name)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            public_names.add(node.name)

    return public_names
