"""Reusable testing framework for Python projects.

This framework provides:
- Auto-discovery of modules and tests
- Configurable fixtures with caching
- Change detection for incremental testing
- Extensible test generation

Usage in another project:
    1. Copy Test/framework/ to your project
    2. Create Test/config/test_config.toml
    3. Adapt Test/conftest.py fixtures
    4. Run: pytest Test/tests/

Framework modules:
    - discovery: Auto-discover modules, files, and changes
    - fixtures: Factory for creating project-specific fixtures
    - markers: Dynamic pytest marker generation
"""

__version__ = "2.1.0"

from . import discovery, generator, inspector

__all__ = ["discovery", "generator", "inspector"]
