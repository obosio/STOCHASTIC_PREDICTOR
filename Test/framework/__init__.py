"""Reusable testing framework for Python projects.

This framework provides:
- Auto-discovery of modules and tests
- Change detection for incremental testing
- Extensible test generation
- Markdown report rendering

Usage in another project:
    1. Copy Test/framework/ to your project
    2. Create Test/config/test_config.toml
    3. Adapt Test/conftest.py fixtures
    4. Run: pytest Test/tests/

Framework modules:
    - discovery: Auto-discover modules, files, and changes
    - generator: Test generation support
    - inspector: Callable inspection utilities
    - reports: Markdown report rendering
"""

__version__ = "2.1.0"

from . import discovery, generator, inspector, reports

__all__ = ["discovery", "generator", "inspector", "reports"]
