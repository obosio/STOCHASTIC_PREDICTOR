"""Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides fixtures
available to ALL test files in the Test/ directory.

Fixtures use scope="session" for maximum efficiency - created once
and shared across all test modules.

For project-specific fixture configuration, see:
    Test/config/fixtures_spec.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure JAX for testing
os.environ["JAX_ENABLE_X64"] = "1"
jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL FIXTURES (scope="session" - created once, reused across all tests)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get absolute path to project root directory.

    Returns:
        Path: Absolute path to STOCHASTIC_PREDICTOR/

    Example:
        def test_config_exists(project_root):
            assert (project_root / "config.toml").exists()
    """
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config_obj():
    """Load predictor configuration (expensive operation - cached).

    Created ONCE per test session and reused across all tests.
    This is the main configuration object used throughout the system.

    Returns:
        PredictorConfig: Validated configuration dataclass

    Raises:
        pytest.skip: If config.toml is incomplete or invalid

    Example:
        def test_sigma_bounds(config_obj):
            assert config_obj.sigma_min > 0
            assert config_obj.sigma_max < 100
    """
    try:
        from Python.api.config import PredictorConfigInjector

        injector = PredictorConfigInjector()
        return injector.create_config()
    except Exception as e:
        pytest.skip(f"Config incomplete or invalid: {e}")


@pytest.fixture(scope="session")
def prng_key():
    """Initialize JAX PRNG key with fixed seed for determinism.

    Created ONCE per test session. All tests use the same seed for
    reproducibility. Individual tests can split this key as needed.

    Returns:
        jax.random.PRNGKey: Initialized PRNG key with seed=42

    Example:
        def test_random_samples(prng_key):
            key1, key2 = jax.random.split(prng_key)
            samples = jax.random.normal(key1, (100,))
            assert samples.shape == (100,)
    """
    from Python.api.prng import initialize_jax_prng

    return initialize_jax_prng(seed=42)


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTION-SCOPED FIXTURES (created per test, for isolation)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def fresh_prng_key():
    """Create a fresh PRNG key for each test (function scope).

    Use this when tests need complete PRNG isolation and don't
    mind the overhead of creating a new key.

    Returns:
        jax.random.PRNGKey: New PRNG key with seed=42

    Example:
        def test_with_isolation(fresh_prng_key):
            # This key is unique to this test
            samples = jax.random.uniform(fresh_prng_key, (10,))
    """
    from Python.api.prng import initialize_jax_prng

    return initialize_jax_prng(seed=42)


# ═══════════════════════════════════════════════════════════════════════════
# PYTEST HOOKS (customize pytest behavior)
# ═══════════════════════════════════════════════════════════════════════════


def pytest_configure(config):
    """Register custom markers for test categorization.

    Markers allow selective test execution:
        pytest -m api          # Run only API tests
        pytest -m "not slow"   # Skip slow tests
    """
    config.addinivalue_line("markers", "api: Tests for Python/api/ module")
    config.addinivalue_line("markers", "core: Tests for Python/core/ module")
    config.addinivalue_line("markers", "kernels: Tests for Python/kernels/ module")
    config.addinivalue_line("markers", "io: Tests for Python/io/ module")
    config.addinivalue_line("markers", "slow: Tests that take >5 seconds")
    config.addinivalue_line("markers", "integration: Integration tests (vs unit tests)")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their file path.

    Tests in Test/tests/api/ automatically get @pytest.mark.api
    This allows running subset of tests without manual marking:
        pytest -m api
    """
    for item in items:
        # Extract module path from test file
        test_file = Path(item.fspath)

        # Auto-mark based on directory structure
        if "tests/api" in str(test_file):
            item.add_marker(pytest.mark.api)
        elif "tests/core" in str(test_file):
            item.add_marker(pytest.mark.core)
        elif "tests/kernels" in str(test_file):
            item.add_marker(pytest.mark.kernels)
        elif "tests/io" in str(test_file):
            item.add_marker(pytest.mark.io)
