"""
Global pytest configuration for cache isolation.

This configuration ensures that all tests run with isolated cache directories
to prevent contamination of the production cache in data/.
"""

import os


def pytest_configure(config):  # pylint: disable=unused-argument
    """
    Global pytest configuration hook that runs before any tests.

    Sets PROVIDER_CACHE_ROOT to redirect all cache operations to temp/pytest/
    instead of the production data/ directory. This prevents test data from
    contaminating the production cache.

    Also disables FLOW_CACHE_ENABLED to prevent workflow caching during tests.

    Configures filterwarnings to suppress external library deprecation warnings.

    For local development: Uses temp/pytest/
    For CI/CD: Environment may override this (GitHub Actions already uses temp/)
    """
    # Configure warning filters to suppress external library warnings
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:workflows")
    # Only set if not already configured (allows CI to override)
    if "PROVIDER_CACHE_ROOT" not in os.environ:
        # Use project temp directory for test isolation
        project_root = os.getcwd()
        test_cache_root = os.path.join(project_root, "temp", "pytest")
        os.environ["PROVIDER_CACHE_ROOT"] = test_cache_root

    # Disable flow caching during tests to prevent interference with mocking
    # and to ensure tests run with fresh data
    if "FLOW_CACHE_ENABLED" not in os.environ:
        os.environ["FLOW_CACHE_ENABLED"] = "False"
