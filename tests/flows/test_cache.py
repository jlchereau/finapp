"""
Unit tests for flow cache utilities.

Tests the apply_flow_cache function and its integration with settings.
"""

import pytest
from unittest.mock import patch

from app.flows.cache import apply_flow_cache


@pytest.mark.asyncio
async def test_apply_flow_cache_enabled():
    """Test that cache decorator is applied when FLOW_CACHE_ENABLED=True."""
    with patch("app.flows.cache.settings") as mock_settings:
        mock_settings.FLOW_CACHE_ENABLED = True
        mock_settings.FLOW_CACHE_TTL = 300

        async def test_func():
            return "test_result"

        cached_func = apply_flow_cache(test_func)

        # Function should be wrapped by aiocache
        assert hasattr(cached_func, "__wrapped__")
        assert cached_func.__name__ == "test_func"

        # Should return expected result
        result = await cached_func()
        assert result == "test_result"


@pytest.mark.asyncio
async def test_apply_flow_cache_disabled():
    """Test that cache decorator is bypassed when FLOW_CACHE_ENABLED=False."""
    with patch("app.flows.cache.settings") as mock_settings:
        mock_settings.FLOW_CACHE_ENABLED = False

        async def test_func():
            return "test_result"

        result_func = apply_flow_cache(test_func)

        # Function should be unchanged (not wrapped)
        assert not hasattr(result_func, "__wrapped__")
        assert result_func is test_func

        # Should return expected result
        result = await result_func()
        assert result == "test_result"


def test_apply_flow_cache_ttl_setting():
    """Test that TTL from settings is used in cache decorator."""
    with patch("app.flows.cache.settings") as mock_settings:
        mock_settings.FLOW_CACHE_ENABLED = True
        mock_settings.FLOW_CACHE_TTL = 600  # Custom TTL

        async def test_func():
            return "test_result"

        cached_func = apply_flow_cache(test_func)

        # Verify function is wrapped (indicates caching applied)
        assert hasattr(cached_func, "__wrapped__")
        # Note: We don't test TTL value directly since it's internal to aiocache


def test_apply_flow_cache_preserves_function_metadata():
    """Test that function metadata is preserved when caching is applied."""
    with patch("app.flows.cache.settings") as mock_settings:
        mock_settings.FLOW_CACHE_ENABLED = True
        mock_settings.FLOW_CACHE_TTL = 300

        async def test_func_with_metadata():
            """Test function with docstring."""
            return "test_result"

        cached_func = apply_flow_cache(test_func_with_metadata)

        # Function name should be preserved
        assert cached_func.__name__ == "test_func_with_metadata"


def test_apply_flow_cache_with_sync_function():
    """Test apply_flow_cache with non-async function (should still work)."""
    with patch("app.flows.cache.settings") as mock_settings:
        mock_settings.FLOW_CACHE_ENABLED = False

        def sync_func():
            return "sync_result"

        result_func = apply_flow_cache(sync_func)

        # When disabled, should return original function
        assert result_func is sync_func
        assert result_func() == "sync_result"
