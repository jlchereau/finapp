"""
Flow cache utilities for aiocache management.

This module provides utilities for applying aiocache decorators conditionally
based on application settings. Used across multiple workflow modules.
"""

from aiocache import cached

from ..lib.settings import settings


def apply_flow_cache(func):
    """
    Apply aiocache decorator based on settings, or return function unchanged.

    This helper function allows conditional caching based on FLOW_CACHE_ENABLED
    setting and uses the configurable FLOW_CACHE_TTL value.

    Args:
        func: The async function to potentially cache

    Returns:
        Cached function if FLOW_CACHE_ENABLED=True, otherwise original function
    """
    if settings.FLOW_CACHE_ENABLED:
        return cached(ttl=settings.FLOW_CACHE_TTL)(func)
    return func
