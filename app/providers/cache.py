"""
Cache decorator for provider _fetch_data methods.
"""

import asyncio
from functools import wraps
import pandas as pd
import orjson

from app.lib.storage import get_cache_file_paths

# In-memory locks for cache files to ensure safe concurrent access
_CACHE_LOCKS: dict[str, asyncio.Lock] = {}


def apply_provider_cache(func):  # decorator for async _fetch_data methods
    """
    Decorator to cache provider _fetch_data outputs.
    Usage: apply to async _fetch_data(self, query, **kwargs).
    Supports DataFrame (parquet) and Pydantic BaseModel (json).
    Optional kwarg cache_date (YYYYMMDD) to read from old cache without writing.
    """

    @wraps(func)
    async def wrapper(self, query, *args, cache_date: str | None = None, **kwargs):
        # If caching is disabled globally or per-provider, bypass cache entirely
        from app.lib.settings import settings

        # Check if cache is enabled globally and for this provider
        cache_enabled = getattr(self.config, "cache_enabled", True)
        if not settings.PROVIDER_CACHE_ENABLED or not cache_enabled:
            # Directly fetch without caching
            return await func(self, query, *args, **kwargs)
        # Get cache file paths using unified storage utility
        json_path, parquet_path = get_cache_file_paths(
            self.provider_type.value, query, cache_date
        )
        # Choose a lock per cache file
        # Determine lock key based on existing cache file or target path
        if json_path.exists() or not parquet_path.exists():
            lock_key = str(json_path)
        else:
            lock_key = str(parquet_path)
        lock = _CACHE_LOCKS.setdefault(lock_key, asyncio.Lock())
        async with lock:
            # Attempt to load DataFrame cache
            if parquet_path.exists():
                try:
                    return pd.read_parquet(parquet_path)
                except (
                    FileNotFoundError,
                    pd.errors.EmptyDataError,
                    pd.errors.ParserError,
                    OSError,
                ):
                    pass
            # Attempt to load BaseModel cache
            if json_path.exists():
                try:
                    raw = open(json_path, "rb").read()
                    obj = orjson.loads(raw)  # pylint: disable=no-member
                    model_name = obj.get("__model__")
                    data = obj.get("data")
                    # Use model_validate instead of parse_obj (Pydantic V2)
                    # The model class should be importable from data structure
                    if model_name and data is not None:
                        # Try to recreate the model instance from cached data
                        # This will be handled by the specific model classes
                        # For now, skip cache loading - models will be recreated
                        # This is a temporary solution
                        pass
                except (
                    FileNotFoundError,
                    OSError,
                    ValueError,  # orjson raises ValueError for JSON errors
                    UnicodeDecodeError,
                ):
                    pass
            # Cache miss or read-only mode: fetch fresh data
            result = await func(self, query, *args, **kwargs)
            # If cache_date specified, do not write new cache
            if cache_date:
                return result
            # Write DataFrame cache
            if isinstance(result, pd.DataFrame):
                try:
                    result.to_parquet(str(parquet_path))
                except (OSError, ValueError):
                    pass
            # Write BaseModel cache
            elif hasattr(result, "model_dump"):
                try:
                    payload = {
                        "__model__": result.__class__.__name__,
                        "data": result.model_dump(),
                    }
                    data_bytes = orjson.dumps(payload)  # pylint: disable=no-member
                    with open(json_path, "wb") as f:
                        f.write(data_bytes)
                except (OSError, TypeError):
                    pass
            return result

    return wrapper
