"""
Cache decorator for provider _fetch_data methods.
"""

import os
import asyncio
from functools import wraps
from datetime import datetime
import pandas as pd
import orjson

from .parsers import MODEL_CACHE

# In-memory locks for cache files to ensure safe concurrent access
_CACHE_LOCKS: dict[str, asyncio.Lock] = {}


def cache(func):  # decorator for async _fetch_data methods
    """
    Decorator to cache provider _fetch_data outputs.
    Usage: apply to async _fetch_data(self, query, **kwargs).
    Supports DataFrame (parquet) and Pydantic BaseModel (json).
    Optional kwarg cache_date (YYYYMMDD) to read from old cache without writing.
    """

    @wraps(func)
    async def wrapper(self, query, *args, cache_date: str | None = None, **kwargs):
        # If caching is disabled globally or per-provider, bypass cache entirely
        from app.core.settings import settings
        # Check if cache is enabled globally and for this provider
        cache_enabled = getattr(self.config, "cache_enabled", True)
        if (
            not settings.CACHE_ENABLED
            or not cache_enabled
        ):
            # Directly fetch without caching
            return await func(self, query, *args, **kwargs)
        # Determine cache directory based on date
        date_str = cache_date or datetime.now().strftime("%Y%m%d")
        base_dir = os.path.join(os.getcwd(), "data", date_str)
        os.makedirs(base_dir, exist_ok=True)
        # Sanitize query for filename
        q = query.upper().strip() if isinstance(query, str) else "none"
        base_name = f"{self.provider_type.value}_{q}"
        json_path = os.path.join(base_dir, base_name + ".json")
        parquet_path = os.path.join(base_dir, base_name + ".parquet")
        # Choose a lock per cache file
        # Determine lock key based on existing cache file or target path
        if os.path.exists(json_path) or not os.path.exists(parquet_path):
            lock_key = json_path
        else:
            lock_key = parquet_path
        lock = _CACHE_LOCKS.setdefault(lock_key, asyncio.Lock())
        async with lock:
            # Attempt to load DataFrame cache
            if os.path.exists(parquet_path):
                try:
                    return pd.read_parquet(parquet_path)
                except Exception:
                    pass
            # Attempt to load BaseModel cache
            if os.path.exists(json_path):
                try:
                    raw = open(json_path, "rb").read()
                    obj = orjson.loads(raw)
                    model_name = obj.get("__model__")
                    data = obj.get("data")
                    model_cls = MODEL_CACHE.get(model_name)
                    if model_cls and data is not None:
                        return model_cls.parse_obj(data)
                except Exception:
                    pass
            # Cache miss or read-only mode: fetch fresh data
            result = await func(self, query, *args, **kwargs)
            # If cache_date specified, do not write new cache
            if cache_date:
                return result
            # Write DataFrame cache
            if isinstance(result, pd.DataFrame):
                try:
                    result.to_parquet(parquet_path)
                except Exception:
                    pass
            # Write BaseModel cache
            elif hasattr(result, "model_dump"):
                try:
                    payload = {
                        "__model__": result.__class__.__name__,
                        "data": result.model_dump(),
                    }
                    data_bytes = orjson.dumps(payload)
                    open(json_path, "wb").write(data_bytes)
                except Exception:
                    pass
            return result

    return wrapper
