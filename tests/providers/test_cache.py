from datetime import datetime, timedelta
import pandas as pd
from pydantic import BaseModel
import pytest

from app.providers.cache import apply_provider_cache
from app.providers.base import BaseProvider, ProviderType, ProviderConfig
from app.lib.storage import get_cache_file_paths


class DummyDFProvider(BaseProvider[pd.DataFrame]):
    def _get_provider_type(self) -> ProviderType:
        return ProviderType.DUMMY

    # pyrefly: ignore[bad-override]
    # @apply_provider_cache triggers pyrefly bad-override - no easy fix
    @apply_provider_cache
    async def _fetch_data(
        self, query: str | None, *args, cache_date: str | None = None, **kwargs
    ) -> pd.DataFrame:
        """Simulate fetching a DataFrame with unique content using query"""
        return pd.DataFrame({"value": [len(query) if query else 0]})


class SimpleModel(BaseModel):
    value: int


class DummyModelProvider(BaseProvider[SimpleModel]):
    def _get_provider_type(self) -> ProviderType:
        return ProviderType.DUMMY

    # pyrefly: ignore[bad-override]
    # @apply_provider_cache triggers pyrefly bad-override - no easy fix
    @apply_provider_cache
    async def _fetch_data(
        self, query: str | None, *args, cache_date: str | None = None, **kwargs
    ) -> SimpleModel:
        return SimpleModel(value=len(query) if query else 0)


@pytest.mark.asyncio
async def test_dataframe_cache():
    provider = DummyDFProvider(ProviderConfig())
    # First fetch should create cache file
    res1 = await provider.get_data("foo")
    # Verify result and extract DataFrame
    assert res1.success
    df1 = res1.data  # type: ignore[attr-defined]
    assert isinstance(df1, pd.DataFrame)

    # Check for cache files using the storage system
    _, parquet_path = get_cache_file_paths("dummy", "foo")
    assert parquet_path.exists(), "Parquet cache file was not created"

    # Second fetch should return cached DataFrame
    res2 = await provider.get_data("foo")
    df2 = res2.data  # type: ignore[attr-defined]
    assert isinstance(df2, pd.DataFrame)
    assert df2.equals(df1)


@pytest.mark.asyncio
async def test_model_cache():
    provider = DummyModelProvider(ProviderConfig())
    # First fetch should create cache file
    res1 = await provider.get_data("bar")
    # Verify result and extract model
    assert res1.success
    model1 = res1.data  # type: ignore[attr-defined]
    assert isinstance(model1, SimpleModel)
    assert model1.value == 3

    # Check for cache files using the storage system
    json_path, _ = get_cache_file_paths("dummy", "bar")
    assert json_path.exists(), "JSON cache file was not created"

    # Second fetch should return cached model
    res2 = await provider.get_data("bar")
    model2 = res2.data  # type: ignore[attr-defined]
    assert isinstance(model2, SimpleModel)
    assert model2.value == model1.value


@pytest.mark.asyncio
async def test_cache_date_read_only():
    provider = DummyDFProvider(ProviderConfig())
    # Provide a past date where no cache exists
    old_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    # Should fetch without error and not write new cache files
    df = await provider._fetch_data("baz", cache_date=old_date)
    assert isinstance(df, pd.DataFrame)

    # Check that cache files were not created for read-only mode
    json_path, parquet_path = get_cache_file_paths("custom", "baz", old_date)
    # Directory may be created by get_cache_paths, but no files should be written
    assert (
        not json_path.exists()
    ), "JSON cache file should not be created in read-only mode"
    assert (
        not parquet_path.exists()
    ), "Parquet cache file should not be created in read-only mode"
