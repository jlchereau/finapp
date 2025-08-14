import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from pydantic import BaseModel
import pytest

from app.models.cache import cache
from app.models.base import BaseProvider, ProviderType, ProviderConfig


class DummyDFProvider(BaseProvider[pd.DataFrame]):
    def __init__(self, config=None):
        super().__init__(config)

    def _get_provider_type(self) -> ProviderType:
        return ProviderType.CUSTOM

    @cache
    async def _fetch_data(self, query: str | None, **kwargs) -> pd.DataFrame:
        """Simulate fetching a DataFrame with unique content using query"""
        return pd.DataFrame({"value": [len(query) if query else 0]})


class SimpleModel(BaseModel):
    value: int


class DummyModelProvider(BaseProvider[SimpleModel]):
    def __init__(self, config=None):
        super().__init__(config)

    def _get_provider_type(self) -> ProviderType:
        return ProviderType.CUSTOM

    @cache
    async def _fetch_data(self, query: str | None, **kwargs) -> SimpleModel:
        return SimpleModel(value=len(query) if query else 0)


@pytest.fixture(autouse=True)
def temp_cache_dir(monkeypatch):
    """Create a temporary cache directory in the workspace temp folder."""
    # Use the project's temp directory to avoid polluting data or OS temp
    project_root = Path(__file__).resolve().parent.parent.parent

    # Create unique temp directory for this test run
    import time

    timestamp = str(int(time.time() * 1000))
    temp_dir = project_root / "temp" / f"test_cache_{timestamp}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Change working directory to temp for cache isolation
    original_cwd = os.getcwd()
    monkeypatch.chdir(temp_dir)

    yield temp_dir

    # Restore original directory
    os.chdir(original_cwd)

    # Cleanup after test
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_dataframe_cache():
    provider = DummyDFProvider(ProviderConfig())
    # First fetch should create cache file
    res1 = await provider.get_data("foo")
    # Verify result and extract DataFrame
    assert res1.success
    df1 = res1.data  # type: ignore[attr-defined]
    assert isinstance(df1, pd.DataFrame)
    date_dir = Path.cwd() / "data" / datetime.now().strftime("%Y%m%d")
    files = list(date_dir.glob("*_FOO.parquet"))
    assert files, "Parquet cache file was not created"
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
    date_dir = Path.cwd() / "data" / datetime.now().strftime("%Y%m%d")
    files = list(date_dir.glob("*_BAR.json"))
    assert files, "JSON cache file was not created"
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
    cache_dir = Path.cwd() / "data" / old_date
    # Directory is created but should remain empty
    assert cache_dir.exists()
    assert not any(
        cache_dir.iterdir()
    ), "Read-only cache directory should have no files"
