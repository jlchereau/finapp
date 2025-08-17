"""
Unit tests for markets workflow functions.

Tests the Buffet Indicator workflow in app/flows/markets.py with mocked data
to avoid external API dependencies.
"""

import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from app.flows.markets import (
    fetch_buffet_indicator_data,
    BuffetIndicatorWorkflow,
)


@pytest.fixture(autouse=True)
def isolate_cache(monkeypatch, tmp_path):
    """Isolate cache to prevent contamination of production cache directory."""
    monkeypatch.setenv("PROVIDER_CACHE_ROOT", str(tmp_path))
    # Disable flow cache for tests to prevent interference with mocking
    monkeypatch.setenv("FLOW_CACHE_ENABLED", "False")


@pytest.fixture
def sample_gdp_data():
    """Create sample GDP data from FRED."""
    dates = pd.date_range(start="2020-01-01", periods=16, freq="QE")
    return pd.DataFrame(
        {
            "value": [20000 + i * 100 for i in range(16)],  # GDP in billions
        },
        index=dates,
    )


@pytest.fixture
def sample_wilshire_data():
    """Create sample Wilshire 5000 data."""
    dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
    return pd.DataFrame(
        {
            "Open": [30000 + i for i in range(1000)],
            "High": [30100 + i for i in range(1000)],
            "Low": [29900 + i for i in range(1000)],
            "Close": [30000 + i for i in range(1000)],
            "Adj Close": [30000 + i for i in range(1000)],
            "Volume": [1000000 + i * 1000 for i in range(1000)],
        },
        index=dates,
    )


@pytest.fixture
def mock_provider_result():
    """Create a mock provider result."""
    result = MagicMock()
    result.success = True
    result.error_message = None
    return result


@pytest.mark.asyncio
async def test_fetch_buffet_indicator_data_success(
    sample_gdp_data, sample_wilshire_data, mock_provider_result
):
    """Test successful Buffet Indicator data fetch and calculation."""
    # Setup mock provider results (create separate objects)
    gdp_result = MagicMock()
    gdp_result.success = True
    gdp_result.data = sample_gdp_data

    wilshire_result = MagicMock()
    wilshire_result.success = True
    wilshire_result.data = sample_wilshire_data

    # Mock the providers
    with patch("app.flows.markets.create_fred_series_provider") as mock_fred, patch(
        "app.flows.markets.create_yahoo_history_provider"
    ) as mock_yahoo:

        # Setup provider mocks
        mock_fred_instance = AsyncMock()
        mock_fred_instance.get_data = AsyncMock(return_value=gdp_result)
        mock_fred.return_value = mock_fred_instance

        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=wilshire_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 1, 1)
        result = await fetch_buffet_indicator_data(base_date)

        # Verify results
        assert "data" in result
        assert "base_date" in result
        assert "latest_value" in result
        assert "data_points" in result

        data = result["data"]
        assert not data.empty
        assert "GDP" in data.columns
        assert "Wilshire_5000" in data.columns
        assert "Buffet_Indicator" in data.columns

        # Verify provider calls
        mock_fred_instance.get_data.assert_called_once_with("GDP")
        mock_yahoo_instance.get_data.assert_called_once_with("^FTW5000")


@pytest.mark.asyncio
async def test_fetch_buffet_indicator_data_fred_error(sample_wilshire_data):
    """Test handling of FRED API errors."""
    # Test the error handling within the workflow
    workflow = BuffetIndicatorWorkflow()

    # Mock the FRED provider to fail
    with patch.object(workflow, "fred_provider") as mock_fred:
        mock_fred.get_data = AsyncMock(side_effect=Exception("FRED API error"))

        # Test the workflow directly
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "FRED API error" in str(e)


@pytest.mark.asyncio
async def test_fetch_buffet_indicator_data_yahoo_error(sample_gdp_data):
    """Test handling of Yahoo Finance errors."""
    # Test the error handling within the workflow
    workflow = BuffetIndicatorWorkflow()

    # Mock the Yahoo provider to fail
    with patch.object(workflow, "yahoo_provider") as mock_yahoo:
        mock_yahoo.get_data = AsyncMock(side_effect=Exception("Yahoo API error"))

        # Test the workflow directly
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "Yahoo API error" in str(e)


@pytest.mark.asyncio
async def test_buffet_indicator_calculation_accuracy(
    sample_gdp_data, sample_wilshire_data, mock_provider_result
):
    """Test the accuracy of Buffet Indicator calculation."""
    # Setup mock provider results (create separate objects)
    gdp_result = MagicMock()
    gdp_result.success = True
    gdp_result.data = sample_gdp_data

    wilshire_result = MagicMock()
    wilshire_result.success = True
    wilshire_result.data = sample_wilshire_data

    # Mock the providers
    with patch("app.flows.markets.create_fred_series_provider") as mock_fred, patch(
        "app.flows.markets.create_yahoo_history_provider"
    ) as mock_yahoo:

        # Setup provider mocks
        mock_fred_instance = AsyncMock()
        mock_fred_instance.get_data = AsyncMock(return_value=gdp_result)
        mock_fred.return_value = mock_fred_instance

        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=wilshire_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 1, 1)
        result = await fetch_buffet_indicator_data(base_date)

        # Verify calculation accuracy
        data = result["data"]

        # Check that Buffet Indicator = (Wilshire / GDP) * 100
        for idx in data.index:
            expected_value = (
                data.loc[idx, "Wilshire_5000"] / data.loc[idx, "GDP"]
            ) * 100
            actual_value = data.loc[idx, "Buffet_Indicator"]
            assert (
                abs(actual_value - expected_value) < 0.01
            )  # Allow small floating point errors


@pytest.mark.asyncio
async def test_buffet_indicator_workflow_direct():
    """Test the BuffetIndicatorWorkflow class directly."""
    # Create sample data
    gdp_data = pd.DataFrame(
        {"value": [20000, 20100, 20200]},
        index=pd.date_range(start="2020-01-01", periods=3, freq="QE"),
    )

    wilshire_data = pd.DataFrame(
        {
            "Close": [30000, 30100, 30200],
            "Volume": [1000000, 1000000, 1000000],
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="QE"),
    )

    # Mock provider results
    gdp_result = MagicMock()
    gdp_result.success = True
    gdp_result.data = gdp_data

    wilshire_result = MagicMock()
    wilshire_result.success = True
    wilshire_result.data = wilshire_data

    # Mock the providers at the class level
    with patch("app.flows.markets.create_fred_series_provider") as mock_fred, patch(
        "app.flows.markets.create_yahoo_history_provider"
    ) as mock_yahoo:

        # Setup provider mocks
        mock_fred_instance = AsyncMock()
        mock_fred_instance.get_data = AsyncMock(return_value=gdp_result)
        mock_fred.return_value = mock_fred_instance

        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=wilshire_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = BuffetIndicatorWorkflow()
        base_date = datetime(2020, 1, 1)

        result = await workflow.run(base_date=base_date)

        # Verify results
        assert "data" in result
        data = result["data"]
        assert not data.empty
        assert len(data) == 3  # Should have 3 quarters of data
        assert "Buffet_Indicator" in data.columns
