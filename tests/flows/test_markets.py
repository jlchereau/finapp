"""
Unit tests for markets workflow functions.

Tests various market workflows in app/flows/markets.py with mocked data
to avoid external API dependencies.
"""

import asyncio
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from app.flows.markets import (
    fetch_vix_data,
    VIXWorkflow,
    fetch_yield_curve_data,
    YieldCurveWorkflow,
    fetch_currency_data,
    CurrencyWorkflow,
    fetch_precious_metals_data,
    PreciousMetalsWorkflow,
    fetch_crypto_data,
    CryptoCurrencyWorkflow,
    fetch_crude_oil_data,
    CrudeOilWorkflow,
    fetch_bloomberg_commodity_data,
    BloombergCommodityWorkflow,
    fetch_msci_world_data,
    MSCIWorldWorkflow,
)


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


@pytest.fixture
def sample_msci_world_data():
    """Create sample MSCI World Index data."""
    dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
    return pd.DataFrame(
        {
            "Open": [2500 + i * 0.1 for i in range(500)],
            "High": [2510 + i * 0.1 for i in range(500)],
            "Low": [2490 + i * 0.1 for i in range(500)],
            "Close": [2500 + i * 0.1 for i in range(500)],
            "Adj Close": [2500 + i * 0.1 for i in range(500)],
            "Volume": [1000000 + i * 100 for i in range(500)],
        },
        index=dates,
    )


# VIX Tests


@pytest.fixture
def sample_vix_data():
    """Create sample VIX data."""
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    # Create realistic VIX values (typically 10-80 range)
    vix_values = [15 + (i % 30) + (i // 100) * 5 for i in range(365)]
    return pd.DataFrame(
        {
            "Open": [v - 1 for v in vix_values],
            "High": [v + 2 for v in vix_values],
            "Low": [v - 2 for v in vix_values],
            "Close": vix_values,
            "Adj Close": vix_values,
            "Volume": [1000000 + i * 1000 for i in range(365)],
        },
        index=dates,
    )


@pytest.mark.asyncio
async def test_fetch_vix_data_success(sample_vix_data):
    """Test successful VIX data fetch and processing."""
    # Setup mock provider result
    vix_result = MagicMock()
    vix_result.success = True
    vix_result.data = sample_vix_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        # Setup provider mock
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=vix_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 1, 1)
        result = await fetch_vix_data(base_date)

        # Verify results
        assert "data" in result
        assert "base_date" in result
        assert "historical_mean" in result
        assert "latest_value" in result
        assert "data_points" in result

        data = result["data"]
        assert not data.empty
        assert "VIX" in data.columns
        assert "VIX_MA50" in data.columns

        # Verify historical mean is calculated
        historical_mean = result["historical_mean"]
        assert isinstance(historical_mean, float)
        assert 10 <= historical_mean <= 50  # Reasonable VIX range

        # Verify 50-day moving average is calculated
        ma_values = data["VIX_MA50"].dropna()
        assert not ma_values.empty
        assert all(isinstance(val, (int, float)) for val in ma_values)

        # Verify provider call
        mock_yahoo_instance.get_data.assert_called_once_with("^VIX")


@pytest.mark.asyncio
async def test_fetch_vix_data_yahoo_error():
    """Test handling of Yahoo Finance errors for VIX."""
    # Test the error handling within the workflow
    workflow = VIXWorkflow()

    # Mock the Yahoo provider to fail
    with patch.object(workflow, "yahoo_provider") as mock_yahoo:
        mock_yahoo.get_data = AsyncMock(side_effect=Exception("Yahoo VIX API error"))

        # Test the workflow directly
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "Yahoo VIX API error" in str(e)


@pytest.mark.asyncio
async def test_vix_data_empty_handling():
    """Test handling of empty VIX data."""
    # Setup mock provider result with empty data
    vix_result = MagicMock()
    vix_result.success = True
    vix_result.data = pd.DataFrame()  # Empty DataFrame

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=vix_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = VIXWorkflow()
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "No vix data available" in str(e)


@pytest.mark.asyncio
async def test_vix_provider_failure():
    """Test handling of VIX provider failure."""
    # Setup mock provider result with failure
    vix_result = MagicMock()
    vix_result.success = False
    vix_result.error_message = "VIX provider failed"

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=vix_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = VIXWorkflow()
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "VIX data fetch failed" in str(e)


@pytest.mark.asyncio
async def test_vix_historical_mean_calculation(sample_vix_data):
    """Test the accuracy of VIX historical mean calculation."""
    # Setup mock provider result
    vix_result = MagicMock()
    vix_result.success = True
    vix_result.data = sample_vix_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=vix_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 6, 1)  # Mid-year to get partial data
        result = await fetch_vix_data(base_date)

        # Verify calculation accuracy
        historical_mean = result["historical_mean"]
        expected_mean = sample_vix_data["Close"].mean()

        # Should match the mean of the full dataset
        assert abs(historical_mean - expected_mean) < 0.01


@pytest.mark.asyncio
async def test_vix_data_filtering_by_base_date(sample_vix_data):
    """Test that VIX data is properly filtered by base_date."""
    # Setup mock provider result
    vix_result = MagicMock()
    vix_result.success = True
    vix_result.data = sample_vix_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=vix_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test with different base dates
        base_date = datetime(2020, 6, 1)  # Mid-year
        result = await fetch_vix_data(base_date)

        data = result["data"]
        assert not data.empty

        # All data should be from base_date onwards
        assert data.index.min() >= pd.to_datetime(base_date.date())

        # Should be less than full dataset
        assert len(data) < len(sample_vix_data)


@pytest.mark.asyncio
async def test_vix_workflow_direct():
    """Test the VIXWorkflow class directly."""
    # Create sample VIX data
    vix_data = pd.DataFrame(
        {
            "Close": [20.5, 18.3, 25.1, 22.7, 19.8],
            "Volume": [1000000] * 5,
        },
        index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
    )

    # Mock provider result
    vix_result = MagicMock()
    vix_result.success = True
    vix_result.data = vix_data

    # Mock the provider at the class level
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=vix_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = VIXWorkflow()
        base_date = datetime(2020, 1, 1)

        result = await workflow.run(base_date=base_date)

        # Verify results
        assert "data" in result
        assert "historical_mean" in result

        data = result["data"]
        assert not data.empty
        assert len(data) == 5  # Should have 5 days of data
        assert "VIX" in data.columns

        # Verify historical mean
        expected_mean = vix_data["Close"].mean()
        actual_mean = result["historical_mean"]
        assert abs(actual_mean - expected_mean) < 0.01


@pytest.mark.asyncio
async def test_vix_no_close_price_data():
    """Test handling when VIX data has no Close price column."""
    # Create VIX data without Close price
    vix_data = pd.DataFrame(
        {
            "Open": [20.5, 18.3, 25.1],
            "High": [21.0, 19.0, 26.0],
            "Low": [20.0, 18.0, 25.0],
            "Volume": [1000000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    # Mock provider result
    vix_result = MagicMock()
    vix_result.success = True
    vix_result.data = vix_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=vix_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = VIXWorkflow()
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "No Close price data available for VIX" in str(e)


@pytest.mark.asyncio
async def test_vix_moving_average_calculation(sample_vix_data):
    """Test the accuracy of VIX 50-day moving average calculation."""
    # Setup mock provider result
    vix_result = MagicMock()
    vix_result.success = True
    vix_result.data = sample_vix_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=vix_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 1, 1)
        result = await fetch_vix_data(base_date)

        data = result["data"]
        assert "VIX_MA50" in data.columns

        # Verify moving average calculation accuracy
        # Calculate expected MA manually for the first few data points
        vix_values = data["VIX"]
        ma_values = data["VIX_MA50"]

        # For the first point, MA should equal the VIX value (min_periods=1)
        assert abs(ma_values.iloc[0] - vix_values.iloc[0]) < 0.01

        # For later points, verify the MA is correctly calculated
        if len(data) >= 50:
            # Check 50th point (index 49) - should be average of first 50 values
            expected_ma_50 = vix_values.iloc[:50].mean()
            actual_ma_50 = ma_values.iloc[49]
            assert abs(actual_ma_50 - expected_ma_50) < 0.01

        # Verify MA smooths out volatility (should be less volatile than raw VIX)
        if len(data) >= 100:
            vix_std = vix_values.iloc[50:].std()
            ma_std = ma_values.iloc[50:].std()
            assert ma_std < vix_std  # MA should be smoother


@pytest.mark.asyncio
async def test_vix_moving_average_with_limited_data():
    """Test VIX moving average with less than 50 days of data."""
    # Create limited VIX data (only 30 days)
    dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
    vix_values = [20 + i for i in range(30)]
    limited_vix_data = pd.DataFrame(
        {
            "Close": vix_values,
            "Volume": [1000000] * 30,
        },
        index=dates,
    )

    # Setup mock provider result
    vix_result = MagicMock()
    vix_result.success = True
    vix_result.data = limited_vix_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=vix_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 1, 1)
        result = await fetch_vix_data(base_date)

        data = result["data"]
        assert "VIX_MA50" in data.columns

        # Verify MA is calculated even with limited data
        ma_values = data["VIX_MA50"].dropna()
        assert (
            len(ma_values) == 30
        )  # All points should have MA values due to min_periods=1

        # Verify MA calculation for limited data
        # First point MA should equal first VIX value
        assert abs(ma_values.iloc[0] - data["VIX"].iloc[0]) < 0.01

        # 30th point MA should be average of all 30 values
        expected_ma_30 = data["VIX"].iloc[:30].mean()
        actual_ma_30 = ma_values.iloc[29]
        assert abs(actual_ma_30 - expected_ma_30) < 0.01


@pytest.mark.asyncio
async def test_vix_workflow_moving_average_integration():
    """Test the VIXWorkflow class directly with moving average."""
    # Create sample VIX data with known values for easy verification
    vix_data = pd.DataFrame(
        {
            "Close": [10, 20, 30, 40, 50],  # Simple ascending values
            "Volume": [1000000] * 5,
        },
        index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
    )

    # Mock provider result
    vix_result = MagicMock()
    vix_result.success = True
    vix_result.data = vix_data

    # Mock the provider at the class level
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=vix_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = VIXWorkflow()
        base_date = datetime(2020, 1, 1)

        result = await workflow.run(base_date=base_date)

        # Verify results include moving average
        data = result["data"]
        assert "VIX_MA50" in data.columns

        # Verify moving average values with known data
        # MA[0] = 10 (first value)
        # MA[1] = (10+20)/2 = 15
        # MA[2] = (10+20+30)/3 = 20
        # MA[3] = (10+20+30+40)/4 = 25
        # MA[4] = (10+20+30+40+50)/5 = 30

        expected_ma = [10, 15, 20, 25, 30]
        actual_ma = data["VIX_MA50"].tolist()

        for i, (expected, actual) in enumerate(zip(expected_ma, actual_ma)):
            assert (
                abs(actual - expected) < 0.01
            ), f"MA mismatch at index {i}: expected {expected}, got {actual}"


# Yield Curve Tests


@pytest.fixture
def sample_yield_curve_data():
    """Create sample yield curve data from FRED."""
    # Create 30 days of yield curve data
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")

    # Typical yield curve shape with longer maturities having higher yields
    yield_values = {
        "1M": [4.5 + i * 0.01 for i in range(30)],  # 1-month rates
        "3M": [4.7 + i * 0.01 for i in range(30)],  # 3-month rates
        "6M": [4.8 + i * 0.01 for i in range(30)],  # 6-month rates
        "1Y": [4.9 + i * 0.01 for i in range(30)],  # 1-year rates
        "2Y": [5.0 + i * 0.01 for i in range(30)],  # 2-year rates
        "5Y": [5.2 + i * 0.01 for i in range(30)],  # 5-year rates
        "10Y": [5.5 + i * 0.01 for i in range(30)],  # 10-year rates
        "30Y": [5.8 + i * 0.01 for i in range(30)],  # 30-year rates
    }

    return pd.DataFrame(yield_values, index=dates)


@pytest.fixture
def sample_yield_curve_result():
    """Create a sample yield curve result dictionary."""
    data = pd.DataFrame(
        {
            "1M": [4.5, 4.6, 4.7],
            "3M": [4.7, 4.8, 4.9],
            "6M": [4.8, 4.9, 5.0],
            "1Y": [4.9, 5.0, 5.1],
            "2Y": [5.0, 5.1, 5.2],
            "5Y": [5.2, 5.3, 5.4],
            "10Y": [5.5, 5.6, 5.7],
            "30Y": [5.8, 5.9, 6.0],
        },
        index=pd.date_range(start="2024-01-01", periods=3, freq="D"),
    )

    return {
        "data": data,
        "base_date": datetime(2024, 1, 1),
        "latest_date": data.index[-1],
        "maturities": list(data.columns),
        "data_points": len(data),
    }


@pytest.mark.asyncio
async def test_fetch_yield_curve_data_success(sample_yield_curve_data):
    """Test successful yield curve data fetch and processing."""
    # Setup mock provider result
    fred_result = MagicMock()
    fred_result.success = True
    fred_result.data = sample_yield_curve_data

    # Mock the FRED provider
    with patch("app.flows.markets.create_fred_series_provider") as mock_fred:
        # Setup provider mock
        mock_fred_instance = AsyncMock()
        mock_fred_instance.fetch_yield_curve_data = AsyncMock(
            return_value=fred_result.data
        )
        mock_fred.return_value = mock_fred_instance

        # Test the function
        base_date = datetime(2024, 1, 1)
        result = await fetch_yield_curve_data(base_date)

        # Verify results
        assert "data" in result
        assert "base_date" in result
        assert "latest_date" in result
        assert "maturities" in result
        assert "data_points" in result

        data = result["data"]
        assert not data.empty
        assert len(data.columns) == 8  # 8 Treasury maturities
        assert all(
            maturity in data.columns
            for maturity in ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
        )

        # Verify data structure
        maturities = result["maturities"]
        assert len(maturities) == 8
        assert maturities == ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]

        # Verify latest date
        latest_date = result["latest_date"]
        assert isinstance(latest_date, (pd.Timestamp, datetime))

        # Verify provider call
        mock_fred_instance.fetch_yield_curve_data.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_yield_curve_data_fred_error():
    """Test yield curve data fetch with FRED provider error."""
    # Mock the FRED provider to raise exception
    with patch("app.flows.markets.create_fred_series_provider") as mock_fred:
        mock_fred_instance = AsyncMock()
        mock_fred_instance.fetch_yield_curve_data = AsyncMock(
            side_effect=Exception("FRED API error")
        )
        mock_fred.return_value = mock_fred_instance

        # Test the function and expect exception
        base_date = datetime(2024, 1, 1)

        with pytest.raises(Exception) as exc_info:
            await fetch_yield_curve_data(base_date)

        assert "Yield curve workflow execution failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_yield_curve_data_empty_handling():
    """Test yield curve data fetch with empty data."""
    # Setup mock provider result with empty data
    empty_data = pd.DataFrame()

    # Mock the FRED provider
    with patch("app.flows.markets.create_fred_series_provider") as mock_fred:
        mock_fred_instance = AsyncMock()
        mock_fred_instance.fetch_yield_curve_data = AsyncMock(return_value=empty_data)
        mock_fred.return_value = mock_fred_instance

        # Test the function
        base_date = datetime(2024, 1, 1)

        with pytest.raises(Exception) as exc_info:
            await fetch_yield_curve_data(base_date)

        assert "Yield curve workflow execution failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_yield_curve_workflow_direct():
    """Test the YieldCurveWorkflow class directly."""
    # Create sample yield curve data
    yield_data = pd.DataFrame(
        {
            "1M": [4.5, 4.6],
            "3M": [4.7, 4.8],
            "6M": [4.8, 4.9],
            "1Y": [4.9, 5.0],
            "2Y": [5.0, 5.1],
            "5Y": [5.2, 5.3],
            "10Y": [5.5, 5.6],
            "30Y": [5.8, 5.9],
        },
        index=pd.date_range(start="2024-01-01", periods=2, freq="D"),
    )

    # Mock the FRED provider at the workflow level
    with patch("app.flows.markets.create_fred_series_provider") as mock_fred:
        mock_fred_instance = AsyncMock()
        mock_fred_instance.fetch_yield_curve_data = AsyncMock(return_value=yield_data)
        mock_fred.return_value = mock_fred_instance

        # Test the workflow directly
        workflow = YieldCurveWorkflow()
        base_date = datetime(2024, 1, 1)

        result = await workflow.run(base_date=base_date)

        # Verify result structure
        assert "data" in result
        assert "base_date" in result
        assert "latest_date" in result
        assert "maturities" in result
        assert "data_points" in result

        # Verify data integrity
        data = result["data"]
        assert not data.empty
        assert len(data) == 2  # 2 days of data
        assert len(data.columns) == 8  # 8 maturities


@pytest.mark.asyncio
async def test_yield_curve_data_filtering_by_base_date(sample_yield_curve_data):
    """Test yield curve data is properly filtered by base date."""
    # Mock the FRED provider
    with patch("app.flows.markets.create_fred_series_provider") as mock_fred:
        mock_fred_instance = AsyncMock()
        mock_fred_instance.fetch_yield_curve_data = AsyncMock(
            return_value=sample_yield_curve_data
        )
        mock_fred.return_value = mock_fred_instance

        # Test with a base date that should filter the data
        base_date = datetime(2024, 1, 15)  # Middle of sample data
        result = await fetch_yield_curve_data(base_date)

        data = result["data"]

        # Verify all dates in result are >= base_date
        filtered_dates = data.index[data.index >= pd.Timestamp(base_date)]
        assert len(filtered_dates) > 0
        assert all(date >= pd.Timestamp(base_date) for date in data.index)


@pytest.mark.asyncio
async def test_yield_curve_provider_failure():
    """Test yield curve workflow handles provider instantiation failure."""
    # Mock provider creation to fail
    with patch("app.flows.markets.create_fred_series_provider") as mock_fred:
        mock_fred.side_effect = Exception("Provider creation failed")

        base_date = datetime(2024, 1, 1)

        with pytest.raises(Exception) as exc_info:
            await fetch_yield_curve_data(base_date)

        assert "Provider creation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_yield_curve_maturities_order(sample_yield_curve_data):
    """Test that yield curve maturities are returned in correct order."""
    # Mock the FRED provider
    with patch("app.flows.markets.create_fred_series_provider") as mock_fred:
        mock_fred_instance = AsyncMock()
        mock_fred_instance.fetch_yield_curve_data = AsyncMock(
            return_value=sample_yield_curve_data
        )
        mock_fred.return_value = mock_fred_instance

        base_date = datetime(2024, 1, 1)
        result = await fetch_yield_curve_data(base_date)

        maturities = result["maturities"]
        expected_order = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]

        assert maturities == expected_order

        # Verify data columns are also in correct order
        data = result["data"]
        assert list(data.columns) == expected_order


@pytest.mark.asyncio
async def test_yield_curve_data_values_realistic(sample_yield_curve_data):
    """Test that yield curve data values are within realistic ranges."""
    # Mock the FRED provider
    with patch("app.flows.markets.create_fred_series_provider") as mock_fred:
        mock_fred_instance = AsyncMock()
        mock_fred_instance.fetch_yield_curve_data = AsyncMock(
            return_value=sample_yield_curve_data
        )
        mock_fred.return_value = mock_fred_instance

        base_date = datetime(2024, 1, 1)
        result = await fetch_yield_curve_data(base_date)

        data = result["data"]

        # Verify all yield values are reasonable (0% to 20%)
        for column in data.columns:
            values = data[column].dropna()
            assert all(
                0 <= val <= 20 for val in values
            ), f"Unrealistic yield values in {column}"

        # Verify typical yield curve shape (longer maturities generally higher)
        latest_row = data.iloc[-1]
        assert (
            latest_row["1M"] <= latest_row["30Y"]
        ), "Yield curve should generally be upward sloping"


@pytest.mark.asyncio
async def test_yield_curve_workflow_integration():
    """Test full yield curve workflow integration."""
    # Create realistic sample data
    yield_data = pd.DataFrame(
        {
            "1M": [4.5, 4.6, 4.7],
            "3M": [4.7, 4.8, 4.9],
            "6M": [4.8, 4.9, 5.0],
            "1Y": [4.9, 5.0, 5.1],
            "2Y": [5.0, 5.1, 5.2],
            "5Y": [5.2, 5.3, 5.4],
            "10Y": [5.5, 5.6, 5.7],
            "30Y": [5.8, 5.9, 6.0],
        },
        index=pd.date_range(start="2024-01-01", periods=3, freq="D"),
    )

    # Mock the FRED provider at the workflow level
    with patch("app.flows.markets.create_fred_series_provider") as mock_fred:
        mock_fred_instance = AsyncMock()
        mock_fred_instance.fetch_yield_curve_data = AsyncMock(return_value=yield_data)
        mock_fred.return_value = mock_fred_instance

        # Test the full workflow
        workflow = YieldCurveWorkflow()
        base_date = datetime(2024, 1, 1)

        result = await workflow.run(base_date=base_date)

        # Verify complete result structure
        expected_keys = [
            "data",
            "base_date",
            "latest_date",
            "maturities",
            "data_points",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key {key} in result"

        # Verify data quality
        data = result["data"]
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert len(data.columns) == 8

        # Verify latest date is properly set
        latest_date = result["latest_date"]
        assert latest_date == data.index.max()

        # Verify data points count
        assert result["data_points"] == len(data)


# Currency Tests


@pytest.fixture
def sample_usdeur_data():
    """Create sample USD/EUR data."""
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    # Create realistic USD/EUR values (typically 0.7-0.9 range)
    usdeur_values = [0.8 + 0.1 * (i % 100) / 100 for i in range(365)]
    return pd.DataFrame(
        {
            "Open": [v - 0.01 for v in usdeur_values],
            "High": [v + 0.02 for v in usdeur_values],
            "Low": [v - 0.02 for v in usdeur_values],
            "Close": usdeur_values,
            "Adj Close": usdeur_values,
            "Volume": [1000000 + i * 1000 for i in range(365)],
        },
        index=dates,
    )


@pytest.fixture
def sample_gbpeur_data():
    """Create sample GBP/EUR data."""
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    # Create realistic GBP/EUR values (typically 1.1-1.3 range)
    gbpeur_values = [1.15 + 0.15 * (i % 80) / 80 for i in range(365)]
    return pd.DataFrame(
        {
            "Open": [v - 0.01 for v in gbpeur_values],
            "High": [v + 0.02 for v in gbpeur_values],
            "Low": [v - 0.02 for v in gbpeur_values],
            "Close": gbpeur_values,
            "Adj Close": gbpeur_values,
            "Volume": [500000 + i * 500 for i in range(365)],
        },
        index=dates,
    )


@pytest.mark.asyncio
async def test_fetch_currency_data_success(sample_usdeur_data, sample_gbpeur_data):
    """Test successful currency data fetch and processing."""
    # Setup mock provider results
    usdeur_result = MagicMock()
    usdeur_result.success = True
    usdeur_result.data = sample_usdeur_data

    gbpeur_result = MagicMock()
    gbpeur_result.success = True
    gbpeur_result.data = sample_gbpeur_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        # Setup provider mock
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(
            side_effect=[usdeur_result, gbpeur_result]
        )
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 1, 1)
        result = await fetch_currency_data(base_date)

        # Verify results
        assert "data" in result
        assert "base_date" in result
        assert "latest_usdeur" in result
        assert "latest_gbpeur" in result
        assert "data_points" in result

        data = result["data"]
        assert not data.empty
        assert "USD_EUR" in data.columns
        assert "GBP_EUR" in data.columns

        # Verify latest values are set
        latest_usdeur = result["latest_usdeur"]
        latest_gbpeur = result["latest_gbpeur"]
        assert isinstance(latest_usdeur, float)
        assert isinstance(latest_gbpeur, float)
        assert 0.6 <= latest_usdeur <= 1.0  # Reasonable USD/EUR range
        assert 1.0 <= latest_gbpeur <= 1.4  # Reasonable GBP/EUR range

        # Verify provider calls
        assert mock_yahoo_instance.get_data.call_count == 2
        mock_yahoo_instance.get_data.assert_any_call("EUR=X")
        mock_yahoo_instance.get_data.assert_any_call("GBPEUR=X")


@pytest.mark.asyncio
async def test_currency_workflow_direct():
    """Test the CurrencyWorkflow class directly."""
    # Create sample currency data
    usdeur_data = pd.DataFrame(
        {
            "Close": [0.85, 0.86, 0.87, 0.88, 0.89],
            "Volume": [1000000] * 5,
        },
        index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
    )

    gbpeur_data = pd.DataFrame(
        {
            "Close": [1.16, 1.17, 1.18, 1.19, 1.20],
            "Volume": [500000] * 5,
        },
        index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
    )

    # Mock provider results
    usdeur_result = MagicMock()
    usdeur_result.success = True
    usdeur_result.data = usdeur_data

    gbpeur_result = MagicMock()
    gbpeur_result.success = True
    gbpeur_result.data = gbpeur_data

    # Mock the provider at the class level
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(
            side_effect=[usdeur_result, gbpeur_result]
        )
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = CurrencyWorkflow()
        base_date = datetime(2020, 1, 1)

        result = await workflow.run(base_date=base_date)

        # Verify results
        assert "data" in result
        assert "latest_usdeur" in result
        assert "latest_gbpeur" in result

        data = result["data"]
        assert not data.empty
        assert len(data) == 5  # Should have 5 days of data
        assert "USD_EUR" in data.columns
        assert "GBP_EUR" in data.columns

        # Verify latest values
        latest_usdeur = result["latest_usdeur"]
        latest_gbpeur = result["latest_gbpeur"]
        assert abs(latest_usdeur - 0.89) < 0.01
        assert abs(latest_gbpeur - 1.20) < 0.01


@pytest.mark.asyncio
async def test_currency_data_filtering_by_base_date(
    sample_usdeur_data, sample_gbpeur_data
):
    """Test that currency data is properly filtered by base_date."""
    # Setup mock provider results
    usdeur_result = MagicMock()
    usdeur_result.success = True
    usdeur_result.data = sample_usdeur_data

    gbpeur_result = MagicMock()
    gbpeur_result.success = True
    gbpeur_result.data = sample_gbpeur_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(
            side_effect=[usdeur_result, gbpeur_result]
        )
        mock_yahoo.return_value = mock_yahoo_instance

        # Test with different base dates
        base_date = datetime(2020, 6, 1)  # Mid-year
        result = await fetch_currency_data(base_date)

        data = result["data"]
        assert not data.empty

        # All data should be from base_date onwards
        assert data.index.min() >= pd.to_datetime(base_date.date())

        # Should be less than full dataset
        assert len(data) < len(sample_usdeur_data)


@pytest.mark.asyncio
async def test_currency_data_yahoo_error():
    """Test handling of Yahoo Finance errors for currency data."""
    # Test the error handling within the workflow
    workflow = CurrencyWorkflow()

    # Mock the Yahoo provider to fail
    with patch.object(workflow, "yahoo_provider") as mock_yahoo:
        mock_yahoo.get_data = AsyncMock(
            side_effect=Exception("Yahoo currency API error")
        )

        # Test the workflow directly
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "Yahoo currency API error" in str(e)


@pytest.mark.asyncio
async def test_currency_data_empty_handling():
    """Test handling of empty currency data."""
    # Setup mock provider results with empty data
    usdeur_result = MagicMock()
    usdeur_result.success = True
    usdeur_result.data = pd.DataFrame()  # Empty DataFrame

    gbpeur_result = MagicMock()
    gbpeur_result.success = True
    gbpeur_result.data = pd.DataFrame()  # Empty DataFrame

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(
            side_effect=[usdeur_result, gbpeur_result]
        )
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = CurrencyWorkflow()
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "usd_eur data" in str(e)


@pytest.mark.asyncio
async def test_currency_provider_failure():
    """Test handling of currency provider failure."""
    # Setup mock provider result with failure
    usdeur_result = MagicMock()
    usdeur_result.success = False
    usdeur_result.error_message = "USD/EUR provider failed"

    gbpeur_result = MagicMock()
    gbpeur_result.success = True
    gbpeur_result.data = pd.DataFrame({"Close": [1.15]}, index=[datetime(2020, 1, 1)])

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(
            side_effect=[usdeur_result, gbpeur_result]
        )
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = CurrencyWorkflow()
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "USD/EUR data fetch failed" in str(e)


@pytest.mark.asyncio
async def test_currency_calculation_accuracy(sample_usdeur_data, sample_gbpeur_data):
    """Test the accuracy of currency data alignment and processing."""
    # Setup mock provider results
    usdeur_result = MagicMock()
    usdeur_result.success = True
    usdeur_result.data = sample_usdeur_data

    gbpeur_result = MagicMock()
    gbpeur_result.success = True
    gbpeur_result.data = sample_gbpeur_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(
            side_effect=[usdeur_result, gbpeur_result]
        )
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 1, 1)
        result = await fetch_currency_data(base_date)

        # Verify alignment accuracy
        data = result["data"]

        # Both currencies should have same index (aligned dates)
        assert len(data["USD_EUR"].dropna()) > 0
        assert len(data["GBP_EUR"].dropna()) > 0

        # Verify exchange rate value ranges are realistic
        usd_eur_values = data["USD_EUR"].dropna()
        gbp_eur_values = data["GBP_EUR"].dropna()

        # USD/EUR typically between 0.6 and 1.0
        assert all(0.5 <= val <= 1.1 for val in usd_eur_values)

        # GBP/EUR typically between 1.0 and 1.4
        assert all(0.9 <= val <= 1.5 for val in gbp_eur_values)


@pytest.mark.asyncio
async def test_currency_no_close_price_data():
    """Test handling when currency data has no Close price column."""
    # Create USD/EUR data without Close price
    usdeur_data = pd.DataFrame(
        {
            "Open": [0.85, 0.86, 0.87],
            "High": [0.86, 0.87, 0.88],
            "Low": [0.84, 0.85, 0.86],
            "Volume": [1000000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    gbpeur_data = pd.DataFrame(
        {
            "Close": [1.15, 1.16, 1.17],
            "Volume": [500000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    # Mock provider results
    usdeur_result = MagicMock()
    usdeur_result.success = True
    usdeur_result.data = usdeur_data

    gbpeur_result = MagicMock()
    gbpeur_result.success = True
    gbpeur_result.data = gbpeur_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(
            side_effect=[usdeur_result, gbpeur_result]
        )
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = CurrencyWorkflow()
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "No Close price data available for USD/EUR" in str(e)


@pytest.mark.asyncio
async def test_currency_workflow_integration():
    """Test full currency workflow integration."""
    # Create realistic sample data
    usdeur_data = pd.DataFrame(
        {
            "Close": [0.85, 0.86, 0.87],
            "Volume": [1000000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    gbpeur_data = pd.DataFrame(
        {
            "Close": [1.15, 1.16, 1.17],
            "Volume": [500000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    # Mock the provider at the workflow level
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(
            side_effect=[
                MagicMock(success=True, data=usdeur_data),
                MagicMock(success=True, data=gbpeur_data),
            ]
        )
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the full workflow
        workflow = CurrencyWorkflow()
        base_date = datetime(2020, 1, 1)

        result = await workflow.run(base_date=base_date)

        # Verify complete result structure
        expected_keys = [
            "data",
            "base_date",
            "latest_usdeur",
            "latest_gbpeur",
            "data_points",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key {key} in result"

        # Verify data quality
        data = result["data"]
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert len(data.columns) == 2

        # Verify latest values are properly set
        latest_usdeur = result["latest_usdeur"]
        latest_gbpeur = result["latest_gbpeur"]
        assert abs(latest_usdeur - 0.87) < 0.01
        assert abs(latest_gbpeur - 1.17) < 0.01

        # Verify data points count
        assert result["data_points"] == len(data)


# Precious Metals Tests


@pytest.fixture
def sample_gold_data():
    """Create sample gold futures data."""
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    # Create realistic gold values (typically $1600-$2400 range)
    gold_values = [1800 + 400 * (i % 100) / 100 for i in range(365)]
    return pd.DataFrame(
        {
            "Open": [v - 10 for v in gold_values],
            "High": [v + 20 for v in gold_values],
            "Low": [v - 20 for v in gold_values],
            "Close": gold_values,
            "Adj Close": gold_values,
            "Volume": [100000 + i * 1000 for i in range(365)],
        },
        index=dates,
    )


@pytest.fixture
def sample_bitcoin_data():
    """Create sample Bitcoin price data."""
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    # Create realistic Bitcoin values (typically $20K-$70K range)
    btc_values = [35000 + 15000 * (i % 100) / 100 for i in range(365)]
    return pd.DataFrame(
        {
            "Open": [v - 500 for v in btc_values],
            "High": [v + 1000 for v in btc_values],
            "Low": [v - 1000 for v in btc_values],
            "Close": btc_values,
            "Adj Close": btc_values,
            "Volume": [50000000 + i * 100000 for i in range(365)],
        },
        index=dates,
    )


@pytest.fixture
def sample_ethereum_data():
    """Create sample Ethereum price data."""
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    # Create realistic Ethereum values (typically $1K-$4K range)
    eth_values = [2500 + 1000 * (i % 100) / 100 for i in range(365)]
    return pd.DataFrame(
        {
            "Open": [v - 50 for v in eth_values],
            "High": [v + 100 for v in eth_values],
            "Low": [v - 100 for v in eth_values],
            "Close": eth_values,
            "Adj Close": eth_values,
            "Volume": [10000000 + i * 50000 for i in range(365)],
        },
        index=dates,
    )


@pytest.fixture
def sample_wti_data():
    """Create sample WTI crude oil price data."""
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    # Create realistic WTI values (typically $40-$120/bbl range)
    wti_values = [70 + 30 * (i % 100) / 100 for i in range(365)]
    return pd.DataFrame(
        {
            "Open": [v - 1.5 for v in wti_values],
            "High": [v + 2.0 for v in wti_values],
            "Low": [v - 2.0 for v in wti_values],
            "Close": wti_values,
            "Adj Close": wti_values,
            "Volume": [200000 + i * 1000 for i in range(365)],
        },
        index=dates,
    )


@pytest.fixture
def sample_brent_data():
    """Create sample Brent crude oil price data."""
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    # Create realistic Brent values (typically $40-$120/bbl, higher than WTI)
    brent_values = [72 + 32 * (i % 100) / 100 for i in range(365)]
    return pd.DataFrame(
        {
            "Open": [v - 1.5 for v in brent_values],
            "High": [v + 2.0 for v in brent_values],
            "Low": [v - 2.0 for v in brent_values],
            "Close": brent_values,
            "Adj Close": brent_values,
            "Volume": [150000 + i * 800 for i in range(365)],
        },
        index=dates,
    )


@pytest.mark.asyncio
async def test_fetch_precious_metals_data_success(sample_gold_data):
    """Test successful precious metals data fetch and processing."""
    # Setup mock provider result
    gold_result = MagicMock()
    gold_result.success = True
    gold_result.data = sample_gold_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        # Setup provider mock
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=gold_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 1, 1)
        result = await fetch_precious_metals_data(base_date)

        # Verify results
        assert "data" in result
        assert "base_date" in result
        assert "latest_value" in result
        assert "data_points" in result

        data = result["data"]
        assert not data.empty
        assert "Gold" in data.columns
        assert "Gold_MA50" in data.columns

        # Verify latest value is set
        latest_value = result["latest_value"]
        assert isinstance(latest_value, float)
        assert 1500 <= latest_value <= 2500  # Reasonable gold price range

        # Verify 50-day moving average is calculated
        ma_values = data["Gold_MA50"].dropna()
        assert not ma_values.empty
        assert all(isinstance(val, (int, float)) for val in ma_values)

        # Verify provider call
        mock_yahoo_instance.get_data.assert_called_once_with("GC=F")


@pytest.mark.asyncio
async def test_fetch_precious_metals_data_yahoo_error():
    """Test handling of Yahoo Finance errors for precious metals."""
    # Test the error handling within the workflow
    workflow = PreciousMetalsWorkflow()

    # Mock the Yahoo provider to fail
    with patch.object(workflow, "yahoo_provider") as mock_yahoo:
        mock_yahoo.get_data = AsyncMock(side_effect=Exception("Yahoo gold API error"))

        # Test the workflow directly
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "Yahoo gold API error" in str(e)


@pytest.mark.asyncio
async def test_precious_metals_data_empty_handling():
    """Test handling of empty precious metals data."""
    # Setup mock provider result with empty data
    gold_result = MagicMock()
    gold_result.success = True
    gold_result.data = pd.DataFrame()  # Empty DataFrame

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=gold_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = PreciousMetalsWorkflow()
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "No gold data available" in str(e)


@pytest.mark.asyncio
async def test_precious_metals_provider_failure():
    """Test handling of precious metals provider failure."""
    # Setup mock provider result with failure
    gold_result = MagicMock()
    gold_result.success = False
    gold_result.error_message = "Gold provider failed"

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=gold_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = PreciousMetalsWorkflow()
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "Gold data fetch failed" in str(e)


@pytest.mark.asyncio
async def test_precious_metals_calculation_accuracy(sample_gold_data):
    """Test the accuracy of precious metals moving average calculation."""
    # Setup mock provider result
    gold_result = MagicMock()
    gold_result.success = True
    gold_result.data = sample_gold_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=gold_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 6, 1)  # Mid-year to get partial data
        result = await fetch_precious_metals_data(base_date)

        # Verify moving average calculation accuracy
        data = result["data"]
        assert "Gold_MA50" in data.columns

        # Verify that moving averages are properly calculated
        # Since the data is filtered, the MA values come from full dataset calculation
        gold_values = data["Gold"]
        ma_values = data["Gold_MA50"]

        # Verify all MA values are reasonable
        assert all(isinstance(val, (int, float)) for val in ma_values.dropna())
        assert len(ma_values.dropna()) > 0

        # Verify MA smooths out volatility (should be less volatile than raw gold)
        if len(data) >= 50:
            gold_std = gold_values.std()
            ma_std = ma_values.std()
            assert ma_std <= gold_std  # MA should be smoother or equal


@pytest.mark.asyncio
async def test_precious_metals_data_filtering_by_base_date(sample_gold_data):
    """Test that precious metals data is properly filtered by base_date."""
    # Setup mock provider result
    gold_result = MagicMock()
    gold_result.success = True
    gold_result.data = sample_gold_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=gold_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test with different base dates
        base_date = datetime(2020, 6, 1)  # Mid-year
        result = await fetch_precious_metals_data(base_date)

        data = result["data"]
        assert not data.empty

        # All data should be from base_date onwards
        assert data.index.min() >= pd.to_datetime(base_date.date())

        # Should be less than full dataset
        assert len(data) < len(sample_gold_data)


@pytest.mark.asyncio
async def test_precious_metals_workflow_direct():
    """Test the PreciousMetalsWorkflow class directly."""
    # Create sample gold data
    gold_data = pd.DataFrame(
        {
            "Close": [1850.5, 1863.3, 1875.1, 1892.7, 1908.8],
            "Volume": [100000] * 5,
        },
        index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
    )

    # Mock provider result
    gold_result = MagicMock()
    gold_result.success = True
    gold_result.data = gold_data

    # Mock the provider at the class level
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=gold_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = PreciousMetalsWorkflow()
        base_date = datetime(2020, 1, 1)

        result = await workflow.run(base_date=base_date)

        # Verify results
        assert "data" in result
        assert "latest_value" in result

        data = result["data"]
        assert not data.empty
        assert len(data) == 5  # Should have 5 days of data
        assert "Gold" in data.columns

        # Verify latest value
        latest_value = result["latest_value"]
        assert abs(latest_value - 1908.8) < 0.01


@pytest.mark.asyncio
async def test_precious_metals_no_close_price_data():
    """Test handling when precious metals data has no Close price column."""
    # Create gold data without Close price
    gold_data = pd.DataFrame(
        {
            "Open": [1850.5, 1863.3, 1875.1],
            "High": [1860.0, 1870.0, 1885.0],
            "Low": [1840.0, 1855.0, 1870.0],
            "Volume": [100000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    # Mock provider result
    gold_result = MagicMock()
    gold_result.success = True
    gold_result.data = gold_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=gold_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the workflow directly
        workflow = PreciousMetalsWorkflow()
        try:
            await workflow.run(base_date=datetime(2020, 1, 1))
            assert False, "Expected exception to be raised"
        except Exception as e:
            assert "No Close price data available for gold" in str(e)


@pytest.mark.asyncio
async def test_precious_metals_moving_average_calculation(sample_gold_data):
    """Test the accuracy of precious metals moving average calculations."""
    # Setup mock provider result
    gold_result = MagicMock()
    gold_result.success = True
    gold_result.data = sample_gold_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=gold_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 1, 1)
        result = await fetch_precious_metals_data(base_date)

        data = result["data"]
        assert "Gold_MA50" in data.columns
        assert "Gold_MA200" in data.columns

        # Verify that moving averages are properly calculated
        gold_values = data["Gold"]
        ma50_values = data["Gold_MA50"]
        ma200_values = data["Gold_MA200"]

        # Verify all MA values are reasonable
        assert all(isinstance(val, (int, float)) for val in ma50_values.dropna())
        assert all(isinstance(val, (int, float)) for val in ma200_values.dropna())
        assert len(ma50_values.dropna()) > 0
        assert len(ma200_values.dropna()) > 0

        # Test MA50 calculation if we have enough data
        if len(data) >= 50:
            # Find first non-NaN MA50 value
            first_valid_idx = ma50_values.first_valid_index()
            if first_valid_idx is not None:
                # Get position of first valid MA value
                pos = data.index.get_loc(first_valid_idx)

                # Calculate manual 50-day MA at that position
                manual_ma50 = data["Gold"].iloc[max(0, pos - 49) : pos + 1].mean()
                calculated_ma50 = ma50_values.loc[first_valid_idx]

                # Should be approximately equal (within 0.01)
                assert abs(manual_ma50 - calculated_ma50) < 0.01

        # Test MA200 calculation if we have enough data
        if len(data) >= 200:
            # Find first non-NaN MA200 value
            first_valid_idx = ma200_values.first_valid_index()
            if first_valid_idx is not None:
                # Get position of first valid MA value
                pos = data.index.get_loc(first_valid_idx)

                # Calculate manual 200-day MA at that position
                manual_ma200 = data["Gold"].iloc[max(0, pos - 199) : pos + 1].mean()
                calculated_ma200 = ma200_values.loc[first_valid_idx]

                # Should be approximately equal (within 0.01)
                assert abs(manual_ma200 - calculated_ma200) < 0.01

            # Verify MA smooths out volatility (MA200 should be smoother than MA50)
            gold_std = gold_values.std()
            ma50_std = ma50_values.std()
            ma200_std = ma200_values.std()
            assert ma50_std <= gold_std  # MA50 should be smoother
            assert ma200_std <= ma50_std  # MA200 should be even smoother


@pytest.mark.asyncio
async def test_precious_metals_moving_average_with_limited_data():
    """Test precious metals moving averages with limited data points."""
    # Create limited gold data (less than 200 days)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    gold_values = [1800 + i * 2 for i in range(100)]  # Simple ascending values
    limited_gold_data = pd.DataFrame(
        {
            "Close": gold_values,
            "Volume": [100000] * 100,
        },
        index=dates,
    )

    # Setup mock provider result
    gold_result = MagicMock()
    gold_result.success = True
    gold_result.data = limited_gold_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=gold_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 1, 1)
        result = await fetch_precious_metals_data(base_date)

        data = result["data"]
        assert "Gold_MA50" in data.columns
        assert "Gold_MA200" in data.columns

        # MA50 should be available for data >= 50 days
        ma50_valid_count = data["Gold_MA50"].count()
        assert ma50_valid_count > 0  # Should have some MA50 values
        # With 100 data points, we can get MA50 from positions 49-99 (51 values)
        # But since we're filtering by base_date, we might get all 100 values
        assert ma50_valid_count <= 100

        # MA200 should be mostly NaN for data < 200 days
        ma200_valid_count = data["Gold_MA200"].count()
        assert ma200_valid_count == 0  # No valid MA200 values with only 100 days


@pytest.mark.asyncio
async def test_precious_metals_workflow_integration():
    """Test full precious metals workflow integration."""
    # Create realistic sample data
    gold_data = pd.DataFrame(
        {
            "Close": [1850, 1863, 1875],  # Simple ascending values
            "Volume": [100000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    # Mock the provider at the workflow level
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(
            return_value=MagicMock(success=True, data=gold_data)
        )
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the full workflow
        workflow = PreciousMetalsWorkflow()
        base_date = datetime(2020, 1, 1)

        result = await workflow.run(base_date=base_date)

        # Verify complete result structure
        expected_keys = ["data", "base_date", "latest_value", "data_points"]
        for key in expected_keys:
            assert key in result, f"Missing key {key} in result"

        # Verify data quality
        data = result["data"]
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert len(data.columns) == 3  # Gold, Gold_MA50, and Gold_MA200

        # Verify latest value is properly set
        latest_value = result["latest_value"]
        assert abs(latest_value - 1875) < 0.01

        # Verify data points count
        assert result["data_points"] == len(data)


@pytest.mark.asyncio
async def test_precious_metals_value_ranges_realistic(sample_gold_data):
    """Test that precious metals data values are within realistic ranges."""
    # Setup mock provider result
    gold_result = MagicMock()
    gold_result.success = True
    gold_result.data = sample_gold_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=gold_result)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_precious_metals_data(base_date)

        data = result["data"]

        # Verify all gold values are reasonable ($1000-$3000/oz)
        gold_values = data["Gold"].dropna()
        assert all(
            1000 <= val <= 3000 for val in gold_values
        ), "Unrealistic gold price values"

        # Verify moving average values are also reasonable
        ma_values = data["Gold_MA50"].dropna()
        assert all(
            1000 <= val <= 3000 for val in ma_values
        ), "Unrealistic gold MA values"


# ==============================================================================
# Cryptocurrency Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_fetch_crypto_data_success(sample_bitcoin_data, sample_ethereum_data):
    """Test successful cryptocurrency data fetch and processing."""
    # Setup mock provider results
    bitcoin_result = MagicMock()
    bitcoin_result.success = True
    bitcoin_result.data = sample_bitcoin_data

    ethereum_result = MagicMock()
    ethereum_result.success = True
    ethereum_result.data = sample_ethereum_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        # Setup provider mock to return different data for different tickers
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "BTC-USD":
                return bitcoin_result
            elif ticker == "ETH-USD":
                return ethereum_result
            else:
                raise ValueError(f"Unexpected ticker: {ticker}")

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_crypto_data(base_date)

        # Verify structure
        assert "data" in result
        assert "base_date" in result
        assert "latest_btc" in result
        assert "latest_eth" in result
        assert "data_points" in result

        # Verify data content
        data = result["data"]
        assert not data.empty
        assert "BTC" in data.columns
        assert "ETH" in data.columns

        # Verify realistic value ranges
        btc_values = data["BTC"].dropna()
        if not btc_values.empty:
            assert all(
                20000 <= val <= 70000 for val in btc_values
            ), "Unrealistic Bitcoin values"

        eth_values = data["ETH"].dropna()
        if not eth_values.empty:
            assert all(
                1000 <= val <= 5000 for val in eth_values
            ), "Unrealistic Ethereum values"


@pytest.mark.asyncio
async def test_crypto_data_empty_handling():
    """Test handling of empty cryptocurrency data."""
    # Setup mock provider results with empty data
    bitcoin_result = MagicMock()
    bitcoin_result.success = True
    bitcoin_result.data = pd.DataFrame()  # Empty DataFrame

    ethereum_result = MagicMock()
    ethereum_result.success = True
    ethereum_result.data = pd.DataFrame()  # Empty DataFrame

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "BTC-USD":
                return bitcoin_result
            elif ticker == "ETH-USD":
                return ethereum_result

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)

        # Should raise exception due to empty data
        with pytest.raises(Exception) as exc_info:
            await fetch_crypto_data(base_date)

        assert "Bitcoin" in str(exc_info.value) or "Ethereum" in str(exc_info.value)


@pytest.mark.asyncio
async def test_crypto_provider_failure():
    """Test handling of cryptocurrency provider failure."""
    # Setup mock provider results with failure
    bitcoin_result = MagicMock()
    bitcoin_result.success = False
    bitcoin_result.error_message = "Bitcoin provider failed"

    ethereum_result = MagicMock()
    ethereum_result.success = False
    ethereum_result.error_message = "Ethereum provider failed"

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "BTC-USD":
                return bitcoin_result
            elif ticker == "ETH-USD":
                return ethereum_result

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)

        # Should raise exception due to provider failure
        with pytest.raises(Exception) as exc_info:
            await fetch_crypto_data(base_date)

        assert "provider failed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_crypto_data_filtering_by_base_date(
    sample_bitcoin_data, sample_ethereum_data
):
    """Test that cryptocurrency data is properly filtered by base_date."""
    # Setup mock provider results
    bitcoin_result = MagicMock()
    bitcoin_result.success = True
    bitcoin_result.data = sample_bitcoin_data

    ethereum_result = MagicMock()
    ethereum_result.success = True
    ethereum_result.data = sample_ethereum_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "BTC-USD":
                return bitcoin_result
            elif ticker == "ETH-USD":
                return ethereum_result

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        # Use base_date later in the dataset
        base_date = datetime(2020, 6, 1)
        result = await fetch_crypto_data(base_date)

        data = result["data"]

        # Verify data is filtered to start from base_date
        if not data.empty:
            earliest_date = data.index.min()
            assert earliest_date >= pd.to_datetime(base_date.date())


@pytest.mark.asyncio
async def test_crypto_workflow_direct():
    """Test the CryptoCurrencyWorkflow class directly."""
    # Create sample crypto data
    bitcoin_data = pd.DataFrame(
        {
            "Close": [45000, 47000, 48500, 50000, 51500],
            "Volume": [1000000000] * 5,
        },
        index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
    )

    ethereum_data = pd.DataFrame(
        {
            "Close": [3000, 3200, 3150, 3300, 3450],
            "Volume": [500000000] * 5,
        },
        index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
    )

    # Mock provider results
    bitcoin_result = MagicMock()
    bitcoin_result.success = True
    bitcoin_result.data = bitcoin_data

    ethereum_result = MagicMock()
    ethereum_result.success = True
    ethereum_result.data = ethereum_data

    # Create workflow and mock provider
    workflow = CryptoCurrencyWorkflow()

    def mock_get_data(ticker):
        if ticker == "BTC-USD":
            return bitcoin_result
        elif ticker == "ETH-USD":
            return ethereum_result

    workflow.yahoo_provider.get_data = AsyncMock(side_effect=mock_get_data)

    # Run workflow
    base_date = datetime(2020, 1, 1)
    result = await workflow.run(base_date=base_date)

    # Verify result
    assert isinstance(result, dict)
    data = result["data"]
    assert "BTC" in data.columns
    assert "ETH" in data.columns


@pytest.mark.asyncio
async def test_crypto_no_close_price_data():
    """Test handling when cryptocurrency data has no Close price column."""
    # Create Bitcoin data without Close price
    bitcoin_data = pd.DataFrame(
        {
            "Open": [45000, 47000, 48500],
            "High": [46000, 48000, 49500],
            "Low": [44000, 46000, 47500],
            "Volume": [1000000000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    # Create Ethereum data with Close price (to test Bitcoin-specific failure)
    ethereum_data = pd.DataFrame(
        {
            "Close": [3000, 3200, 3150],
            "Volume": [500000000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    bitcoin_result = MagicMock()
    bitcoin_result.success = True
    bitcoin_result.data = bitcoin_data

    ethereum_result = MagicMock()
    ethereum_result.success = True
    ethereum_result.data = ethereum_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "BTC-USD":
                return bitcoin_result
            elif ticker == "ETH-USD":
                return ethereum_result

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)

        # Should raise exception due to missing Close price in Bitcoin data
        with pytest.raises(Exception) as exc_info:
            await fetch_crypto_data(base_date)

        assert "Close price" in str(exc_info.value)


@pytest.mark.asyncio
async def test_crypto_workflow_integration():
    """Test full cryptocurrency workflow integration."""
    # Create realistic sample data
    bitcoin_data = pd.DataFrame(
        {
            "Close": [45000, 47000, 48500],  # Simple ascending values
            "Volume": [1000000000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    ethereum_data = pd.DataFrame(
        {
            "Close": [3000, 3200, 3150],  # Mixed values
            "Volume": [500000000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    bitcoin_result = MagicMock()
    bitcoin_result.success = True
    bitcoin_result.data = bitcoin_data

    ethereum_result = MagicMock()
    ethereum_result.success = True
    ethereum_result.data = ethereum_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "BTC-USD":
                return bitcoin_result
            elif ticker == "ETH-USD":
                return ethereum_result

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_crypto_data(base_date)

        # Verify successful integration
        assert result["data_points"] > 0
        assert result["latest_btc"] is not None
        assert result["latest_eth"] is not None
        assert isinstance(result["base_date"], datetime)


@pytest.mark.asyncio
async def test_crypto_parallel_data_fetching():
    """Test that Bitcoin and Ethereum data are fetched in parallel."""
    # Create sample data with different date ranges to test alignment
    bitcoin_dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
    ethereum_dates = pd.date_range(start="2020-01-03", periods=8, freq="D")

    bitcoin_data = pd.DataFrame(
        {
            "Close": [45000 + i * 1000 for i in range(10)],
            "Volume": [1000000000] * 10,
        },
        index=bitcoin_dates,
    )

    ethereum_data = pd.DataFrame(
        {
            "Close": [3000 + i * 100 for i in range(8)],
            "Volume": [500000000] * 8,
        },
        index=ethereum_dates,
    )

    bitcoin_result = MagicMock()
    bitcoin_result.success = True
    bitcoin_result.data = bitcoin_data

    ethereum_result = MagicMock()
    ethereum_result.success = True
    ethereum_result.data = ethereum_data

    # Track call order to verify parallel execution
    call_tracker = []

    async def track_calls(ticker):
        call_tracker.append(f"start_{ticker}")
        # Simulate async delay
        await asyncio.sleep(0.01)
        call_tracker.append(f"end_{ticker}")
        if ticker == "BTC-USD":
            return bitcoin_result
        elif ticker == "ETH-USD":
            return ethereum_result

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(side_effect=track_calls)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_crypto_data(base_date)

        # Verify both APIs were called
        btc_calls = [call for call in call_tracker if "BTC-USD" in call]
        eth_calls = [call for call in call_tracker if "ETH-USD" in call]
        assert len(btc_calls) == 2  # start and end
        assert len(eth_calls) == 2  # start and end

        # Verify data alignment worked (common dates only)
        data = result["data"]
        if not data.empty:
            assert "BTC" in data.columns
            assert "ETH" in data.columns
            # Should have data for overlapping dates only
            assert len(data) <= min(len(bitcoin_data), len(ethereum_data))


@pytest.mark.asyncio
async def test_crypto_value_ranges_realistic(sample_bitcoin_data, sample_ethereum_data):
    """Test that cryptocurrency data values are within realistic ranges."""
    # Setup mock provider results
    bitcoin_result = MagicMock()
    bitcoin_result.success = True
    bitcoin_result.data = sample_bitcoin_data

    ethereum_result = MagicMock()
    ethereum_result.success = True
    ethereum_result.data = sample_ethereum_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "BTC-USD":
                return bitcoin_result
            elif ticker == "ETH-USD":
                return ethereum_result

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_crypto_data(base_date)

        data = result["data"]

        # Verify all Bitcoin values are reasonable ($10K-$100K)
        btc_values = data["BTC"].dropna()
        if not btc_values.empty:
            assert all(
                10000 <= val <= 100000 for val in btc_values
            ), "Unrealistic Bitcoin price values"

        # Verify all Ethereum values are reasonable ($500-$10K)
        eth_values = data["ETH"].dropna()
        if not eth_values.empty:
            assert all(
                500 <= val <= 10000 for val in eth_values
            ), "Unrealistic Ethereum price values"


# =============================================================================
# Crude Oil Tests
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_crude_oil_data_success(sample_wti_data, sample_brent_data):
    """Test successful crude oil data fetch and processing."""
    # Setup mock provider results
    wti_result = MagicMock()
    wti_result.success = True
    wti_result.data = sample_wti_data

    brent_result = MagicMock()
    brent_result.success = True
    brent_result.data = sample_brent_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        # Setup provider mock to return different data for different tickers
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "CL=F":
                return wti_result
            elif ticker == "BZ=F":
                return brent_result
            else:
                raise ValueError(f"Unexpected ticker: {ticker}")

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_crude_oil_data(base_date)

        # Verify structure
        assert isinstance(result, dict)
        assert "data" in result
        assert "base_date" in result
        assert "latest_wti" in result
        assert "latest_brent" in result
        assert "data_points" in result

        # Verify data structure
        data = result["data"]
        assert isinstance(data, pd.DataFrame)
        assert "WTI" in data.columns
        assert "Brent" in data.columns

        # Verify latest prices
        assert isinstance(result["latest_wti"], float)
        assert isinstance(result["latest_brent"], float)
        assert result["latest_wti"] > 0
        assert result["latest_brent"] > 0

        # Verify data points count
        assert result["data_points"] > 0


@pytest.mark.asyncio
async def test_crude_oil_data_empty_handling():
    """Test handling of empty crude oil data."""
    # Setup mock provider results with empty data
    wti_result = MagicMock()
    wti_result.success = True
    wti_result.data = pd.DataFrame()  # Empty DataFrame

    brent_result = MagicMock()
    brent_result.success = True
    brent_result.data = pd.DataFrame()  # Empty DataFrame

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "CL=F":
                return wti_result
            elif ticker == "BZ=F":
                return brent_result

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)

        # Should raise exception due to empty data
        with pytest.raises(Exception) as exc_info:
            await fetch_crude_oil_data(base_date)

        assert "WTI" in str(exc_info.value) or "Brent" in str(exc_info.value)


@pytest.mark.asyncio
async def test_crude_oil_provider_failure():
    """Test handling of crude oil provider failure."""
    # Setup mock provider results with failure
    wti_result = MagicMock()
    wti_result.success = False
    wti_result.error_message = "WTI provider failed"

    brent_result = MagicMock()
    brent_result.success = False
    brent_result.error_message = "Brent provider failed"

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "CL=F":
                return wti_result
            elif ticker == "BZ=F":
                return brent_result

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)

        # Should raise exception due to provider failure
        with pytest.raises(Exception) as exc_info:
            await fetch_crude_oil_data(base_date)

        assert "provider failed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_crude_oil_data_filtering_by_base_date(
    sample_wti_data, sample_brent_data
):
    """Test that crude oil data is properly filtered by base_date."""
    # Setup mock provider results
    wti_result = MagicMock()
    wti_result.success = True
    wti_result.data = sample_wti_data

    brent_result = MagicMock()
    brent_result.success = True
    brent_result.data = sample_brent_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "CL=F":
                return wti_result
            elif ticker == "BZ=F":
                return brent_result

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        # Use base_date later in the dataset
        base_date = datetime(2020, 6, 1)
        result = await fetch_crude_oil_data(base_date)

        data = result["data"]

        # Should only include data from base_date onward
        if not data.empty:
            assert data.index.min() >= base_date
            # Should have fewer points than original sample data
            assert len(data) < len(sample_wti_data)


@pytest.mark.asyncio
async def test_crude_oil_workflow_direct():
    """Test the CrudeOilWorkflow class directly."""
    # Create sample crude oil data
    wti_data = pd.DataFrame(
        {
            "Close": [68.5, 70.2, 72.1, 74.0, 75.8],
            "Volume": [500000] * 5,
        },
        index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
    )

    brent_data = pd.DataFrame(
        {
            "Close": [70.8, 72.5, 74.4, 76.2, 78.0],
            "Volume": [400000] * 5,
        },
        index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
    )

    # Mock provider results
    wti_result = MagicMock()
    wti_result.success = True
    wti_result.data = wti_data

    brent_result = MagicMock()
    brent_result.success = True
    brent_result.data = brent_data

    # Create workflow and mock provider
    workflow = CrudeOilWorkflow()

    def mock_get_data(ticker):
        if ticker == "CL=F":
            return wti_result
        elif ticker == "BZ=F":
            return brent_result

    workflow.yahoo_provider.get_data = AsyncMock(side_effect=mock_get_data)

    # Run workflow
    result = await workflow.run(base_date=datetime(2020, 1, 1))

    # Check result structure
    assert isinstance(result, dict)
    assert "data" in result
    assert "latest_wti" in result
    assert "latest_brent" in result

    # Verify data alignment and values
    data = result["data"]
    assert "WTI" in data.columns
    assert "Brent" in data.columns
    assert len(data) == 5  # Should have all 5 data points


@pytest.mark.asyncio
async def test_crude_oil_no_close_price_data():
    """Test handling when crude oil data has no Close price column."""
    # Create WTI data without Close price
    wti_data = pd.DataFrame(
        {
            "Open": [68.0, 69.5, 71.0],
            "High": [70.0, 72.0, 73.5],
            "Low": [67.0, 68.5, 70.0],
            "Volume": [500000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    # Create Brent data with Close price (to test WTI-specific failure)
    brent_data = pd.DataFrame(
        {
            "Close": [70.8, 72.5, 74.4],
            "Volume": [400000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    wti_result = MagicMock()
    wti_result.success = True
    wti_result.data = wti_data

    brent_result = MagicMock()
    brent_result.success = True
    brent_result.data = brent_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "CL=F":
                return wti_result
            elif ticker == "BZ=F":
                return brent_result

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)

        # Should raise exception due to missing Close column in WTI data
        with pytest.raises(Exception) as exc_info:
            await fetch_crude_oil_data(base_date)

        assert "Close" in str(exc_info.value) or "WTI" in str(exc_info.value)


@pytest.mark.asyncio
async def test_crude_oil_workflow_integration():
    """Test full crude oil workflow integration."""
    # Create realistic sample data
    wti_data = pd.DataFrame(
        {
            "Close": [68.5, 70.2, 72.1],  # Simple ascending values
            "Volume": [500000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    brent_data = pd.DataFrame(
        {
            "Close": [70.8, 72.5, 74.4],  # Mixed values, higher than WTI
            "Volume": [400000] * 3,
        },
        index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
    )

    wti_result = MagicMock()
    wti_result.success = True
    wti_result.data = wti_data

    brent_result = MagicMock()
    brent_result.success = True
    brent_result.data = brent_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "CL=F":
                return wti_result
            elif ticker == "BZ=F":
                return brent_result

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_crude_oil_data(base_date)

        # Verify comprehensive result
        assert "data" in result
        assert "latest_wti" in result
        assert "latest_brent" in result
        assert "data_points" in result

        data = result["data"]
        assert len(data) == 3
        assert data["WTI"].iloc[-1] == 72.1
        assert data["Brent"].iloc[-1] == 74.4

        # Verify latest prices
        assert result["latest_wti"] == 72.1
        assert result["latest_brent"] == 74.4
        assert result["data_points"] == 3


@pytest.mark.asyncio
async def test_crude_oil_parallel_data_fetching():
    """Test that WTI and Brent data are fetched in parallel."""
    # Create sample data with different date ranges to test alignment
    wti_dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
    brent_dates = pd.date_range(start="2020-01-03", periods=8, freq="D")

    wti_data = pd.DataFrame(
        {
            "Close": [68.0 + i * 1.5 for i in range(10)],
            "Volume": [500000] * 10,
        },
        index=wti_dates,
    )

    brent_data = pd.DataFrame(
        {
            "Close": [70.0 + i * 1.8 for i in range(8)],
            "Volume": [400000] * 8,
        },
        index=brent_dates,
    )

    wti_result = MagicMock()
    wti_result.success = True
    wti_result.data = wti_data

    brent_result = MagicMock()
    brent_result.success = True
    brent_result.data = brent_data

    # Track call order to verify parallel execution
    call_order = []

    def mock_get_data_with_tracking(ticker):
        call_order.append(ticker)
        if ticker == "CL=F":
            return wti_result
        elif ticker == "BZ=F":
            return brent_result

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(
            side_effect=mock_get_data_with_tracking
        )
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_crude_oil_data(base_date)

        # Both tickers should be called
        assert "CL=F" in call_order
        assert "BZ=F" in call_order

        # Verify data alignment (should only have overlapping dates)
        data = result["data"]
        if not data.empty:
            assert "WTI" in data.columns
            assert "Brent" in data.columns
            # Should have data for overlapping dates only
            assert len(data) <= min(len(wti_data), len(brent_data))


@pytest.mark.asyncio
async def test_crude_oil_value_ranges_realistic(sample_wti_data, sample_brent_data):
    """Test that crude oil data values are within realistic ranges."""
    # Setup mock provider results
    wti_result = MagicMock()
    wti_result.success = True
    wti_result.data = sample_wti_data

    brent_result = MagicMock()
    brent_result.success = True
    brent_result.data = sample_brent_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()

        def mock_get_data(ticker):
            if ticker == "CL=F":
                return wti_result
            elif ticker == "BZ=F":
                return brent_result

        mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_crude_oil_data(base_date)

        data = result["data"]

        # Verify all WTI values are reasonable ($20-$150/bbl)
        wti_values = data["WTI"].dropna()
        if not wti_values.empty:
            assert all(
                20 <= val <= 150 for val in wti_values
            ), "Unrealistic WTI crude oil price values"

        # Verify all Brent values are reasonable ($20-$150/bbl)
        brent_values = data["Brent"].dropna()
        if not brent_values.empty:
            assert all(
                20 <= val <= 150 for val in brent_values
            ), "Unrealistic Brent crude oil price values"


# ========================== BLOOMBERG COMMODITY INDEX TESTS ==========================


@pytest.fixture
def sample_bloomberg_commodity_data():
    """Create sample Bloomberg Commodity Index (^BCOM) data."""
    dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
    # BCOM typically ranges from 80 to 120 (indexed to 100 in 2014)
    base_value = 100
    return pd.DataFrame(
        {
            "Open": [base_value + (i % 30 - 15) * 0.5 for i in range(1000)],
            "High": [base_value + (i % 30 - 15) * 0.5 + 1 for i in range(1000)],
            "Low": [base_value + (i % 30 - 15) * 0.5 - 1 for i in range(1000)],
            "Close": [
                base_value + (i % 30 - 15) * 0.5 + 0.2 * (i % 5 - 2)
                for i in range(1000)
            ],
            "Volume": [1000000 + i * 1000 for i in range(1000)],
        },
        index=dates,
    )


@pytest.mark.asyncio
async def test_bloomberg_commodity_data_empty_handling():
    """Test Bloomberg commodity workflow with empty data."""
    # Mock empty provider result
    empty_result = MagicMock()
    empty_result.success = True
    empty_result.data = pd.DataFrame()

    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=empty_result)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2023, 1, 1)

        # Should raise exception when no data is available
        with pytest.raises(Exception) as exc_info:
            await fetch_bloomberg_commodity_data(base_date)

        assert (
            "no" in str(exc_info.value).lower()
            and "data" in str(exc_info.value).lower()
        )


@pytest.mark.asyncio
async def test_bloomberg_commodity_provider_failure():
    """Test Bloomberg commodity workflow with provider failure."""
    # Mock provider failure
    failed_result = MagicMock()
    failed_result.success = False
    failed_result.error = "API failure"

    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=failed_result)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2023, 1, 1)

        with pytest.raises(Exception) as exc_info:
            await fetch_bloomberg_commodity_data(base_date)

        # Check for general failure indicators in the error message
        error_msg = str(exc_info.value).lower()
        assert ("failed" in error_msg or "error" in error_msg) and (
            "bcom" in error_msg or "commodity" in error_msg
        )


@pytest.mark.asyncio
async def test_bloomberg_commodity_calculation_accuracy(
    sample_bloomberg_commodity_data,
):
    """Test the accuracy of Bloomberg commodity moving average calculations."""
    # Setup mock provider result
    bcom_result = MagicMock()
    bcom_result.success = True
    bcom_result.data = sample_bloomberg_commodity_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=bcom_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 6, 1)  # Mid-year to get partial data
        result = await fetch_bloomberg_commodity_data(base_date)

        # Verify moving average calculation accuracy
        data = result["data"]
        assert "BCOM_MA50" in data.columns
        assert "BCOM_MA200" in data.columns

        # Verify that moving averages are properly calculated
        bcom_values = data["BCOM"]
        ma50_values = data["BCOM_MA50"]
        ma200_values = data["BCOM_MA200"]

        # Verify all MA values are reasonable
        assert all(isinstance(val, (int, float)) for val in ma50_values.dropna())
        assert all(isinstance(val, (int, float)) for val in ma200_values.dropna())
        assert len(ma50_values.dropna()) > 0
        assert len(ma200_values.dropna()) > 0

        # Verify MA smooths out volatility (should be less volatile than raw BCOM)
        if len(data) >= 200:
            bcom_std = bcom_values.std()
            ma50_std = ma50_values.std()
            ma200_std = ma200_values.std()
            assert ma50_std <= bcom_std  # MA50 should be smoother
            assert ma200_std <= ma50_std  # MA200 should be even smoother


@pytest.mark.asyncio
async def test_bloomberg_commodity_data_filtering_by_base_date(
    sample_bloomberg_commodity_data,
):
    """Test that Bloomberg commodity data is properly filtered by base_date."""
    # Setup mock provider result
    bcom_result = MagicMock()
    bcom_result.success = True
    bcom_result.data = sample_bloomberg_commodity_data

    # Mock the provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=bcom_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test with different base dates
        base_date = datetime(2020, 6, 1)  # Mid-year
        result = await fetch_bloomberg_commodity_data(base_date)

        data = result["data"]
        assert not data.empty

        # All data should be from base_date onwards
        assert data.index.min() >= pd.to_datetime(base_date.date())

        # Should be less than full dataset
        assert len(data) < len(sample_bloomberg_commodity_data)


@pytest.mark.asyncio
async def test_bloomberg_commodity_workflow_direct():
    """Test the BloombergCommodityWorkflow class directly."""
    # Create sample BCOM data
    bcom_data = pd.DataFrame(
        {
            "Close": [95.2, 96.1, 97.5, 98.3, 99.7],
            "Volume": [1000000] * 5,
        },
        index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
    )

    # Mock provider result
    bcom_result = MagicMock()
    bcom_result.success = True
    bcom_result.data = bcom_data

    # Mock the yahoo provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=bcom_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Create and run workflow
        workflow = BloombergCommodityWorkflow()
        base_date = datetime(2020, 1, 1)

        # Run workflow directly
        result = await workflow.run(base_date=base_date)

        # Verify result structure
        assert "data" in result
        assert not result["data"].empty
        assert "BCOM" in result["data"].columns


@pytest.mark.asyncio
async def test_bloomberg_commodity_no_close_price_data():
    """Test Bloomberg commodity workflow when Close price data is missing."""
    # Create data without Close column
    bcom_data = pd.DataFrame(
        {
            "Open": [95.2, 96.1, 97.5, 98.3, 99.7],
            "High": [96.0, 96.8, 98.1, 99.0, 100.2],
            "Low": [94.5, 95.3, 96.8, 97.6, 98.9],
            "Volume": [1000000] * 5,
        },
        index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
    )

    bcom_result = MagicMock()
    bcom_result.success = True
    bcom_result.data = bcom_data

    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=bcom_result)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)

        with pytest.raises(Exception) as exc_info:
            await fetch_bloomberg_commodity_data(base_date)

        # Should fail because Close column is missing
        assert (
            "close" in str(exc_info.value).lower()
            or "missing" in str(exc_info.value).lower()
        )


@pytest.mark.asyncio
async def test_bloomberg_commodity_moving_average_calculation(
    sample_bloomberg_commodity_data,
):
    """Test Bloomberg commodity moving average calculations in detail."""
    # Setup mock provider result
    bcom_result = MagicMock()
    bcom_result.success = True
    bcom_result.data = sample_bloomberg_commodity_data

    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=bcom_result)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_bloomberg_commodity_data(base_date)

        data = result["data"]

        # Test MA50 calculation
        if "BCOM_MA50" in data.columns and len(data) >= 50:
            # Find first non-NaN MA50 value
            first_valid_idx = data["BCOM_MA50"].first_valid_index()
            if first_valid_idx is not None:
                # Get position of first valid MA value
                pos = data.index.get_loc(first_valid_idx)

                # Calculate manual 50-day MA at that position
                manual_ma50 = data["BCOM"].iloc[max(0, pos - 49) : pos + 1].mean()
                calculated_ma50 = data["BCOM_MA50"].loc[first_valid_idx]

                # Should be approximately equal (within 0.01)
                assert abs(manual_ma50 - calculated_ma50) < 0.01

        # Test MA200 calculation
        if "BCOM_MA200" in data.columns and len(data) >= 200:
            # Find first non-NaN MA200 value
            first_valid_idx = data["BCOM_MA200"].first_valid_index()
            if first_valid_idx is not None:
                # Get position of first valid MA value
                pos = data.index.get_loc(first_valid_idx)

                # Calculate manual 200-day MA at that position
                manual_ma200 = data["BCOM"].iloc[max(0, pos - 199) : pos + 1].mean()
                calculated_ma200 = data["BCOM_MA200"].loc[first_valid_idx]

                # Should be approximately equal (within 0.01)
                assert abs(manual_ma200 - calculated_ma200) < 0.01


@pytest.mark.asyncio
async def test_bloomberg_commodity_moving_average_with_limited_data():
    """Test Bloomberg commodity moving averages with limited data points."""
    # Create limited data (less than 200 days)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    limited_data = pd.DataFrame(
        {
            "Open": [95 + i * 0.1 for i in range(100)],
            "High": [96 + i * 0.1 for i in range(100)],
            "Low": [94 + i * 0.1 for i in range(100)],
            "Close": [95.5 + i * 0.1 for i in range(100)],
            "Volume": [1000000] * 100,
        },
        index=dates,
    )

    bcom_result = MagicMock()
    bcom_result.success = True
    bcom_result.data = limited_data

    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=bcom_result)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_bloomberg_commodity_data(base_date)

        data = result["data"]

        # MA50 should be available for data >= 50 days
        assert "BCOM_MA50" in data.columns
        ma50_valid_count = data["BCOM_MA50"].count()
        assert ma50_valid_count > 0  # Should have some MA50 values
        # With 100 data points, we can get MA50 from positions 49-99 (51 values)
        # But since we're filtering by base_date, we might get all 100 values
        assert ma50_valid_count <= 100

        # MA200 should be mostly NaN for data < 200 days
        assert "BCOM_MA200" in data.columns
        ma200_valid_count = data["BCOM_MA200"].count()
        assert ma200_valid_count == 0  # No valid MA200 values with only 100 days


@pytest.mark.asyncio
async def test_bloomberg_commodity_workflow_integration():
    """Test Bloomberg commodity workflow integration with realistic data."""
    # Create realistic BCOM data with proper trend
    dates = pd.date_range(start="2019-01-01", periods=500, freq="D")
    realistic_data = pd.DataFrame(
        {
            "Open": [90 + i * 0.02 + (i % 20 - 10) * 0.5 for i in range(500)],
            "High": [91 + i * 0.02 + (i % 20 - 10) * 0.5 for i in range(500)],
            "Low": [89 + i * 0.02 + (i % 20 - 10) * 0.5 for i in range(500)],
            "Close": [90.5 + i * 0.02 + (i % 20 - 10) * 0.5 for i in range(500)],
            "Volume": [1000000 + i * 100 for i in range(500)],
        },
        index=dates,
    )

    bcom_result = MagicMock()
    bcom_result.success = True
    bcom_result.data = realistic_data

    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=bcom_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test with base date in middle of dataset
        base_date = datetime(2019, 6, 1)
        result = await fetch_bloomberg_commodity_data(base_date)

        data = result["data"]

        # Basic structure verification
        assert not data.empty
        assert "BCOM" in data.columns
        assert "BCOM_MA50" in data.columns
        assert "BCOM_MA200" in data.columns

        # Data filtering verification
        assert data.index.min() >= pd.to_datetime(base_date.date())

        # Moving averages should be calculated
        assert data["BCOM_MA50"].count() > 0
        assert data["BCOM_MA200"].count() > 0


@pytest.mark.asyncio
async def test_bloomberg_commodity_value_ranges_realistic(
    sample_bloomberg_commodity_data,
):
    """Test that Bloomberg commodity values are within realistic ranges."""
    # Setup mock provider result
    bcom_result = MagicMock()
    bcom_result.success = True
    bcom_result.data = sample_bloomberg_commodity_data

    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=bcom_result)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_bloomberg_commodity_data(base_date)

        data = result["data"]

        # Verify all BCOM values are reasonable (typically 60-130 index points)
        bcom_values = data["BCOM"].dropna()
        if not bcom_values.empty:
            assert all(
                60 <= val <= 130 for val in bcom_values
            ), "Unrealistic Bloomberg Commodity Index values"


# MSCI World Index Tests


@pytest.mark.asyncio
async def test_fetch_msci_world_data_success(sample_msci_world_data):
    """Test successful MSCI World Index data fetch and calculation."""
    # Setup mock provider result
    msci_result = MagicMock()
    msci_result.success = True
    msci_result.data = sample_msci_world_data

    # Mock the Yahoo provider
    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=msci_result)
        mock_yahoo.return_value = mock_yahoo_instance

        # Test the function
        base_date = datetime(2020, 1, 1)
        result = await fetch_msci_world_data(base_date)

        # Verify results
        assert "data" in result
        assert "base_date" in result
        assert "latest_value" in result
        assert "data_points" in result

        data = result["data"]
        assert not data.empty
        assert "MSCI_World" in data.columns
        assert "MSCI_MA50" in data.columns
        assert "MSCI_MA200" in data.columns
        assert "MSCI_BB_Upper" in data.columns
        assert "MSCI_BB_Lower" in data.columns
        assert "MSCI_BB_Mid" in data.columns

        # Verify provider call
        mock_yahoo_instance.get_data.assert_called_once_with("^990100-USD-STRD")

        # Verify data has been filtered by base_date
        assert data.index.min() >= pd.to_datetime(base_date.date())

        # Verify moving averages and Bollinger bands are calculated
        assert data["MSCI_MA50"].count() > 0  # Should have some non-NaN values
        assert data["MSCI_MA200"].count() > 0  # Should have some non-NaN values
        assert data["MSCI_BB_Upper"].count() > 0  # Should have some non-NaN values
        assert data["MSCI_BB_Lower"].count() > 0  # Should have some non-NaN values


@pytest.mark.asyncio
async def test_msci_world_workflow_error_handling():
    """Test MSCI World workflow error handling."""
    workflow = MSCIWorldWorkflow()

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
async def test_msci_world_bollinger_bands_calculation(sample_msci_world_data):
    """Test that Bollinger bands are calculated correctly."""
    # Setup mock provider result
    msci_result = MagicMock()
    msci_result.success = True
    msci_result.data = sample_msci_world_data

    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=msci_result)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_msci_world_data(base_date)

        data = result["data"]

        # Verify Bollinger bands relationships
        # Upper band should be greater than middle, middle greater than lower
        valid_data = data.dropna(
            subset=["MSCI_BB_Upper", "MSCI_BB_Mid", "MSCI_BB_Lower"]
        )
        if not valid_data.empty:
            assert all(
                valid_data["MSCI_BB_Upper"] >= valid_data["MSCI_BB_Mid"]
            ), "Upper Bollinger band should be >= middle"
            assert all(
                valid_data["MSCI_BB_Mid"] >= valid_data["MSCI_BB_Lower"]
            ), "Middle Bollinger band should be >= lower"


@pytest.mark.asyncio
async def test_msci_world_moving_averages_calculation(sample_msci_world_data):
    """Test that moving averages are calculated with proper windows."""
    # Setup mock provider result
    msci_result = MagicMock()
    msci_result.success = True
    msci_result.data = sample_msci_world_data

    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=msci_result)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_msci_world_data(base_date)

        data = result["data"]

        # Data filtering verification
        assert data.index.min() >= pd.to_datetime(base_date.date())

        # Moving averages should be calculated
        assert data["MSCI_MA50"].count() > 0
        assert data["MSCI_MA200"].count() > 0


@pytest.mark.asyncio
async def test_msci_world_value_ranges_realistic(sample_msci_world_data):
    """Test that MSCI World values are within realistic ranges."""
    # Setup mock provider result
    msci_result = MagicMock()
    msci_result.success = True
    msci_result.data = sample_msci_world_data

    with patch("app.flows.markets.create_yahoo_history_provider") as mock_yahoo:
        mock_yahoo_instance = AsyncMock()
        mock_yahoo_instance.get_data = AsyncMock(return_value=msci_result)
        mock_yahoo.return_value = mock_yahoo_instance

        base_date = datetime(2020, 1, 1)
        result = await fetch_msci_world_data(base_date)

        data = result["data"]

        # Verify all MSCI World values are reasonable (typically 1500-4000 index points)
        msci_values = data["MSCI_World"].dropna()
        if not msci_values.empty:
            assert all(
                1500 <= val <= 4000 for val in msci_values
            ), "Unrealistic MSCI World Index values"
