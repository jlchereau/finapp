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
    fetch_vix_data,
    VIXWorkflow,
    fetch_yield_curve_data,
    YieldCurveWorkflow,
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
            assert "No VIX data available" in str(e)


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
