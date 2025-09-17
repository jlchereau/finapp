"""
Test yield curve workflow functionality.

Tests for the YieldCurveWorkflow class and fetch_yield_curve_data function,
verifying proper data processing and error handling for US Treasury yield curves.
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, AsyncMock

from app.flows.markets.yield_curve import (
    fetch_yield_curve_data,
    YieldCurveWorkflow,
    YIELD_CURVE_SERIES,
)
from app.providers.base import ProviderResult, ProviderType


# Yield Curve Tests


def create_mock_series_provider(sample_data):
    """Helper function to create a mock FRED provider for testing."""
    # Create individual series data
    series_data = {}
    for series_id, maturity_label in YIELD_CURVE_SERIES.items():
        if maturity_label in sample_data.columns:
            series_df = pd.DataFrame(
                {"value": sample_data[maturity_label]}, index=sample_data.index
            )
            series_data[series_id] = series_df
        else:
            # Return empty DataFrame if maturity not present
            series_data[series_id] = pd.DataFrame()

    # Mock get_data for individual series
    async def mock_get_data(query, **kwargs):
        data = series_data.get(query, pd.DataFrame())
        if not data.empty:
            return ProviderResult(
                success=True,
                data=data,
                provider_type=ProviderType.FRED_SERIES,
                query=query,
            )
        else:
            return ProviderResult(
                success=False,
                error_message="No data found",
                provider_type=ProviderType.FRED_SERIES,
                query=query,
            )

    mock_fred_instance = AsyncMock()
    mock_fred_instance.get_data = AsyncMock(side_effect=mock_get_data)
    return mock_fred_instance


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
    # Mock the FRED provider
    with patch("app.flows.markets.yield_curve.create_fred_series_provider") as mock_fred:
        # Setup provider mock
        mock_fred_instance = AsyncMock()

        # Create individual series data for each Treasury series
        series_data = {}
        for series_id, maturity_label in YIELD_CURVE_SERIES.items():
            # Create a subset of the sample data for this maturity
            series_df = pd.DataFrame(
                {"value": sample_yield_curve_data[maturity_label]},
                index=sample_yield_curve_data.index,
            )
            series_data[series_id] = series_df

        # Mock get_data to return appropriate ProviderResult for each series
        async def mock_get_data(query, **kwargs):
            data = series_data.get(query, pd.DataFrame())
            if not data.empty:
                return ProviderResult(
                    success=True,
                    data=data,
                    provider_type=ProviderType.FRED_SERIES,
                    query=query,
                )
            else:
                return ProviderResult(
                    success=False,
                    error_message="No data found",
                    provider_type=ProviderType.FRED_SERIES,
                    query=query,
                )

        mock_fred_instance.get_data = AsyncMock(side_effect=mock_get_data)
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

        # Verify provider calls - should be called for each Treasury series
        assert mock_fred_instance.get_data.call_count == len(YIELD_CURVE_SERIES)


@pytest.mark.asyncio
async def test_fetch_yield_curve_data_fred_error():
    """Test yield curve data fetch with FRED provider error."""
    # Mock the FRED provider to raise exception
    with patch("app.flows.markets.yield_curve.create_fred_series_provider") as mock_fred:
        mock_fred_instance = AsyncMock()
        # Return failed ProviderResult instead of exception
        mock_fred_instance.get_data = AsyncMock(
            return_value=ProviderResult(
                success=False,
                error_message="FRED API error",
                provider_type=ProviderType.FRED_SERIES,
            )
        )
        mock_fred.return_value = mock_fred_instance

        # Test the function and expect exception
        base_date = datetime(2024, 1, 1)

        with pytest.raises(Exception) as exc_info:
            await fetch_yield_curve_data(base_date)

        assert "No yield curve data retrieved" in str(exc_info.value)


@pytest.mark.asyncio
async def test_yield_curve_data_empty_handling():
    """Test yield curve data fetch with empty data."""
    # Mock the FRED provider to return empty data for all series
    with patch("app.flows.markets.yield_curve.create_fred_series_provider") as mock_fred:
        mock_fred_instance = AsyncMock()
        mock_fred_instance.get_data = AsyncMock(
            return_value=ProviderResult(
                success=False,
                error_message="No data",
                provider_type=ProviderType.FRED_SERIES,
            )
        )
        mock_fred.return_value = mock_fred_instance

        # Test the function
        base_date = datetime(2024, 1, 1)

        with pytest.raises(Exception) as exc_info:
            await fetch_yield_curve_data(base_date)

        assert "No yield curve data retrieved" in str(exc_info.value)


@pytest.mark.asyncio
async def test_yield_curve_workflow_direct():
    """Test the YieldCurveWorkflow class directly."""
    # Create sample yield curve data for individual series
    sample_data = pd.DataFrame(
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
    with patch("app.flows.markets.yield_curve.create_fred_series_provider") as mock_fred:
        mock_fred_instance = AsyncMock()

        # Create individual series data
        series_data = {}
        for series_id, maturity_label in YIELD_CURVE_SERIES.items():
            series_df = pd.DataFrame(
                {"value": sample_data[maturity_label]}, index=sample_data.index
            )
            series_data[series_id] = series_df

        # Mock get_data for individual series
        async def mock_get_data(query, **kwargs):
            data = series_data.get(query, pd.DataFrame())
            if not data.empty:
                return ProviderResult(
                    success=True,
                    data=data,
                    provider_type=ProviderType.FRED_SERIES,
                    query=query,
                )
            else:
                return ProviderResult(
                    success=False,
                    error_message="No data found",
                    provider_type=ProviderType.FRED_SERIES,
                    query=query,
                )

        mock_fred_instance.get_data = AsyncMock(side_effect=mock_get_data)
        mock_fred.return_value = mock_fred_instance

        # Test the workflow directly using FlowRunner
        from app.flows.base import FlowRunner

        workflow = YieldCurveWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)
        base_date = datetime(2024, 1, 1)

        result = await runner.run(base_date=base_date)

        # Verify result structure - should be FlowResult from FlowRunner
        assert result.success
        assert result.data is not None

        # Verify data integrity
        data = result.data
        assert not data.empty
        assert len(data) == 2  # 2 days of data
        assert len(data.columns) == 8  # 8 maturities

        # Verify metadata
        metadata = result.metadata
        assert "latest_date" in metadata
        assert "maturities" in metadata
        assert "data_points" in metadata


@pytest.mark.asyncio
async def test_yield_curve_data_filtering_by_base_date(sample_yield_curve_data):
    """Test yield curve data is properly filtered by base date."""
    # Mock the FRED provider
    with patch("app.flows.markets.yield_curve.create_fred_series_provider") as mock_fred:
        mock_fred_instance = create_mock_series_provider(sample_yield_curve_data)
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
    with patch("app.flows.markets.yield_curve.create_fred_series_provider") as mock_fred:
        mock_fred.side_effect = Exception("Provider creation failed")

        base_date = datetime(2024, 1, 1)

        with pytest.raises(Exception) as exc_info:
            await fetch_yield_curve_data(base_date)

        assert "Provider creation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_yield_curve_maturities_order(sample_yield_curve_data):
    """Test that yield curve maturities are returned in correct order."""
    # Mock the FRED provider
    with patch("app.flows.markets.yield_curve.create_fred_series_provider") as mock_fred:
        mock_fred_instance = create_mock_series_provider(sample_yield_curve_data)
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
    with patch("app.flows.markets.yield_curve.create_fred_series_provider") as mock_fred:
        mock_fred_instance = create_mock_series_provider(sample_yield_curve_data)
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
    sample_data = pd.DataFrame(
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
    with patch("app.flows.markets.yield_curve.create_fred_series_provider") as mock_fred:
        mock_fred_instance = create_mock_series_provider(sample_data)
        mock_fred.return_value = mock_fred_instance

        # Test the full workflow using FlowRunner
        from app.flows.base import FlowRunner

        workflow = YieldCurveWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)
        base_date = datetime(2024, 1, 1)

        result = await runner.run(base_date=base_date)

        # Verify complete result structure - should be FlowResult
        assert result.success
        assert result.data is not None

        # Verify data quality
        data = result.data
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert len(data.columns) == 8

        # Verify metadata
        metadata = result.metadata
        assert "latest_date" in metadata
        assert "maturities" in metadata
        assert "data_points" in metadata

        # Verify latest date is properly set
        latest_date = metadata["latest_date"]
        assert latest_date == data.index.max()

        # Verify data points count
        assert metadata["data_points"] == len(data)
