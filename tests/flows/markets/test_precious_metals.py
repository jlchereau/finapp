"""
Unit tests for Precious Metals workflow and FlowRunner integration.
"""

import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from app.flows.markets.precious_metals import (
    fetch_precious_metals_data,
    PreciousMetalsWorkflow,
)
from app.flows.base import FlowRunner, FlowResultEvent


@pytest.fixture
def sample_gold_data():
    """Create sample gold futures data."""
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    # Create realistic gold prices (typically $1200-$2000 range)
    gold_values = [1500 + (i % 100) * 5 + (i // 100) * 10 for i in range(365)]
    return pd.DataFrame(
        {
            "Open": [v - 10 for v in gold_values],
            "High": [v + 20 for v in gold_values],
            "Low": [v - 20 for v in gold_values],
            "Close": gold_values,
            "Adj Close": gold_values,
            "Volume": [100000 + i * 100 for i in range(365)],
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


class TestPreciousMetalsWorkflow:
    """Test the PreciousMetalsWorkflow class."""

    def test_workflow_initialization(self):
        """Test that workflow initializes correctly."""
        workflow = PreciousMetalsWorkflow()
        assert workflow is not None
        assert hasattr(workflow, "yahoo_provider")

    @pytest.mark.asyncio
    async def test_fetch_and_process_gold_data_success(self):
        """Test successful gold data fetching and processing step."""
        workflow = PreciousMetalsWorkflow()

        # Create mock gold data
        gold_data = pd.DataFrame(
            {"Close": [1500.0, 1520.0, 1540.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        # Mock provider result
        gold_result = MagicMock()
        gold_result.success = True
        gold_result.data = gold_data

        # Mock the provider method
        workflow.yahoo_provider.get_data = AsyncMock(return_value=gold_result)

        # Create start event
        start_event = MagicMock()
        start_event.base_date = datetime(2020, 1, 1)

        # Execute the step
        result = await workflow.fetch_and_process_gold_data(start_event)

        # Verify result
        assert isinstance(result, FlowResultEvent)
        assert result.success is True
        assert result.data is not None
        assert not result.data.empty
        assert "Gold" in result.data.columns
        assert "Gold_MA50" in result.data.columns
        assert "Gold_MA200" in result.data.columns

    @pytest.mark.asyncio
    async def test_fetch_and_process_gold_data_provider_failure(self):
        """Test handling of Yahoo Finance provider failure."""
        workflow = PreciousMetalsWorkflow()

        # Mock provider result with failure
        gold_result = MagicMock()
        gold_result.success = False
        gold_result.error_message = "Gold provider failed"

        # Mock the provider method
        workflow.yahoo_provider.get_data = AsyncMock(return_value=gold_result)

        # Create start event
        start_event = MagicMock()
        start_event.base_date = datetime(2020, 1, 1)

        # Execute the step and expect exception
        with pytest.raises(Exception) as exc_info:
            await workflow.fetch_and_process_gold_data(start_event)

        assert "Gold data fetch failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_and_process_gold_data_no_close_column(self):
        """Test error handling when gold data has no Close column."""
        workflow = PreciousMetalsWorkflow()

        # Create gold data without Close column
        gold_data = pd.DataFrame({"Open": [1500.0, 1520.0, 1540.0]})

        # Mock provider result
        gold_result = MagicMock()
        gold_result.success = True
        gold_result.data = gold_data

        # Mock the provider method
        workflow.yahoo_provider.get_data = AsyncMock(return_value=gold_result)

        # Create start event
        start_event = MagicMock()
        start_event.base_date = datetime(2020, 1, 1)

        # Execute the step and expect exception
        with pytest.raises(Exception) as exc_info:
            await workflow.fetch_and_process_gold_data(start_event)

        assert "No Close price data available for gold" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_and_process_gold_data_empty_data(self):
        """Test error handling when gold data is empty."""
        workflow = PreciousMetalsWorkflow()

        # Create empty gold data
        gold_data = pd.DataFrame()

        # Mock provider result
        gold_result = MagicMock()
        gold_result.success = True
        gold_result.data = gold_data

        # Mock the provider method
        workflow.yahoo_provider.get_data = AsyncMock(return_value=gold_result)

        # Create start event
        start_event = MagicMock()
        start_event.base_date = datetime(2020, 1, 1)

        # Execute the step and expect exception
        with pytest.raises(Exception) as exc_info:
            await workflow.fetch_and_process_gold_data(start_event)

        assert "No gold data available" in str(exc_info.value)


class TestPreciousMetalsFlowRunner:
    """Test the Precious Metals workflow with FlowRunner."""

    @pytest.mark.asyncio
    async def test_flowrunner_integration_success(self):
        """Test successful workflow execution with FlowRunner."""
        workflow = PreciousMetalsWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        # Mock the workflow to return a FlowResultEvent
        mock_result = FlowResultEvent.success_result(
            data=pd.DataFrame(
                {
                    "Gold": [1500.0, 1520.0],
                    "Gold_MA50": [1500.0, 1510.0],
                    "Gold_MA200": [1485.0, 1495.0],
                }
            ),
            metadata={
                "latest_value": 1520.0,
                "data_points": 2,
            },
        )

        # Mock workflow.run to return the mock result
        workflow.run = AsyncMock(return_value=mock_result)

        # Execute workflow through FlowRunner
        result = await runner.run(base_date=datetime(2020, 1, 1))

        # Verify FlowResultEvent structure
        assert isinstance(result, FlowResultEvent)
        assert result.success is True
        assert result.data is not None
        assert not result.data.empty
        assert "Gold" in result.data.columns
        assert "Gold_MA50" in result.data.columns
        assert "Gold_MA200" in result.data.columns

    @pytest.mark.asyncio
    async def test_flowrunner_integration_workflow_exception(self):
        """Test FlowRunner handling of Exception."""
        workflow = PreciousMetalsWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        # Mock workflow to raise Exception
        workflow.run = AsyncMock(side_effect=Exception("Test gold workflow error"))

        # Execute and expect FlowResultEvent with error
        result = await runner.run(base_date=datetime(2020, 1, 1))

        assert isinstance(result, FlowResultEvent)
        assert result.success is False
        assert result.data is None
        assert (
            result.error_message is not None
            and "Test gold workflow error" in result.error_message
        )


class TestFetchPreciousMetalsData:
    """Test the fetch_precious_metals_data function."""

    @pytest.mark.asyncio
    async def test_fetch_precious_metals_data_success(self, sample_gold_data):
        """Test successful precious metals data fetch and calculation."""
        # Setup mock provider result
        gold_result = MagicMock()
        gold_result.success = True
        gold_result.data = sample_gold_data

        # Mock the provider
        with patch(
            "app.flows.markets.precious_metals.create_yahoo_history_provider"
        ) as mock_yahoo:
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
            assert "Gold_MA200" in data.columns

            # Verify latest value is within reasonable range
            latest_value = result["latest_value"]
            assert isinstance(latest_value, (float, int)) or hasattr(
                latest_value, "dtype"
            )
            assert 1000 <= latest_value <= 3000  # Reasonable gold price range

            # Verify moving averages are calculated
            ma50_values = data["Gold_MA50"].dropna()
            ma200_values = data["Gold_MA200"].dropna()

            # Moving averages should have some non-null values
            if len(data) >= 50:
                assert not ma50_values.empty
            if len(data) >= 200:
                assert not ma200_values.empty

            # Verify provider call
            mock_yahoo_instance.get_data.assert_called_once_with("GC=F")

    @pytest.mark.asyncio
    async def test_fetch_precious_metals_data_yahoo_error(self):
        """Test handling of Yahoo Finance errors for gold."""
        # Test the error handling within the workflow
        workflow = PreciousMetalsWorkflow()

        # Mock the Yahoo provider to fail
        with patch.object(workflow, "yahoo_provider") as mock_yahoo:
            mock_yahoo.get_data = AsyncMock(
                side_effect=Exception("Yahoo Gold API error")
            )

            # Test the workflow directly
            try:
                await workflow.run(base_date=datetime(2020, 1, 1))
                assert False, "Expected exception to be raised"
            except Exception as e:
                assert "Yahoo Gold API error" in str(e)

    @pytest.mark.asyncio
    async def test_gold_data_empty_handling(self):
        """Test handling of empty gold data."""
        # Setup mock provider result with empty data
        gold_result = MagicMock()
        gold_result.success = True
        gold_result.data = pd.DataFrame()  # Empty DataFrame

        # Mock the provider
        with patch(
            "app.flows.markets.precious_metals.create_yahoo_history_provider"
        ) as mock_yahoo:
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
    async def test_gold_provider_failure(self):
        """Test handling of gold provider failure."""
        # Setup mock provider result with failure
        gold_result = MagicMock()
        gold_result.success = False
        gold_result.error_message = "Gold provider failed"

        # Mock the provider
        with patch(
            "app.flows.markets.precious_metals.create_yahoo_history_provider"
        ) as mock_yahoo:
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
    async def test_gold_moving_averages_calculation(self, sample_gold_data):
        """Test the accuracy of gold moving averages calculation."""
        # Setup mock provider result
        gold_result = MagicMock()
        gold_result.success = True
        gold_result.data = sample_gold_data

        # Mock the provider
        with patch(
            "app.flows.markets.precious_metals.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=gold_result)
            mock_yahoo.return_value = mock_yahoo_instance

            # Test the function
            base_date = datetime(2020, 1, 1)
            result = await fetch_precious_metals_data(base_date)

            data = result["data"]
            assert "Gold_MA50" in data.columns
            assert "Gold_MA200" in data.columns

            # Verify moving average calculation accuracy for a full year of data
            gold_values = data["Gold"]
            ma50_values = data["Gold_MA50"]
            ma200_values = data["Gold_MA200"]

            # Check that we have reasonable MA values
            ma50_non_null = ma50_values.dropna()
            ma200_non_null = ma200_values.dropna()

            if len(data) >= 50:
                assert not ma50_non_null.empty
                # MA50 should be calculated after 50 periods
                assert len(ma50_non_null) == len(data) - 49  # 50 min_periods - 1

            if len(data) >= 200:
                assert not ma200_non_null.empty
                # MA200 should be calculated after 200 periods
                assert len(ma200_non_null) == len(data) - 199  # 200 min_periods - 1

            # Verify MA smooths out volatility (should be less volatile than raw gold)
            if len(ma50_non_null) > 50:
                gold_std = gold_values.iloc[50:].std()
                ma50_std = ma50_non_null.iloc[1:].std()  # Skip first value
                assert ma50_std < gold_std  # MA should be smoother

    @pytest.mark.asyncio
    async def test_gold_data_filtering_by_base_date(self, sample_gold_data):
        """Test that gold data is properly filtered by base_date."""
        # Setup mock provider result
        gold_result = MagicMock()
        gold_result.success = True
        gold_result.data = sample_gold_data

        # Mock the provider
        with patch(
            "app.flows.markets.precious_metals.create_yahoo_history_provider"
        ) as mock_yahoo:
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
    async def test_gold_workflow_direct(self):
        """Test the PreciousMetalsWorkflow class directly."""
        # Create sample gold data
        gold_data = pd.DataFrame(
            {
                "Close": [1500.0, 1520.0, 1540.0, 1560.0, 1580.0],
                "Volume": [100000] * 5,
            },
            index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
        )

        # Mock provider result
        gold_result = MagicMock()
        gold_result.success = True
        gold_result.data = gold_data

        # Mock the provider at the class level
        with patch(
            "app.flows.markets.precious_metals.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=gold_result)
            mock_yahoo.return_value = mock_yahoo_instance

            # Test the workflow directly
            workflow = PreciousMetalsWorkflow()
            base_date = datetime(2020, 1, 1)

            result = await workflow.run(base_date=base_date)

            # Verify results
            assert result.data is not None
            assert result.metadata is not None
            assert "latest_value" in result.metadata

            data = result.data
            assert not data.empty
            assert len(data) == 5  # Should have 5 days of data
            assert "Gold" in data.columns

            # Verify latest value
            latest_value = result.metadata["latest_value"]
            assert latest_value == 1580.0  # Last value in gold data

    @pytest.mark.asyncio
    async def test_gold_no_close_price_data(self):
        """Test handling when gold data has no Close price column."""
        # Create gold data without Close price
        gold_data = pd.DataFrame(
            {
                "Open": [1500.0, 1520.0, 1540.0],
                "High": [1520.0, 1540.0, 1560.0],
                "Low": [1480.0, 1500.0, 1520.0],
                "Volume": [100000] * 3,
            },
            index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
        )

        # Mock provider result
        gold_result = MagicMock()
        gold_result.success = True
        gold_result.data = gold_data

        # Mock the provider
        with patch(
            "app.flows.markets.precious_metals.create_yahoo_history_provider"
        ) as mock_yahoo:
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
    async def test_gold_moving_average_with_limited_data(self):
        """Test gold moving average with less than 50/200 days of data."""
        # Create limited gold data (only 30 days)
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        gold_values = [1500 + i * 10 for i in range(30)]
        limited_gold_data = pd.DataFrame(
            {
                "Close": gold_values,
                "Volume": [100000] * 30,
            },
            index=dates,
        )

        # Setup mock provider result
        gold_result = MagicMock()
        gold_result.success = True
        gold_result.data = limited_gold_data

        # Mock the provider
        with patch(
            "app.flows.markets.precious_metals.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=gold_result)
            mock_yahoo.return_value = mock_yahoo_instance

            # Test the function
            base_date = datetime(2020, 1, 1)
            result = await fetch_precious_metals_data(base_date)

            data = result["data"]
            assert "Gold_MA50" in data.columns
            assert "Gold_MA200" in data.columns

            # Verify MA columns exist but may be NaN for insufficient data
            # MA50 should be NaN for first 49 values (min_periods=50)
            # MA200 should be all NaN (min_periods=200, but only 30 data points)
            ma50_values = data["Gold_MA50"]
            ma200_values = data["Gold_MA200"]

            # All MA50 should be NaN due to min_periods=50 requirement
            assert ma50_values.isna().all()

            # All MA200 should be NaN due to min_periods=200 requirement
            assert ma200_values.isna().all()
