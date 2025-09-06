"""
Unit tests for VIX workflow and FlowRunner integration.
"""

import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from app.flows.markets_vix import (
    fetch_vix_data,
    VIXWorkflow,
    VIXEvent,
)
from app.flows.base import FlowRunner, FlowResult
from app.lib.exceptions import WorkflowException


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


@pytest.fixture
def mock_provider_result():
    """Create a mock provider result."""
    result = MagicMock()
    result.success = True
    result.error_message = None
    return result


class TestVIXWorkflow:
    """Test the VIXWorkflow class."""

    def test_workflow_initialization(self):
        """Test that workflow initializes correctly."""
        workflow = VIXWorkflow()
        assert workflow is not None
        assert hasattr(workflow, "yahoo_provider")

    @pytest.mark.asyncio
    async def test_fetch_vix_data_step_success(self):
        """Test successful VIX data fetching step."""
        workflow = VIXWorkflow()

        # Create mock VIX data
        vix_data = pd.DataFrame(
            {"Close": [20.5, 18.3, 25.1]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        # Mock provider result
        vix_result = MagicMock()
        vix_result.success = True
        vix_result.data = vix_data

        # Mock the provider method
        workflow.yahoo_provider.get_data = AsyncMock(return_value=vix_result)

        # Create start event
        start_event = MagicMock()
        start_event.base_date = datetime(2020, 1, 1)

        # Execute the step
        result = await workflow.fetch_vix_data(start_event)

        # Verify result
        assert isinstance(result, VIXEvent)
        assert result.base_date == datetime(2020, 1, 1)
        pd.testing.assert_frame_equal(result.vix_data, vix_data)

    @pytest.mark.asyncio
    async def test_process_vix_data_success(self):
        """Test successful VIX data processing step."""
        workflow = VIXWorkflow()

        # Create realistic test data
        base_date = datetime(2020, 1, 1)
        vix_data = pd.DataFrame(
            {"Close": [20.5, 18.3, 25.1, 22.7, 19.8]},
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )

        # Create VIXEvent
        event = VIXEvent(vix_data=vix_data, base_date=base_date)

        # Execute the processing step
        result = await workflow.process_vix_data(event)

        # Verify result structure
        assert hasattr(result, "result")
        assert isinstance(result.result, FlowResult)
        assert result.result.success is True
        assert result.result.data is not None
        assert not result.result.data.empty

        # Verify data columns
        data = result.result.data
        assert "VIX" in data.columns
        assert "VIX_MA50" in data.columns

        # Verify metadata
        metadata = result.result.metadata
        assert "historical_mean" in metadata
        assert "latest_value" in metadata
        assert "data_points" in metadata

    @pytest.mark.asyncio
    async def test_process_vix_data_no_close_column(self):
        """Test error handling when VIX data has no Close column."""
        workflow = VIXWorkflow()

        vix_data = pd.DataFrame({"Open": [20.5, 18.3, 25.1]})
        event = VIXEvent(vix_data=vix_data, base_date=datetime(2020, 1, 1))

        with pytest.raises(WorkflowException) as exc_info:
            await workflow.process_vix_data(event)

        assert exc_info.value.workflow == "VIXWorkflow"
        assert exc_info.value.step == "process_vix_data"
        assert "No Close price data available for VIX" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_process_vix_data_empty_data(self):
        """Test error handling when VIX data is empty."""
        workflow = VIXWorkflow()

        vix_data = pd.DataFrame({"Close": []})
        event = VIXEvent(vix_data=vix_data, base_date=datetime(2020, 1, 1))

        with pytest.raises(WorkflowException) as exc_info:
            await workflow.process_vix_data(event)

        assert exc_info.value.workflow == "VIXWorkflow"
        assert exc_info.value.step == "process_vix_data"
        assert "No VIX data available" in exc_info.value.message


class TestVIXFlowRunner:
    """Test the VIX workflow with FlowRunner."""

    @pytest.mark.asyncio
    async def test_flowrunner_integration_success(self):
        """Test successful workflow execution with FlowRunner."""
        workflow = VIXWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        # Mock the workflow to return a FlowResult
        mock_result = MagicMock()
        mock_result.result = FlowResult.success_result(
            data=pd.DataFrame(
                {
                    "VIX": [20.5, 18.3],
                    "VIX_MA50": [20.5, 19.4],
                }
            ),
            metadata={
                "historical_mean": 19.9,
                "latest_value": 18.3,
                "data_points": 2,
            },
        )

        # Mock workflow.run to return the mock result
        workflow.run = AsyncMock(return_value=mock_result)

        # Execute workflow through FlowRunner
        result = await runner.run(base_date=datetime(2020, 1, 1))

        # Verify FlowResult structure
        assert isinstance(result, FlowResult)
        assert result.success is True
        assert result.data is not None
        assert not result.data.empty
        assert "VIX" in result.data.columns
        assert "VIX_MA50" in result.data.columns

    @pytest.mark.asyncio
    async def test_flowrunner_integration_workflow_exception(self):
        """Test FlowRunner handling of WorkflowException."""
        workflow = VIXWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        # Mock workflow to raise WorkflowException
        workflow.run = AsyncMock(
            side_effect=WorkflowException(
                workflow="VIXWorkflow",
                step="test_step",
                message="Test workflow error",
            )
        )

        # Execute and expect FlowResult with error
        result = await runner.run(base_date=datetime(2020, 1, 1))

        assert isinstance(result, FlowResult)
        assert result.success is False
        assert result.data is None
        assert (
            result.error_message is not None
            and "Test workflow error" in result.error_message
        )


class TestFetchVIXData:
    """Test the fetch_vix_data function."""

    @pytest.mark.asyncio
    async def test_fetch_vix_data_success(self, sample_vix_data):
        """Test successful VIX data fetch and calculation."""
        # Setup mock provider result
        vix_result = MagicMock()
        vix_result.success = True
        vix_result.data = sample_vix_data

        # Mock the provider
        with patch("app.flows.markets_vix.create_yahoo_history_provider") as mock_yahoo:
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
    async def test_fetch_vix_data_yahoo_error(self):
        """Test handling of Yahoo Finance errors for VIX."""
        # Test the error handling within the workflow
        workflow = VIXWorkflow()

        # Mock the Yahoo provider to fail
        with patch.object(workflow, "yahoo_provider") as mock_yahoo:
            mock_yahoo.get_data = AsyncMock(
                side_effect=Exception("Yahoo VIX API error")
            )

            # Test the workflow directly
            try:
                await workflow.run(base_date=datetime(2020, 1, 1))
                assert False, "Expected exception to be raised"
            except Exception as e:
                assert "Yahoo VIX API error" in str(e)

    @pytest.mark.asyncio
    async def test_vix_data_empty_handling(self):
        """Test handling of empty VIX data."""
        # Setup mock provider result with empty data
        vix_result = MagicMock()
        vix_result.success = True
        vix_result.data = pd.DataFrame()  # Empty DataFrame

        # Mock the provider
        with patch("app.flows.markets_vix.create_yahoo_history_provider") as mock_yahoo:
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
    async def test_vix_provider_failure(self):
        """Test handling of VIX provider failure."""
        # Setup mock provider result with failure
        vix_result = MagicMock()
        vix_result.success = False
        vix_result.error_message = "VIX provider failed"

        # Mock the provider
        with patch("app.flows.markets_vix.create_yahoo_history_provider") as mock_yahoo:
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
    async def test_vix_historical_mean_calculation(self, sample_vix_data):
        """Test the accuracy of VIX historical mean calculation."""
        # Setup mock provider result
        vix_result = MagicMock()
        vix_result.success = True
        vix_result.data = sample_vix_data

        # Mock the provider
        with patch("app.flows.markets_vix.create_yahoo_history_provider") as mock_yahoo:
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
    async def test_vix_data_filtering_by_base_date(self, sample_vix_data):
        """Test that VIX data is properly filtered by base_date."""
        # Setup mock provider result
        vix_result = MagicMock()
        vix_result.success = True
        vix_result.data = sample_vix_data

        # Mock the provider
        with patch("app.flows.markets_vix.create_yahoo_history_provider") as mock_yahoo:
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
    async def test_vix_workflow_direct(self):
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
        with patch("app.flows.markets_vix.create_yahoo_history_provider") as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=vix_result)
            mock_yahoo.return_value = mock_yahoo_instance

            # Test the workflow directly
            workflow = VIXWorkflow()
            base_date = datetime(2020, 1, 1)

            result = await workflow.run(base_date=base_date)

            # Verify results
            assert result.data is not None
            assert result.metadata is not None
            assert "historical_mean" in result.metadata

            data = result.data
            assert not data.empty
            assert len(data) == 5  # Should have 5 days of data
            assert "VIX" in data.columns

            # Verify historical mean
            expected_mean = vix_data["Close"].mean()
            actual_mean = result.metadata["historical_mean"]
            assert abs(actual_mean - expected_mean) < 0.01

    @pytest.mark.asyncio
    async def test_vix_no_close_price_data(self):
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
        with patch("app.flows.markets_vix.create_yahoo_history_provider") as mock_yahoo:
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
    async def test_vix_moving_average_calculation(self, sample_vix_data):
        """Test the accuracy of VIX 50-day moving average calculation."""
        # Setup mock provider result
        vix_result = MagicMock()
        vix_result.success = True
        vix_result.data = sample_vix_data

        # Mock the provider
        with patch("app.flows.markets_vix.create_yahoo_history_provider") as mock_yahoo:
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
    async def test_vix_moving_average_with_limited_data(self):
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
        with patch("app.flows.markets_vix.create_yahoo_history_provider") as mock_yahoo:
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
    async def test_vix_workflow_moving_average_integration(self):
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
        with patch("app.flows.markets_vix.create_yahoo_history_provider") as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=vix_result)
            mock_yahoo.return_value = mock_yahoo_instance

            # Test the workflow directly
            workflow = VIXWorkflow()
            base_date = datetime(2020, 1, 1)

            result = await workflow.run(base_date=base_date)

            # Verify results include moving average
            data = result.data
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
