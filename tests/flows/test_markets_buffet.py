"""
Unit tests for the Buffet Indicator workflow and FlowRunner integration.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, MagicMock, patch
import pandas as pd

from app.flows.markets_buffet import (
    BuffetIndicatorWorkflow,
    BuffetIndicatorEvent,
    fetch_buffet_indicator_data,
)
from app.flows.base import FlowRunner, FlowResult
from app.lib.exceptions import WorkflowException
from app.providers.base import ProviderResult, ProviderType


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


class TestBuffetIndicatorWorkflow:
    """Test the BuffetIndicatorWorkflow class."""

    def test_workflow_initialization(self):
        """Test that workflow initializes correctly."""
        workflow = BuffetIndicatorWorkflow()
        assert workflow is not None
        assert hasattr(workflow, "fred_provider")
        assert hasattr(workflow, "yahoo_provider")

    @pytest.mark.asyncio
    async def test_fetch_economic_data_success(self):
        """Test successful data fetching step."""
        workflow = BuffetIndicatorWorkflow()

        # Create mock data
        gdp_data = pd.DataFrame(
            {"value": [20000, 21000, 22000]},
            index=pd.date_range("2020-01-01", periods=3, freq="QS"),
        )

        wilshire_data = pd.DataFrame(
            {"Close": [30000, 32000, 34000]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        # Mock provider results
        gdp_result = ProviderResult(
            success=True, data=gdp_data, provider_type=ProviderType.FRED_SERIES
        )
        wilshire_result = ProviderResult(
            success=True, data=wilshire_data, provider_type=ProviderType.YAHOO_HISTORY
        )

        # Mock the provider methods
        workflow.fred_provider.get_data = AsyncMock(return_value=gdp_result)
        workflow.yahoo_provider.get_data = AsyncMock(return_value=wilshire_result)

        # Create start event
        start_event = Mock()
        start_event.base_date = datetime(2020, 1, 1)
        start_event.original_period = "1Y"

        # Execute the step
        result = await workflow.fetch_economic_data(start_event)

        # Verify result
        assert isinstance(result, BuffetIndicatorEvent)
        assert result.base_date == datetime(2020, 1, 1)
        assert result.original_period == "1Y"
        pd.testing.assert_frame_equal(result.gdp_data, gdp_data)
        pd.testing.assert_frame_equal(result.wilshire_data, wilshire_data)

    @pytest.mark.asyncio
    async def test_fetch_economic_data_gdp_failure(self):
        """Test handling of GDP data fetch failure."""
        workflow = BuffetIndicatorWorkflow()

        # Mock successful Wilshire result but failed GDP result
        wilshire_data = pd.DataFrame(
            {"Close": [30000, 32000, 34000]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        wilshire_result = ProviderResult(
            success=True, data=wilshire_data, provider_type=ProviderType.YAHOO_HISTORY
        )

        # Mock failed GDP result
        gdp_result = ProviderResult(
            success=False,
            error_message="GDP API unavailable",
            provider_type=ProviderType.FRED_SERIES,
        )

        workflow.fred_provider.get_data = AsyncMock(return_value=gdp_result)
        workflow.yahoo_provider.get_data = AsyncMock(return_value=wilshire_result)

        # Create start event
        start_event = Mock()
        start_event.base_date = datetime(2020, 1, 1)
        start_event.original_period = "1Y"

        # Execute the step and expect WorkflowException
        with pytest.raises(WorkflowException) as exc_info:
            await workflow.fetch_economic_data(start_event)

        assert exc_info.value.workflow == "BuffetIndicatorWorkflow"
        assert exc_info.value.step == "fetch_economic_data"
        assert "GDP data fetch failed" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_calculate_buffet_indicator_success(self):
        """Test successful Buffet Indicator calculation."""
        workflow = BuffetIndicatorWorkflow()

        # Create realistic test data
        base_date = datetime(2020, 1, 1)

        # GDP data (quarterly, in billions)
        gdp_data = pd.DataFrame(
            {"value": [20000, 20500, 21000, 21500]},
            index=pd.date_range("2020-01-01", periods=4, freq="QS"),
        )

        # Wilshire data (daily, index values)
        wilshire_data = pd.DataFrame(
            {"Close": [30000, 31000, 32000, 33000]},
            index=pd.date_range("2020-01-01", periods=4, freq="QS"),
        )

        # Create BuffetIndicatorEvent
        event = BuffetIndicatorEvent(
            gdp_data=gdp_data,
            wilshire_data=wilshire_data,
            base_date=base_date,
            original_period="1Y",
        )

        # Mock the trend calculation and period adjustment functions
        with (
            patch("app.flows.markets_buffet.calculate_exponential_trend") as mock_trend,
            patch("app.flows.markets_buffet.ensure_minimum_data_points") as mock_ensure,
            patch(
                "app.flows.markets_buffet.filter_trend_data_to_period"
            ) as mock_filter,
        ):

            # Setup mock returns
            mock_trend.return_value = pd.DataFrame({"trend": [1.5, 1.6, 1.7, 1.8]})
            mock_ensure.return_value = (
                pd.DataFrame(
                    {
                        "GDP": [20000, 20500, 21000, 21500],
                        "Wilshire_5000": [30000, 31000, 32000, 33000],
                        "Buffet_Indicator": [150, 151, 152, 153],
                    }
                ),
                "1Y",
                False,
            )
            mock_filter.return_value = pd.DataFrame({"trend": [1.5, 1.6, 1.7, 1.8]})

            # Execute the calculation step
            result = await workflow.calculate_buffet_indicator(event)

            # Verify result structure
            assert hasattr(result, "result")
            assert isinstance(result.result, FlowResult)
            assert result.result.success is True
            assert result.result.data is not None
            assert not result.result.data.empty

            # Verify metadata
            metadata = result.result.metadata
            assert "trend_data" in metadata
            assert "original_period" in metadata
            assert "actual_period" in metadata
            assert "was_adjusted" in metadata
            assert "latest_value" in metadata
            assert "data_points" in metadata

    @pytest.mark.asyncio
    async def test_calculate_buffet_indicator_no_gdp_data(self):
        """Test error handling when GDP data is empty."""
        workflow = BuffetIndicatorWorkflow()

        # Empty GDP data
        gdp_data = pd.DataFrame({"value": []})
        wilshire_data = pd.DataFrame({"Close": [30000, 31000, 32000]})

        event = BuffetIndicatorEvent(
            gdp_data=gdp_data,
            wilshire_data=wilshire_data,
            base_date=datetime(2020, 1, 1),
            original_period="1Y",
        )

        with pytest.raises(WorkflowException) as exc_info:
            await workflow.calculate_buffet_indicator(event)

        assert exc_info.value.workflow == "BuffetIndicatorWorkflow"
        assert exc_info.value.step == "calculate_buffet_indicator"
        assert "No GDP data available" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_calculate_buffet_indicator_no_wilshire_close_column(self):
        """Test error handling when Wilshire data has no Close column."""
        workflow = BuffetIndicatorWorkflow()

        gdp_data = pd.DataFrame({"value": [20000, 21000, 22000]})
        wilshire_data = pd.DataFrame({"Open": [30000, 31000, 32000]})  # No Close column

        event = BuffetIndicatorEvent(
            gdp_data=gdp_data,
            wilshire_data=wilshire_data,
            base_date=datetime(2020, 1, 1),
            original_period="1Y",
        )

        with pytest.raises(WorkflowException) as exc_info:
            await workflow.calculate_buffet_indicator(event)

        assert exc_info.value.workflow == "BuffetIndicatorWorkflow"
        assert exc_info.value.step == "calculate_buffet_indicator"
        assert (
            "No Close price data available for Wilshire 5000" in exc_info.value.message
        )


class TestBuffetIndicatorFlowRunner:
    """Test the Buffet Indicator workflow with FlowRunner."""

    @pytest.mark.asyncio
    async def test_flowrunner_integration_success(self):
        """Test successful workflow execution with FlowRunner."""
        workflow = BuffetIndicatorWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        # Mock the workflow to return a FlowResult
        mock_result = Mock()
        mock_result.result = FlowResult.success_result(
            data=pd.DataFrame(
                {
                    "GDP": [20000, 21000],
                    "Wilshire_5000": [30000, 32000],
                    "Buffet_Indicator": [150, 152],
                }
            ),
            metadata={
                "trend_data": pd.DataFrame({"trend": [1.5, 1.6]}),
                "original_period": "1Y",
                "actual_period": "1Y",
                "was_adjusted": False,
                "latest_value": 152.0,
                "data_points": 2,
            },
        )

        # Mock workflow.run to return the mock result
        workflow.run = AsyncMock(return_value=mock_result)

        # Execute workflow through FlowRunner
        result = await runner.run(base_date=datetime(2020, 1, 1), original_period="1Y")

        # Verify FlowResult structure
        assert isinstance(result, FlowResult)
        assert result.success is True
        assert result.data is not None
        assert not result.data.empty
        assert "GDP" in result.data.columns
        assert "Wilshire_5000" in result.data.columns
        assert "Buffet_Indicator" in result.data.columns

    @pytest.mark.asyncio
    async def test_flowrunner_integration_workflow_exception(self):
        """Test FlowRunner handling of WorkflowException."""
        workflow = BuffetIndicatorWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        # Mock workflow to raise WorkflowException
        workflow.run = AsyncMock(
            side_effect=WorkflowException(
                workflow="BuffetIndicatorWorkflow",
                step="test_step",
                message="Test workflow error",
            )
        )

        # Execute and expect FlowResult with error
        result = await runner.run(base_date=datetime(2020, 1, 1), original_period="1Y")

        assert isinstance(result, FlowResult)
        assert result.success is False
        assert result.data is None
        assert (
            result.error_message is not None
            and "Test workflow error" in result.error_message
        )


class TestFetchBuffetIndicatorData:
    """Test the fetch_buffet_indicator_data function."""

    @pytest.mark.asyncio
    async def test_fetch_buffet_indicator_data_success(
        self, sample_gdp_data, sample_wilshire_data
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
        with (
            patch("app.flows.markets_buffet.create_fred_series_provider") as mock_fred,
            patch(
                "app.flows.markets_buffet.create_yahoo_history_provider"
            ) as mock_yahoo,
        ):

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
    async def test_fetch_buffet_indicator_data_fred_error(self, sample_wilshire_data):
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
