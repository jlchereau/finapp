"""
Unit tests for the Buffet Indicator workflow and FlowRunner integration.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, MagicMock, patch
import pandas as pd

from app.flows.markets.buffet import (
    BuffetIndicatorWorkflow,
    FetchGDPEvent,
    FetchWilshireEvent,
    GDPResultEvent,
    WilshireResultEvent,
    fetch_buffet_indicator_data,
)
from app.flows.base import FlowRunner, FlowResultEvent
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
    async def test_initiate_data_fetch_success(self):
        """Test successful data fetch dispatch step."""
        workflow = BuffetIndicatorWorkflow()

        # Create start event
        start_event = Mock()
        start_event.base_date = datetime(2020, 1, 1)
        start_event.original_period = "1Y"

        # Create mock context (send_event stays as Mock since it's sync)
        mock_ctx = MagicMock()
        mock_ctx.send_event = Mock()

        # Mock store.set as async function
        async def mock_set(key, value):
            pass

        mock_ctx.store.set = mock_set

        # Execute the step
        result = await workflow.initiate_data_fetch(mock_ctx, start_event)

        # Verify result is dummy FetchGDPEvent (following yield pattern)
        assert isinstance(result, FetchGDPEvent)
        assert result.observation_start == "dummy"  # Still used for workflow structure

        # Verify events were sent
        assert mock_ctx.send_event.call_count == 2

        # Note: We can't easily assert on async function calls with our simple mock
        # The important thing is that the step executed successfully

    @pytest.mark.asyncio
    async def test_fetch_gdp_data_success(self):
        """Test successful GDP data fetching step."""
        workflow = BuffetIndicatorWorkflow()

        # Create mock GDP data
        gdp_data = pd.DataFrame(
            {"value": [20000, 21000, 22000]},
            index=pd.date_range("2020-01-01", periods=3, freq="QS"),
        )

        # Mock provider result
        gdp_result = ProviderResult(
            success=True, data=gdp_data, provider_type=ProviderType.FRED_SERIES
        )

        # Mock the provider method
        workflow.fred_provider.get_data = AsyncMock(return_value=gdp_result)

        # Create GDP fetch event
        fetch_event = FetchGDPEvent(
            base_date=datetime(2020, 1, 1),
            observation_start=None,  # No longer limits data collection
        )

        # Execute the step
        result = await workflow.fetch_gdp_data(fetch_event)

        # Verify result
        assert isinstance(result, GDPResultEvent)
        assert result.error is None
        pd.testing.assert_frame_equal(result.data, gdp_data)

        # Verify provider was called correctly
        workflow.fred_provider.get_data.assert_called_once_with(
            query="GDP"  # No longer passes period limitations
        )

    @pytest.mark.asyncio
    async def test_fetch_wilshire_data_success(self):
        """Test successful Wilshire data fetching step."""
        workflow = BuffetIndicatorWorkflow()

        # Create mock Wilshire data
        wilshire_data = pd.DataFrame(
            {"Close": [30000, 32000, 34000]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        # Mock provider result
        wilshire_result = ProviderResult(
            success=True, data=wilshire_data, provider_type=ProviderType.YAHOO_HISTORY
        )

        # Mock the provider method
        workflow.yahoo_provider.get_data = AsyncMock(return_value=wilshire_result)

        # Create Wilshire fetch event
        fetch_event = FetchWilshireEvent(
            base_date=datetime(2020, 1, 1),
            period="max",
        )

        # Execute the step
        result = await workflow.fetch_wilshire_data(fetch_event)

        # Verify result
        assert isinstance(result, WilshireResultEvent)
        assert result.error is None
        pd.testing.assert_frame_equal(result.data, wilshire_data)

        # Verify provider was called correctly
        workflow.yahoo_provider.get_data.assert_called_once_with(query="^FTW5000")

    @pytest.mark.asyncio
    async def test_fetch_gdp_data_provider_failure(self):
        """Test handling of GDP data fetch provider failure."""
        workflow = BuffetIndicatorWorkflow()

        # Mock failed GDP result
        gdp_result = ProviderResult(
            success=False,
            error_message="GDP API unavailable",
            provider_type=ProviderType.FRED_SERIES,
        )

        workflow.fred_provider.get_data = AsyncMock(return_value=gdp_result)

        # Create GDP fetch event
        fetch_event = FetchGDPEvent(
            base_date=datetime(2020, 1, 1),
            observation_start=None,  # No longer limits data collection
        )

        # Execute the step
        result = await workflow.fetch_gdp_data(fetch_event)

        # Verify error result
        assert isinstance(result, GDPResultEvent)
        assert result.data is None
        assert result.error == "GDP API unavailable"

    @pytest.mark.asyncio
    async def test_fetch_wilshire_data_provider_failure(self):
        """Test handling of Wilshire data fetch provider failure."""
        workflow = BuffetIndicatorWorkflow()

        # Mock failed Wilshire result
        wilshire_result = ProviderResult(
            success=False,
            error_message="Wilshire API unavailable",
            provider_type=ProviderType.YAHOO_HISTORY,
        )

        workflow.yahoo_provider.get_data = AsyncMock(return_value=wilshire_result)

        # Create Wilshire fetch event
        fetch_event = FetchWilshireEvent(
            base_date=datetime(2020, 1, 1),
            period="max",
        )

        # Execute the step
        result = await workflow.fetch_wilshire_data(fetch_event)

        # Verify error result
        assert isinstance(result, WilshireResultEvent)
        assert result.data is None
        assert result.error == "Wilshire API unavailable"

    @pytest.mark.asyncio
    async def test_calculate_buffet_indicator_success(self):
        """Test successful Buffet Indicator calculation with collector pattern."""
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

        # Create result events
        gdp_result = GDPResultEvent(data=gdp_data, error=None)
        wilshire_result = WilshireResultEvent(data=wilshire_data, error=None)

        # Create mock context (use MagicMock with async methods where needed)
        mock_ctx = MagicMock()

        async def mock_get(key):
            return {
                "base_date": base_date,
                "original_period": "1Y",
            }[key]

        mock_ctx.store.get = mock_get

        # Mock collect_events to return both results immediately (synchronous)
        mock_ctx.collect_events.return_value = [gdp_result, wilshire_result]

        # Mock the trend calculation and period adjustment functions
        with (
            patch("app.flows.markets.buffet.calculate_exponential_trend") as mock_trend,
            patch("app.flows.markets.buffet.ensure_minimum_data_points") as mock_ensure,
            patch(
                "app.flows.markets.buffet.filter_trend_data_to_period"
            ) as mock_filter,
        ):

            # Setup mock returns
            mock_trend.return_value = pd.DataFrame({"trend": [1.5, 1.6, 1.7, 1.8]})
            mock_ensure.return_value = pd.DataFrame(
                {
                    "GDP": [20000, 20500, 21000, 21500],
                    "Wilshire_5000": [30000, 31000, 32000, 33000],
                    "Buffet_Indicator": [150, 151, 152, 153],
                }
            )
            mock_filter.return_value = pd.DataFrame({"trend": [1.5, 1.6, 1.7, 1.8]})

            # Execute the calculation step with GDP result event (collector will get
            # both)
            result = await workflow.calculate_buffet_indicator(mock_ctx, gdp_result)

            # Verify result structure
            assert isinstance(result, FlowResultEvent)
            assert result.success is True
            assert result.data is not None
            assert not result.data.empty

            # Verify metadata
            metadata = result.metadata
            assert "trend_data" in metadata
            assert "original_period" in metadata
            assert "latest_value" in metadata
            assert "data_points" in metadata

            # Verify collector pattern was used
            mock_ctx.collect_events.assert_called_once_with(
                gdp_result, [GDPResultEvent, WilshireResultEvent]
            )

    @pytest.mark.asyncio
    async def test_calculate_buffet_indicator_gdp_error(self):
        """Test error handling when GDP data fetch failed."""
        workflow = BuffetIndicatorWorkflow()

        # GDP error and valid Wilshire data
        gdp_result = GDPResultEvent(data=None, error="GDP API error")
        wilshire_data = pd.DataFrame({"Close": [30000, 31000, 32000]})
        wilshire_result = WilshireResultEvent(data=wilshire_data, error=None)

        # Create mock context (use MagicMock with async methods where needed)
        mock_ctx = MagicMock()

        async def mock_get(key):
            return {
                "base_date": datetime(2020, 1, 1),
                "original_period": "1Y",
            }[key]

        mock_ctx.store.get = mock_get

        # Mock collect_events to return both results (synchronous)
        mock_ctx.collect_events.return_value = [gdp_result, wilshire_result]

        with pytest.raises(Exception) as exc_info:
            await workflow.calculate_buffet_indicator(mock_ctx, gdp_result)

        assert "GDP data fetch failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_calculate_buffet_indicator_wilshire_error(self):
        """Test error handling when Wilshire data fetch failed."""
        workflow = BuffetIndicatorWorkflow()

        # Valid GDP data and Wilshire error
        gdp_data = pd.DataFrame({"value": [20000, 21000, 22000]})
        gdp_result = GDPResultEvent(data=gdp_data, error=None)
        wilshire_result = WilshireResultEvent(data=None, error="Wilshire API error")

        # Create mock context (use MagicMock with async methods where needed)
        mock_ctx = MagicMock()

        async def mock_get(key):
            return {
                "base_date": datetime(2020, 1, 1),
                "original_period": "1Y",
            }[key]

        mock_ctx.store.get = mock_get

        # Mock collect_events to return both results (synchronous)
        mock_ctx.collect_events.return_value = [gdp_result, wilshire_result]

        with pytest.raises(Exception) as exc_info:
            await workflow.calculate_buffet_indicator(mock_ctx, gdp_result)

        assert "Wilshire data fetch failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_calculate_buffet_indicator_collector_not_ready(self):
        """Test collector pattern when not all results are ready."""
        workflow = BuffetIndicatorWorkflow()

        gdp_result = GDPResultEvent(data=pd.DataFrame({"value": [20000]}), error=None)

        # Create mock context (use MagicMock with async methods where needed)
        mock_ctx = MagicMock()

        async def mock_get(key):
            return {
                "base_date": datetime(2020, 1, 1),
                "original_period": "1Y",
            }[key]

        mock_ctx.store.get = mock_get

        # Mock collect_events to return None (not all collected yet, synchronous)
        mock_ctx.collect_events.return_value = None

        # Execute the calculation step
        result = await workflow.calculate_buffet_indicator(mock_ctx, gdp_result)

        # Should return None when not all results are collected
        assert result is None


class TestBuffetIndicatorFlowRunner:
    """Test the Buffet Indicator workflow with FlowRunner."""

    @pytest.mark.asyncio
    async def test_flowrunner_integration_success(self):
        """Test successful workflow execution with FlowRunner."""
        # Create mock data
        gdp_data = pd.DataFrame(
            {"value": [20000, 21000]},
            index=pd.date_range("2020-01-01", periods=2, freq="QS"),
        )
        wilshire_data = pd.DataFrame(
            {"Close": [30000, 32000]},
            index=pd.date_range("2020-01-01", periods=2, freq="QS"),
        )

        # Mock provider results
        gdp_result = ProviderResult(
            success=True, data=gdp_data, provider_type=ProviderType.FRED_SERIES
        )
        wilshire_result = ProviderResult(
            success=True, data=wilshire_data, provider_type=ProviderType.YAHOO_HISTORY
        )

        # Create workflow and runner
        workflow = BuffetIndicatorWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        # Mock the providers
        workflow.fred_provider.get_data = AsyncMock(return_value=gdp_result)
        workflow.yahoo_provider.get_data = AsyncMock(return_value=wilshire_result)

        # Execute workflow through FlowRunner
        result = await runner.run(base_date=datetime(2020, 1, 1), original_period="1Y")

        # Verify FlowResultEvent structure
        assert isinstance(result, FlowResultEvent)
        assert result.success is True
        assert result.data is not None
        assert not result.data.empty
        assert "GDP" in result.data.columns
        assert "Wilshire_5000" in result.data.columns
        assert "Buffet_Indicator" in result.data.columns

    @pytest.mark.asyncio
    async def test_flowrunner_integration_workflow_exception(self):
        """Test FlowRunner handling of FlowException."""
        workflow = BuffetIndicatorWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        # Mock GDP provider to fail
        gdp_result = ProviderResult(
            success=False,
            error_message="GDP API error",
            provider_type=ProviderType.FRED_SERIES,
        )
        workflow.fred_provider.get_data = AsyncMock(return_value=gdp_result)

        # Mock Wilshire provider with valid data
        wilshire_data = pd.DataFrame(
            {"Close": [30000, 32000]},
            index=pd.date_range("2020-01-01", periods=2, freq="QS"),
        )
        wilshire_result = ProviderResult(
            success=True, data=wilshire_data, provider_type=ProviderType.YAHOO_HISTORY
        )
        workflow.yahoo_provider.get_data = AsyncMock(return_value=wilshire_result)

        # Execute and expect FlowResult with error
        result = await runner.run(base_date=datetime(2020, 1, 1), original_period="1Y")

        assert isinstance(result, FlowResultEvent)
        assert result.success is False
        assert result.data is None
        assert (
            result.error_message is not None
            and "GDP data fetch failed" in result.error_message
        )


class TestFetchBuffetIndicatorData:
    """Test the fetch_buffet_indicator_data function."""

    @pytest.mark.asyncio
    async def test_fetch_buffet_indicator_data_success(
        self, sample_gdp_data, sample_wilshire_data
    ):
        """Test successful Buffet Indicator data fetch and calculation."""
        # Setup mock provider results as ProviderResult objects
        gdp_result = ProviderResult(
            success=True, data=sample_gdp_data, provider_type=ProviderType.FRED_SERIES
        )
        wilshire_result = ProviderResult(
            success=True,
            data=sample_wilshire_data,
            provider_type=ProviderType.YAHOO_HISTORY,
        )

        # Mock the providers
        with (
            patch("app.flows.markets.buffet.create_fred_series_provider") as mock_fred,
            patch(
                "app.flows.markets.buffet.create_yahoo_history_provider"
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

            # Verify provider calls - now called through the workflow steps
            mock_fred_instance.get_data.assert_called()
            mock_yahoo_instance.get_data.assert_called()

    @pytest.mark.asyncio
    async def test_fetch_buffet_indicator_data_fred_error(self, sample_wilshire_data):
        """Test handling of FRED API errors."""
        # Setup mock provider results - GDP fails, Wilshire succeeds
        gdp_result = ProviderResult(
            success=False,
            error_message="FRED API error",
            provider_type=ProviderType.FRED_SERIES,
        )
        wilshire_result = ProviderResult(
            success=True,
            data=sample_wilshire_data,
            provider_type=ProviderType.YAHOO_HISTORY,
        )

        # Mock the providers
        with (
            patch("app.flows.markets.buffet.create_fred_series_provider") as mock_fred,
            patch(
                "app.flows.markets.buffet.create_yahoo_history_provider"
            ) as mock_yahoo,
        ):
            # Setup provider mocks
            mock_fred_instance = AsyncMock()
            mock_fred_instance.get_data = AsyncMock(return_value=gdp_result)
            mock_fred.return_value = mock_fred_instance

            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=wilshire_result)
            mock_yahoo.return_value = mock_yahoo_instance

            # Test the function and expect exception
            base_date = datetime(2020, 1, 1)
            with pytest.raises(Exception) as exc_info:
                await fetch_buffet_indicator_data(base_date)

            assert "GDP data fetch failed" in str(exc_info.value)
