"""
Unit tests for TimeSeriesWorkflow and related functions.

Tests the TimeSeriesWorkflow class and fetch_time_series_data function
with mocked data to avoid external API dependencies.
"""

import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from app.flows.compare.time_series import (
    TimeSeriesWorkflow,
    fetch_time_series_data,
)
from app.flows.base import FlowResult


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "Open": [100 + i for i in range(100)],
            "High": [105 + i for i in range(100)],
            "Low": [95 + i for i in range(100)],
            "Close": [102 + i for i in range(100)],
            "Adj Close": [102 + i for i in range(100)],
            "Volume": [1000000 + i * 1000 for i in range(100)],
        },
        index=dates,
    )


@pytest.fixture
def mock_provider_result():
    """Create a mock provider result."""
    result = MagicMock()
    result.success = True
    return result


class TestFetchTimeSeriesData:
    """Test fetch_time_series_data function."""

    @pytest.mark.asyncio
    async def test_fetch_time_series_data_success(
        self, sample_ohlcv_data, mock_provider_result
    ):
        """Test successful raw data fetching."""
        mock_provider_result.data = sample_ohlcv_data

        with patch(
            "app.flows.compare.time_series.create_yahoo_history_provider"
        ) as mock_create:
            mock_provider = AsyncMock()
            mock_provider.get_data.return_value = mock_provider_result
            mock_create.return_value = mock_provider

            # Test with cache disabled to avoid aiocache in tests
            with patch("app.flows.cache.settings") as mock_settings:
                mock_settings.FLOW_CACHE_ENABLED = False

                result = await fetch_time_series_data(
                    ["AAPL", "MSFT"], datetime(2024, 1, 1)
                )

                assert "AAPL" in result
                assert "MSFT" in result
                assert isinstance(result["AAPL"], pd.DataFrame)
                assert len(result["AAPL"]) == 100

    @pytest.mark.asyncio
    async def test_fetch_time_series_data_empty_tickers(self):
        """Test with empty ticker list."""
        result = await fetch_time_series_data([], datetime(2024, 1, 1))
        assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_time_series_data_provider_failure(self):
        """Test handling of provider failures."""
        with patch(
            "app.flows.compare.time_series.create_yahoo_history_provider"
        ) as mock_create:
            mock_provider = AsyncMock()
            mock_provider.get_data.side_effect = Exception("API Error")
            mock_create.return_value = mock_provider

            with patch("app.flows.cache.settings") as mock_settings:
                mock_settings.FLOW_CACHE_ENABLED = False

                # Should raise FlowException when all tickers fail
                with pytest.raises(Exception) as exc_info:
                    await fetch_time_series_data(["INVALID"], datetime(2024, 1, 1))

                # Check it's the expected exception type (FlowException format)
                assert "Time series data fetch workflow failed" in str(exc_info.value)


class TestTimeSeriesWorkflow:
    """Test TimeSeriesWorkflow class."""

    def test_workflow_initialization(self):
        """Test workflow can be initialized."""
        with patch(
            "app.flows.compare.time_series.create_yahoo_history_provider"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            workflow = TimeSeriesWorkflow()
            assert workflow.yahoo_history is not None

    @pytest.mark.asyncio
    async def test_workflow_dispatch_step(self):
        """Test the dispatch step."""
        with patch(
            "app.flows.compare.time_series.create_yahoo_history_provider"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            workflow = TimeSeriesWorkflow()

            # Create mock context and start event
            ctx = MagicMock()
            ctx.store.set = AsyncMock()
            ctx.send_event = MagicMock()

            start_event = MagicMock()
            start_event.tickers = ["AAPL", "MSFT"]
            start_event.base_date = datetime(2024, 1, 1)

            # Call dispatch step
            result = await workflow.dispatch(ctx, start_event)

            # Verify context storage
            ctx.store.set.assert_any_call("tickers", ["AAPL", "MSFT"])
            ctx.store.set.assert_any_call("base_date", datetime(2024, 1, 1))
            ctx.store.set.assert_any_call("num_to_collect", 2)

            # Verify events sent
            assert ctx.send_event.call_count == 2

            # Verify return type
            assert result.__class__.__name__ == "DispatchEvent"

    @pytest.mark.asyncio
    async def test_workflow_fetch_ticker_data_step(
        self, sample_ohlcv_data, mock_provider_result
    ):
        """Test the fetch_ticker_data step."""
        mock_provider_result.data = sample_ohlcv_data

        with patch(
            "app.flows.compare.time_series.create_yahoo_history_provider"
        ) as mock_create:
            mock_provider = AsyncMock()
            mock_provider.get_data.return_value = mock_provider_result
            mock_create.return_value = mock_provider

            workflow = TimeSeriesWorkflow()

            # Create mock fetch event
            fetch_event = MagicMock()
            fetch_event.ticker = "AAPL"

            result = await workflow.fetch_ticker_data(fetch_event)

            assert result.ticker == "AAPL"
            assert result.success is True
            assert isinstance(result.data, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_workflow_normalize_data_step_success(self, sample_ohlcv_data):
        """Test the normalize_data step with successful data."""
        with patch(
            "app.flows.compare.time_series.create_yahoo_history_provider"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            workflow = TimeSeriesWorkflow()

            # Create mock context
            ctx = MagicMock()
            ctx.store.get = AsyncMock()
            ctx.store.get.side_effect = lambda key: {
                "tickers": ["AAPL"],
                "base_date": datetime(2024, 1, 1),
                "start_time": 0.0,
                "num_to_collect": 1,
            }[key]

            # Create mock response event
            response_event = MagicMock()
            response_event.ticker = "AAPL"
            response_event.success = True
            response_event.data = sample_ohlcv_data

            # Mock collect_events to return our event
            ctx.collect_events.return_value = [response_event]

            result = await workflow.normalize_data(ctx, response_event)

            # Verify result
            assert result.__class__.__name__ == "StopEvent"
            assert isinstance(result.result, FlowResult)
            assert result.result.success is True
            assert isinstance(result.result.data, pd.DataFrame)
            assert "AAPL" in result.result.data.columns
