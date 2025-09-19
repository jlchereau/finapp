"""
Unit tests for MetricsWorkflow and related functions.

Tests the MetricsWorkflow class with mocked data to avoid external API dependencies.
"""

import pandas as pd
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.flows.compare.metrics import (
    MetricsWorkflow,
    _convert_metrics_to_dataframe,
)
from app.providers.yahoo import YahooInfoModel
from app.flows.base import FlowResult


@pytest.fixture
def sample_yahoo_info_data():
    """Create sample Yahoo info data for testing."""
    return {
        "symbol": "AAPL",
        "longName": "Apple Inc.",
        "regularMarketPrice": 150.0,
        "regularMarketChange": 2.5,
        "regularMarketChangePercent": 1.7,
        "regularMarketVolume": 50000000,
        "marketCap": 2500000000000,
        "trailingPE": 25.5,
        "dividendYield": 0.006,
        "beta": 1.2,
        "fiftyTwoWeekHigh": 200.0,
        "fiftyTwoWeekLow": 120.0,
        "currency": "USD",
        "exchange": "NMS",
        "sector": "Technology",
        "industry": "Consumer Electronics",
    }


@pytest.fixture
def sample_yahoo_info_model(sample_yahoo_info_data):
    """Create a YahooInfoModel instance for testing."""
    return YahooInfoModel(**sample_yahoo_info_data)


@pytest.fixture
def mock_provider_result():
    """Create a mock provider result."""
    result = MagicMock()
    result.success = True
    return result


class TestConvertMetricsToDataFrame:
    """Test _convert_metrics_to_dataframe helper function."""

    def test_convert_metrics_to_dataframe_success(self, sample_yahoo_info_model):
        """Test successful conversion to DataFrame."""
        metrics_data = {"AAPL": sample_yahoo_info_model}

        df = _convert_metrics_to_dataframe(metrics_data)

        assert isinstance(df, pd.DataFrame)
        assert "AAPL" in df.columns
        assert "Company Name" in df.index
        assert df.loc["Company Name", "AAPL"] == "Apple Inc."
        assert df.loc["Price", "AAPL"] == 150.0

    def test_convert_metrics_to_dataframe_multiple_tickers(
        self, sample_yahoo_info_data
    ):
        """Test conversion with multiple tickers."""
        # Create second ticker data
        msft_data = sample_yahoo_info_data.copy()
        msft_data.update(
            {
                "symbol": "MSFT",
                "longName": "Microsoft Corporation",
                "regularMarketPrice": 300.0,
            }
        )

        metrics_data = {
            "AAPL": YahooInfoModel(**sample_yahoo_info_data),
            "MSFT": YahooInfoModel(**msft_data),
        }

        df = _convert_metrics_to_dataframe(metrics_data)

        assert isinstance(df, pd.DataFrame)
        assert "AAPL" in df.columns
        assert "MSFT" in df.columns
        assert df.loc["Company Name", "AAPL"] == "Apple Inc."
        assert df.loc["Company Name", "MSFT"] == "Microsoft Corporation"

    def test_convert_metrics_to_dataframe_empty_data(self):
        """Test conversion with empty data."""
        df = _convert_metrics_to_dataframe({})

        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestMetricsWorkflow:
    """Test MetricsWorkflow class."""

    def test_workflow_initialization(self):
        """Test workflow can be initialized."""
        with patch(
            "app.flows.compare.metrics.create_yahoo_info_provider"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            workflow = MetricsWorkflow()
            assert workflow.yahoo_info is not None

    @pytest.mark.asyncio
    async def test_workflow_dispatch_step(self):
        """Test the dispatch step."""
        with patch(
            "app.flows.compare.metrics.create_yahoo_info_provider"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            workflow = MetricsWorkflow()

            # Create mock context and start event
            ctx = MagicMock()
            ctx.store.set = AsyncMock()
            ctx.send_event = MagicMock()

            start_event = MagicMock()
            start_event.tickers = ["AAPL", "MSFT"]

            # Call dispatch step
            result = await workflow.dispatch(ctx, start_event)

            # Verify context storage
            ctx.store.set.assert_any_call("tickers", ["AAPL", "MSFT"])
            ctx.store.set.assert_any_call("num_to_collect", 2)

            # Verify events sent
            assert ctx.send_event.call_count == 2

            # Verify return type
            assert result.__class__.__name__ == "DispatchEvent"

    @pytest.mark.asyncio
    async def test_workflow_fetch_ticker_metrics_step(
        self, sample_yahoo_info_model, mock_provider_result
    ):
        """Test the fetch_ticker_metrics step."""
        mock_provider_result.data = sample_yahoo_info_model

        with patch(
            "app.flows.compare.metrics.create_yahoo_info_provider"
        ) as mock_create:
            mock_provider = AsyncMock()
            mock_provider.get_data.return_value = mock_provider_result
            mock_create.return_value = mock_provider

            workflow = MetricsWorkflow()

            # Create mock fetch event
            fetch_event = MagicMock()
            fetch_event.ticker = "AAPL"

            result = await workflow.fetch_ticker_metrics(fetch_event)

            assert result.ticker == "AAPL"
            assert result.success is True
            assert isinstance(result.data, YahooInfoModel)

    @pytest.mark.asyncio
    async def test_workflow_fetch_ticker_metrics_step_failure(self):
        """Test the fetch_ticker_metrics step with provider failure."""
        mock_provider_result = MagicMock()
        mock_provider_result.success = False
        mock_provider_result.error_message = "API Error"

        with patch(
            "app.flows.compare.metrics.create_yahoo_info_provider"
        ) as mock_create:
            mock_provider = AsyncMock()
            mock_provider.get_data.return_value = mock_provider_result
            mock_create.return_value = mock_provider

            workflow = MetricsWorkflow()

            # Create mock fetch event
            fetch_event = MagicMock()
            fetch_event.ticker = "INVALID"

            result = await workflow.fetch_ticker_metrics(fetch_event)

            assert result.ticker == "INVALID"
            assert result.success is False
            assert "API Error" in result.error_message

    @pytest.mark.asyncio
    async def test_workflow_compile_metrics_step_success(self, sample_yahoo_info_model):
        """Test the compile_metrics step with successful data."""
        with patch(
            "app.flows.compare.metrics.create_yahoo_info_provider"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            workflow = MetricsWorkflow()

            # Create mock context
            ctx = MagicMock()
            ctx.store.get = AsyncMock()
            ctx.store.get.side_effect = lambda key: {
                "tickers": ["AAPL"],
                "start_time": 0.0,
                "num_to_collect": 1,
            }[key]

            # Create mock response event
            response_event = MagicMock()
            response_event.ticker = "AAPL"
            response_event.success = True
            response_event.data = sample_yahoo_info_model

            # Mock collect_events to return our event
            ctx.collect_events.return_value = [response_event]

            result = await workflow.compile_metrics(ctx, response_event)

            # Verify result
            assert result.__class__.__name__ == "StopEvent"
            assert isinstance(result.result, FlowResult)
            assert result.result.success is True
            assert isinstance(result.result.data, dict)
            assert "comparison_df" in result.result.data
            assert "raw_metrics" in result.result.data

    @pytest.mark.asyncio
    async def test_workflow_compile_metrics_step_dispatch_event(self):
        """Test the compile_metrics step with DispatchEvent (should return None)."""
        with patch(
            "app.flows.compare.metrics.create_yahoo_info_provider"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            workflow = MetricsWorkflow()

            # Create mock context and dispatch event
            ctx = MagicMock()

            # Import the actual DispatchEvent class and use isinstance check
            from app.flows.compare.metrics import DispatchEvent

            dispatch_event = DispatchEvent()

            result = await workflow.compile_metrics(ctx, dispatch_event)

            # Should return None for DispatchEvent
            assert result is None

    @pytest.mark.asyncio
    async def test_workflow_compile_metrics_step_all_failures(self):
        """Test the compile_metrics step when all tickers fail."""
        with patch(
            "app.flows.compare.metrics.create_yahoo_info_provider"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            workflow = MetricsWorkflow()

            # Create mock context
            ctx = MagicMock()
            ctx.store.get = AsyncMock()
            ctx.store.get.side_effect = lambda key: {
                "tickers": ["INVALID"],
                "start_time": 0.0,
                "num_to_collect": 1,
            }[key]

            # Create mock response event with failure
            response_event = MagicMock()
            response_event.ticker = "INVALID"
            response_event.success = False
            response_event.error_message = "Not found"

            # Mock collect_events to return our event
            ctx.collect_events.return_value = [response_event]

            # Should raise FlowException when all tickers fail
            with pytest.raises(Exception) as exc_info:
                await workflow.compile_metrics(ctx, response_event)

            # Check it's the expected exception type (FlowException format)
            assert "Failed to fetch metrics for all" in str(exc_info.value)
