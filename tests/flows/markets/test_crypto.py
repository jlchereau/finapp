"""
Unit tests for Cryptocurrency workflow and FlowRunner integration.
"""

import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from app.flows.markets.crypto import (
    fetch_crypto_data,
    CryptoCurrencyWorkflow,
)
from app.flows.base import FlowRunner, FlowResultEvent


@pytest.fixture
def sample_bitcoin_data():
    """Create sample Bitcoin data."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    btc_values = [30000 + (i % 50) * 500 + (i // 50) * 1000 for i in range(100)]
    return pd.DataFrame(
        {
            "Open": [v - 500 for v in btc_values],
            "High": [v + 1000 for v in btc_values],
            "Low": [v - 1000 for v in btc_values],
            "Close": btc_values,
            "Adj Close": btc_values,
            "Volume": [1000000 + i * 10000 for i in range(100)],
        },
        index=dates,
    )


@pytest.fixture
def sample_ethereum_data():
    """Create sample Ethereum data."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    eth_values = [2000 + (i % 30) * 50 + (i // 30) * 100 for i in range(100)]
    return pd.DataFrame(
        {
            "Open": [v - 50 for v in eth_values],
            "High": [v + 100 for v in eth_values],
            "Low": [v - 100 for v in eth_values],
            "Close": eth_values,
            "Adj Close": eth_values,
            "Volume": [500000 + i * 5000 for i in range(100)],
        },
        index=dates,
    )


class TestCryptoCurrencyWorkflow:
    """Test the CryptoCurrencyWorkflow class."""

    def test_workflow_initialization(self):
        """Test that workflow initializes correctly."""
        workflow = CryptoCurrencyWorkflow()
        assert workflow is not None
        assert hasattr(workflow, "yahoo_provider")

    @pytest.mark.asyncio
    async def test_initiate_crypto_fetch(self):
        """Test the dispatch step that sends parallel events."""
        workflow = CryptoCurrencyWorkflow()

        # Create mock context
        ctx = MagicMock()
        ctx.store.set = AsyncMock()
        ctx.send_event = MagicMock()

        # Create start event
        start_event = MagicMock()
        start_event.base_date = datetime(2020, 1, 1)

        # Execute the step
        result = await workflow.initiate_crypto_fetch(ctx, start_event)

        # Verify context operations
        ctx.store.set.assert_any_call("base_date", start_event.base_date)
        assert ctx.send_event.call_count == 2  # Bitcoin and Ethereum events
        assert result is not None

    @pytest.mark.asyncio
    async def test_fetch_bitcoin_data_success(self):
        """Test successful Bitcoin data fetching step."""
        workflow = CryptoCurrencyWorkflow()

        btc_data = pd.DataFrame(
            {"Close": [30000.0, 31000.0, 32000.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        btc_result = MagicMock()
        btc_result.success = True
        btc_result.data = btc_data

        workflow.yahoo_provider.get_data = AsyncMock(return_value=btc_result)

        fetch_event = MagicMock()
        fetch_event.base_date = datetime(2020, 1, 1)

        result = await workflow.fetch_bitcoin_data(fetch_event)

        assert result.data is not None
        assert not result.data.empty
        assert result.error is None

    @pytest.mark.asyncio
    async def test_fetch_ethereum_data_success(self):
        """Test successful Ethereum data fetching step."""
        workflow = CryptoCurrencyWorkflow()

        eth_data = pd.DataFrame(
            {"Close": [2000.0, 2100.0, 2200.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        eth_result = MagicMock()
        eth_result.success = True
        eth_result.data = eth_data

        workflow.yahoo_provider.get_data = AsyncMock(return_value=eth_result)

        fetch_event = MagicMock()
        fetch_event.base_date = datetime(2020, 1, 1)

        result = await workflow.fetch_ethereum_data(fetch_event)

        assert result.data is not None
        assert not result.data.empty
        assert result.error is None


class TestCryptoCurrencyFlowRunner:
    """Test the Crypto workflow with FlowRunner."""

    @pytest.mark.asyncio
    async def test_flowrunner_integration_success(self):
        """Test successful workflow execution with FlowRunner."""
        workflow = CryptoCurrencyWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        mock_result = FlowResultEvent.success_result(
            data=pd.DataFrame(
                {
                    "BTC": [30000.0, 31000.0],
                    "ETH": [2000.0, 2100.0],
                }
            ),
            metadata={
                "latest_btc": 31000.0,
                "latest_eth": 2100.0,
                "data_points": 2,
            },
        )

        workflow.run = AsyncMock(return_value=mock_result)

        result = await runner.run(base_date=datetime(2020, 1, 1))

        assert isinstance(result, FlowResultEvent)
        assert result.success is True
        assert result.data is not None
        assert "BTC" in result.data.columns
        assert "ETH" in result.data.columns


class TestFetchCryptoData:
    """Test the fetch_crypto_data function."""

    @pytest.mark.asyncio
    async def test_fetch_crypto_data_success(
        self, sample_bitcoin_data, sample_ethereum_data
    ):
        """Test successful crypto data fetch and calculation."""
        btc_result = MagicMock()
        btc_result.success = True
        btc_result.data = sample_bitcoin_data

        eth_result = MagicMock()
        eth_result.success = True
        eth_result.data = sample_ethereum_data

        with patch(
            "app.flows.markets.crypto.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()

            def mock_get_data(query):
                if query == "BTC-USD":
                    return btc_result
                elif query == "ETH-USD":
                    return eth_result
                else:
                    raise ValueError(f"Unexpected query: {query}")

            mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
            mock_yahoo.return_value = mock_yahoo_instance

            base_date = datetime(2020, 1, 1)
            result = await fetch_crypto_data(base_date)

            assert "data" in result
            assert "base_date" in result
            assert "latest_btc" in result
            assert "latest_eth" in result
            assert "data_points" in result

            data = result["data"]
            assert not data.empty
            assert "BTC" in data.columns
            assert "ETH" in data.columns

            latest_btc = result["latest_btc"]
            latest_eth = result["latest_eth"]
            assert isinstance(latest_btc, (float, int)) or hasattr(latest_btc, "dtype")
            assert isinstance(latest_eth, (float, int)) or hasattr(latest_eth, "dtype")
            assert 10000 <= latest_btc <= 100000  # Reasonable BTC range
            assert 1000 <= latest_eth <= 10000  # Reasonable ETH range

    @pytest.mark.asyncio
    async def test_crypto_data_filtering_by_base_date(
        self, sample_bitcoin_data, sample_ethereum_data
    ):
        """Test that crypto data is properly filtered by base_date."""
        btc_result = MagicMock()
        btc_result.success = True
        btc_result.data = sample_bitcoin_data

        eth_result = MagicMock()
        eth_result.success = True
        eth_result.data = sample_ethereum_data

        with patch(
            "app.flows.markets.crypto.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()

            def mock_get_data(query):
                if query == "BTC-USD":
                    return btc_result
                elif query == "ETH-USD":
                    return eth_result
                else:
                    raise ValueError(f"Unexpected query: {query}")

            mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
            mock_yahoo.return_value = mock_yahoo_instance

            base_date = datetime(2020, 2, 15)  # Mid-period
            result = await fetch_crypto_data(base_date)

            data = result["data"]
            assert not data.empty

            # All data should be from base_date onwards
            assert data.index.min() >= pd.to_datetime(base_date.date())

            # Should be less than full dataset
            assert len(data) < 100

    @pytest.mark.asyncio
    async def test_crypto_workflow_direct(self):
        """Test the CryptoCurrencyWorkflow class directly."""
        btc_data = pd.DataFrame(
            {
                "Close": [30000.0, 31000.0, 32000.0, 33000.0, 34000.0],
                "Volume": [1000000] * 5,
            },
            index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
        )

        eth_data = pd.DataFrame(
            {
                "Close": [2000.0, 2100.0, 2200.0, 2300.0, 2400.0],
                "Volume": [500000] * 5,
            },
            index=pd.date_range(start="2020-01-01", periods=5, freq="D"),
        )

        btc_result = MagicMock()
        btc_result.success = True
        btc_result.data = btc_data

        eth_result = MagicMock()
        eth_result.success = True
        eth_result.data = eth_data

        with patch(
            "app.flows.markets.crypto.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()

            def mock_get_data(query):
                if query == "BTC-USD":
                    return btc_result
                elif query == "ETH-USD":
                    return eth_result
                else:
                    raise ValueError(f"Unexpected query: {query}")

            mock_yahoo_instance.get_data = AsyncMock(side_effect=mock_get_data)
            mock_yahoo.return_value = mock_yahoo_instance

            workflow = CryptoCurrencyWorkflow()
            base_date = datetime(2020, 1, 1)

            result = await workflow.run(base_date=base_date)

            assert result.data is not None
            assert result.metadata is not None
            assert "latest_btc" in result.metadata
            assert "latest_eth" in result.metadata

            data = result.data
            assert not data.empty
            assert len(data) == 5
            assert "BTC" in data.columns
            assert "ETH" in data.columns

            latest_btc = result.metadata["latest_btc"]
            latest_eth = result.metadata["latest_eth"]
            assert latest_btc == 34000.0
            assert latest_eth == 2400.0

    @pytest.mark.asyncio
    async def test_crypto_provider_failure(self):
        """Test handling of crypto provider failure."""
        btc_result = MagicMock()
        btc_result.success = False
        btc_result.error_message = "Bitcoin provider failed"

        with patch(
            "app.flows.markets.crypto.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=btc_result)
            mock_yahoo.return_value = mock_yahoo_instance

            workflow = CryptoCurrencyWorkflow()
            try:
                await workflow.run(base_date=datetime(2020, 1, 1))
                assert False, "Expected exception to be raised"
            except Exception as e:
                assert "Bitcoin data fetch failed" in str(e)

    @pytest.mark.asyncio
    async def test_crypto_no_close_price_data(self):
        """Test handling when crypto data has no Close price column."""
        btc_data = pd.DataFrame(
            {
                "Open": [30000.0, 31000.0, 32000.0],
                "High": [31000.0, 32000.0, 33000.0],
                "Low": [29000.0, 30000.0, 31000.0],
                "Volume": [1000000] * 3,
            },
            index=pd.date_range(start="2020-01-01", periods=3, freq="D"),
        )

        btc_result = MagicMock()
        btc_result.success = True
        btc_result.data = btc_data

        with patch(
            "app.flows.markets.crypto.create_yahoo_history_provider"
        ) as mock_yahoo:
            mock_yahoo_instance = AsyncMock()
            mock_yahoo_instance.get_data = AsyncMock(return_value=btc_result)
            mock_yahoo.return_value = mock_yahoo_instance

            workflow = CryptoCurrencyWorkflow()
            try:
                await workflow.run(base_date=datetime(2020, 1, 1))
                assert False, "Expected exception to be raised"
            except Exception as e:
                assert "No Close price data available for Bitcoin" in str(e)
