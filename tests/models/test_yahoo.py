#!/usr/bin/env pytest
# pylint: skip-file
# flake8: noqa
# type: ignore
#!/usr/bin/env pytest
# pylint: skip-file
# flake8: noqa
# type: ignore
"""
Unit tests for the Yahoo provider module.
Tests both YahooHistoryProvider and YahooInfoProvider.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pandas import DataFrame
from pydantic import BaseModel

from app.models.yahoo import (
    YahooHistoryProvider,
    YahooInfoProvider,
    create_yahoo_history_provider,
    create_yahoo_info_provider,
    YAHOO_INFO_CONFIG,
)
from app.models.base import ProviderType, ProviderConfig


class TestYahooHistoryProvider:
    """Test cases for YahooHistoryProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = YahooHistoryProvider()

    def test_provider_type(self):
        """Test that provider returns correct type."""
        assert self.provider.provider_type == ProviderType.YAHOO_HISTORY

    def test_provider_initialization(self):
        """Test provider initialization."""
        assert isinstance(self.provider.config, ProviderConfig)
        assert hasattr(self.provider, "logger")

    def test_provider_initialization_with_custom_config(self):
        """Test provider initialization with custom config."""
        config = ProviderConfig(
            timeout=60.0, retries=5, extra_config={"period": "2y", "interval": "1h"}
        )
        provider = YahooHistoryProvider(config)

        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5
        assert provider.config.extra_config["period"] == "2y"
        assert provider.config.extra_config["interval"] == "1h"

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_fetch_data_success(self, mock_ticker_class):
        """Test successful data fetching."""
        # Mock the yfinance response
        mock_data = DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 100.0],
                "Close": [104.0, 105.0],
                "Volume": [1000000, 1100000],
            }
        )
        mock_data.index.name = "Date"

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("AAPL")

        assert result.success is True
        assert isinstance(result.data, DataFrame)
        assert len(result.data) == 2
        assert result.ticker == "AAPL"
        assert result.provider_type == ProviderType.YAHOO_HISTORY

        # Verify yfinance was called correctly
        mock_ticker_class.assert_called_once_with("AAPL")
        mock_ticker.history.assert_called_once()

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_fetch_data_with_custom_parameters(self, mock_ticker_class):
        """Test data fetching with custom period and interval."""
        mock_data = DataFrame({"Close": [100.0]})
        mock_data.index.name = "Date"

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("AAPL", period="6mo", interval="1h")

        assert result.success is True
        mock_ticker.history.assert_called_once_with(period="6mo", interval="1h")

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_fetch_data_with_date_range(self, mock_ticker_class):
        """Test data fetching with start/end dates."""
        mock_data = DataFrame({"Close": [100.0]})
        mock_data.index.name = "Date"

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data(
            "AAPL", start="2023-01-01", end="2023-12-31", interval="1d"
        )

        assert result.success is True
        mock_ticker.history.assert_called_once_with(
            start="2023-01-01", end="2023-12-31", interval="1d"
        )

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_fetch_data_empty_response(self, mock_ticker_class):
        """Test handling of empty data response."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = DataFrame()  # Empty DataFrame
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("INVALID")

        assert result.success is False
        assert "No historical data found" in result.error_message
        assert result.error_code == "ValueError"

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_fetch_data_yfinance_exception(self, mock_ticker_class):
        """Test handling of yfinance exceptions."""
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("yfinance error")
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("AAPL")

        assert result.success is False
        assert "yfinance error" in result.error_message

    def test_get_data_sync(self):
        """Test synchronous wrapper."""
        with patch("yfinance.Ticker") as mock_ticker_class:
            mock_data = DataFrame({"Close": [100.0]})
            mock_data.index.name = "Date"

            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_data
            mock_ticker_class.return_value = mock_ticker

            result = self.provider.get_data_sync("AAPL")

            assert result.success is True


class TestYahooInfoProvider:
    """Test cases for YahooInfoProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = YahooInfoProvider()

    def test_provider_type(self):
        """Test that provider returns correct type."""
        assert self.provider.provider_type == ProviderType.YAHOO_INFO

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_fetch_data_success(self, mock_ticker_class):
        """Test successful info data fetching."""
        mock_info_data = {
            "symbol": "AAPL",
            "longName": "Apple Inc.",
            "regularMarketPrice": 150.50,
            "regularMarketChange": 2.50,
            "regularMarketChangePercent": 0.0168,
            "regularMarketVolume": 50000000,
            "marketCap": 2500000000000,
            "trailingPE": 25.5,
            "dividendYield": 0.006,
            "beta": 1.2,
            "fiftyTwoWeekHigh": 180.0,
            "fiftyTwoWeekLow": 120.0,
            "currency": "USD",
            "exchange": "NMS",
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }

        mock_ticker = MagicMock()
        mock_ticker.info = mock_info_data
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("AAPL")

        assert result.success is True
        assert isinstance(result.data, BaseModel)
        assert result.ticker == "AAPL"
        assert result.provider_type == ProviderType.YAHOO_INFO

        # Check parsed data fields (access via getattr due to dynamic model)
        assert getattr(result.data, "ticker") == "AAPL"
        assert getattr(result.data, "company_name") == "Apple Inc."
        assert getattr(result.data, "price") == 150.50

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_fetch_data_empty_response(self, mock_ticker_class):
        """Test handling of empty info response."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}  # Empty dict
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("INVALID")

        assert result.success is False
        assert "No info data found" in result.error_message

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_fetch_data_none_response(self, mock_ticker_class):
        """Test handling of None info response."""
        mock_ticker = MagicMock()
        mock_ticker.info = None
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("INVALID")

        assert result.success is False
        assert "No info data found" in result.error_message

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_fetch_data_partial_data(self, mock_ticker_class):
        """Test handling of partial info data."""
        mock_info_data = {
            "symbol": "AAPL",
            "regularMarketPrice": 150.50,
            # Missing many fields
        }

        mock_ticker = MagicMock()
        mock_ticker.info = mock_info_data
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("AAPL")

        # Should succeed due to non-strict parsing
        assert result.success is True
        assert getattr(result.data, "ticker") == "AAPL"
        assert getattr(result.data, "price") == 150.50
        # Missing fields should be None (default)
        assert getattr(result.data, "company_name") is None

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_fetch_data_yfinance_exception(self, mock_ticker_class):
        """Test handling of yfinance exceptions."""
        mock_ticker = MagicMock()

        # Make the info property raise an exception when accessed
        type(mock_ticker).info = PropertyMock(
            side_effect=Exception("yfinance info error")
        )
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("AAPL")

        assert result.success is False
        assert "yfinance info error" in result.error_message


class TestYahooFactoryFunctions:
    """Test cases for Yahoo provider factory functions."""

    def test_create_yahoo_history_provider_defaults(self):
        """Test factory function with default parameters."""
        provider = create_yahoo_history_provider()

        assert isinstance(provider, YahooHistoryProvider)
        assert provider.config.timeout == 30.0
        assert provider.config.retries == 3
        assert provider.config.extra_config["period"] == "1y"
        assert provider.config.extra_config["interval"] == "1d"

    def test_create_yahoo_history_provider_custom(self):
        """Test factory function with custom parameters."""
        provider = create_yahoo_history_provider(
            period="2y", interval="1h", timeout=60.0, retries=5
        )

        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5
        assert provider.config.extra_config["period"] == "2y"
        assert provider.config.extra_config["interval"] == "1h"

    def test_create_yahoo_info_provider_defaults(self):
        """Test info provider factory with default parameters."""
        provider = create_yahoo_info_provider()

        assert isinstance(provider, YahooInfoProvider)
        assert provider.config.timeout == 30.0
        assert provider.config.retries == 3

    def test_create_yahoo_info_provider_custom(self):
        """Test info provider factory with custom parameters."""
        provider = create_yahoo_info_provider(timeout=45.0, retries=2)

        assert provider.config.timeout == 45.0
        assert provider.config.retries == 2


class TestYahooInfoConfig:
    """Test cases for Yahoo info configuration."""

    def test_yahoo_info_config_structure(self):
        """Test that Yahoo info config has expected structure."""
        assert YAHOO_INFO_CONFIG.name == "YahooInfoModel"
        assert YAHOO_INFO_CONFIG.strict_mode is False
        assert YAHOO_INFO_CONFIG.default_value is None

        # Check some key fields
        fields = YAHOO_INFO_CONFIG.fields
        assert "ticker" in fields
        assert "company_name" in fields
        assert "price" in fields
        assert "market_cap" in fields

        # Check expressions
        assert fields["ticker"]["expr"] == "symbol"
        assert fields["company_name"]["expr"] == "longName"
        assert fields["price"]["expr"] == "regularMarketPrice"

    def test_yahoo_info_config_all_fields_have_defaults(self):
        """Test that all fields have default values."""
        for field_name, field_config in YAHOO_INFO_CONFIG.fields.items():
            assert "default" in field_config
            # Most should have None default, except currency
            if field_name == "currency":
                assert field_config["default"] == "USD"
            else:
                assert field_config["default"] is None


class TestYahooProviderIntegration:
    """Integration tests for Yahoo providers."""

    @pytest.mark.asyncio
    async def test_history_and_info_providers_together(self):
        """Test using both providers together."""
        with patch("yfinance.Ticker") as mock_ticker_class:
            # Setup mock data
            mock_history = DataFrame({"Close": [100.0, 101.0]})
            mock_history.index.name = "Date"
            mock_info = {"symbol": "AAPL", "regularMarketPrice": 100.0}

            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_history
            mock_ticker.info = mock_info
            mock_ticker_class.return_value = mock_ticker

            # Test both providers
            history_provider = YahooHistoryProvider()
            info_provider = YahooInfoProvider()

            history_result = await history_provider.get_data("AAPL")
            info_result = await info_provider.get_data("AAPL")

            assert history_result.success is True
            assert info_result.success is True
            assert isinstance(history_result.data, DataFrame)
            assert isinstance(info_result.data, BaseModel)

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent requests to Yahoo providers."""
        with patch("yfinance.Ticker") as mock_ticker_class:
            mock_data = DataFrame({"Close": [100.0]})
            mock_data.index.name = "Date"

            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_data
            mock_ticker_class.return_value = mock_ticker

            provider = YahooHistoryProvider()

            # Make multiple concurrent requests
            tasks = [
                provider.get_data("AAPL"),
                provider.get_data("GOOGL"),
                provider.get_data("MSFT"),
            ]

            results = await asyncio.gather(*tasks)

            # All should succeed
            assert all(result.success for result in results)
            assert len(results) == 3
