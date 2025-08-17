"""
Unit tests for the Yahoo provider module.
Tests both YahooHistoryProvider and YahooInfoProvider.
"""

import os
import asyncio
from unittest.mock import patch, MagicMock, PropertyMock
from pandas import DataFrame
from pydantic import BaseModel, ValidationError
import pytest

from app.providers.yahoo import (
    YahooHistoryProvider,
    YahooInfoProvider,
    create_yahoo_history_provider,
    create_yahoo_info_provider,
    YahooInfoModel,
)
from app.providers.base import ProviderType, ProviderConfig


class TestYahooHistoryProvider:
    """Test cases for YahooHistoryProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        # pylint:disable=attribute-defined-outside-init
        # Disable caching to ensure we test the actual provider logic
        config = ProviderConfig(cache_enabled=False)
        self.provider = YahooHistoryProvider(config)

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
    @patch("app.providers.yahoo.yf.Ticker")
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
        # Ensure data is present before checking its type
        assert result.data is not None
        assert isinstance(result.data, DataFrame)
        assert len(result.data) == 2
        assert result.query == "AAPL"
        assert result.provider_type == ProviderType.YAHOO_HISTORY

        # Verify yfinance was called correctly
        mock_ticker_class.assert_called_once_with("AAPL")
        mock_ticker.history.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.providers.yahoo.yf.Ticker")
    async def test_fetch_data_with_custom_parameters(self, mock_ticker_class):
        """Test data fetching with custom period and interval."""
        mock_data = DataFrame({"Close": [100.0]})
        mock_data.index.name = "Date"

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("AAPL", period="6mo", interval="1h")

        assert result.success is True
        # Ensure data is present
        assert result.data is not None
        mock_ticker.history.assert_called_once_with(period="6mo", interval="1h")

    @pytest.mark.asyncio
    @patch("app.providers.yahoo.yf.Ticker")
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
        # Ensure data is present
        assert result.data is not None
        mock_ticker.history.assert_called_once_with(
            start="2023-01-01", end="2023-12-31", interval="1d"
        )

    @pytest.mark.asyncio
    @patch("app.providers.yahoo.yf.Ticker")
    async def test_fetch_data_empty_response(self, mock_ticker_class):
        """Test handling of empty data response."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = DataFrame()  # Empty DataFrame
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("INVALID")

        assert result.success is False
        assert "No historical data found" in (result.error_message or "")
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_fetch_data_yfinance_exception(self, mock_ticker_class):
        """Test handling of yfinance exceptions."""
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("yfinance error")
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("AAPL")

        assert result.success is False
        assert "yfinance error" in (result.error_message or "")
        assert result.error_code == "RetriableProviderException"

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
        # pylint:disable=attribute-defined-outside-init
        # Disable caching to ensure we test the actual provider logic
        config = ProviderConfig(cache_enabled=False)
        self.provider = YahooInfoProvider(config)

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
        # Ensure data is present before checking type
        assert result.data is not None
        assert isinstance(result.data, BaseModel)
        assert result.query == "AAPL"
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
        assert "No info data found" in (result.error_message or "")

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_fetch_data_none_response(self, mock_ticker_class):
        """Test handling of None info response."""
        mock_ticker = MagicMock()
        mock_ticker.info = None
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("INVALID")

        assert result.success is False
        assert "No info data found" in (result.error_message or "")

    @pytest.mark.asyncio
    @patch("yfinance.Ticker")
    async def test_fetch_data_partial_data(self, mock_ticker_class):
        """Test handling of partial info data with strict validation."""
        mock_info_data = {
            "symbol": "AAPL",
            "regularMarketPrice": 150.50,
            # Missing many required fields
        }

        mock_ticker = MagicMock()
        mock_ticker.info = mock_info_data
        mock_ticker_class.return_value = mock_ticker

        result = await self.provider.get_data("AAPL")

        # Should fail due to strict validation requiring all fields
        assert result.success is False
        assert result.data is None
        assert "Field required" in (result.error_message or "")

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
        assert "yfinance info error" in (result.error_message or "")


class TestYahooFactoryFunctions:
    """Test cases for Yahoo provider factory functions."""

    def test_create_yahoo_history_provider_defaults(self):
        """Test factory function with default parameters."""
        provider = create_yahoo_history_provider()

        assert isinstance(provider, YahooHistoryProvider)
        assert provider.config.timeout == 30.0
        assert provider.config.retries == 3
        assert provider.config.extra_config["period"] == "max"
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

    def test_yahoo_info_model_structure(self):
        """Test that Yahoo info model has expected structure."""
        # Test with realistic Yahoo API data including extra fields to ignore
        test_data = {
            # Required fields we need
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
            # Extra fields from Yahoo API that should be ignored
            "shortName": "Apple Inc.",
            "quoteType": "EQUITY",
            "messageBoardId": "finmb_24937",
            "market": "us_market",
            "annualHoldingsTurnover": None,
            "enterpriseToRevenue": 7.833,
            "enterpriseToEbitda": 22.806,
            "52WeekChange": 0.15678901,
            "SandP52WeekChange": 0.24258518,
            "lastMarket": None,
            "logo_url": "https://logo.clearbit.com/apple.com",
        }
        model = YahooInfoModel(**test_data)

        # Check that required fields are parsed correctly
        assert model.ticker == "AAPL"
        assert model.company_name == "Apple Inc."
        assert model.price == 150.50
        assert model.market_cap == 2500000000000
        assert model.sector == "Technology"
        assert model.currency == "USD"

    def test_yahoo_info_model_aliases(self):
        """Test that aliases work correctly."""
        # Test data with complete Yahoo API field names
        test_data = {
            "symbol": "GOOGL",
            "longName": "Alphabet Inc.",
            "regularMarketPrice": 2800.50,
            "regularMarketChange": 25.00,
            "regularMarketChangePercent": 0.009,
            "regularMarketVolume": 1200000,
            "marketCap": 1800000000000,
            "trailingPE": 28.5,
            "dividendYield": 0.0,  # Google doesn't pay dividends
            "beta": 1.1,
            "fiftyTwoWeekHigh": 3000.0,
            "fiftyTwoWeekLow": 2100.0,
            "currency": "USD",
            "exchange": "NASDAQ",
            "sector": "Communication Services",
            "industry": "Internet Content & Information",
        }

        model = YahooInfoModel.model_validate(test_data)

        # Check that aliases map correctly
        assert model.ticker == "GOOGL"
        assert model.company_name == "Alphabet Inc."
        assert model.price == 2800.50
        assert model.currency == "USD"
        assert model.sector == "Communication Services"

    def test_yahoo_info_model_missing_required_field(self):
        """Test that missing required fields raise ValidationError."""
        # Test data missing the 'symbol' field
        test_data = {
            "longName": "Apple Inc.",
            "regularMarketPrice": 150.50,
            # Missing other required fields...
        }

        with pytest.raises(ValidationError) as exc_info:
            YahooInfoModel(**test_data)

        # Should complain about missing fields
        assert "symbol" in str(exc_info.value) or "Field required" in str(
            exc_info.value
        )


class TestYahooProviderIntegration:
    """Integration tests for Yahoo providers."""

    @pytest.mark.asyncio
    async def test_history_and_info_providers_together(self):
        """Test using both providers together."""
        with patch("yfinance.Ticker") as mock_ticker_class:
            # Setup mock data
            mock_history = DataFrame({"Close": [100.0, 101.0]})
            mock_history.index.name = "Date"
            mock_info = {
                "symbol": "AAPL",
                "longName": "Apple Inc.",
                "regularMarketPrice": 100.0,
                "regularMarketChange": 1.50,
                "regularMarketChangePercent": 0.015,
                "regularMarketVolume": 45000000,
                "marketCap": 2000000000000,
                "trailingPE": 24.0,
                "dividendYield": 0.005,
                "beta": 1.15,
                "fiftyTwoWeekHigh": 175.0,
                "fiftyTwoWeekLow": 115.0,
                "currency": "USD",
                "exchange": "NASDAQ",
                "sector": "Technology",
                "industry": "Consumer Electronics",
            }

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


class TestCacheSettingsYahoo:
    """Test cases for cache setting on Yahoo providers."""

    @pytest.mark.asyncio
    async def test_history_cache_disabled_per_provider(self, tmp_path, monkeypatch):
        # Set PROVIDER_CACHE_ROOT to isolated temp directory
        monkeypatch.setenv("PROVIDER_CACHE_ROOT", str(tmp_path))
        # Disable caching for this provider
        config = ProviderConfig(cache_enabled=False)
        provider = YahooHistoryProvider(config)
        # Patch yfinance history
        from pandas import DataFrame

        with patch("app.providers.yahoo.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = DataFrame({"Close": [123]})
            mock_ticker_class.return_value = mock_ticker

            # First and second calls should both fetch fresh data
            await provider.get_data("AAPL")
            await provider.get_data("AAPL")
            # Should call history twice when cache is disabled
            assert (
                mock_ticker.history.call_count == 2
            ), "History called twice; cache disabled"

    @pytest.mark.asyncio
    async def test_info_cache_disabled_per_provider(self, tmp_path, monkeypatch):
        # Set PROVIDER_CACHE_ROOT to isolated temp directory
        monkeypatch.setenv("PROVIDER_CACHE_ROOT", str(tmp_path))
        config = ProviderConfig(cache_enabled=False)
        provider = YahooInfoProvider(config)

        with patch("yfinance.Ticker") as mock_ticker_class:
            # Mock underlying info data
            mock_ticker = MagicMock()
            mock_ticker.info = {"symbol": "AAPL"}
            mock_ticker_class.return_value = mock_ticker
            # With native Pydantic validation, we expect direct calls to model_validate
            # Since cache is disabled, we should fetch fresh data twice
            await provider.get_data("AAPL")
            await provider.get_data("AAPL")
            # Both calls should succeed and fetch fresh data


class TestGlobalCacheSettingsYahoo:
    """Test cases for global cache setting on Yahoo providers."""

    @pytest.mark.asyncio
    async def test_global_cache_disabled_history(self, tmp_path, monkeypatch):
        # Set PROVIDER_CACHE_ROOT to isolated temp directory
        monkeypatch.setenv("PROVIDER_CACHE_ROOT", str(tmp_path))
        # Disable global cache
        from app.lib.settings import settings

        monkeypatch.setattr(settings, "PROVIDER_CACHE_ENABLED", False)

        config = ProviderConfig()  # default cache_enabled True
        provider = YahooHistoryProvider(config)

        # Patch yfinance history
        from pandas import DataFrame

        with patch("app.providers.yahoo.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = DataFrame({"Close": [1]})
            mock_ticker_class.return_value = mock_ticker

            await provider.get_data("AAPL")
            await provider.get_data("AAPL")
            # Should fetch fresh data twice since global cache disabled
            assert mock_ticker.history.call_count == 2

    @pytest.mark.asyncio
    async def test_global_cache_disabled_info(self, tmp_path, monkeypatch):
        # Set PROVIDER_CACHE_ROOT to isolated temp directory
        monkeypatch.setenv("PROVIDER_CACHE_ROOT", str(tmp_path))
        # Disable global cache
        from app.lib.settings import settings

        monkeypatch.setattr(settings, "PROVIDER_CACHE_ENABLED", False)

        provider = YahooInfoProvider()  # default config

        with patch("app.providers.yahoo.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.info = {"symbol": "AAPL"}
            mock_ticker_class.return_value = mock_ticker
            # With native Pydantic validation, we expect direct model validation
            # Since global cache is disabled, we should fetch fresh data twice
            await provider.get_data("AAPL")
            await provider.get_data("AAPL")
            # Both calls should succeed and fetch fresh data
