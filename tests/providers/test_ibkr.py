"""
Unit tests for the IBKR provider module.
Tests IBKRApp, IBKRPositionsProvider, and IBKRCashProvider.
"""

import asyncio
import threading
from unittest.mock import patch, MagicMock
from pandas import DataFrame
import pytest

from app.providers.ibkr import (
    IBKRApp,
    IBKRPositionsProvider,
    IBKRCashProvider,
    create_ibkr_positions_provider,
    create_ibkr_cash_provider,
    run_ibkr_loop,
)
from app.providers.base import ProviderType, ProviderConfig


class TestIBKRApp:
    """Test cases for IBKRApp class."""

    def setup_method(self):
        """Set up test fixtures."""
        # pylint:disable=attribute-defined-outside-init
        self.app = IBKRApp()

    def test_ibkr_app_initialization(self):
        """Test IBKRApp initialization."""
        assert self.app.positions_data == []
        assert self.app.cash_data == []
        assert isinstance(self.app.positions_ready, threading.Event)
        assert isinstance(self.app.cash_ready, threading.Event)
        assert isinstance(self.app.connected, threading.Event)
        assert self.app.next_order_id is None

    def test_error_handling_informational_messages(self):
        """Test that informational error codes are logged as debug."""
        # Test informational error codes that should be logged as debug
        informational_codes = [2104, 2106, 2107, 2158]

        for error_code in informational_codes:
            with patch("app.providers.ibkr.logger.debug") as mock_debug:
                self.app.error(1, error_code, f"Info message {error_code}")
                mock_debug.assert_called_once()

    def test_error_handling_actual_errors(self):
        """Test that actual errors are logged as warnings."""
        with patch("app.providers.ibkr.logger.warning") as mock_warning:
            self.app.error(1, 500, "Real error message")
            mock_warning.assert_called_once_with("IBKR error 1 500: Real error message")

    def test_nextValidId_callback(self):
        """Test nextValidId callback sets connection event."""
        with patch("app.providers.ibkr.logger.info") as mock_info:
            self.app.nextValidId(123)

            assert self.app.next_order_id == 123
            assert self.app.connected.is_set()
            mock_info.assert_called_once_with(
                "IBKR connected, next valid order ID: 123"
            )

    def test_position_callback(self):
        """Test position callback accumulates data."""
        # Mock contract object
        mock_contract = MagicMock()
        mock_contract.symbol = "AAPL"
        mock_contract.secType = "STK"
        mock_contract.currency = "USD"
        mock_contract.exchange = "NASDAQ"

        self.app.position("U123456", mock_contract, 100.0, 150.50)

        assert len(self.app.positions_data) == 1
        position = self.app.positions_data[0]
        assert position["account"] == "U123456"
        assert position["symbol"] == "AAPL"
        assert position["secType"] == "STK"
        assert position["currency"] == "USD"
        assert position["exchange"] == "NASDAQ"
        assert position["position"] == 100.0
        assert position["avgCost"] == 150.50

    def test_positionEnd_callback(self):
        """Test positionEnd callback sets event and logs."""
        with patch("app.providers.ibkr.logger.info") as mock_info:
            # Add some test data first
            self.app.positions_data = [{"test": "data"}]

            self.app.positionEnd()

            assert self.app.positions_ready.is_set()
            mock_info.assert_called_once_with("Received 1 positions from IBKR")

    def test_updateAccountValue_callback_cash_balance(self):
        """Test updateAccountValue callback for cash balance."""
        self.app.updateAccountValue("CashBalance", "10000.50", "USD", "U123456")

        assert len(self.app.cash_data) == 1
        cash_entry = self.app.cash_data[0]
        assert cash_entry["account"] == "U123456"
        assert cash_entry["currency"] == "USD"
        assert cash_entry["value"] == 10000.50

    def test_updateAccountValue_callback_non_cash(self):
        """Test updateAccountValue callback ignores non-cash values."""
        self.app.updateAccountValue("NetLiquidation", "50000.00", "USD", "U123456")

        # Should not add anything to cash_data
        assert len(self.app.cash_data) == 0

    def test_accountDownloadEnd_callback(self):
        """Test accountDownloadEnd callback sets event and logs."""
        with patch("app.providers.ibkr.logger.info") as mock_info:
            self.app.accountDownloadEnd("U123456")

            assert self.app.cash_ready.is_set()
            mock_info.assert_called_once_with("Account download complete for: U123456")


class TestIBKRPositionsProvider:
    """Test cases for IBKRPositionsProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        # pylint:disable=attribute-defined-outside-init
        # Disable caching to ensure we test the actual provider logic
        config = ProviderConfig(cache_enabled=False)
        self.provider = IBKRPositionsProvider(config)

    def test_provider_type(self):
        """Test that provider returns correct type."""
        assert self.provider.provider_type == ProviderType.IBKR_POSITIONS

    def test_provider_initialization(self):
        """Test provider initialization."""
        assert isinstance(self.provider.config, ProviderConfig)
        assert hasattr(self.provider, "logger")

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_fetch_data_success(self, mock_to_thread):
        """Test successful positions data fetching."""
        # Mock the return value of asyncio.to_thread
        expected_data = DataFrame(
            [
                {
                    "account": "U123456",
                    "symbol": "AAPL",
                    "secType": "STK",
                    "currency": "USD",
                    "exchange": "NASDAQ",
                    "position": 100.0,
                    "avgCost": 150.50,
                }
            ]
        )
        mock_to_thread.return_value = expected_data

        result = await self.provider.get_data(None)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, DataFrame)
        assert len(result.data) == 1
        assert result.data.iloc[0]["symbol"] == "AAPL"
        assert result.data.iloc[0]["position"] == 100.0
        assert result.provider_type == ProviderType.IBKR_POSITIONS

        # Verify asyncio.to_thread was called
        mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_get_data_without_query_parameter(self, mock_to_thread):
        """Test that get_data works without providing query parameter."""
        expected_data = DataFrame([{"symbol": "AAPL", "position": 100.0}])
        mock_to_thread.return_value = expected_data

        # Test async method without query parameter
        result = await self.provider.get_data()

        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 1

    @patch("asyncio.to_thread")
    def test_get_data_sync_without_query_parameter(self, mock_to_thread):
        """Test that get_data_sync works without providing query parameter."""
        expected_data = DataFrame([{"symbol": "AAPL", "position": 100.0}])
        mock_to_thread.return_value = expected_data

        # Test sync method without query parameter
        result = self.provider.get_data_sync()

        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 1

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_fetch_data_empty_positions(self, mock_to_thread):
        """Test handling of empty positions response."""
        # Mock empty DataFrame
        expected_columns = [
            "account",
            "symbol",
            "secType",
            "currency",
            "exchange",
            "position",
            "avgCost",
        ]
        empty_data = DataFrame(columns=expected_columns)
        mock_to_thread.return_value = empty_data

        result = await self.provider.get_data(None)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, DataFrame)
        assert len(result.data) == 0
        # Check that DataFrame has expected columns
        assert list(result.data.columns) == expected_columns

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_fetch_data_connection_failure(self, mock_to_thread):
        """Test handling of connection failure."""
        # Mock connection error
        mock_to_thread.side_effect = ConnectionError(
            "Failed to connect to IBKR Gateway within 10 seconds"
        )

        result = await self.provider.get_data(None)

        assert result.success is False
        assert "Failed to connect to IBKR Gateway" in (result.error_message or "")
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_fetch_data_positions_timeout(self, mock_to_thread):
        """Test handling of positions data timeout."""
        # Mock timeout error
        mock_to_thread.side_effect = TimeoutError(
            "Timeout waiting for positions data from IBKR"
        )

        result = await self.provider.get_data(None)

        assert result.success is False
        assert "Timeout waiting for positions data" in (result.error_message or "")
        assert result.error_code == "RetriableProviderException"

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_fetch_data_ibkr_exception(self, mock_to_thread):
        """Test handling of IBKR API exceptions."""
        # Mock general exception
        mock_to_thread.side_effect = Exception("IBKR API error")

        result = await self.provider.get_data(None)

        assert result.success is False
        assert "IBKR API error" in (result.error_message or "")
        assert result.error_code == "RetriableProviderException"


class TestIBKRCashProvider:
    """Test cases for IBKRCashProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        # pylint:disable=attribute-defined-outside-init
        config = ProviderConfig(cache_enabled=False)
        self.provider = IBKRCashProvider(config)

    def test_provider_type(self):
        """Test that provider returns correct type."""
        assert self.provider.provider_type == ProviderType.IBKR_CASH

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_fetch_data_success(self, mock_to_thread):
        """Test successful cash data fetching."""
        # Mock successful cash data
        expected_data = DataFrame(
            [
                {"account": "U123456", "currency": "USD", "value": 50000.0},
                {"account": "U123456", "currency": "EUR", "value": 10000.0},
            ]
        )
        mock_to_thread.return_value = expected_data

        result = await self.provider.get_data(None)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, DataFrame)
        assert len(result.data) == 2
        assert result.provider_type == ProviderType.IBKR_CASH

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_get_data_without_query_parameter(self, mock_to_thread):
        """Test that get_data works without providing query parameter."""
        expected_data = DataFrame(
            [{"account": "U123456", "currency": "USD", "value": 50000.0}]
        )
        mock_to_thread.return_value = expected_data

        # Test async method without query parameter
        result = await self.provider.get_data()

        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 1

    @patch("asyncio.to_thread")
    def test_get_data_sync_without_query_parameter(self, mock_to_thread):
        """Test that get_data_sync works without providing query parameter."""
        expected_data = DataFrame(
            [{"account": "U123456", "currency": "USD", "value": 50000.0}]
        )
        mock_to_thread.return_value = expected_data

        # Test sync method without query parameter
        result = self.provider.get_data_sync()

        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 1

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_fetch_data_empty_cash(self, mock_to_thread):
        """Test handling of empty cash response."""
        # Mock empty DataFrame
        expected_columns = ["account", "currency", "value"]
        empty_data = DataFrame(columns=expected_columns)
        mock_to_thread.return_value = empty_data

        result = await self.provider.get_data(None)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, DataFrame)
        assert len(result.data) == 0
        # Check DataFrame has expected columns
        assert list(result.data.columns) == expected_columns

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_fetch_data_cash_timeout(self, mock_to_thread):
        """Test handling of cash data timeout."""
        # Mock timeout error
        mock_to_thread.side_effect = TimeoutError(
            "Timeout waiting for account data from IBKR"
        )

        result = await self.provider.get_data(None)

        assert result.success is False
        assert "Timeout waiting for account data" in (result.error_message or "")
        assert result.error_code == "RetriableProviderException"


class TestIBKRFactoryFunctions:
    """Test cases for IBKR provider factory functions."""

    def test_create_ibkr_positions_provider_defaults(self):
        """Test factory function with default parameters."""
        provider = create_ibkr_positions_provider()

        assert isinstance(provider, IBKRPositionsProvider)
        assert provider.config.timeout == 60.0
        assert provider.config.retries == 3

    def test_create_ibkr_positions_provider_custom(self):
        """Test factory function with custom parameters."""
        provider = create_ibkr_positions_provider(timeout=120.0, retries=5)

        assert provider.config.timeout == 120.0
        assert provider.config.retries == 5

    def test_create_ibkr_cash_provider_defaults(self):
        """Test cash provider factory with default parameters."""
        provider = create_ibkr_cash_provider()

        assert isinstance(provider, IBKRCashProvider)
        assert provider.config.timeout == 60.0
        assert provider.config.retries == 3

    def test_create_ibkr_cash_provider_custom(self):
        """Test cash provider factory with custom parameters."""
        provider = create_ibkr_cash_provider(timeout=90.0, retries=2)

        assert provider.config.timeout == 90.0
        assert provider.config.retries == 2


class TestIBKRProviderIntegration:
    """Integration tests for IBKR providers."""

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_concurrent_requests(self, mock_to_thread):
        """Test concurrent requests to IBKR providers."""
        # Mock successful empty responses
        mock_to_thread.return_value = DataFrame()

        pos_provider = IBKRPositionsProvider()
        cash_provider = IBKRCashProvider()

        # Make concurrent requests
        tasks = [
            pos_provider.get_data(None),
            cash_provider.get_data(None),
        ]

        results = await asyncio.gather(*tasks)

        # Both should succeed
        assert all(result.success for result in results)
        assert len(results) == 2

    def test_run_ibkr_loop_helper(self):
        """Test the run_ibkr_loop helper function."""
        mock_app = MagicMock()

        # Call the helper function
        run_ibkr_loop(mock_app)

        # Verify it calls app.run()
        mock_app.run.assert_called_once()


class TestCacheSettingsIBKR:
    """Test cases for cache settings on IBKR providers."""

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_positions_cache_disabled_per_provider(
        self, mock_to_thread, tmp_path, monkeypatch
    ):
        """Test positions provider with cache disabled."""
        monkeypatch.setenv("PROVIDER_CACHE_ROOT", str(tmp_path))
        config = ProviderConfig(cache_enabled=False)
        provider = IBKRPositionsProvider(config)

        # Mock successful response
        mock_to_thread.return_value = DataFrame([{"symbol": "AAPL"}])

        # First and second calls should both fetch fresh data
        await provider.get_data(None)
        await provider.get_data(None)

        # Should call asyncio.to_thread twice when cache disabled
        assert mock_to_thread.call_count == 2

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_cash_global_cache_disabled(
        self, mock_to_thread, tmp_path, monkeypatch
    ):
        """Test cash provider with global cache disabled."""
        monkeypatch.setenv("PROVIDER_CACHE_ROOT", str(tmp_path))
        from app.lib.settings import settings

        monkeypatch.setattr(settings, "PROVIDER_CACHE_ENABLED", False)

        provider = IBKRCashProvider()

        # Mock successful response
        mock_to_thread.return_value = DataFrame([{"account": "U123456"}])

        await provider.get_data(None)
        await provider.get_data(None)

        # Should fetch fresh data twice since global cache disabled
        assert mock_to_thread.call_count == 2
