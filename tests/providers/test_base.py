"""
Unit tests for the base provider module.
Tests the abstract base classes and core functionality.
"""

import asyncio
from unittest.mock import patch
from datetime import datetime
import pytest
from pandas import DataFrame

from app.providers.base import (
    BaseProvider,
    ProviderType,
    ProviderStatus,
    ProviderResult,
    ProviderConfig,
)


class MockProvider(BaseProvider[DataFrame]):
    """Mock provider for testing."""

    def _get_provider_type(self) -> ProviderType:
        return ProviderType.DUMMY

    async def _fetch_data(
        self, query: str | None, *args, cache_date: str | None = None, **kwargs
    ) -> DataFrame:
        """Mock implementation that returns a simple DataFrame."""
        # Providers expect non-null query for testing; None support not simulated here
        if query == "ERROR":
            raise ValueError("Mock error for testing")
        if query == "TIMEOUT":
            await asyncio.sleep(100)  # This will timeout
        return DataFrame({"price": [100.0], "volume": [1000]})


class TestProviderConfig:
    """Test cases for ProviderConfig."""

    def test_provider_config_defaults(self):
        """Test ProviderConfig with default values."""
        config = ProviderConfig()

        assert config.timeout == 30.0
        assert config.retries == 3
        assert config.retry_delay == 1.0
        assert config.cache_enabled is True
        assert config.cache_ttl == 3600
        assert config.rate_limit is None
        assert config.user_agent == "FinApp/1.0"
        assert config.extra_config == {}

    def test_provider_config_custom_values(self):
        """Test ProviderConfig with custom values."""
        config = ProviderConfig(
            timeout=60.0,
            retries=5,
            retry_delay=2.0,
            cache_enabled=False,
            cache_ttl=7200,
            rate_limit=0.5,
            user_agent="TestApp/2.0",
            extra_config={"test": "value"},
        )

        assert config.timeout == 60.0
        assert config.retries == 5
        assert config.retry_delay == 2.0
        assert config.cache_enabled is False
        assert config.cache_ttl == 7200
        assert config.rate_limit == 0.5
        assert config.user_agent == "TestApp/2.0"
        assert config.extra_config == {"test": "value"}


class TestProviderResult:
    """Test cases for ProviderResult."""

    def test_provider_result_success(self):
        """Test successful ProviderResult."""
        data = DataFrame({"price": [100.0]})
        result = ProviderResult[DataFrame](
            success=True,
            data=data,
            provider_type=ProviderType.YAHOO_HISTORY,
            query="AAPL",
        )

        assert result.success is True
        # Ensure data is present before calling equals()
        assert isinstance(result.data, DataFrame)
        assert DataFrame(result.data).equals(data)
        assert result.error_message is None
        assert result.error_code is None
        assert result.provider_type == ProviderType.YAHOO_HISTORY
        assert result.query == "AAPL"
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.metadata, dict)

    def test_provider_result_error(self):
        """Test error ProviderResult."""
        result = ProviderResult[DataFrame](
            success=False,
            error_message="Test error",
            error_code="ValueError",
            provider_type=ProviderType.YAHOO_HISTORY,
            query="INVALID",
        )

        assert result.success is False
        assert result.data is None
        assert result.error_message == "Test error"
        assert result.error_code == "ValueError"
        assert result.provider_type == ProviderType.YAHOO_HISTORY
        assert result.query == "INVALID"


class TestBaseProvider:
    """Test cases for BaseProvider abstract class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = MockProvider()  # pylint:disable=attribute-defined-outside-init

    def test_provider_initialization(self):
        """Test provider initialization."""
        assert isinstance(self.provider.config, ProviderConfig)
        assert self.provider.provider_type == ProviderType.DUMMY
        assert hasattr(self.provider, "logger")
        assert hasattr(self.provider, "_semaphore")

    def test_provider_initialization_with_custom_config(self):
        """Test provider initialization with custom config."""
        config = ProviderConfig(timeout=60.0, retries=5)
        provider = MockProvider(config)

        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5

    @pytest.mark.asyncio
    async def test_get_data_success(self):
        """Test successful data retrieval."""
        result = await self.provider.get_data("AAPL")

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, DataFrame)
        assert result.error_message is None
        assert result.provider_type == ProviderType.DUMMY
        assert result.query == "AAPL"
        assert result.execution_time is not None
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_get_data_error(self):
        """Test error handling in data retrieval."""
        result = await self.provider.get_data("ERROR")

        assert result.success is False
        assert result.data is None
        assert "Mock error for testing" in (result.error_message or "")
        assert result.error_code == "ValueError"
        assert result.provider_type == ProviderType.DUMMY
        assert result.query == "ERROR"

    @pytest.mark.asyncio
    async def test_get_data_timeout(self):
        """Test timeout handling."""
        # Use a very short timeout for testing
        config = ProviderConfig(timeout=0.1, retries=0)
        provider = MockProvider(config)

        result = await provider.get_data("TIMEOUT")

        assert result.success is False
        assert result.data is None
        # The error message should contain information about the failure
        assert result.error_message is not None
        assert "Failed after" in (result.error_message or "")
        assert result.error_code in ["TimeoutError", "CancelledError"]

    @pytest.mark.asyncio
    async def test_get_data_retries(self):
        """Test retry mechanism."""
        config = ProviderConfig(retries=2, retry_delay=0.01)
        provider = MockProvider(config)

        result = await provider.get_data("ERROR")

        assert result.success is False
        assert result.error_message is not None
        assert "Failed after 3 attempts" in (result.error_message or "")
        assert result.metadata["total_attempts"] == 3

    @pytest.mark.asyncio
    async def test_get_data_non_string_query(self):
        """Test validation of non-string query type."""
        # Passing non-string should error
        result = await self.provider.get_data(123)  # type: ignore

        assert result.success is False
        assert result.error_message is not None
        assert "Query must be a string or None" in (result.error_message or "")
        assert result.error_code == "ValueError"

    @pytest.mark.asyncio
    async def test_get_data_empty_string(self):
        """Test empty string query is passed to provider."""
        result = await self.provider.get_data("")

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, DataFrame)
        assert result.query == ""

    @pytest.mark.asyncio
    async def test_get_data_query_passthrough(self):
        """Test query is passed through unmodified without normalization."""
        raw_query = "  aapl  "
        result = await self.provider.get_data(raw_query)

        assert result.success is True
        assert result.query == raw_query

    def test_get_data_sync(self):
        """Test synchronous wrapper."""
        result = self.provider.get_data_sync("AAPL")

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, DataFrame)

    def test_get_data_sync_error(self):
        """Test synchronous wrapper with error."""
        result = self.provider.get_data_sync("ERROR")

        assert result.success is False
        assert result.error_message is not None
        assert "Mock error for testing" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality."""
        config = ProviderConfig(rate_limit=2.0)  # 2 requests per second
        provider = MockProvider(config)

        start_time = asyncio.get_event_loop().time()

        # Make two requests
        result1 = await provider.get_data("AAPL")
        result2 = await provider.get_data("AAPL")

        end_time = asyncio.get_event_loop().time()

        # Should take at least 0.5 seconds due to rate limiting
        assert end_time - start_time >= 0.4  # Small tolerance for timing
        assert result1.success is True
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_semaphore_concurrency_control(self):
        """Test that semaphore limits concurrent operations."""
        # Create provider with very small semaphore for testing
        provider = MockProvider()
        # Only 1 concurrent operation
        provider._semaphore = asyncio.Semaphore(1)  # pylint: disable=protected-access

        # Start multiple tasks
        tasks = [provider.get_data("AAPL") for _ in range(3)]
        results = await asyncio.gather(*tasks)

        # All should succeed despite concurrency limit
        assert all(result.success for result in results)

    @pytest.mark.asyncio
    async def test_logger_usage(self):
        """Test that logger is used properly."""
        with (
            patch("app.providers.base.logger.info") as mock_info,
            patch("app.providers.base.logger.warning") as mock_warning,
        ):

            # Test successful case
            await self.provider.get_data("AAPL")
            mock_info.assert_called()

            # Test error case - should use warning for retry failures
            await self.provider.get_data("ERROR")
            mock_warning.assert_called()


class TestProviderTypes:
    """Test cases for provider type enums."""

    def test_provider_type_values(self):
        """Test that all provider types have expected values."""
        # assert ProviderType.ALPHA_VANTAGE == "alpha_vantage"
        assert ProviderType.BLACKROCK_HOLDINGS == "blackrock_holdings"
        assert ProviderType.FRED_SERIES == "fred_series"
        assert ProviderType.IBKR_POSITIONS == "ibkr_positions"
        assert ProviderType.IBKR_CASH == "ibkr_cash"
        assert ProviderType.SHILLER_CAPE == "shiller_cape"
        assert ProviderType.TIPRANKS_DATA == "tipranks_data"
        assert ProviderType.TIPRANKS_NEWS_SENTIMENT == "tipranks_news_sentiment"
        assert ProviderType.YAHOO_HISTORY == "yahoo_history"
        assert ProviderType.YAHOO_INFO == "yahoo_info"
        assert ProviderType.ZACKS == "zacks"
        # ------------------------------------------------------
        assert ProviderType.DUMMY == "dummy"  # For testing purposes only

    def test_provider_status_values(self):
        """Test that all provider status values are correct."""
        assert ProviderStatus.IDLE == "idle"
        assert ProviderStatus.RUNNING == "running"
        assert ProviderStatus.SUCCESS == "success"
        assert ProviderStatus.ERROR == "error"
        assert ProviderStatus.TIMEOUT == "timeout"
