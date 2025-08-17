"""
Unit tests for the FredSeriesProvider.

These tests verify the functionality of the FRED API provider including:
- Valid series data fetching
- Error handling for invalid series IDs
- API key validation
- Parameter handling
- Cache behavior
"""

import os
import pytest
import httpx
import pandas as pd
from unittest.mock import AsyncMock, patch, MagicMock

from app.providers.fred import FredSeriesProvider, create_fred_series_provider
from app.providers.base import ProviderConfig


@pytest.fixture
def mock_fred_response():
    """Mock successful FRED API response."""
    return {
        "realtime_start": "2024-01-01",
        "realtime_end": "2024-01-01",
        "observation_start": "1947-01-01",
        "observation_end": "2023-10-01",
        "units": "Billions of Chained 2012 Dollars",
        "output_type": 1,
        "file_type": "json",
        "order_by": "observation_date",
        "sort_order": "asc",
        "count": 3,
        "offset": 0,
        "limit": 100000,
        "observations": [
            {
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-01-01",
                "date": "2023-01-01",
                "value": "26854.6",
            },
            {
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-01-01",
                "date": "2023-04-01",
                "value": "27063.0",
            },
            {
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-01-01",
                "date": "2023-07-01",
                "value": "27610.1",
            },
        ],
    }


@pytest.fixture
def mock_fred_error_response():
    """Mock FRED API error response."""
    return {
        "error_code": 400,
        "error_message": "Bad Request. The series does not exist.",
    }


@pytest.fixture
def mock_fred_empty_response():
    """Mock FRED API response with no observations."""
    return {
        "realtime_start": "2024-01-01",
        "realtime_end": "2024-01-01",
        "observation_start": "1947-01-01",
        "observation_end": "2023-10-01",
        "units": "Billions of Chained 2012 Dollars",
        "output_type": 1,
        "file_type": "json",
        "order_by": "observation_date",
        "sort_order": "asc",
        "count": 0,
        "offset": 0,
        "limit": 100000,
        "observations": [],
    }


class TestFredSeriesProvider:
    """Test cases for FredSeriesProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        # Disable caching to ensure we test the actual provider logic
        config = ProviderConfig(cache_enabled=False)
        self.provider = FredSeriesProvider(config)

    def test_provider_type(self):
        """Test that provider returns correct type."""
        from app.providers.base import ProviderType

        # pylint: disable=protected-access
        assert self.provider._get_provider_type() == ProviderType.FRED_SERIES

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        """Test error handling when API key is missing."""
        with patch("app.providers.fred.settings") as mock_settings:
            mock_settings.FRED_API_KEY = ""
            result = await self.provider.get_data("GDPC1")
            assert not result.success
            assert "FRED_API_KEY is required" in result.error_message
            assert result.error_code == "ValueError"

    @pytest.mark.asyncio
    async def test_missing_series_id(self):
        """Test error handling when series ID is missing."""
        with patch("app.providers.fred.settings") as mock_settings:
            mock_settings.FRED_API_KEY = "test_key"
            result = await self.provider.get_data(None)
            assert not result.success
            assert "Series ID must be provided" in result.error_message
            assert result.error_code == "ValueError"

    @pytest.mark.asyncio
    @patch("app.providers.fred.settings")
    @patch("httpx.AsyncClient")
    async def test_successful_data_fetch(
        self, mock_client, mock_settings, mock_fred_response
    ):
        """Test successful data fetching from FRED API."""
        # Mock settings
        mock_settings.FRED_API_KEY = "test_api_key_12345678901234567890"

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_fred_response
        mock_response.raise_for_status.return_value = None

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Test the fetch
        result = await self.provider.get_data("GDPC1")

        assert result.success
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 3
        assert result.data.index.name == "date"
        assert "value" in result.data.columns
        assert result.data.index.dtype.kind == "M"  # datetime type

        # Check data values
        expected_dates = pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01"])
        expected_values = [26854.6, 27063.0, 27610.1]

        # Reset index name for comparison
        expected_dates.name = "date"
        pd.testing.assert_index_equal(result.data.index, expected_dates)
        assert result.data["value"].tolist() == expected_values

    @pytest.mark.asyncio
    @patch("app.providers.fred.settings")
    @patch("httpx.AsyncClient")
    async def test_invalid_series_id(
        self, mock_client, mock_settings, mock_fred_error_response
    ):
        """Test error handling for invalid series ID."""
        # Mock settings
        mock_settings.FRED_API_KEY = "test_api_key_12345678901234567890"

        # Mock HTTP response with error
        mock_response = MagicMock()
        mock_response.json.return_value = mock_fred_error_response
        mock_response.raise_for_status.return_value = None

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Test the fetch
        result = await self.provider.get_data("INVALID_SERIES")

        assert not result.success
        assert result.error_code == "RetriableProviderException"
        assert "Invalid series ID" in result.error_message

    @pytest.mark.asyncio
    @patch("app.providers.fred.settings")
    @patch("httpx.AsyncClient")
    async def test_empty_response(
        self, mock_client, mock_settings, mock_fred_empty_response
    ):
        """Test handling of empty observations response."""
        # Mock settings
        mock_settings.FRED_API_KEY = "test_api_key_12345678901234567890"

        # Mock HTTP response with no observations
        mock_response = MagicMock()
        mock_response.json.return_value = mock_fred_empty_response
        mock_response.raise_for_status.return_value = None

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Test the fetch
        result = await self.provider.get_data("EMPTY_SERIES")

        assert not result.success
        assert result.error_code == "NonRetriableProviderException"
        assert "No data found for series ID" in result.error_message

    @pytest.mark.asyncio
    @patch("app.providers.fred.settings")
    @patch("httpx.AsyncClient")
    async def test_http_error_400(self, mock_client, mock_settings):
        """Test handling of HTTP 400 error."""
        # Mock settings
        mock_settings.FRED_API_KEY = "test_api_key_12345678901234567890"

        # Mock HTTP 400 error
        mock_client_instance = AsyncMock()
        error_response = MagicMock()
        error_response.status_code = 400
        mock_client_instance.get.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=error_response
        )
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Test the fetch
        result = await self.provider.get_data("BAD_SERIES")

        assert not result.success
        assert result.error_code == "NonRetriableProviderException"
        assert "Invalid request" in result.error_message

    @pytest.mark.asyncio
    @patch("app.providers.fred.settings")
    @patch("httpx.AsyncClient")
    async def test_http_error_403(self, mock_client, mock_settings):
        """Test handling of HTTP 403 (authentication) error."""
        # Mock settings
        mock_settings.FRED_API_KEY = "invalid_api_key"

        # Mock HTTP 403 error
        mock_client_instance = AsyncMock()
        error_response = MagicMock()
        error_response.status_code = 403
        mock_client_instance.get.side_effect = httpx.HTTPStatusError(
            "Forbidden", request=MagicMock(), response=error_response
        )
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Test the fetch
        result = await self.provider.get_data("GDPC1")

        assert not result.success
        assert result.error_code == "NonRetriableProviderException"
        assert "Invalid FRED API key" in result.error_message

    @pytest.mark.asyncio
    @patch("app.providers.fred.settings")
    @patch("httpx.AsyncClient")
    async def test_http_error_500(self, mock_client, mock_settings):
        """Test handling of HTTP 500 (server) error."""
        # Mock settings
        mock_settings.FRED_API_KEY = "test_api_key_12345678901234567890"

        # Mock HTTP 500 error
        mock_client_instance = AsyncMock()
        error_response = MagicMock()
        error_response.status_code = 500
        mock_client_instance.get.side_effect = httpx.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=error_response
        )
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Test the fetch
        result = await self.provider.get_data("GDPC1")

        assert not result.success
        assert result.error_code == "RetriableProviderException"
        assert "HTTP error" in result.error_message

    @pytest.mark.asyncio
    @patch("app.providers.fred.settings")
    @patch("httpx.AsyncClient")
    async def test_network_error(self, mock_client, mock_settings):
        """Test handling of network connection error."""
        # Mock settings
        mock_settings.FRED_API_KEY = "test_api_key_12345678901234567890"

        # Mock network error
        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = httpx.RequestError("Connection failed")
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Test the fetch
        result = await self.provider.get_data("GDPC1")

        assert not result.success
        assert result.error_code == "RetriableProviderException"
        assert "Network error" in result.error_message

    @pytest.mark.asyncio
    @patch("app.providers.fred.settings")
    @patch("httpx.AsyncClient")
    async def test_with_date_parameters(
        self, mock_client, mock_settings, mock_fred_response
    ):
        """Test data fetching with date parameters."""
        # Mock settings
        mock_settings.FRED_API_KEY = "test_api_key_12345678901234567890"

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_fred_response
        mock_response.raise_for_status.return_value = None

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Test with date parameters
        result = await self.provider.get_data(
            "GDPC1", observation_start="2023-01-01", observation_end="2023-12-31"
        )

        assert result.success

        # Verify the API was called with correct parameters
        call_args = mock_client_instance.get.call_args
        assert call_args[1]["params"]["observation_start"] == "2023-01-01"
        assert call_args[1]["params"]["observation_end"] == "2023-12-31"

    @pytest.mark.asyncio
    @patch("app.providers.fred.settings")
    @patch("httpx.AsyncClient")
    async def test_missing_values_handling(self, mock_client, mock_settings):
        """Test handling of missing values (represented as '.' in FRED API)."""
        # Mock settings
        mock_settings.FRED_API_KEY = "test_api_key_12345678901234567890"

        # Mock response with missing values
        mock_response_data = {
            "observations": [
                {"date": "2023-01-01", "value": "100.0"},
                {"date": "2023-02-01", "value": "."},  # Missing value
                {"date": "2023-03-01", "value": "102.0"},
                {"date": "2023-04-01", "value": "."},  # Missing value
                {"date": "2023-05-01", "value": "103.0"},
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        mock_client_instance = AsyncMock()
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Test the fetch
        result = await self.provider.get_data("TEST_SERIES")

        assert result.success
        assert len(result.data) == 3  # Only non-missing values
        expected_values = [100.0, 102.0, 103.0]
        assert result.data["value"].tolist() == expected_values

    def test_factory_function(self):
        """Test the factory function for creating provider instances."""
        provider = create_fred_series_provider(timeout=20.0, retries=5)

        assert isinstance(provider, FredSeriesProvider)
        assert provider.config.timeout == 20.0
        assert provider.config.retries == 5

    @pytest.mark.asyncio
    @patch("app.providers.fred.settings")
    async def test_series_id_normalization(self, mock_settings):
        """Test that series IDs are properly normalized (uppercase, stripped)."""
        mock_settings.FRED_API_KEY = "test_api_key_12345678901234567890"

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"observations": []}
            mock_response.raise_for_status.return_value = None

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            # Test with lowercase and whitespace
            await self.provider.get_data("  gdpc1  ")

            # Verify the API was called with normalized series ID
            call_args = mock_client_instance.get.call_args
            assert call_args[1]["params"]["series_id"] == "GDPC1"


class TestFredSeriesIntegration:
    """Integration tests that test the full provider workflow."""

    @pytest.mark.asyncio
    @patch("app.providers.fred.settings")
    async def test_full_workflow_success(self, mock_settings):
        """Test the complete workflow from provider creation to data return."""
        # Mock settings
        mock_settings.FRED_API_KEY = "test_api_key_12345678901234567890"

        # Create provider with cache disabled
        config = ProviderConfig(timeout=15.0, retries=1, cache_enabled=False)
        provider = FredSeriesProvider(config)

        # Mock successful API response
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "observations": [
                    {"date": "2023-01-01", "value": "100.0"},
                    {"date": "2023-02-01", "value": "101.0"},
                ]
            }
            mock_response.raise_for_status.return_value = None

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            # Test full workflow
            result = await provider.get_data("GDPC1")

            assert result.success
            assert isinstance(result.data, pd.DataFrame)
            assert len(result.data) == 2
            assert result.provider_type.value == "fred_series"
            assert result.query == "GDPC1"
            assert result.execution_time is not None
