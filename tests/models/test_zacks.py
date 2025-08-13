"""
Unit tests for the Zacks provider module.
Tests ZacksProvider for fetching financial data from Zacks API.
"""

import asyncio
import os
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
from pydantic import BaseModel, ValidationError
import pytest

from app.models.zacks import (
    ZacksProvider,
    create_zacks_provider,
    ZacksModel,
)
from app.models.base import ProviderType, ProviderConfig


os.environ["PYTEST_DEBUG_TEMPROOT"] = os.getcwd() + "/temp/"


@pytest.fixture(autouse=True)
def isolate_cwd(tmp_path, monkeypatch):
    # Use a temp cwd to avoid cache pollution and disable global cache
    monkeypatch.chdir(tmp_path)
    # from app.core.settings import settings
    # monkeypatch.setattr(settings, 'CACHE_ENABLED', False)


class TestZacksProvider:
    """Test cases for ZacksProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = ZacksProvider()  # pylint:disable=attribute-defined-outside-init

    def test_provider_type(self):
        """Test that provider returns correct type."""
        assert self.provider.provider_type == ProviderType.ZACKS

    def test_provider_initialization(self):
        """Test provider initialization."""
        assert isinstance(self.provider.config, ProviderConfig)
        assert hasattr(self.provider, "logger")

    def test_provider_initialization_with_custom_config(self):
        """Test provider initialization with custom config."""
        config = ProviderConfig(
            timeout=60.0, retries=5, rate_limit=0.5, user_agent="TestApp/1.0"
        )
        provider = ZacksProvider(config)

        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5
        assert provider.config.rate_limit == 0.5
        assert provider.config.user_agent == "TestApp/1.0"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fetch_data_success(self, mock_client_class):
        """Test successful data fetching from Zacks API."""
        # Mock the httpx response
        mock_response_data = {
            "ticker": "AAPL",
            "price": 150.50,
            "change": 2.50,
            "percentChange": 1.68,
            "volume": 50000000,
            "high": 152.0,
            "low": 149.0,
            "open": 151.0,
            "previousClose": 148.0,
            "marketCap": 2500000000000,
            "peRatio": 25.5,
            "zacksRank": 2,
            "valueScore": "B",
            "growthScore": "A",
            "momentumScore": "B",
            "vgmScore": "B",
        }

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        assert result.success is True
        assert isinstance(result.data, BaseModel)
        assert result.query == "AAPL"
        assert result.provider_type == ProviderType.ZACKS

        # Check parsed data fields
        assert getattr(result.data, "ticker") == "AAPL"
        assert getattr(result.data, "price") == 150.50
        assert getattr(result.data, "zacks_rank") == 2

        # Verify HTTP call was made correctly
        expected_url = "https://quote-feed.zacks.com/index?t=AAPL"
        mock_client.get.assert_called_once_with(expected_url)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fetch_data_http_404_error(self, mock_client_class):
        """Test handling of HTTP 404 errors (ticker not found)."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        http_error = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.get.side_effect = http_error
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("INVALID")

        assert result.success is False
        assert "Ticker not found in Zacks" in (result.error_message or "")
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fetch_data_http_429_error(self, mock_client_class):
        """Test handling of HTTP 429 errors (rate limit exceeded)."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        http_error = httpx.HTTPStatusError(
            "Too Many Requests", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.get.side_effect = http_error
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        assert result.success is False
        assert "Rate limit exceeded" in (result.error_message or "")
        assert result.error_code == "RetriableProviderException"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fetch_data_http_500_error(self, mock_client_class):
        """Test handling of HTTP 500 errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        http_error = httpx.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.get.side_effect = http_error
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        assert result.success is False
        assert "HTTP 500 error" in (result.error_message or "")
        assert result.error_code == "RetriableProviderException"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fetch_data_network_error(self, mock_client_class):
        """Test handling of network connection errors."""
        network_error = httpx.RequestError("Connection failed")

        mock_client = AsyncMock()
        mock_client.get.side_effect = network_error
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        assert result.success is False
        assert "Network error connecting to Zacks API" in (result.error_message or "")
        assert result.error_code == "RetriableProviderException"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fetch_data_invalid_json_response(self, mock_client_class):
        """Test handling of invalid JSON response."""
        mock_response_data = None  # Invalid response

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        assert result.success is False
        assert "Invalid response from Zacks API" in (result.error_message or "")
        assert result.error_code == "ValueError"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fetch_data_empty_dict_response(self, mock_client_class):
        """Test handling of empty dictionary response."""
        mock_response_data = {}  # Empty dict

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        assert result.success is False
        assert "Invalid response from Zacks API" in (result.error_message or "")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fetch_data_partial_data(self, mock_client_class):
        """Test handling of partial data response with strict validation."""
        mock_response_data = {
            "ticker": "AAPL",
            "price": 150.50,
            # Missing many required fields
        }

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        # Should fail due to strict validation requiring all fields
        assert result.success is False
        assert result.data is None
        assert "Field required" in (result.error_message or "")

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fetch_data_timeout_handling(self, mock_client_class):
        """Test timeout configuration is applied."""
        config = ProviderConfig(timeout=0.1)  # Very short timeout
        provider = ZacksProvider(config)

        # --- stub out the response so that raise_for_status() is a normal method ---
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "ticker": "AAPL",
            "price": 150.0,
            "change": 2.5,
            "percentChange": 1.68,
            "volume": 50000000,
            "high": 152.0,
            "low": 149.0,
            "open": 151.0,
            "previousClose": 148.0,
            "marketCap": 2500000000000,
            "peRatio": 25.5,
            "zacksRank": 2,
            "valueScore": "B",
            "growthScore": "A",
            "momentumScore": "B",
            "vgmScore": "B",
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        # Verify timeout is passed to httpx
        await provider.get_data("AAPL")

        # Check that AsyncClient was created with correct timeout
        _, kwargs = mock_client_class.call_args
        assert "timeout" in kwargs
        assert kwargs["timeout"].connect == 0.1

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_fetch_data_user_agent_header(self, mock_client_class):
        """Test that custom user agent is used."""
        config = ProviderConfig(user_agent="CustomAgent/1.0")
        provider = ZacksProvider(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ticker": "AAPL",
            "price": 150.0,
            "change": 2.5,
            "percentChange": 1.68,
            "volume": 50000000,
            "high": 152.0,
            "low": 149.0,
            "open": 151.0,
            "previousClose": 148.0,
            "marketCap": 2500000000000,
            "peRatio": 25.5,
            "zacksRank": 2,
            "valueScore": "B",
            "growthScore": "A",
            "momentumScore": "B",
            "vgmScore": "B",
        }
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        await provider.get_data("AAPL")

        # Check that AsyncClient was created with correct headers
        _, kwargs = mock_client_class.call_args
        assert "headers" in kwargs
        assert kwargs["headers"]["User-Agent"] == "CustomAgent/1.0"

    def test_get_data_sync(self):
        """Test synchronous wrapper."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "ticker": "AAPL",
                "price": 150.0,
                "change": 2.5,
                "percentChange": 1.68,
                "volume": 50000000,
                "high": 152.0,
                "low": 149.0,
                "open": 151.0,
                "previousClose": 148.0,
                "marketCap": 2500000000000,
                "peRatio": 25.5,
                "zacksRank": 2,
                "valueScore": "B",
                "growthScore": "A",
                "momentumScore": "B",
                "vgmScore": "B",
            }
            mock_response.raise_for_status.return_value = None

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = self.provider.get_data_sync("AAPL")

            assert result.success is True


class TestZacksFactoryFunction:
    """Test cases for Zacks provider factory function."""

    def test_create_zacks_provider_defaults(self):
        """Test factory function with default parameters."""
        provider = create_zacks_provider()

        assert isinstance(provider, ZacksProvider)
        assert provider.config.timeout == 30.0
        assert provider.config.retries == 3
        assert provider.config.rate_limit == 1.0
        assert "Zacks Data Provider" in provider.config.user_agent

    def test_create_zacks_provider_custom(self):
        """Test factory function with custom parameters."""
        provider = create_zacks_provider(timeout=60.0, retries=5, rate_limit=0.5)

        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5
        assert provider.config.rate_limit == 0.5


class TestZacksConfig:
    """Test cases for Zacks configuration."""

    def test_zacks_model_structure(self):
        """Test that Zacks model has expected structure."""
        # Test with realistic Zacks API data including extra fields to ignore
        test_data = {
            # Required fields we need
            "ticker": "AAPL",
            "price": 150.50,
            "change": 2.50,
            "percentChange": 1.68,
            "volume": 50000000,
            "high": 152.0,
            "low": 149.0,
            "open": 151.0,
            "previousClose": 148.0,
            "marketCap": 2500000000000,
            "peRatio": 25.5,
            "zacksRank": 2,
            "valueScore": "B",
            "growthScore": "A",
            "momentumScore": "B",
            "vgmScore": "B",
            # Extra fields from Zacks API that should be ignored
            "lastUpdated": "2023-10-15T10:30:00Z",
            "analystConsensus": "Buy",
            "priceTarget": 175.0,
            "earnings": {"nextDate": "2024-01-25", "estimate": 2.11},
            "institutional_ownership": 58.4,
            "short_interest": 0.45,
            "insider_trading": "neutral",
        }
        model = ZacksModel(**test_data)

        # Check that required fields are parsed correctly
        assert model.ticker == "AAPL"
        assert model.price == 150.50
        assert model.zacks_rank == 2
        assert model.value_score == "B"
        assert model.growth_score == "A"
        assert model.volume == 50000000

    def test_zacks_model_aliases(self):
        """Test that aliases work correctly."""
        # Test data with complete Zacks API field names
        test_data = {
            "ticker": "AAPL",
            "price": 150.50,
            "change": 2.50,
            "percentChange": 1.68,
            "volume": 50000000,
            "high": 152.0,
            "low": 149.0,
            "open": 151.0,
            "previousClose": 148.0,
            "marketCap": 2500000000000,
            "peRatio": 25.5,
            "zacksRank": 2,
            "valueScore": "B",
            "growthScore": "A",
            "momentumScore": "B",
            "vgmScore": "B",
        }

        model = ZacksModel.model_validate(test_data)

        # Check that aliases map correctly
        assert model.ticker == "AAPL"
        assert model.price == 150.50
        assert model.percent_change == 1.68
        assert model.zacks_rank == 2
        assert model.open_price == 151.0
        assert model.previous_close == 148.0

    def test_zacks_model_missing_required_field(self):
        """Test that missing required fields raise ValidationError."""
        # Test data missing required fields
        test_data = {
            "ticker": "AAPL",
            "price": 150.50,
            # Missing many other required fields...
        }

        with pytest.raises(ValidationError) as exc_info:
            ZacksModel(**test_data)

        # Should complain about missing fields
        assert "Field required" in str(exc_info.value)


class TestZacksProviderIntegration:
    """Integration tests for Zacks provider."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Test multiple concurrent requests to Zacks provider."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "ticker": "TEST",
                "price": 100.0,
                "change": 1.5,
                "percentChange": 1.52,
                "volume": 30000000,
                "high": 101.0,
                "low": 99.0,
                "open": 100.5,
                "previousClose": 98.5,
                "marketCap": 1500000000000,
                "peRatio": 20.0,
                "zacksRank": 3,
                "valueScore": "A",
                "growthScore": "B",
                "momentumScore": "A",
                "vgmScore": "A",
            }
            mock_response.raise_for_status.return_value = None

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            provider = ZacksProvider()

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

    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self):
        """Test that rate limiting is applied."""
        config = ProviderConfig(rate_limit=2.0)  # 2 requests per second
        provider = ZacksProvider(config)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "ticker": "AAPL",
                "price": 150.0,
                "change": 2.5,
                "percentChange": 1.68,
                "volume": 50000000,
                "high": 152.0,
                "low": 149.0,
                "open": 151.0,
                "previousClose": 148.0,
                "marketCap": 2500000000000,
                "peRatio": 25.5,
                "zacksRank": 2,
                "valueScore": "B",
                "growthScore": "A",
                "momentumScore": "B",
                "vgmScore": "B",
            }
            mock_response.raise_for_status.return_value = None

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            start_time = asyncio.get_event_loop().time()

            # Make two requests
            result1 = await provider.get_data("AAPL")
            result2 = await provider.get_data("AAPL")

            end_time = asyncio.get_event_loop().time()

            # Should take at least 0.5 seconds due to rate limiting
            assert end_time - start_time >= 0.4  # Small tolerance
            assert result1.success is True
            assert result2.success is True

    @pytest.mark.asyncio
    async def test_error_handling_in_concurrent_requests(self):
        """Test error handling when some concurrent requests fail."""
        with patch("httpx.AsyncClient") as mock_client_class:
            # First request succeeds, second fails
            success_response = MagicMock()
            success_response.json.return_value = {
                "ticker": "AAPL",
                "price": 150.0,
                "change": 2.5,
                "percentChange": 1.68,
                "volume": 50000000,
                "high": 152.0,
                "low": 149.0,
                "open": 151.0,
                "previousClose": 148.0,
                "marketCap": 2500000000000,
                "peRatio": 25.5,
                "zacksRank": 2,
                "valueScore": "B",
                "growthScore": "A",
                "momentumScore": "B",
                "vgmScore": "B",
            }
            success_response.raise_for_status.return_value = None

            error_response = MagicMock()
            error_response.status_code = 404
            http_error = httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=error_response
            )

            # Track call count to return different responses
            call_count = 0

            def get_side_effect(*args, **kwargs):  # pylint:disable=unused-argument
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return success_response
                else:
                    raise http_error

            # Create a single mock client that behaves differently on each call
            mock_client = AsyncMock()
            mock_client.get.side_effect = get_side_effect
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            provider = ZacksProvider()

            tasks = [
                provider.get_data("AAPL"),  # Should succeed
                provider.get_data("INVALID"),  # Should fail
            ]

            results = await asyncio.gather(*tasks)

            assert results[0].success is True
            assert results[1].success is False
            assert "Ticker not found in Zacks" in (results[1].error_message or "")

    class TestCacheSettingsZacks:
        """Test cases for cache setting on Zacks provider."""

        @pytest.mark.asyncio
        async def test_cache_disabled_per_provider(self, tmp_path, monkeypatch):
            # Use isolated temp cwd
            monkeypatch.chdir(tmp_path)
            # Disable cache in provider config
            config = ProviderConfig(cache_enabled=False)
            provider = ZacksProvider(config)

            # Patch AsyncClient to track calls
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_resp = MagicMock()
                mock_resp.raise_for_status.return_value = None
                mock_resp.json.return_value = {
                    "ticker": "AAPL",
                    "price": 100.0,
                    "change": 1.5,
                    "percentChange": 1.52,
                    "volume": 30000000,
                    "high": 101.0,
                    "low": 99.0,
                    "open": 100.5,
                    "previousClose": 98.5,
                    "marketCap": 1500000000000,
                    "peRatio": 20.0,
                    "zacksRank": 3,
                    "valueScore": "A",
                    "growthScore": "B",
                    "momentumScore": "A",
                    "vgmScore": "A",
                }
                mock_client.get.return_value = mock_resp
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client_cls.return_value = mock_client

                # First and second calls should fetch fresh data
                await provider.get_data("AAPL")
                await provider.get_data("AAPL")
                # AsyncClient.get called twice due to cache disabled
                assert mock_client.get.call_count == 2


class TestGlobalCacheSettingsZacks:
    """Test cases for global cache setting on Zacks provider."""

    @pytest.mark.asyncio
    async def test_global_cache_disabled(self, tmp_path, monkeypatch):
        # Use isolated temp cwd
        monkeypatch.chdir(tmp_path)
        # Disable global cache
        from app.core.settings import settings

        monkeypatch.setattr(settings, "CACHE_ENABLED", False)

        provider = ZacksProvider(ProviderConfig())

        # Patch AsyncClient to track calls
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {
                "ticker": "AAPL",
                "price": 100.0,
                "change": 1.5,
                "percentChange": 1.52,
                "volume": 30000000,
                "high": 101.0,
                "low": 99.0,
                "open": 100.5,
                "previousClose": 98.5,
                "marketCap": 1500000000000,
                "peRatio": 20.0,
                "zacksRank": 3,
                "valueScore": "A",
                "growthScore": "B",
                "momentumScore": "A",
                "vgmScore": "A",
            }
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_cls.return_value = mock_client

            # First and second calls should fetch fresh data
            await provider.get_data("AAPL")
            await provider.get_data("AAPL")
            # AsyncClient.get should be called twice when global cache disabled
            assert mock_client.get.call_count == 2
