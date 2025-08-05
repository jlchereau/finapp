"""
Unit tests for the Zacks provider module.
Tests ZacksProvider for fetching financial data from Zacks API.
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
from pydantic import BaseModel

from app.models.zacks import (
    ZacksProvider,
    create_zacks_provider,
    ZACKS_CONFIG,
)
from app.models.base import ProviderType, ProviderConfig


class TestZacksProvider:
    """Test cases for ZacksProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = ZacksProvider()

    def test_provider_type(self):
        """Test that provider returns correct type."""
        assert self.provider.provider_type == ProviderType.ZACKS

    def test_provider_initialization(self):
        """Test provider initialization."""
        assert isinstance(self.provider.config, ProviderConfig)
        assert hasattr(self.provider, 'logger')

    def test_provider_initialization_with_custom_config(self):
        """Test provider initialization with custom config."""
        config = ProviderConfig(
            timeout=60.0,
            retries=5,
            rate_limit=0.5,
            user_agent="TestApp/1.0"
        )
        provider = ZacksProvider(config)
        
        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5
        assert provider.config.rate_limit == 0.5
        assert provider.config.user_agent == "TestApp/1.0"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_fetch_data_success(self, mock_client_class):
        """Test successful data fetching from Zacks API."""
        # Mock the httpx response
        mock_response_data = {
            'ticker': 'AAPL',
            'price': 150.50,
            'change': 2.50,
            'percentChange': 1.68,
            'volume': 50000000,
            'high': 152.0,
            'low': 149.0,
            'open': 151.0,
            'previousClose': 148.0,
            'marketCap': 2500000000000,
            'peRatio': 25.5,
            'zacksRank': 2,
            'valueScore': 'B',
            'growthScore': 'A',
            'momentumScore': 'B',
            'vgmScore': 'B'
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
        assert result.ticker == "AAPL"
        assert result.provider_type == ProviderType.ZACKS
        
        # Check parsed data fields
        assert getattr(result.data, 'ticker') == 'AAPL'
        assert getattr(result.data, 'price') == 150.50
        assert getattr(result.data, 'zacks_rank') == 2
        
        # Verify HTTP call was made correctly
        expected_url = "https://quote-feed.zacks.com/index?t=AAPL"
        mock_client.get.assert_called_once_with(expected_url)

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_fetch_data_http_404_error(self, mock_client_class):
        """Test handling of HTTP 404 errors (ticker not found)."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        http_error = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_response
        )
        
        mock_client = AsyncMock()
        mock_client.get.side_effect = http_error
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client
        
        result = await self.provider.get_data("INVALID")
        
        assert result.success is False
        assert "Ticker not found in Zacks" in result.error_message
        assert result.error_code == "ValueError"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_fetch_data_http_429_error(self, mock_client_class):
        """Test handling of HTTP 429 errors (rate limit exceeded)."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        
        http_error = httpx.HTTPStatusError(
            "Too Many Requests",
            request=MagicMock(),
            response=mock_response
        )
        
        mock_client = AsyncMock()
        mock_client.get.side_effect = http_error
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client
        
        result = await self.provider.get_data("AAPL")
        
        assert result.success is False
        assert "Rate limit exceeded" in result.error_message
        assert result.error_code == "HTTPError"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_fetch_data_http_500_error(self, mock_client_class):
        """Test handling of HTTP 500 errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        
        http_error = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=mock_response
        )
        
        mock_client = AsyncMock()
        mock_client.get.side_effect = http_error
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client
        
        result = await self.provider.get_data("AAPL")
        
        assert result.success is False
        assert "HTTP 500 error" in result.error_message
        assert result.error_code == "HTTPError"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
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
        assert "Network error connecting to Zacks API" in result.error_message
        assert result.error_code == "ConnectionError"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
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
        assert "Invalid response from Zacks API" in result.error_message
        assert result.error_code == "ValueError"

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
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
        assert "Invalid response from Zacks API" in result.error_message

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_fetch_data_partial_data(self, mock_client_class):
        """Test handling of partial data response."""
        mock_response_data = {
            'ticker': 'AAPL',
            'price': 150.50,
            # Missing many other fields
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
        
        # Should succeed due to non-strict parsing
        assert result.success is True
        assert getattr(result.data, 'ticker') == 'AAPL'
        assert getattr(result.data, 'price') == 150.50
        # Missing fields should be None (default)
        assert getattr(result.data, 'volume') is None

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_fetch_data_timeout_handling(self, mock_client_class):
        """Test timeout configuration is applied."""
        config = ProviderConfig(timeout=0.1)  # Very short timeout
        provider = ZacksProvider(config)
        
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client
        
        # Verify timeout is passed to httpx
        await provider.get_data("AAPL")
        
        # Check that AsyncClient was created with correct timeout
        args, kwargs = mock_client_class.call_args
        assert 'timeout' in kwargs
        assert kwargs['timeout'].connect == 0.1

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_fetch_data_user_agent_header(self, mock_client_class):
        """Test that custom user agent is used."""
        config = ProviderConfig(user_agent="CustomAgent/1.0")
        provider = ZacksProvider(config)
        
        mock_response = MagicMock()
        mock_response.json.return_value = {'ticker': 'AAPL', 'price': 150.0}
        mock_response.raise_for_status.return_value = None
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client
        
        await provider.get_data("AAPL")
        
        # Check that AsyncClient was created with correct headers
        args, kwargs = mock_client_class.call_args
        assert 'headers' in kwargs
        assert kwargs['headers']['User-Agent'] == "CustomAgent/1.0"

    def test_get_data_sync(self):
        """Test synchronous wrapper."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_response = MagicMock()
            mock_response.json.return_value = {'ticker': 'AAPL', 'price': 150.0}
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
        provider = create_zacks_provider(
            timeout=60.0,
            retries=5,
            rate_limit=0.5
        )
        
        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5
        assert provider.config.rate_limit == 0.5


class TestZacksConfig:
    """Test cases for Zacks configuration."""

    def test_zacks_config_structure(self):
        """Test that Zacks config has expected structure."""
        assert ZACKS_CONFIG.name == "ZacksModel"
        assert ZACKS_CONFIG.strict_mode is False
        assert ZACKS_CONFIG.default_value is None
        
        # Check some key fields
        fields = ZACKS_CONFIG.fields
        assert "ticker" in fields
        assert "price" in fields
        assert "zacks_rank" in fields
        assert "value_score" in fields
        
        # Check expressions
        assert fields["ticker"]["expr"] == "ticker"
        assert fields["price"]["expr"] == "price"
        assert fields["zacks_rank"]["expr"] == "zacksRank"

    def test_zacks_config_all_fields_have_defaults(self):
        """Test that all fields have default values."""
        for field_name, field_config in ZACKS_CONFIG.fields.items():
            assert "default" in field_config
            assert field_config["default"] is None


class TestZacksProviderIntegration:
    """Integration tests for Zacks provider."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Test multiple concurrent requests to Zacks provider."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_response = MagicMock()
            mock_response.json.return_value = {'ticker': 'TEST', 'price': 100.0}
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
                provider.get_data("MSFT")
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
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_response = MagicMock()
            mock_response.json.return_value = {'ticker': 'AAPL', 'price': 150.0}
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
        with patch('httpx.AsyncClient') as mock_client_class:
            # First request succeeds, second fails
            success_response = MagicMock()
            success_response.json.return_value = {'ticker': 'AAPL', 'price': 150}
            success_response.raise_for_status.return_value = None
            
            error_response = MagicMock()
            error_response.status_code = 404
            http_error = httpx.HTTPStatusError(
                "Not Found",
                request=MagicMock(),
                response=error_response
            )
            
            # Track call count to return different responses
            call_count = 0
            
            def get_side_effect(*args, **kwargs):
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
                provider.get_data("AAPL"),    # Should succeed
                provider.get_data("INVALID")  # Should fail
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert results[0].success is True
            assert results[1].success is False
            assert "Ticker not found in Zacks" in results[1].error_message
