"""
Unit tests for the Tipranks provider module.
Tests TipranksDataProvider for fetching analyst data from Tipranks website.
"""

import asyncio
from unittest.mock import patch, MagicMock
import httpx
import pytest
from pydantic import ValidationError

from app.providers.tipranks import (
    TipranksDataProvider,
    TipranksDataModel,
    TipranksNewsSentimentProvider,
    TipranksNewsSentimentModel,
    create_tipranks_data_provider,
    create_tipranks_news_sentiment_provider,
)
from app.providers.base import ProviderType, ProviderConfig


class TestTipranksDataModel:
    """Test cases for TipranksDataModel."""

    def test_model_missing_required_fields_raises_validation_error(self):
        """Test that missing required fields raise ValidationError."""
        # Test data missing required fields
        test_data = {
            "ticker": "AAPL",
            "companyName": "Apple Inc",
            # Missing many other required fields...
        }

        with pytest.raises(ValidationError) as exc_info:
            TipranksDataModel(**test_data)

        # Should complain about missing fields
        assert "Field required" in str(exc_info.value)

    def test_model_null_values_raise_validation_error(self):
        """Test that None values in required fields raise ValidationError."""
        # Test data with None values in required fields
        test_data = {
            "ticker": "AAPL",
            "companyName": "Apple Inc",
            "consensus_rating": None,  # Should raise ValidationError
            "numOfAnalysts": 25,
            "buy_count": 18,
            "hold_count": 5,
            "sell_count": 2,
            "price_target": 180.50,
            "price_target_high": 200.0,
            "price_target_low": 150.0,
            "smart_score": 8,
            "marketCap": 3000000000,
        }

        with pytest.raises(ValidationError) as exc_info:
            TipranksDataModel(**test_data)

        # Should complain about None value in required field
        assert "Required" in str(exc_info.value) and "cannot be None" in str(
            exc_info.value
        )

    def test_model_initialization_with_aliases(self):
        """Test model initialization using field aliases."""
        data = {
            "ticker": "AAPL",
            "companyName": "Apple Inc",
            "consensus_rating": 5,
            "numOfAnalysts": 25,
            "buy_count": 18,
            "hold_count": 5,
            "sell_count": 2,
            "price_target": 180.50,
            "price_target_high": 200.0,
            "price_target_low": 150.0,
            "smart_score": 8,
            "marketCap": 3000000000,
        }
        model = TipranksDataModel(**data)

        assert model.ticker == "AAPL"
        assert model.company_name == "Apple Inc"
        assert model.consensus_rating == 5
        assert model.analyst_count == 25
        assert model.buy_count == 18
        assert model.hold_count == 5
        assert model.sell_count == 2
        assert model.price_target == 180.50
        assert model.price_target_high == 200.0
        assert model.price_target_low == 150.0
        assert model.smart_score == 8
        assert model.market_cap == 3000000000

    def test_model_with_extra_fields(self):
        """Test that model ignores extra fields."""
        data = {
            "ticker": "AAPL",
            "companyName": "Apple Inc",
            "consensus_rating": 5,
            "numOfAnalysts": 25,
            "buy_count": 18,
            "hold_count": 5,
            "sell_count": 2,
            "price_target": 180.50,
            "price_target_high": 200.0,
            "price_target_low": 150.0,
            "smart_score": 8,
            "marketCap": 3000000000,
            "extraField": "ignored",
            "anotherExtra": 123,
        }
        model = TipranksDataModel(**data)

        assert model.ticker == "AAPL"
        assert model.consensus_rating == 5
        assert not hasattr(model, "extraField")
        assert not hasattr(model, "anotherExtra")


class TestTipranksDataProvider:
    """Test cases for TipranksDataProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        # Disable caching to ensure we test the actual provider logic
        config = ProviderConfig(cache_enabled=False)
        self.provider = TipranksDataProvider(config)

    def test_provider_type(self):
        """Test that provider returns correct type."""
        assert self.provider.provider_type == ProviderType.TIPRANKS_DATA

    def test_provider_initialization(self):
        """Test provider initialization."""
        assert isinstance(self.provider.config, ProviderConfig)
        assert hasattr(self.provider, "logger")

    def test_provider_initialization_with_custom_config(self):
        """Test provider initialization with custom config."""
        config = ProviderConfig(
            timeout=60.0, retries=5, rate_limit=0.5, user_agent="TestApp/1.0"
        )
        provider = TipranksDataProvider(config)

        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5
        assert provider.config.rate_limit == 0.5
        assert provider.config.user_agent == "TestApp/1.0"

    @pytest.mark.asyncio
    async def test_fetch_data_no_query(self):
        """Test handling when no query is provided."""
        result = await self.provider.get_data(None)

        assert result.success is False
        assert "Query must be provided for TipranksDataProvider" in (
            result.error_message or ""
        )
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    async def test_fetch_data_empty_query(self):
        """Test handling when empty query is provided."""
        result = await self.provider.get_data("")

        assert result.success is False
        assert "Query must be provided for TipranksDataProvider" in (
            result.error_message or ""
        )
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    @patch("httpx.Client")
    async def test_fetch_data_http_error(self, mock_client_class):
        """Test handling of HTTP errors."""
        # Mock HTTP error
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=MagicMock(status_code=404)
        )
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        assert result.success is False
        assert result.error_code == "RetriableProviderException"

    @pytest.mark.asyncio
    @patch("httpx.Client")
    async def test_fetch_data_invalid_json(self, mock_client_class):
        """Test handling when Tipranks returns invalid JSON."""
        # Mock invalid JSON response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = None  # Invalid JSON
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        assert result.success is False
        assert "No Tipranks data found for query: AAPL" in (result.error_message or "")
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    @patch("httpx.Client")
    async def test_fetch_data_success(self, mock_client_class):
        """Test successful data fetching and parsing."""
        # Mock successful Tipranks response
        mock_tipranks_data = {
            "ticker": "AAPL",
            "companyName": "Apple Inc",
            "numOfAnalysts": 25,
            "marketCap": 3000000000,
            "consensuses": [
                {
                    "rating": 5,
                    "nB": 18,
                    "nH": 5,
                    "nS": 2,
                    "isLatest": 1,
                }
            ],
            "ptConsensus": [
                {
                    "priceTarget": 180.50,
                    "high": 200.0,
                    "low": 150.0,
                }
            ],
            "tipranksStockScore": {"score": 8},
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_tipranks_data
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, TipranksDataModel)
        assert result.data.ticker == "AAPL"
        assert result.data.company_name == "Apple Inc"
        assert result.data.consensus_rating == 5
        assert result.data.analyst_count == 25
        assert result.data.buy_count == 18
        assert result.data.price_target == 180.50

    @pytest.mark.asyncio
    @patch("httpx.Client")
    async def test_fetch_data_missing_consensus_data(self, mock_client_class):
        """Test handling when consensus data is missing from API response."""
        # Mock Tipranks response with missing consensus data
        mock_tipranks_data = {
            "ticker": "AAPL",
            "companyName": "Apple Inc",
            "numOfAnalysts": 15,
            "consensuses": [],  # Empty consensus data
            "ptConsensus": [],  # Empty price target data
            "tipranksStockScore": {},  # Empty stock score data
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_tipranks_data
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        # Should fail due to missing required nested data structures
        assert result.success is False
        assert "No latest consensus data found" in (result.error_message or "")
        assert result.error_code == "NonRetriableProviderException"

    def test_get_data_sync(self):
        """Test synchronous wrapper."""
        with patch("httpx.Client") as mock_client_class:
            # Mock successful response
            mock_tipranks_data = {
                "ticker": "AAPL",
                "companyName": "Apple Inc",
                "numOfAnalysts": 25,
                "marketCap": 3000000000,
                "consensuses": [
                    {
                        "rating": 5,
                        "nB": 18,
                        "nH": 5,
                        "nS": 2,
                        "isLatest": 1,
                    }
                ],
                "ptConsensus": [
                    {
                        "priceTarget": 180.50,
                        "high": 200.0,
                        "low": 150.0,
                    }
                ],
                "tipranksStockScore": {"score": 8},
            }

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_tipranks_data
            mock_response.raise_for_status.return_value = None
            mock_client.get.return_value = mock_response
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = None
            mock_client_class.return_value = mock_client

            result = self.provider.get_data_sync("AAPL")

            assert result.success is True

    @pytest.mark.asyncio
    async def test_get_data_without_query_raises_error(self):
        """Test that calling get_data() without query raises appropriate error."""
        result = await self.provider.get_data()  # No query parameter

        assert result.success is False
        assert "must be provided" in (result.error_message or "").lower()
        assert result.error_code == "NonRetriableProviderException"

    def test_get_data_sync_without_query_raises_error(self):
        """Test that calling get_data_sync() without query raises appropriate error."""
        result = self.provider.get_data_sync()  # No query parameter

        assert result.success is False
        assert "must be provided" in (result.error_message or "").lower()
        assert result.error_code == "NonRetriableProviderException"


class TestTipranksFactoryFunction:
    """Test cases for Tipranks provider factory function."""

    def test_create_tipranks_data_provider_defaults(self):
        """Test factory function with default parameters."""
        provider = create_tipranks_data_provider()

        assert isinstance(provider, TipranksDataProvider)
        assert provider.config.timeout == 30.0
        assert provider.config.retries == 3

    def test_create_tipranks_data_provider_custom(self):
        """Test factory function with custom parameters."""
        provider = create_tipranks_data_provider(timeout=60.0, retries=5)

        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5


class TestTipranksProviderIntegration:
    """Integration tests for Tipranks provider."""

    @pytest.mark.asyncio
    @patch("httpx.Client")
    async def test_multiple_concurrent_requests(self, mock_client_class):
        """Test multiple concurrent requests to Tipranks provider."""
        # Mock successful response for all requests
        mock_tipranks_data = {
            "ticker": "AAPL",
            "companyName": "Apple Inc",
            "numOfAnalysts": 25,
            "marketCap": 3000000000,
            "consensuses": [
                {
                    "rating": 5,
                    "nB": 18,
                    "nH": 5,
                    "nS": 2,
                    "isLatest": 1,
                }
            ],
            "ptConsensus": [
                {
                    "priceTarget": 180.50,
                    "high": 200.0,
                    "low": 150.0,
                }
            ],
            "tipranksStockScore": {"score": 8},
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_tipranks_data
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        provider = TipranksDataProvider()

        # Make multiple concurrent requests
        tasks = [
            provider.get_data("AAPL"),
            provider.get_data("MSFT"),
            provider.get_data("GOOGL"),
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(result.success for result in results)
        assert len(results) == 3

    @pytest.mark.asyncio
    @patch("httpx.Client")
    async def test_error_handling_in_concurrent_requests(self, mock_client_class):
        """Test error handling when some concurrent requests fail."""
        # Track call count to return different responses
        call_count = 0

        def get_side_effect(*args, **kwargs):  # pylint: disable=unused-argument
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            if call_count == 1:
                # First request succeeds
                response.json.return_value = {
                    "ticker": "AAPL",
                    "companyName": "Apple Inc",
                    "numOfAnalysts": 25,
                    "marketCap": 3000000000,
                    "consensuses": [
                        {
                            "rating": 5,
                            "nB": 18,
                            "nH": 5,
                            "nS": 2,
                            "isLatest": 1,
                        }
                    ],
                    "ptConsensus": [
                        {
                            "priceTarget": 180.50,
                            "high": 200.0,
                            "low": 150.0,
                        }
                    ],
                    "tipranksStockScore": {"score": 8},
                }
                response.raise_for_status.return_value = None
                return response
            else:
                # Second request fails with HTTP error
                response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Not Found",
                    request=MagicMock(),
                    response=MagicMock(status_code=404),
                )
                return response

        mock_client = MagicMock()
        mock_client.get.side_effect = get_side_effect
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        provider = TipranksDataProvider()

        tasks = [
            provider.get_data("AAPL"),  # Should succeed
            provider.get_data("INVALID"),  # Should fail
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error_code == "RetriableProviderException"


class TestCacheSettingsTipranks:
    """Test cases for cache setting on Tipranks provider."""

    @pytest.mark.asyncio
    @patch("httpx.Client")
    async def test_cache_disabled_per_provider(
        self, mock_client_class, tmp_path, monkeypatch
    ):
        # Set PROVIDER_CACHE_ROOT to isolated temp directory
        monkeypatch.setenv("PROVIDER_CACHE_ROOT", str(tmp_path))
        # Disable cache in provider config
        config = ProviderConfig(cache_enabled=False)
        provider = TipranksDataProvider(config)

        # Mock successful response
        mock_tipranks_data = {
            "ticker": "AAPL",
            "companyName": "Apple Inc",
            "numOfAnalysts": 15,
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_tipranks_data
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        # First and second calls should fetch fresh data
        await provider.get_data("AAPL")
        await provider.get_data("AAPL")

        # Should call twice due to cache disabled
        assert mock_client.get.call_count == 2


class TestGlobalCacheSettingsTipranks:
    """Test cases for global cache setting on Tipranks provider."""

    @pytest.mark.asyncio
    @patch("httpx.Client")
    async def test_global_cache_disabled(
        self, mock_client_class, tmp_path, monkeypatch
    ):
        # Set PROVIDER_CACHE_ROOT to isolated temp directory
        monkeypatch.setenv("PROVIDER_CACHE_ROOT", str(tmp_path))
        # Disable global cache
        from app.lib.settings import settings

        monkeypatch.setattr(settings, "PROVIDER_CACHE_ENABLED", False)

        provider = TipranksDataProvider(ProviderConfig())

        # Mock successful response
        mock_tipranks_data = {
            "ticker": "AAPL",
            "companyName": "Apple Inc",
            "numOfAnalysts": 15,
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_tipranks_data
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        # First and second calls should fetch fresh data
        await provider.get_data("AAPL")
        await provider.get_data("AAPL")

        # Should call twice when global cache disabled
        assert mock_client.get.call_count == 2


class TestTipranksNewsSentimentModel:
    """Test cases for TipranksNewsSentimentModel."""

    def test_model_missing_required_fields_raises_validation_error(self):
        """Test that missing required fields raise ValidationError."""
        # Test data missing required fields
        test_data = {
            "ticker": "AAPL",
            "companyName": "Apple Inc",
            # Missing many other required fields...
        }

        with pytest.raises(ValidationError) as exc_info:
            TipranksNewsSentimentModel(**test_data)

        # Should complain about missing fields
        assert "Field required" in str(exc_info.value)

    def test_model_null_values_raise_validation_error(self):
        """Test that None values in required fields raise ValidationError."""
        # Test data with None values in required fields
        test_data = {
            "ticker": "AAPL",
            "companyName": "Apple Inc",
            "score": None,  # Should raise ValidationError
            "bullish_percent": 0.85,
            "bearish_percent": 0.15,
            "articles_last_week": 84,
            "weekly_average": 132.25,
            "buzz_score": 0.635,
            "sectorAverageBullishPercent": 0.587,
            "word_cloud_count": 12,
        }

        with pytest.raises(ValidationError) as exc_info:
            TipranksNewsSentimentModel(**test_data)

        # Should complain about None value in required field
        assert "Required" in str(exc_info.value) and "cannot be None" in str(
            exc_info.value
        )

    def test_model_initialization_with_full_data(self):
        """Test model initialization with full data."""
        data = {
            "ticker": "MSFT",
            "companyName": "Microsoft",
            "score": 0.75,
            "bullish_percent": 0.85,
            "bearish_percent": 0.15,
            "articles_last_week": 84,
            "weekly_average": 132.25,
            "buzz_score": 0.635,
            "sectorAverageBullishPercent": 0.587,
            "word_cloud_count": 12,
        }
        model = TipranksNewsSentimentModel(**data)

        assert model.ticker == "MSFT"
        assert model.company_name == "Microsoft"
        assert model.sentiment_score == 0.75
        assert model.bullish_percent == 0.85
        assert model.bearish_percent == 0.15
        assert model.articles_last_week == 84
        assert model.weekly_average == 132.25
        assert model.buzz_score == 0.635
        assert model.sector_avg_bullish_percent == 0.587
        assert model.word_cloud_count == 12

    def test_model_with_extra_fields(self):
        """Test that model ignores extra fields."""
        data = {
            "ticker": "AAPL",
            "companyName": "Apple Inc",
            "score": 0.8,
            "bullish_percent": 0.85,
            "bearish_percent": 0.15,
            "articles_last_week": 84,
            "weekly_average": 132.25,
            "buzz_score": 0.635,
            "sectorAverageBullishPercent": 0.587,
            "word_cloud_count": 12,
            "extraField": "ignored",
            "anotherExtra": 123,
        }
        model = TipranksNewsSentimentModel(**data)

        assert model.ticker == "AAPL"
        assert model.sentiment_score == 0.8
        assert not hasattr(model, "extraField")
        assert not hasattr(model, "anotherExtra")


class TestTipranksNewsSentimentProvider:
    """Test cases for TipranksNewsSentimentProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        # Disable caching to ensure we test the actual provider logic
        config = ProviderConfig(cache_enabled=False)
        self.provider = TipranksNewsSentimentProvider(config)

    def test_provider_type(self):
        """Test that provider returns correct type."""
        assert self.provider.provider_type == ProviderType.TIPRANKS_NEWS_SENTIMENT

    def test_provider_initialization(self):
        """Test provider initialization."""
        assert isinstance(self.provider.config, ProviderConfig)
        assert hasattr(self.provider, "logger")

    def test_provider_initialization_with_custom_config(self):
        """Test provider initialization with custom config."""
        config = ProviderConfig(
            timeout=60.0, retries=5, rate_limit=0.5, user_agent="TestApp/1.0"
        )
        provider = TipranksNewsSentimentProvider(config)

        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5
        assert provider.config.rate_limit == 0.5
        assert provider.config.user_agent == "TestApp/1.0"

    @pytest.mark.asyncio
    async def test_fetch_data_no_query(self):
        """Test handling when no query is provided."""
        result = await self.provider.get_data(None)

        assert result.success is False
        assert "Query must be provided for TipranksNewsSentimentProvider" in (
            result.error_message or ""
        )
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    async def test_fetch_data_empty_query(self):
        """Test handling when empty query is provided."""
        result = await self.provider.get_data("")

        assert result.success is False
        assert "Query must be provided for TipranksNewsSentimentProvider" in (
            result.error_message or ""
        )
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    @patch("httpx.Client")
    async def test_fetch_data_http_error(self, mock_client_class):
        """Test handling of HTTP errors."""
        # Mock HTTP error
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=MagicMock(status_code=404)
        )
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        assert result.success is False
        assert result.error_code == "RetriableProviderException"

    @pytest.mark.asyncio
    @patch("httpx.Client")
    async def test_fetch_data_invalid_json(self, mock_client_class):
        """Test handling when Tipranks returns invalid JSON."""
        # Mock invalid JSON response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = None  # Invalid JSON
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        assert result.success is False
        assert "No Tipranks news sentiment data found for query: AAPL" in (
            result.error_message or ""
        )
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    @patch("httpx.Client")
    async def test_fetch_data_success(self, mock_client_class):
        """Test successful data fetching and parsing."""
        # Mock successful Tipranks response
        mock_tipranks_data = {
            "ticker": "MSFT",
            "companyName": "Microsoft",
            "score": 0.7352,
            "buzz": {
                "articlesInLastWeek": 84,
                "weeklyAverage": 132.25,
                "buzz": 0.6351,
            },
            "sentiment": {"bullishPercent": 1.0, "bearishPercent": 0.0},
            "sectorAverageBullishPercent": 0.5872,
            "wordCloud": [
                {"text": "hitting $368 billion", "grade": 5},
                {"text": "losing its leader", "grade": 11},
            ],
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_tipranks_data
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("MSFT")

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, TipranksNewsSentimentModel)
        assert result.data.ticker == "MSFT"
        assert result.data.company_name == "Microsoft"
        assert result.data.sentiment_score == 0.7352
        assert result.data.bullish_percent == 1.0
        assert result.data.bearish_percent == 0.0
        assert result.data.articles_last_week == 84
        assert result.data.weekly_average == 132.25
        assert result.data.buzz_score == 0.6351
        assert result.data.sector_avg_bullish_percent == 0.5872
        assert result.data.word_cloud_count == 2

    @pytest.mark.asyncio
    @patch("httpx.Client")
    async def test_fetch_data_missing_sentiment_data(self, mock_client_class):
        """Test handling when sentiment data is missing from API response."""
        # Mock Tipranks response with missing sentiment data
        mock_tipranks_data = {
            "ticker": "AAPL",
            "companyName": "Apple Inc",
            "score": 0.65,
            "sentiment": {},  # Empty sentiment data
            "buzz": {},  # Empty buzz data
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_tipranks_data
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("AAPL")

        # Should fail due to missing required nested data structures
        assert result.success is False
        assert "No sentiment data found" in (result.error_message or "")
        assert result.error_code == "NonRetriableProviderException"

    def test_get_data_sync(self):
        """Test synchronous wrapper."""
        with patch("httpx.Client") as mock_client_class:
            # Mock successful response
            mock_tipranks_data = {
                "ticker": "AAPL",
                "companyName": "Apple Inc",
                "score": 0.8,
                "sentiment": {
                    "bullishPercent": 0.85,
                    "bearishPercent": 0.15,
                },
                "buzz": {
                    "articlesInLastWeek": 84,
                    "weeklyAverage": 132.25,
                    "buzz": 0.635,
                },
                "sectorAverageBullishPercent": 0.587,
                "wordCloud": [{"text": "test"}],
            }

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_tipranks_data
            mock_response.raise_for_status.return_value = None
            mock_client.get.return_value = mock_response
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = None
            mock_client_class.return_value = mock_client

            result = self.provider.get_data_sync("AAPL")

            assert result.success is True


class TestTipranksNewsSentimentFactoryFunction:
    """Test cases for Tipranks news sentiment provider factory function."""

    def test_create_tipranks_news_sentiment_provider_defaults(self):
        """Test factory function with default parameters."""
        provider = create_tipranks_news_sentiment_provider()

        assert isinstance(provider, TipranksNewsSentimentProvider)
        assert provider.config.timeout == 30.0
        assert provider.config.retries == 3

    def test_create_tipranks_news_sentiment_provider_custom(self):
        """Test factory function with custom parameters."""
        provider = create_tipranks_news_sentiment_provider(timeout=60.0, retries=5)

        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5
