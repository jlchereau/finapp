"""
Unit tests for the BlackRock provider module.
Tests BlackrockHoldingsProvider for fetching ETF holdings data from BlackRock website.
"""

import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
import pandas as pd
import pytest

from app.providers.blackrock import (
    BlackrockHoldingsProvider,
    create_blackrock_holdings_provider,
)
from app.providers.base import ProviderType, ProviderConfig


class TestBlackrockHoldingsProvider:
    """Test cases for BlackrockHoldingsProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        # Disable caching to ensure we test the actual provider logic
        config = ProviderConfig(cache_enabled=False)
        self.provider = BlackrockHoldingsProvider(config)

    def test_provider_type(self):
        """Test that provider returns correct type."""
        assert self.provider.provider_type == ProviderType.BLACKROCK_HOLDINGS

    def test_provider_initialization(self):
        """Test provider initialization."""
        assert isinstance(self.provider.config, ProviderConfig)
        assert hasattr(self.provider, "logger")

    def test_provider_initialization_with_custom_config(self):
        """Test provider initialization with custom config."""
        config = ProviderConfig(
            timeout=60.0, retries=5, rate_limit=0.5, user_agent="TestApp/1.0"
        )
        provider = BlackrockHoldingsProvider(config)

        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5
        assert provider.config.rate_limit == 0.5
        assert provider.config.user_agent == "TestApp/1.0"

    @pytest.mark.asyncio
    async def test_fetch_data_no_query(self):
        """Test handling when no query is provided."""
        result = await self.provider.get_data(None)

        assert result.success is False
        assert "ETF ticker must be provided" in (result.error_message or "")
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    async def test_fetch_data_empty_query(self):
        """Test handling when empty query is provided."""
        result = await self.provider.get_data("")

        assert result.success is False
        assert "ETF ticker must be provided" in (result.error_message or "")
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    @patch("app.providers.blackrock.os.getenv")
    @patch("httpx.AsyncClient")
    async def test_fetch_data_no_search_results(self, mock_client_class, mock_getenv):
        """Test handling when Serper API returns no results."""
        # Mock Serper API key
        mock_getenv.return_value = "test_api_key"

        # Mock empty search results
        mock_response = MagicMock()
        mock_response.json.return_value = {"organic": []}
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("INVALID")

        assert result.success is False
        assert "not found on BlackRock website" in (result.error_message or "")
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    @patch("app.providers.blackrock.os.getenv")
    @patch("httpx.AsyncClient")
    async def test_fetch_data_no_valid_etf_url(self, mock_client_class, mock_getenv):
        """Test handling when search results don't contain valid ETF URLs."""
        # Mock Serper API key
        mock_getenv.return_value = "test_api_key"

        # Mock search results without valid BlackRock ETF URLs
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic": [
                {"link": "https://example.com/not-blackrock"},
                {"link": "https://blackrock.com/not-etf-page"},
            ]
        }
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("INVALID")

        assert result.success is False
        assert "No valid BlackRock ETF page found" in (result.error_message or "")
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    @patch("app.providers.blackrock.os.getenv")
    @patch("httpx.AsyncClient")
    async def test_fetch_data_http_error(self, mock_client_class, mock_getenv):
        """Test handling of HTTP errors when fetching ETF page."""
        # Mock Serper API key
        mock_getenv.return_value = "test_api_key"

        # Mock search results
        search_response = MagicMock()
        search_response.json.return_value = {
            "organic": [{"link": "https://blackrock.com/us/products/123/test-etf"}]
        }
        search_response.raise_for_status.return_value = None

        # Mock HTTP error when fetching ETF page
        mock_client = AsyncMock()
        mock_client.post.return_value = search_response  # Serper API call succeeds
        mock_client.get.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=MagicMock(status_code=404)
        )
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("TEST")

        assert result.success is False
        assert result.error_code == "RetriableProviderException"

    @pytest.mark.asyncio
    @patch("app.providers.blackrock.os.getenv")
    @patch("httpx.AsyncClient")
    async def test_fetch_data_no_download_link(self, mock_client_class, mock_getenv):
        """Test handling when no Excel download link is found."""
        # Mock Serper API key
        mock_getenv.return_value = "test_api_key"

        # Mock search results
        search_response = MagicMock()
        search_response.json.return_value = {
            "organic": [{"link": "https://blackrock.com/us/products/123/test-etf"}]
        }
        search_response.raise_for_status.return_value = None

        # Mock HTTP response with HTML that has no download link
        etf_page_response = MagicMock()
        etf_page_response.text = "<html><body>No download link here</body></html>"
        etf_page_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.post.return_value = search_response  # Serper API call
        mock_client.get.return_value = etf_page_response  # ETF page fetch
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("TEST")

        assert result.success is False
        assert "Holdings download link not found" in (result.error_message or "")
        assert result.error_code == "RetriableProviderException"

    @pytest.mark.asyncio
    @patch("app.providers.blackrock.os.getenv")
    @patch("httpx.AsyncClient")
    async def test_fetch_data_success(self, mock_client_class, mock_getenv):
        """Test successful data fetching and parsing."""
        # Mock Serper API key
        mock_getenv.return_value = "test_api_key"

        # Mock search results
        search_response = MagicMock()
        search_response.json.return_value = {
            "organic": [{"link": "https://blackrock.com/us/products/123/test-etf"}]
        }
        search_response.raise_for_status.return_value = None

        # Mock HTML response with download link
        html_content = """
        <html>
            <body>
                <a class="icon-xls-export"
                   href="/download.ajax?fileType=xls&dataType=fund">
                    Download Holdings
                </a>
            </body>
        </html>
        """

        # Mock XML content (BlackRock SpreadsheetML format)
        xml_content = """<?xml version="1.0"?>
<ss:Workbook xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">
<ss:Worksheet ss:Name="Holdings">
<ss:Table>
<ss:Row>
<ss:Cell><ss:Data ss:Type="String">Name</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">Weight (%)</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">Shares</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">Market Value</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">Sector</ss:Data></ss:Cell>
</ss:Row>
<ss:Row>
<ss:Cell><ss:Data ss:Type="String">Apple Inc</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">5.2%</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">1000000</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">$520M</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">Technology</ss:Data></ss:Cell>
</ss:Row>
<ss:Row>
<ss:Cell><ss:Data ss:Type="String">Microsoft Corp</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">4.8%</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">800000</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">$480M</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">Technology</ss:Data></ss:Cell>
</ss:Row>
<ss:Row>
<ss:Cell><ss:Data ss:Type="String">Amazon.com Inc</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">3.1%</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">500000</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">$310M</ss:Data></ss:Cell>
<ss:Cell><ss:Data ss:Type="String">Consumer Discretionary</ss:Data></ss:Cell>
</ss:Row>
</ss:Table>
</ss:Worksheet>
</ss:Workbook>""".encode(
            "utf-8"
        )

        # Mock ETF page response
        etf_page_response = MagicMock()
        etf_page_response.text = html_content
        etf_page_response.raise_for_status.return_value = None

        # Mock XML download response
        xml_download_response = MagicMock()
        xml_download_response.content = xml_content
        xml_download_response.raise_for_status.return_value = None

        # Setup mock client to return different responses for different calls
        mock_client = AsyncMock()

        # First call is Serper search, second is ETF page, third is XML download
        mock_client.post.return_value = search_response  # Serper API call
        mock_client.get.side_effect = [
            etf_page_response,
            xml_download_response,
        ]  # HTTP GET calls

        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.provider.get_data("TEST")

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 3
        assert "ticker" in result.data.columns
        assert result.data["ticker"].iloc[0] == "TEST"
        assert "holding_name" in result.data.columns
        assert "weight" in result.data.columns

    def test_extract_download_url_with_icon_xls_export(self):
        """Test extracting download URL with icon-xls-export class."""
        from bs4 import BeautifulSoup

        html = """
        <a class="icon-xls-export" href="/download.ajax?fileType=xls&dataType=fund">
            Download
        </a>
        """
        soup = BeautifulSoup(html, "lxml")
        base_url = "https://blackrock.com/us/products/123/test-etf"

        url = self.provider._extract_download_url(
            soup, base_url
        )  # pylint: disable=protected-access

        assert url == "https://blackrock.com/download.ajax?fileType=xls&dataType=fund"

    def test_extract_download_url_with_ajax_href(self):
        """Test extracting download URL with ajax and fileType in href."""
        from bs4 import BeautifulSoup

        html = """
        <a href="/fund/123.ajax?fileType=xls&fileName=test_fund&dataType=fund">
            Download Holdings
        </a>
        """
        soup = BeautifulSoup(html, "lxml")
        base_url = "https://blackrock.com/us/products/123/test-etf"

        url = self.provider._extract_download_url(
            soup, base_url
        )  # pylint: disable=protected-access

        expected_url = (
            "https://blackrock.com/fund/123.ajax?"
            "fileType=xls&fileName=test_fund&dataType=fund"
        )
        assert url == expected_url

    def test_extract_download_url_no_match(self):
        """Test extracting download URL when no matching link is found."""
        from bs4 import BeautifulSoup

        html = """
        <a href="/some-other-link">Not a download link</a>
        """
        soup = BeautifulSoup(html, "lxml")
        base_url = "https://blackrock.com/us/products/123/test-etf"

        url = self.provider._extract_download_url(
            soup, base_url
        )  # pylint: disable=protected-access

        assert url is None

    def test_clean_holdings_data(self):
        """Test cleaning and standardizing holdings data."""
        # Create mock raw data with proper column names
        raw_data = pd.DataFrame(
            {
                "Name": ["Apple Inc", "Microsoft Corp"],
                "Weight (%)": ["5.2%", "4.8%"],
                "Shares": ["1000000", "800000"],
                "Market Value": ["$520M", "$480M"],
            }
        )

        cleaned_data = self.provider._clean_holdings_data(
            raw_data, "TEST"
        )  # pylint: disable=protected-access

        assert len(cleaned_data) == 2  # Two holdings
        assert "ticker" in cleaned_data.columns
        assert "holding_name" in cleaned_data.columns
        assert "weight" in cleaned_data.columns
        assert cleaned_data["ticker"].iloc[0] == "TEST"
        assert cleaned_data["holding_name"].iloc[0] == "Apple Inc"
        assert cleaned_data["weight"].iloc[0] == 5.2  # Converted to float

    @patch("app.providers.blackrock.os.getenv")
    def test_get_data_sync(self, mock_getenv):
        """Test synchronous wrapper."""
        # Mock Serper API key
        mock_getenv.return_value = "test_api_key"

        with patch("httpx.AsyncClient") as mock_client_class:

            # Mock search results
            search_response = MagicMock()
            search_response.json.return_value = {
                "organic": [{"link": "https://blackrock.com/us/products/123/test-etf"}]
            }
            search_response.raise_for_status.return_value = None

            # Mock HTML response
            html_content = """
            <a class="icon-xls-export" href="/download.ajax?fileType=csv&dataType=fund">
                Download
            </a>
            """
            etf_page_response = MagicMock()
            etf_page_response.text = html_content
            etf_page_response.raise_for_status.return_value = None

            # Mock CSV content
            csv_content = (
                "Name,Weight (%),Shares,Market Value\n" "Apple Inc,5.2%,1000000,$520M\n"
            ).encode("utf-8")
            csv_download_response = MagicMock()
            csv_download_response.content = csv_content
            csv_download_response.raise_for_status.return_value = None

            mock_client = AsyncMock()
            mock_client.post.return_value = search_response  # Serper API call
            mock_client.get.side_effect = [
                etf_page_response,
                csv_download_response,
            ]  # ETF page and CSV download
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = self.provider.get_data_sync("TEST")

            assert result.success is True

    @pytest.mark.asyncio
    async def test_get_data_without_query_raises_error(self):
        """Test that calling get_data() without query raises appropriate error."""
        result = await self.provider.get_data()  # No query parameter

        assert result.success is False
        assert result.error_message is not None
        assert "must be provided" in result.error_message.lower()
        assert result.error_code == "NonRetriableProviderException"

    def test_get_data_sync_without_query_raises_error(self):
        """Test that calling get_data_sync() without query raises appropriate error."""
        result = self.provider.get_data_sync()  # No query parameter

        assert result.success is False
        assert result.error_message is not None
        assert "must be provided" in result.error_message.lower()
        assert result.error_code == "NonRetriableProviderException"


class TestBlackrockFactoryFunction:
    """Test cases for BlackRock provider factory function."""

    def test_create_blackrock_holdings_provider_defaults(self):
        """Test factory function with default parameters."""
        provider = create_blackrock_holdings_provider()

        assert isinstance(provider, BlackrockHoldingsProvider)
        assert provider.config.timeout == 30.0
        assert provider.config.retries == 3
        assert provider.config.rate_limit == 1.0
        assert "BlackRock Holdings Provider" in provider.config.user_agent

    def test_create_blackrock_holdings_provider_custom(self):
        """Test factory function with custom parameters."""
        provider = create_blackrock_holdings_provider(
            timeout=60.0, retries=5, rate_limit=0.5
        )

        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5
        assert provider.config.rate_limit == 0.5


class TestBlackrockProviderIntegration:
    """Integration tests for BlackRock provider."""

    @pytest.mark.asyncio
    @patch("app.providers.blackrock.os.getenv")
    async def test_multiple_concurrent_requests(self, mock_getenv):
        """Test multiple concurrent requests to BlackRock provider."""
        # Mock Serper API key
        mock_getenv.return_value = "test_api_key"

        with patch("httpx.AsyncClient") as mock_client_class:

            # Mock search results
            search_response = MagicMock()
            search_response.json.return_value = {
                "organic": [{"link": "https://blackrock.com/us/products/123/test-etf"}]
            }
            search_response.raise_for_status.return_value = None

            # Mock responses
            html_content = """
            <a class="icon-xls-export" href="/download.ajax?fileType=csv&dataType=fund">
                Download
            </a>
            """
            etf_page_response = MagicMock()
            etf_page_response.text = html_content
            etf_page_response.raise_for_status.return_value = None

            # Mock CSV content
            csv_content = "Name,Weight (%)\nTest Holding,5.0%\n".encode("utf-8")
            csv_download_response = MagicMock()
            csv_download_response.content = csv_content
            csv_download_response.raise_for_status.return_value = None

            mock_client = AsyncMock()
            mock_client.post.return_value = search_response  # Serper API call
            mock_client.get.side_effect = [
                etf_page_response,
                csv_download_response,
            ] * 3  # For 3 concurrent requests
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Disable caching to ensure we test the actual provider logic
            config = ProviderConfig(cache_enabled=False)
            provider = BlackrockHoldingsProvider(config)

            # Make multiple concurrent requests
            tasks = [
                provider.get_data("TEST1"),
                provider.get_data("TEST2"),
                provider.get_data("TEST3"),
            ]

            results = await asyncio.gather(*tasks)

            # All should succeed
            assert all(result.success for result in results)
            assert len(results) == 3

    @pytest.mark.asyncio
    @patch("app.providers.blackrock.os.getenv")
    async def test_error_handling_in_concurrent_requests(self, mock_getenv):
        """Test error handling when some concurrent requests fail."""
        # Mock Serper API key
        mock_getenv.return_value = "test_api_key"

        with patch("httpx.AsyncClient") as mock_client_class:

            # Track call count to return different responses
            call_count = 0

            def search_side_effect(*args, **kwargs):  # pylint: disable=unused-argument
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First request succeeds
                    response = MagicMock()
                    response.json.return_value = {
                        "organic": [
                            {"link": "https://blackrock.com/us/products/123/test-etf"}
                        ]
                    }
                    response.raise_for_status.return_value = None
                    return response
                else:
                    # Second request fails - no results
                    response = MagicMock()
                    response.json.return_value = {"organic": []}
                    response.raise_for_status.return_value = None
                    return response

            # Mock successful response for first request
            html_content = """
            <a class="icon-xls-export" href="/download.ajax?fileType=csv&dataType=fund">
                Download
            </a>
            """
            etf_page_response = MagicMock()
            etf_page_response.text = html_content
            etf_page_response.raise_for_status.return_value = None

            # Mock CSV content
            csv_content = "Name,Weight (%)\nTest Holding,5.0%\n".encode("utf-8")
            csv_download_response = MagicMock()
            csv_download_response.content = csv_content
            csv_download_response.raise_for_status.return_value = None

            mock_client = AsyncMock()
            mock_client.post.side_effect = search_side_effect  # Serper API calls
            mock_client.get.side_effect = [
                etf_page_response,
                csv_download_response,
            ]  # Only for successful request
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # Disable caching to ensure we test the actual provider logic
            config = ProviderConfig(cache_enabled=False)
            provider = BlackrockHoldingsProvider(config)

            tasks = [
                provider.get_data("TEST1"),  # Should succeed
                provider.get_data("INVALID"),  # Should fail
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 2
            assert results[0].success is True
            assert results[1].success is False
            assert "not found on BlackRock website" in (results[1].error_message or "")


class TestCacheSettingsBlackrock:
    """Test cases for cache setting on BlackRock provider."""

    @pytest.mark.asyncio
    @patch("app.providers.blackrock.os.getenv")
    async def test_cache_disabled_per_provider(
        self, mock_getenv, tmp_path, monkeypatch
    ):
        # Mock Serper API key
        mock_getenv.return_value = "test_api_key"

        # Set PROVIDER_CACHE_ROOT to isolated temp directory
        monkeypatch.setenv("PROVIDER_CACHE_ROOT", str(tmp_path))
        # Disable cache in provider config
        config = ProviderConfig(cache_enabled=False)
        provider = BlackrockHoldingsProvider(config)

        with patch("httpx.AsyncClient") as mock_client_class:

            # Mock search results
            search_response = MagicMock()
            search_response.json.return_value = {
                "organic": [{"link": "https://blackrock.com/us/products/123/test-etf"}]
            }
            search_response.raise_for_status.return_value = None

            # Mock responses
            html_content = """
            <a class="icon-xls-export" href="/download.ajax?fileType=csv&dataType=fund">
                Download
            </a>
            """
            etf_page_response = MagicMock()
            etf_page_response.text = html_content
            etf_page_response.raise_for_status.return_value = None

            # Mock CSV content
            csv_content = "Name,Weight (%)\nTest Holding,5.0%\n".encode("utf-8")
            csv_download_response = MagicMock()
            csv_download_response.content = csv_content
            csv_download_response.raise_for_status.return_value = None

            mock_client = AsyncMock()
            mock_client.post.return_value = search_response  # Serper API call
            mock_client.get.side_effect = [
                etf_page_response,
                csv_download_response,
            ] * 2  # For two calls
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # First and second calls should fetch fresh data
            await provider.get_data("TEST")
            await provider.get_data("TEST")

            # Should call twice due to cache disabled
            assert mock_client.post.call_count == 2


class TestGlobalCacheSettingsBlackrock:
    """Test cases for global cache setting on BlackRock provider."""

    @pytest.mark.asyncio
    @patch("app.providers.blackrock.os.getenv")
    async def test_global_cache_disabled(self, mock_getenv, tmp_path, monkeypatch):
        # Mock Serper API key
        mock_getenv.return_value = "test_api_key"

        # Set PROVIDER_CACHE_ROOT to isolated temp directory
        monkeypatch.setenv("PROVIDER_CACHE_ROOT", str(tmp_path))
        # Disable global cache
        from app.lib.settings import settings

        monkeypatch.setattr(settings, "PROVIDER_CACHE_ENABLED", False)

        provider = BlackrockHoldingsProvider(ProviderConfig())

        with patch("httpx.AsyncClient") as mock_client_class:

            # Mock search results
            search_response = MagicMock()
            search_response.json.return_value = {
                "organic": [{"link": "https://blackrock.com/us/products/123/test-etf"}]
            }
            search_response.raise_for_status.return_value = None

            # Mock responses
            html_content = """
            <a class="icon-xls-export" href="/download.ajax?fileType=csv&dataType=fund">
                Download
            </a>
            """
            etf_page_response = MagicMock()
            etf_page_response.text = html_content
            etf_page_response.raise_for_status.return_value = None

            # Mock CSV content
            csv_content = "Name,Weight (%)\nTest Holding,5.0%\n".encode("utf-8")
            csv_download_response = MagicMock()
            csv_download_response.content = csv_content
            csv_download_response.raise_for_status.return_value = None

            mock_client = AsyncMock()
            mock_client.post.return_value = search_response  # Serper API call
            mock_client.get.side_effect = [
                etf_page_response,
                csv_download_response,
            ] * 2  # For two calls
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # First and second calls should fetch fresh data
            await provider.get_data("TEST")
            await provider.get_data("TEST")

            # Should call twice when global cache disabled
            assert mock_client.post.call_count == 2
