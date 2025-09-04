"""
Unit tests for the Shiller provider module.
Tests ShillerCAPEProvider for fetching CAPE data from Shiller website.
"""

import math
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import pytest

from app.providers.shiller import ShillerCAPEProvider, create_shiller_cape_provider
from app.providers.base import ProviderType, ProviderConfig


# Sample Excel data for testing - create DataFrame structure for mocking
def create_sample_dataframe():
    """Create sample DataFrame with proper Excel structure."""
    # Create a DataFrame with enough columns (up to column Q = 16)
    # Only populate the columns we need for testing
    data = {
        0: [2023.01, 2023.02, 2023.03, 2024.01],  # Column A (dates)
        12: [28.5, 29.1, 27.8, 30.2],  # Column M (CAPE values)
        16: [2.1, 2.3, 1.9, 2.5],  # Column Q (Excess CAPE yield)
    }

    # Create DataFrame with these columns and fill missing columns with NaN
    df = pd.DataFrame(index=range(4))  # 4 rows
    for col in range(17):  # Up to column Q (16)
        if col in data:
            df[col] = data[col]
        else:
            df[col] = None

    return df


EXPECTED_PROCESSED_DATA = pd.DataFrame(
    {"CAPE": [28.5, 29.1, 27.8, 30.2], "Excess_CAPE_Yield": [2.1, 2.3, 1.9, 2.5]},
    index=pd.DatetimeIndex(
        [
            datetime(2023, 1, 1),
            datetime(2023, 2, 1),
            datetime(2023, 3, 1),
            datetime(2024, 1, 1),
        ],
        name="Date",
    ),
)


class TestShillerCAPEProvider:
    """Test cases for ShillerCAPEProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        # Disable caching and retries to ensure we test the actual provider logic
        config = ProviderConfig(cache_enabled=False, retries=0)
        self.provider = ShillerCAPEProvider(config)

    def test_provider_type(self):
        """Test that provider returns correct type."""
        assert self.provider.provider_type == ProviderType.SHILLER_CAPE

    def test_provider_initialization(self):
        """Test provider initialization."""
        assert isinstance(self.provider.config, ProviderConfig)
        assert hasattr(self.provider, "logger")

    def test_provider_initialization_with_custom_config(self):
        """Test provider initialization with custom config."""
        config = ProviderConfig(
            timeout=60.0, retries=5, rate_limit=0.5, user_agent="TestApp/1.0"
        )
        provider = ShillerCAPEProvider(config)

        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5
        assert provider.config.rate_limit == 0.5
        assert provider.config.user_agent == "TestApp/1.0"

    @pytest.mark.asyncio
    @patch("app.providers.shiller.WebPageScrapper")
    @patch("httpx.AsyncClient")
    @patch("app.providers.shiller.xlrd.open_workbook")
    @patch("pandas.read_excel")
    async def test_fetch_data_success(
        self, mock_read_excel, mock_xlrd_open, mock_httpx_client, mock_scrapper
    ):
        """Test successful data fetching and processing."""
        # Mock WebPageScrapper
        mock_scrapper_instance = AsyncMock()
        mock_scrapper_instance.extract_href.return_value = [
            "//img1.wsimg.com/downloads/ie_data.xls?ver=123456789"
        ]
        mock_scrapper.return_value.__aenter__.return_value = mock_scrapper_instance

        # Mock httpx download
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"fake_excel_content"
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Mock xlrd workbook
        mock_workbook = MagicMock()
        mock_xlrd_open.return_value = mock_workbook

        # Mock pandas read_excel
        mock_df = create_sample_dataframe()
        mock_read_excel.return_value = mock_df

        # Test the provider
        result = await self.provider.get_data(None)

        # Verify the result
        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, pd.DataFrame)

        # Check structure
        assert list(result.data.columns) == ["CAPE", "Excess_CAPE_Yield"]
        assert len(result.data) == 4
        assert result.data.index.name == "Date"

        # Verify method calls
        mock_scrapper.assert_called_once_with("https://shillerdata.com/", timeout=30.0)
        mock_scrapper_instance.extract_href.assert_called_once_with(
            'a[href*="ie_data.xls"]'
        )
        mock_client.get.assert_called_once_with(
            "https://img1.wsimg.com/downloads/ie_data.xls?ver=123456789"
        )
        mock_xlrd_open.assert_called_once_with(file_contents=b"fake_excel_content")
        mock_read_excel.assert_called_once_with(mock_workbook, sheet_name="Data", skiprows=8, engine="xlrd")

    @pytest.mark.asyncio
    @patch("app.providers.shiller.WebPageScrapper")
    async def test_fetch_data_no_excel_link_found(self, mock_scrapper):
        """Test handling when no Excel download link is found."""
        # Mock WebPageScrapper returning empty results
        mock_scrapper_instance = AsyncMock()
        mock_scrapper_instance.extract_href.return_value = []
        mock_scrapper.return_value.__aenter__.return_value = mock_scrapper_instance

        result = await self.provider.get_data(None)

        assert result.success is False
        assert "Could not find Excel download link" in (result.error_message or "")
        assert result.error_code == "NonRetriableProviderException"

    @pytest.mark.asyncio
    @patch("app.providers.shiller.WebPageScrapper")
    @patch("httpx.AsyncClient")
    async def test_fetch_data_download_error(self, mock_httpx_client, mock_scrapper):
        """Test handling of HTTP download errors."""
        # Mock WebPageScrapper
        mock_scrapper_instance = AsyncMock()
        mock_scrapper_instance.extract_href.return_value = ["/downloads/ie_data.xls"]
        mock_scrapper.return_value.__aenter__.return_value = mock_scrapper_instance

        # Mock httpx error
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Download failed")
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        result = await self.provider.get_data(None)

        assert result.success is False
        assert result.error_code == "RetriableProviderException"

    @pytest.mark.asyncio
    @patch("app.providers.shiller.WebPageScrapper")
    @patch("httpx.AsyncClient")
    @patch("app.providers.shiller.xlrd.open_workbook")
    @patch("pandas.read_excel")
    async def test_url_handling_relative_urls(
        self, mock_read_excel, mock_xlrd_open, mock_httpx_client, mock_scrapper
    ):
        """Test handling of relative URLs from scraping."""
        # Mock WebPageScrapper with relative URL
        mock_scrapper_instance = AsyncMock()
        mock_scrapper_instance.extract_href.return_value = ["/downloads/ie_data.xls"]
        mock_scrapper.return_value.__aenter__.return_value = mock_scrapper_instance

        # Mock httpx and pandas
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"fake_content"
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        mock_read_excel.return_value = create_sample_dataframe()

        await self.provider.get_data(None)

        # Verify the relative URL was made absolute
        mock_client.get.assert_called_once_with(
            "https://shillerdata.com/downloads/ie_data.xls"
        )

    @pytest.mark.asyncio
    @patch("app.providers.shiller.WebPageScrapper")
    @patch("httpx.AsyncClient")
    @patch("app.providers.shiller.xlrd.open_workbook")
    @patch("pandas.read_excel")
    async def test_url_handling_protocol_relative_urls(
        self, mock_read_excel, mock_xlrd_open, mock_httpx_client, mock_scrapper
    ):
        """Test handling of protocol-relative URLs."""
        # Mock WebPageScrapper with protocol-relative URL
        mock_scrapper_instance = AsyncMock()
        mock_scrapper_instance.extract_href.return_value = [
            "//img1.wsimg.com/downloads/ie_data.xls"
        ]
        mock_scrapper.return_value.__aenter__.return_value = mock_scrapper_instance

        # Mock httpx and pandas
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"fake_content"
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        mock_read_excel.return_value = create_sample_dataframe()

        await self.provider.get_data(None)

        # Verify the protocol-relative URL was made absolute
        mock_client.get.assert_called_once_with(
            "https://img1.wsimg.com/downloads/ie_data.xls"
        )

    def test_date_parsing_logic(self):
        """Test the date parsing logic using math.modf."""
        # Test cases from the comments: 2024.01 = Jan 2024, 2024.10 = Oct 2024
        test_cases = [
            (2024.01, datetime(2024, 1, 1)),
            (2024.10, datetime(2024, 10, 1)),
            (2023.12, datetime(2023, 12, 1)),
            (2020.05, datetime(2020, 5, 1)),
        ]

        for date_val, expected_date in test_cases:
            # Replicate the date parsing logic from the provider
            month_part, year_part = math.modf(float(date_val))
            year = int(year_part)
            month = int(round(month_part * 100)) if month_part > 0 else 1

            # Handle edge cases
            if month < 1 or month > 12:
                month = 1

            result_date = datetime(year, month, 1)
            assert result_date == expected_date, f"Failed for {date_val}"

    def test_date_parsing_edge_cases(self):
        """Test date parsing edge cases."""
        # Test invalid month handling
        month_part, year_part = math.modf(2024.13)  # Invalid month 13
        year = int(year_part)
        month = int(month_part * 100) if month_part > 0 else 1

        if month < 1 or month > 12:
            month = 1  # Should default to 1

        assert month == 1
        assert year == 2024

    @pytest.mark.asyncio
    @patch("app.providers.shiller.WebPageScrapper")
    @patch("httpx.AsyncClient")
    @patch("app.providers.shiller.xlrd.open_workbook")
    @patch("pandas.read_excel")
    async def test_data_cleaning(
        self, mock_read_excel, mock_xlrd_open, mock_httpx_client, mock_scrapper
    ):
        """Test that NaN values are properly handled."""
        # Mock WebPageScrapper and httpx
        mock_scrapper_instance = AsyncMock()
        mock_scrapper_instance.extract_href.return_value = ["/downloads/ie_data.xls"]
        mock_scrapper.return_value.__aenter__.return_value = mock_scrapper_instance

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"fake_content"
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Mock DataFrame with NaN values
        data_with_nans = create_sample_dataframe()
        data_with_nans[12] = pd.Series(
            [28.5, None, 27.8, 30.2], dtype=float
        )  # Add NaN in CAPE values

        mock_read_excel.return_value = data_with_nans

        result = await self.provider.get_data(None)

        assert result.success is True
        # Should still have 4 rows since we only drop rows with ALL NaN values
        assert result.data is not None
        assert len(result.data) == 4


    def test_get_data_sync(self):
        """Test synchronous wrapper."""
        with patch.object(
            self.provider, "get_data", return_value=MagicMock(success=True)
        ) as mock_get_data:
            result = self.provider.get_data_sync(None)
            assert result.success is True
            mock_get_data.assert_called_once_with(None)


class TestShillerFactoryFunction:
    """Test cases for Shiller provider factory function."""

    def test_create_shiller_cape_provider_defaults(self):
        """Test factory function with default parameters."""
        provider = create_shiller_cape_provider()

        assert isinstance(provider, ShillerCAPEProvider)
        assert provider.config.timeout == 30.0
        assert provider.config.retries == 3

    def test_create_shiller_cape_provider_custom(self):
        """Test factory function with custom parameters."""
        provider = create_shiller_cape_provider(timeout=60.0, retries=5)

        assert provider.config.timeout == 60.0
        assert provider.config.retries == 5


class TestShillerCAPEProviderIntegration:
    """Integration tests for ShillerCAPEProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        config = ProviderConfig(cache_enabled=False)
        self.provider = ShillerCAPEProvider(config)

    @pytest.mark.asyncio
    @patch("app.providers.shiller.WebPageScrapper")
    @patch("httpx.AsyncClient")
    @patch("app.providers.shiller.xlrd.open_workbook")
    @patch("pandas.read_excel")
    async def test_full_workflow_integration(
        self, mock_read_excel, mock_xlrd_open, mock_httpx_client, mock_scrapper
    ):
        """Test the complete workflow from scraping to data processing."""
        # Mock the complete chain
        mock_scrapper_instance = AsyncMock()
        mock_scrapper_instance.extract_href.return_value = [
            "//img1.wsimg.com/blobby/downloads/ie_data.xls?ver=1234567890"
        ]
        mock_scrapper.return_value.__aenter__.return_value = mock_scrapper_instance

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"fake_excel_binary_content"
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Realistic Excel-like data structure using helper
        def create_integration_dataframe():
            data = {
                0: [2020.01, 2020.02, 2020.03, 2020.04, 2020.05],  # Dates
                12: [31.2, 28.9, 25.1, 22.8, 27.5],  # CAPE values
                16: [1.8, 2.1, 2.7, 3.2, 2.4],  # Excess yield values
            }
            df = pd.DataFrame(index=range(5))
            for col in range(17):
                if col in data:
                    df[col] = data[col]
                else:
                    df[col] = None
            return df

        mock_read_excel.return_value = create_integration_dataframe()

        # Execute the provider
        result = await self.provider.get_data(None)

        # Comprehensive verification
        assert result.success is True
        assert isinstance(result.data, pd.DataFrame)

        # Verify data structure
        assert len(result.data) == 5
        assert list(result.data.columns) == ["CAPE", "Excess_CAPE_Yield"]
        assert result.data.index.name == "Date"

        # Verify first and last entries
        assert result.data.iloc[0]["CAPE"] == 31.2
        assert result.data.iloc[0]["Excess_CAPE_Yield"] == 1.8
        assert result.data.iloc[-1]["CAPE"] == 27.5
        assert result.data.iloc[-1]["Excess_CAPE_Yield"] == 2.4

        # Verify dates were parsed correctly
        expected_dates = [
            datetime(2020, 1, 1),
            datetime(2020, 2, 1),
            datetime(2020, 3, 1),
            datetime(2020, 4, 1),
            datetime(2020, 5, 1),
        ]
        assert list(result.data.index) == expected_dates

        # Verify all the mocks were called correctly
        mock_scrapper.assert_called_once()
        mock_scrapper_instance.extract_href.assert_called_once_with(
            'a[href*="ie_data.xls"]'
        )
        mock_client.get.assert_called_once_with(
            "https://img1.wsimg.com/blobby/downloads/ie_data.xls?ver=1234567890"
        )
        mock_read_excel.assert_called_once()

        # Check pandas.read_excel was called with correct parameters
        call_args = mock_read_excel.call_args
        assert call_args.kwargs["sheet_name"] == "Data"
        assert call_args.kwargs["skiprows"] == 8
        assert call_args.kwargs["engine"] == "xlrd"
