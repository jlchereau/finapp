"""
Shiller provider module
This module provides functionality to fetch data from ShillerData.

See:
    - http://www.econ.yale.edu/~shiller/data.htm
    - https://shillerdata.com/
"""

import math
from datetime import datetime
import httpx
import pandas as pd
import xlrd
from pandas import DataFrame

from app.lib.logger import logger
from app.lib.scrapper import WebPageScrapper
from .base import (
    BaseProvider,
    ProviderType,
    ProviderConfig,
    NonRetriableProviderException,
    RetriableProviderException,
)
from .cache import apply_provider_cache


class ShillerCAPEProvider(BaseProvider[DataFrame]):
    """
    Provider for fetching CAPE data from ShillerData.

    This provider scrapes https://shillerdata.com/ to find the Excel download
    link, downloads the Excel file, and processes the CAPE data.
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.SHILLER_CAPE

    @apply_provider_cache
    # @apply_provider_cache triggers pyrefly bad-override - no easy fix
    # pyrefly: ignore[bad-override]
    async def _fetch_data(self, query: str | None, *args, **kwargs) -> DataFrame:
        """
        Fetch CAPE data from ShillerData.

        Args:
            query: Not used for Shiller data (can be None)
            **kwargs: Additional parameters (currently unused)

        Returns:
            DataFrame containing CAPE data with date index

        Note:
            The query parameter is not used for Shiller data since it fetches
            a standard dataset, but is kept for provider interface consistency.
        """
        base_url = "https://shillerdata.com/"

        # Step 1: Fetch the main page and find the Excel download link
        logger.info("Fetching Shiller data page to find Excel download link")
        async with WebPageScrapper(base_url, timeout=self.config.timeout) as scrapper:
            # Step 2: Find the temporary link containing 'ie_data.xls'
            excel_urls = await scrapper.extract_href('a[href*="ie_data.xls"]')

        if not excel_urls:
            raise NonRetriableProviderException(
                "Could not find Excel download link on Shiller website"
            )

        excel_url = excel_urls[0]  # Take the first matching URL

        # Handle relative URLs by making them absolute
        if excel_url.startswith("//"):
            excel_url = "https:" + excel_url
        elif excel_url.startswith("/"):
            excel_url = base_url.rstrip("/") + excel_url

        logger.info(f"Found Excel download URL: {excel_url}")

        # Step 3: Download and process the Excel file
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                logger.info("Downloading Excel file from Shiller website")
                response = await client.get(excel_url)
                response.raise_for_status()

            # Read Excel file directly from memory using xlrd
            logger.info("Processing Excel file data")
            workbook = xlrd.open_workbook(file_contents=response.content)
            df = pd.read_excel(workbook, sheet_name="Data", skiprows=8, engine="xlrd")
        except httpx.RequestError as e:
            raise RetriableProviderException(
                f"Failed to download Excel file: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise RetriableProviderException(
                f"HTTP error downloading Excel file: {e}"
            ) from e
        except Exception as e:
            raise RetriableProviderException(
                f"Unexpected error downloading Excel file: {e}"
            ) from e

        # Step 4: Process the data
        # Column A has dates in format like 2024.01 (year.month)
        # Column M has CAPE values
        # Column Q has excess CAPE yield

        # Get the date column (first column, index 0)
        date_values = df.iloc[:, 0].dropna()  # Column A (0-indexed)

        # Parse dates using math.modf as suggested in comments
        dates = []
        for date_val in date_values:
            if pd.isna(date_val) or not isinstance(date_val, (int, float)):
                continue

            # Use math.modf to split float into year and month parts
            month_part, year_part = math.modf(float(date_val))
            year = int(year_part)
            month = int(round(month_part * 100)) if month_part > 0 else 1

            # Handle edge cases
            if month < 1 or month > 12:
                month = 1
            if year < 1800 or year > 3000:  # Sanity check
                continue

            dates.append(datetime(year, month, 1))

        # Extract CAPE and excess yield columns
        cape_values = df.iloc[: len(dates), 12]  # Column M (0-indexed = 12)
        excess_yield_values = df.iloc[: len(dates), 16]  # Column Q (0-indexed = 16)

        # Create result DataFrame
        result_df = pd.DataFrame(
            {
                "CAPE": cape_values.values,
                "Excess_CAPE_Yield": excess_yield_values.values,
            },
            index=pd.DatetimeIndex(dates, name="Date"),
        )

        # Clean up - remove rows with all NaN values
        result_df = result_df.dropna(how="all")

        logger.info(f"Successfully processed Shiller data: {len(result_df)} records")
        return result_df


# Factory function for easy provider creation
def create_shiller_cape_provider(
    timeout: float = 30.0,
    retries: int = 3,
) -> ShillerCAPEProvider:
    """
    Factory function to create a Shiller provider with custom settings.

    Args:
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        Configured ShillerProvider instance
    """
    config = ProviderConfig(timeout=timeout, retries=retries)
    return ShillerCAPEProvider(config)
