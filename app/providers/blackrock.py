"""
Blackrock provider module
This module provides functionality to fetch ETF holdings data from BlackRock website.

See https://www.blackrock.com/
"""

import asyncio
import os
from io import StringIO
from xml.etree import ElementTree as ET

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from pandas import DataFrame

from app.lib.logger import logger
from .base import (
    BaseProvider,
    ProviderType,
    ProviderConfig,
    NonRetriableProviderException,
    RetriableProviderException,
)
from .cache import cache


class BlackrockHoldingsProvider(BaseProvider[DataFrame]):
    """
    Provider for fetching ETF holdings data from BlackRock website.

    This provider searches for BlackRock ETF pages, extracts Excel download links,
    and parses the holdings data into a standardized DataFrame format.
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.BLACKROCK_HOLDINGS

    @cache
    # @cache loses pyrefly - no easy fix
    # pyrefly: ignore[bad-override]
    async def _fetch_data(self, query: str | None, *args, **kwargs) -> DataFrame:
        """
        Fetch ETF holdings data from BlackRock website.

        Args:
            query: ETF ticker symbol (must be non-null)
            **kwargs: Additional parameters (currently unused)

        Returns:
            DataFrame with ETF holdings data

        Raises:
            NonRetriableProviderException: If ticker is invalid or not found
            RetriableProviderException: For network/parsing errors
        """
        if not query:
            raise NonRetriableProviderException("ETF ticker must be provided")

        ticker = query.upper().strip()

        try:
            # Step 1: Search for BlackRock ETF page using Serper API
            search_query = f"site:blackrock.com {ticker} holdings fund"
            search_results = await self._search_with_serper(search_query)

            if not search_results:
                raise NonRetriableProviderException(
                    f"ETF {ticker} not found on BlackRock website"
                )

            # Step 2: Try each search result to find a valid ETF page
            etf_url = None
            for result in search_results:
                url = result.get("link", "")
                if ("/products/" in url or "/fund/" in url) and "blackrock.com" in url:
                    etf_url = url
                    break

            if not etf_url:
                raise NonRetriableProviderException(
                    f"No valid BlackRock ETF page found for {ticker}"
                )

            # Step 3: Fetch the ETF page and parse HTML
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                headers={"User-Agent": self.config.user_agent},
            ) as client:
                response = await client.get(etf_url)
                response.raise_for_status()
                html_content = response.text

            # Step 4: Parse HTML to find Excel download link
            soup = BeautifulSoup(html_content, "lxml")
            download_url = self._extract_download_url(soup, etf_url)

            if not download_url:
                raise RetriableProviderException(
                    f"Holdings download link not found for {ticker}"
                )

            # Step 5: Download and parse Excel file
            holdings_df = await self._download_and_parse_excel(download_url, ticker)

            return holdings_df

        except NonRetriableProviderException:
            raise
        except RetriableProviderException:
            raise
        except Exception as e:
            raise RetriableProviderException(
                f"Error fetching {ticker} holdings: {str(e)}"
            ) from e

    async def _search_with_serper(self, query: str) -> list[dict]:
        """
        Search using Serper API instead of DuckDuckGo to avoid rate limits.

        Args:
            query: Search query string

        Returns:
            List of search results with 'link' and 'title' keys

        Raises:
            RetriableProviderException: If search fails
        """
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            raise RetriableProviderException(
                "SERPER_API_KEY environment variable not set"
            )

        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": 5}

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                headers={"User-Agent": self.config.user_agent},
            ) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()

                search_data = response.json()
                organic_results = search_data.get("organic", [])

                return organic_results

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RetriableProviderException(
                    "Serper API rate limit exceeded"
                ) from e
            elif e.response.status_code == 401:
                raise RetriableProviderException("Invalid Serper API key") from e
            else:
                raise RetriableProviderException(
                    f"Serper API error: {e.response.status_code}"
                ) from e
        except httpx.RequestError as e:
            raise RetriableProviderException(
                f"Network error connecting to Serper API: {str(e)}"
            ) from e
        except Exception as e:
            raise RetriableProviderException(
                f"Unexpected error during search: {str(e)}"
            ) from e

    def _extract_download_url(self, soup: BeautifulSoup, base_url: str) -> str | None:
        """
        Extract the holdings download URL from BlackRock ETF page HTML.

        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for resolving relative links

        Returns:
            Complete download URL or None if not found
        """
        # Look for holdings download links with various possible formats
        selectors = [
            "a.icon-xls-export",
            "a.icon-csv-export",
            'a[href*=".ajax"][href*="fileType=xls"]',
            'a[href*=".ajax"][href*="fileType=csv"]',
            'a[href*="fund"][href*="xls"]',
            'a[href*="fund"][href*="csv"]',
            'a[href*="holdings"]',
            'a[data-link-event*="download"]',
        ]

        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get("href")
                if (
                    href
                    and isinstance(href, str)
                    and (
                        "fileType=xls" in href
                        or "fileType=csv" in href
                        or "dataType=fund" in href
                        or "holdings" in href.lower()
                    )
                ):
                    # Convert relative URL to absolute
                    if href.startswith("/"):
                        from urllib.parse import urljoin

                        return urljoin(base_url, href)
                    return href

        return None

    async def _download_and_parse_excel(
        self, download_url: str, ticker: str
    ) -> DataFrame:
        """
        Download and parse BlackRock holdings file (CSV or XML format).

        Args:
            download_url: URL to download holdings file
            ticker: ETF ticker for reference

        Returns:
            DataFrame with parsed holdings data
        """
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            headers={"User-Agent": self.config.user_agent},
        ) as client:
            response = await client.get(download_url)
            response.raise_for_status()
            file_data = response.content

        # Determine file format and parse accordingly
        try:
            # Try to decode as text first
            try:
                text_content = file_data.decode("utf-8-sig")  # Handle BOM automatically
            except UnicodeDecodeError:
                text_content = file_data.decode("utf-8")

            # Check if it's CSV format
            if "fileType=csv" in download_url or text_content.strip().startswith(
                ("Ticker,", "Fund Holdings as of")
            ):
                df = await self._parse_csv_data(text_content, ticker)
            else:
                # Try XML parsing
                # Remove any remaining BOM characters
                clean_data = file_data
                while clean_data.startswith(b"\xef\xbb\xbf"):
                    clean_data = clean_data[3:]

                # Parse XML
                root = ET.fromstring(clean_data)
                df = self._parse_spreadsheet_xml(root, ticker)

            if df.empty:
                raise ValueError("No holdings data found in file")

            return df

        except Exception as e:
            raise RetriableProviderException(
                f"Error parsing holdings file for {ticker}: {str(e)}"
            ) from e

    async def _parse_csv_data(self, text_content: str, ticker: str) -> DataFrame:
        """
        Parse CSV format holdings data from BlackRock.

        Args:
            text_content: CSV text content
            ticker: ETF ticker for reference

        Returns:
            DataFrame with holdings data
        """

        # Find the start of the actual CSV data
        lines = text_content.strip().split("\n")
        csv_start_idx = 0

        # Look for the header row with ticker/holdings columns
        for i, line in enumerate(lines):
            if any(
                keyword in line.lower()
                for keyword in ["ticker,", "symbol,", "name,", "holding,"]
            ):
                csv_start_idx = i
                break

        # Extract CSV data from the start index
        csv_data = "\n".join(lines[csv_start_idx:])

        # Parse CSV
        df = await asyncio.to_thread(lambda: pd.read_csv(StringIO(csv_data)))

        # Add ticker column
        df["ticker"] = ticker

        # Clean and standardize the data
        df = self._clean_holdings_data(df, ticker)

        return df

    def _parse_spreadsheet_xml(self, root: ET.Element, ticker: str) -> DataFrame:
        """
        Parse Microsoft Excel XML (SpreadsheetML) format and extract holdings data.

        Args:
            root: XML root element
            ticker: ETF ticker for reference

        Returns:
            DataFrame with holdings data
        """
        namespaces = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}

        # Find all worksheets
        worksheets = root.findall(".//ss:Worksheet", namespaces)

        if not worksheets:
            raise ValueError("No worksheets found in XML")

        # Look for holdings data in worksheets (usually in later sheets)
        holdings_data = []

        for worksheet in worksheets:
            rows = worksheet.findall(".//ss:Row", namespaces)

            # Look for a worksheet that contains holdings data
            header_row_idx = None
            header_columns = []

            for i, row in enumerate(rows):
                cells = row.findall(".//ss:Cell", namespaces)
                cell_values = []
                for cell in cells:
                    data = cell.find(".//ss:Data", namespaces)
                    if data is not None and data.text:
                        cell_values.append(data.text.strip())
                    else:
                        cell_values.append("")

                # Look for header row with holdings-related columns
                if cell_values:
                    row_text = " ".join(cell_values).lower()
                    if any(
                        keyword in row_text
                        for keyword in [
                            "name",
                            "weight",
                            "holding",
                            "symbol",
                            "ticker",
                            "isin",
                            "cusip",
                            "nom",
                            "poids",
                            "titre",
                            "valeur",
                            "pondération",  # French terms
                        ]
                    ):
                        header_row_idx = i
                        header_columns = cell_values
                        break

            # If we found a header row, extract the data rows
            if header_row_idx is not None:
                for i, row in enumerate(rows[header_row_idx + 1 :], header_row_idx + 1):
                    cells = row.findall(".//ss:Cell", namespaces)
                    cell_values = []
                    for cell in cells:
                        data = cell.find(".//ss:Data", namespaces)
                        if data is not None and data.text:
                            cell_values.append(data.text.strip())
                        else:
                            cell_values.append("")

                    # Skip empty rows or rows with only one value
                    if len([v for v in cell_values if v]) >= 2:
                        # Pad cell_values to match header length
                        while len(cell_values) < len(header_columns):
                            cell_values.append("")

                        # Create row dict
                        row_dict = dict(zip(header_columns, cell_values))
                        row_dict["ticker"] = ticker
                        holdings_data.append(row_dict)

                # If we found holdings data, break (don't process other worksheets)
                if holdings_data:
                    break

        if not holdings_data:
            raise ValueError("No holdings data found in any worksheet")

        # Convert to DataFrame
        df = pd.DataFrame(holdings_data)

        # Clean and standardize the data
        df = self._clean_holdings_data(df, ticker)

        return df

    def _clean_holdings_data(self, df: DataFrame, ticker: str) -> DataFrame:
        """
        Clean and standardize holdings data from XML-parsed DataFrame.

        Args:
            df: Raw DataFrame from XML parsing
            ticker: ETF ticker

        Returns:
            Cleaned DataFrame with standardized columns
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Standardize column names (support multiple languages and formats)
        column_mapping = {
            # English column names (CSV and XML)
            "Ticker": "symbol",
            "Name": "holding_name",
            "Holding": "holding_name",
            "Security Name": "holding_name",
            "Asset Name": "holding_name",
            "Weight (%)": "weight",
            "Weight": "weight",
            "% Net Assets": "weight",
            "Net Assets %": "weight",
            "Market Value": "market_value",
            "Notional Value": "market_value",
            "Shares": "shares",
            "Units": "shares",
            "Quantity": "shares",
            "Price": "price",
            "CUSIP": "cusip",
            "ISIN": "isin",
            "Sector": "sector",
            "Asset Class": "asset_class",
            "Location": "country",
            "Country": "country",
            "Exchange": "exchange",
            "Market Currency": "market_currency",
            # French column names (common in BlackRock French sites)
            "Nom": "holding_name",
            "Titre": "holding_name",
            "Valeur": "holding_name",
            "Poids (%)": "weight",
            "Poids": "weight",
            "Pondération": "weight",
            "% Actifs nets": "weight",
            "Valeur de marché": "market_value",
            "Parts": "shares",
            "Quantité": "shares",
            "Prix": "price",
            "Secteur": "sector",
            "Classe d'actifs": "asset_class",
            "Pays": "country",
            "Localisation": "country",
        }

        # Apply column mapping
        df.columns = [
            column_mapping.get(col, str(col).lower().replace(" ", "_"))
            for col in df.columns
        ]

        # Remove duplicate columns immediately after mapping
        df = df.loc[:, ~df.columns.duplicated()]

        # Ensure ticker column exists
        if "ticker" not in df.columns:
            df["ticker"] = ticker

        # Clean weight column (remove % and convert to float)
        weight_cols = ["weight", "poids", "pondération"]
        for col in weight_cols:
            if col in df.columns:
                # Convert to string and clean in a single operation
                cleaned_series = (
                    df[col]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .str.replace(" ", "", regex=False)
                )
                # Convert to numeric and assign once
                numeric_series = pd.to_numeric(cleaned_series, errors="coerce")

                if col != "weight":  # Rename to standard column name
                    df["weight"] = numeric_series
                    df = df.drop(columns=[col])
                else:
                    df[col] = numeric_series

        # Clean numeric columns
        numeric_cols = ["market_value", "shares", "price"]
        for col in numeric_cols:
            if col in df.columns:
                # Convert to string and clean in a single operation
                cleaned_series = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.replace("€", "", regex=False)
                    .str.replace(" ", "", regex=False)
                )
                # Convert to numeric and assign once
                df[col] = pd.to_numeric(cleaned_series, errors="coerce")

        # Filter out rows without holding names
        if "holding_name" in df.columns:
            df = df[
                df["holding_name"].notna()
                & (df["holding_name"] != "")
                & (df["holding_name"] != "0")
            ]
            # Remove rows that look like totals or summaries
            # (only if column contains strings)
            if df["holding_name"].dtype == "object":
                df = df[
                    ~df["holding_name"]
                    .astype(str)
                    .str.lower()
                    .str.contains("total|sum|autre|other", na=False)
                ]

        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        return df


# Factory function for easy provider creation
def create_blackrock_holdings_provider(
    timeout: float = 30.0,
    retries: int = 3,
    rate_limit: float = 1.0,
) -> BlackrockHoldingsProvider:
    """
    Factory function to create a BlackRock Holdings provider with custom settings.

    Args:
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        rate_limit: Rate limit in requests per second

    Returns:
        Configured BlackrockHoldingsProvider instance
    """
    logger.debug(
        f"Creating BlackrockHoldingsProvider: timeout={timeout}s, "
        f"retries={retries}, rate_limit={rate_limit}"
    )
    config = ProviderConfig(
        timeout=timeout,
        retries=retries,
        rate_limit=rate_limit,
        user_agent="FinApp/1.0 (BlackRock Holdings Provider)",
    )
    return BlackrockHoldingsProvider(config)
