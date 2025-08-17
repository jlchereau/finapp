"""
Zacks provider module
This module provides functionality to fetch data from Zacks API.
"""

import httpx

from pydantic import BaseModel, Field, ValidationError
from .base import (
    BaseProvider,
    ProviderType,
    ProviderConfig,
    NonRetriableProviderException,
    RetriableProviderException,
)
from .cache import cache
from ..lib.logger import logger


class ZacksModel(BaseModel):
    """Pydantic model for Zacks API data."""

    ticker: str = Field(alias="ticker")
    price: float = Field(alias="last")
    change: float = Field(alias="net_change")
    percent_change: float = Field(alias="percent_net_change")
    volume: int = Field(alias="volume")
    previous_close: float = Field(alias="previous_close")
    pe_ratio: float = Field(alias="pe_f1")
    zacks_rank: int = Field(alias="zacks_rank")
    zacks_rank_text: str = Field(alias="zacks_rank_text")
    dividend_yield: float = Field(alias="dividend_yield")
    market_status: str = Field(alias="market_status")
    company_name: str = Field(alias="name")

    # Score fields
    value_score: str = Field(alias="valueScore")
    growth_score: str = Field(alias="growthScore")
    momentum_score: str = Field(alias="momentumScore")
    vgm_score: str = Field(alias="vgmScore")

    # Price fields
    high: float = Field(alias="high")
    low: float = Field(alias="low")
    open_price: float = Field(alias="open")
    market_cap: float = Field(alias="marketCap")

    # Optional fields that might not always be present
    market_time: str = Field(default="", alias="market_time")
    updated: str = Field(default="", alias="updated")

    model_config = {"populate_by_name": True, "extra": "ignore"}


class ZacksProvider(BaseProvider[BaseModel]):
    """
    Provider for fetching financial data from Zacks API.

    This provider fetches real-time quotes, financial metrics,
    and Zacks-specific scores and rankings.
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.ZACKS

    @cache
    async def _fetch_data(self, query: str | None, **kwargs) -> BaseModel:
        """
        Fetch data from Zacks API.

        Args:
            query: Stock ticker symbol to fetch (must be non-null)
            **kwargs: Additional parameters (currently unused)

        Returns:
            Pydantic model instance with Zacks data

        Raises:
            HTTPError: If the HTTP request fails
            ValueError: If no data is returned or response is invalid
            Exception: For other network-related errors
        """
        # Prepare and validate query
        if query is None:
            logger.error("Query cannot be None for ZacksProvider")
            raise NonRetriableProviderException(
                "Query must be provided for ZacksProvider"
            )
        ticker = query.upper().strip()
        # HTTP request with exception mapping
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.timeout,
                read=self.config.timeout,
                write=self.config.timeout,
                pool=self.config.timeout,
            ),
            headers={"User-Agent": self.config.user_agent},
        ) as client:
            url = f"https://quote-feed.zacks.com/index?t={ticker}"
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 404:
                    raise NonRetriableProviderException(
                        "Ticker not found in Zacks"
                    ) from e
                # Retryable for rate limits and server errors
                raise RetriableProviderException(
                    "Rate limit exceeded" if status == 429 else f"HTTP {status} error"
                ) from e
            except httpx.RequestError as e:
                # Network issues are retryable
                raise RetriableProviderException(
                    "Network error connecting to Zacks API"
                ) from e

        # Validate and parse JSON data
        json_data = response.json()
        if not json_data:
            logger.warning(f"Empty response from Zacks API for ticker: {ticker}")
            raise ValueError("Invalid response from Zacks API")

        # Extract ticker data from nested structure
        # API returns {"TICKER": {ticker_data}} format
        try:
            if isinstance(json_data, dict) and ticker in json_data:
                ticker_data = json_data[ticker]
            else:
                # If direct format, use as-is
                ticker_data = json_data

            # Process data for type conversion
            # Convert string numbers to appropriate types
            if "net_change" in ticker_data and ticker_data["net_change"] != "NULL":
                ticker_data["net_change"] = float(ticker_data["net_change"])
            else:
                ticker_data["net_change"] = 0.0

            if (
                "percent_net_change" in ticker_data
                and ticker_data["percent_net_change"] != "NULL"
            ):
                ticker_data["percent_net_change"] = float(
                    ticker_data["percent_net_change"]
                )
            else:
                ticker_data["percent_net_change"] = 0.0

            if "last" in ticker_data and ticker_data["last"] != "NULL":
                ticker_data["last"] = float(ticker_data["last"])
            else:
                # Fallback to previous_close if last is not available
                if "previous_close" in ticker_data:
                    ticker_data["last"] = float(ticker_data["previous_close"])
                else:
                    ticker_data["last"] = 0.0

            if "volume" in ticker_data and ticker_data["volume"] != "NULL":
                ticker_data["volume"] = int(ticker_data["volume"])
            else:
                ticker_data["volume"] = 0

            if (
                "previous_close" in ticker_data
                and ticker_data["previous_close"] != "NULL"
            ):
                ticker_data["previous_close"] = float(ticker_data["previous_close"])
            else:
                ticker_data["previous_close"] = 0.0

            if "pe_f1" in ticker_data and ticker_data["pe_f1"] != "NULL":
                ticker_data["pe_f1"] = float(ticker_data["pe_f1"])
            else:
                ticker_data["pe_f1"] = 0.0

            if "zacks_rank" in ticker_data and ticker_data["zacks_rank"] != "NULL":
                ticker_data["zacks_rank"] = int(ticker_data["zacks_rank"])
            else:
                ticker_data["zacks_rank"] = 3  # Default neutral rank

            if (
                "dividend_yield" in ticker_data
                and ticker_data["dividend_yield"] != "NULL"
            ):
                ticker_data["dividend_yield"] = float(ticker_data["dividend_yield"])
            else:
                ticker_data["dividend_yield"] = 0.0

        except Exception as e:
            logger.error(f"Error extracting ticker data for {ticker}: {e}")
            raise NonRetriableProviderException(
                f"Failed to extract ticker data: {e}"
            ) from e

        # Parse the JSON data using the Pydantic model (strict validation)
        try:
            result = ZacksModel(**ticker_data)
            return result
        except ValidationError as e:
            logger.error(
                f"Zacks model validation failed for {ticker}: "
                "Pydantic model doesn't match API response structure"
            )
            raise NonRetriableProviderException(
                f"Zacks model validation failed: API response structure doesn't "
                f"match expected model for {ticker}"
            ) from e


# Factory function for easy provider creation
def create_zacks_provider(
    timeout: float = 30.0,
    retries: int = 3,
    rate_limit: float = 1.0,  # 1 request per second
) -> ZacksProvider:
    """
    Factory function to create a Zacks provider with custom settings.

    Args:
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        rate_limit: Rate limit in requests per second

    Returns:
        Configured ZacksProvider instance
    """
    config = ProviderConfig(
        timeout=timeout,
        retries=retries,
        rate_limit=rate_limit,
        user_agent="FinApp/1.0 (Zacks Data Provider)",
    )
    return ZacksProvider(config)
