"""
Zacks provider module
This module provides functionality to fetch data from Zacks API.
"""

import httpx

from pydantic import BaseModel, Field
from .base import (
    BaseProvider,
    ProviderType,
    ProviderConfig,
    NonRetriableProviderException,
    RetriableProviderException,
)
from .cache import cache


class ZacksModel(BaseModel):
    """Pydantic model for Zacks API data."""

    ticker: str = Field(alias="ticker")
    price: float = Field(alias="price")
    change: float = Field(alias="change")
    percent_change: float = Field(alias="percentChange")
    volume: int = Field(alias="volume")
    high: float = Field(alias="high")
    low: float = Field(alias="low")
    open_price: float = Field(alias="open")
    previous_close: float = Field(alias="previousClose")
    market_cap: int = Field(alias="marketCap")
    pe_ratio: float = Field(alias="peRatio")
    zacks_rank: int = Field(alias="zacksRank")
    value_score: str = Field(alias="valueScore")
    growth_score: str = Field(alias="growthScore")
    momentum_score: str = Field(alias="momentumScore")
    vgm_score: str = Field(alias="vgmScore")

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
            raise ValueError("Invalid response from Zacks API")

        # Parse the JSON data using the Pydantic model (strict validation)
        result = ZacksModel(**json_data)
        return result


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
