"""
Zacks provider module
This module provides functionality to fetch data from Zacks API.
"""

import httpx
from pydantic import BaseModel
from .base import (
    BaseProvider,
    ProviderType,
    ProviderConfig,
    NonRetriableProviderException,
    RetriableProviderException,
)
from .cache import cache
from .parsers import PydanticJSONParser, ParserConfig


# Configuration for Zacks API data parsing
ZACKS_CONFIG = ParserConfig(
    name="ZacksModel",
    fields={
        "ticker": {"expr": "ticker", "default": None},
        "price": {"expr": "price", "default": None},
        "change": {"expr": "change", "default": None},
        "percent_change": {"expr": "percentChange", "default": None},
        "volume": {"expr": "volume", "default": None},
        "high": {"expr": "high", "default": None},
        "low": {"expr": "low", "default": None},
        "open_price": {"expr": "open", "default": None},
        "previous_close": {"expr": "previousClose", "default": None},
        "market_cap": {"expr": "marketCap", "default": None},
        "pe_ratio": {"expr": "peRatio", "default": None},
        "zacks_rank": {"expr": "zacksRank", "default": None},
        "value_score": {"expr": "valueScore", "default": None},
        "growth_score": {"expr": "growthScore", "default": None},
        "momentum_score": {"expr": "momentumScore", "default": None},
        "vgm_score": {"expr": "vgmScore", "default": None},
    },
    strict_mode=False,
    default_value=None,
)


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
        parser = PydanticJSONParser(ZACKS_CONFIG)
        result = await parser.parse_async(json_data)
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
