"""
Zacks provider module
This module provides functionality to fetch data from Zacks API.
"""

import httpx
import asyncio
from httpx import HTTPError
from pydantic import BaseModel
from .base import BaseProvider, ProviderType, ProviderConfig
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

    async def _fetch_data(self, ticker: str, **kwargs) -> BaseModel:
        """
        Fetch data from Zacks API.

        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters (currently unused)

        Returns:
            Pydantic model instance with Zacks data

        Raises:
            HTTPError: If the HTTP request fails
            ValueError: If no data is returned or response is invalid
            Exception: For other network-related errors
        """
        url = f"https://quote-feed.zacks.com/index?t={ticker}"

        # Use httpx for async HTTP requests
        timeout = httpx.Timeout(self.config.timeout)

        async with httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": self.config.user_agent}
        ) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()

                # Parse JSON in a separate thread to avoid blocking
                json_data = await asyncio.to_thread(response.json)

                if not json_data or not isinstance(json_data, dict):
                    raise ValueError(
                        f"Invalid response from Zacks API for ticker: {ticker}"
                    )

                # Parse the JSON data using our parser
                parser = PydanticJSONParser(ZACKS_CONFIG)
                result = await parser.parse_async(json_data)

                return result

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise ValueError(f"Ticker not found in Zacks: {ticker}") from e
                elif e.response.status_code == 429:
                    raise HTTPError("Rate limit exceeded for Zacks API") from e
                else:
                    raise HTTPError(
                        f"HTTP {e.response.status_code} error from Zacks API"
                    ) from e
            except httpx.RequestError as e:
                raise ConnectionError(
                    f"Network error connecting to Zacks API: {e}"
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
