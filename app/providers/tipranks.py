"""
Tipranks provider module
This module provides functionality to fetch data from Tipranks.
"""

import asyncio
import time

import httpx
from pydantic import BaseModel, Field, field_validator

from app.lib.logger import logger
from .base import (
    BaseProvider,
    ProviderType,
    ProviderConfig,
    NonRetriableProviderException,
    RetriableProviderException,
)
from .cache import apply_provider_cache
from .headers import get_random_user_agent


class TipranksDataModel(BaseModel):
    """Pydantic model for Tipranks data."""

    # Important: Do not set default values for required fields.
    # This would prevent us from detecting API changes.
    # If the API changes and fields are missing,
    # Pydantic should raise a validation error.

    ticker: str
    company_name: str = Field(alias="companyName")
    consensus_rating: int
    analyst_count: int = Field(alias="numOfAnalysts")
    buy_count: int
    hold_count: int
    sell_count: int
    price_target: float
    price_target_high: float
    price_target_low: float
    smart_score: int
    market_cap: int = Field(alias="marketCap")

    @field_validator(
        "consensus_rating",
        "analyst_count",
        "buy_count",
        "hold_count",
        "sell_count",
        "smart_score",
        "market_cap",
        mode="before",
    )
    @classmethod
    def convert_null_ints(cls, v):
        if v is None:
            raise ValueError("Required int field cannot be None")
        return int(v)

    @field_validator(
        "price_target", "price_target_high", "price_target_low", mode="before"
    )
    @classmethod
    def convert_null_floats(cls, v):
        if v is None:
            raise ValueError("Required float field cannot be None")
        return float(v)

    model_config = {"populate_by_name": True, "extra": "ignore"}


class TipranksNewsSentimentModel(BaseModel):
    """Pydantic model for Tipranks news sentiment data."""

    # Important: Do not set default values for required fields.
    # This would prevent us from detecting API changes.
    # If the API changes and fields are missing,
    # Pydantic should raise a validation error.

    ticker: str
    company_name: str = Field(alias="companyName")
    sentiment_score: float = Field(alias="score")
    bullish_percent: float
    bearish_percent: float
    articles_last_week: int
    weekly_average: float
    buzz_score: float
    sector_avg_bullish_percent: float = Field(alias="sectorAverageBullishPercent")
    word_cloud_count: int

    @field_validator("articles_last_week", "word_cloud_count", mode="before")
    @classmethod
    def convert_null_ints(cls, v):
        if v is None:
            raise ValueError("Required int field cannot be None")
        return int(v)

    @field_validator(
        "sentiment_score",
        "bullish_percent",
        "bearish_percent",
        "weekly_average",
        "buzz_score",
        "sector_avg_bullish_percent",
        mode="before",
    )
    @classmethod
    def convert_null_floats(cls, v):
        if v is None:
            raise ValueError("Required float field cannot be None")
        return float(v)

    model_config = {"populate_by_name": True, "extra": "ignore"}


class TipranksDataProvider(BaseProvider[BaseModel]):
    """
    Provider for fetching analyst data from Tipranks.

    This provider fetches analyst ratings, price targets,
    and consensus information for stocks.
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.TIPRANKS_DATA

    @apply_provider_cache
    # @apply_provider_cache triggers pyrefly bad-override - no easy fix
    # pyrefly: ignore[bad-override]
    async def _fetch_data(self, query: str | None, *args, **kwargs) -> BaseModel:
        """
        Fetch analyst data for a ticker from Tipranks.

        Args:
            query: Stock ticker symbol to fetch (must be non-null)
            **kwargs: Additional parameters (currently unused)

        Returns:
            Pydantic model instance with Tipranks data

        Raises:
            ValueError: If no data is returned or ticker is invalid
            Exception: For other Tipranks-related errors
        """
        ticker = None  # Initialize ticker to avoid UnboundLocalError
        try:
            # Validate query
            if query is None or query.strip() == "":
                logger.error("Query cannot be None or empty for TipranksDataProvider")
                raise ValueError("Query must be provided for TipranksDataProvider")
            ticker = query.upper().strip()

            # Tipranks API configuration
            base_url = "https://www.tipranks.com/api/stocks/"
            user_agent = get_random_user_agent()
            headers = {"User-Agent": user_agent}

            # Run HTTP request in a separate thread to avoid blocking
            def fetch_tipranks_data():
                timestamp = int(time.time())  # Unix epoch in seconds
                url = (
                    f"{base_url}getData/?name={ticker}&benchmark=1&period=3"
                    f"&break={timestamp}"
                )
                logger.info(f"Calling TipRanks API for {ticker}: {url}")

                with httpx.Client(timeout=self.config.timeout) as client:
                    response = client.get(url, headers=headers)
                    response.raise_for_status()
                    return response.json()

            json_data = await asyncio.to_thread(fetch_tipranks_data)

            if not json_data or not isinstance(json_data, dict):
                logger.warning(f"No valid TipRanks data returned for {ticker}")
                raise ValueError(f"No Tipranks data found for query: {query}")

            # Transform the API response structure to match Pydantic model field names
            # Extract consensus data from latest consensus (isLatest=1)
            consensuses = json_data.get("consensuses", [])
            latest_consensus = next(
                (c for c in consensuses if c.get("isLatest") == 1), None
            )
            if latest_consensus is None:
                raise ValueError("No latest consensus data found in API response")

            # Extract price target data
            pt_consensus = json_data.get("ptConsensus", [])
            if not pt_consensus or len(pt_consensus) == 0:
                raise ValueError("No price target consensus data found in API response")
            pt_data = pt_consensus[0]

            # Extract smart score
            stock_score = json_data.get("tipranksStockScore", {})
            if not stock_score:
                raise ValueError("No stock score data found in API response")

            # Build data structure matching Pydantic model aliases and field names
            transformed_data = {
                "ticker": ticker,
                "companyName": json_data.get("companyName"),
                "consensus_rating": latest_consensus.get("rating"),
                "numOfAnalysts": json_data.get("numOfAnalysts"),
                "buy_count": latest_consensus.get("nB"),
                "hold_count": latest_consensus.get("nH"),
                "sell_count": latest_consensus.get("nS"),
                "price_target": pt_data.get("priceTarget"),
                "price_target_high": pt_data.get("high"),
                "price_target_low": pt_data.get("low"),
                "smart_score": stock_score.get("score"),
                "marketCap": json_data.get("marketCap"),
            }

            # Parse the JSON data using the Pydantic model (strict validation)
            result = TipranksDataModel(**transformed_data)
            return result

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            # HTTP errors are generally retriable
            query_info = ticker or query or "unknown"
            logger.warning(f"HTTP error fetching TipRanks data for {query_info}: {e}")
            raise RetriableProviderException(
                f"HTTP error fetching Tipranks data: {e}"
            ) from e
        except ValueError as e:
            # Non-retriable errors (e.g., empty data, invalid ticker)
            query_info = ticker or query or "unknown"
            logger.error(
                f"Non-retriable error in TipranksDataProvider for {query_info}: {e}"
            )
            raise NonRetriableProviderException(str(e)) from e
        except Exception as e:
            # Other errors retriable
            query_info = ticker or query or "unknown"
            logger.warning(
                f"Retriable error in TipranksDataProvider for {query_info}: {e}"
            )
            raise RetriableProviderException(str(e)) from e


class TipranksNewsSentimentProvider(BaseProvider[BaseModel]):
    """
    Provider for fetching news sentiment data from Tipranks.

    This provider fetches news sentiment, buzz metrics,
    and word cloud data for stocks.
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.TIPRANKS_NEWS_SENTIMENT

    @apply_provider_cache
    # @apply_provider_cache triggers pyrefly bad-override - no easy fix
    # pyrefly: ignore[bad-override]
    async def _fetch_data(self, query: str | None, *args, **kwargs) -> BaseModel:
        """
        Fetch news sentiment data for a ticker from Tipranks.

        Args:
            query: Stock ticker symbol to fetch (must be non-null)
            **kwargs: Additional parameters (currently unused)

        Returns:
            Pydantic model instance with Tipranks news sentiment data

        Raises:
            ValueError: If no data is returned or ticker is invalid
            Exception: For other Tipranks-related errors
        """
        ticker = None  # Initialize ticker to avoid UnboundLocalError
        try:
            # Validate query
            if query is None or query.strip() == "":
                logger.error(
                    "Query cannot be None or empty for TipranksNewsSentimentProvider"
                )
                raise ValueError(
                    "Query must be provided for TipranksNewsSentimentProvider"
                )
            ticker = query.upper().strip()

            # Tipranks API configuration
            base_url = "https://www.tipranks.com/api/stocks/"
            user_agent = get_random_user_agent()
            headers = {"User-Agent": user_agent}

            # Run HTTP request in a separate thread to avoid blocking
            def fetch_news_sentiment_data():
                timestamp = int(time.time())  # Unix epoch in seconds
                url = f"{base_url}getNewsSentiments/?ticker={ticker}&break={timestamp}"
                logger.info(f"Calling TipRanks News Sentiment API for {ticker}: {url}")

                with httpx.Client(timeout=self.config.timeout) as client:
                    response = client.get(url, headers=headers)
                    response.raise_for_status()
                    return response.json()

            json_data = await asyncio.to_thread(fetch_news_sentiment_data)

            if not json_data or not isinstance(json_data, dict):
                logger.warning(
                    f"No valid TipRanks news sentiment data returned for {ticker}"
                )
                raise ValueError(
                    f"No Tipranks news sentiment data found for query: {query}"
                )

            # Validate required nested structures exist
            sentiment = json_data.get("sentiment", {})
            if not sentiment:
                raise ValueError("No sentiment data found in API response")

            buzz = json_data.get("buzz", {})
            if not buzz:
                raise ValueError("No buzz data found in API response")

            # Build data structure matching Pydantic model aliases and field names
            transformed_data = {
                "ticker": ticker,
                "companyName": json_data.get("companyName"),
                "score": json_data.get("score"),
                "bullish_percent": sentiment.get("bullishPercent"),
                "bearish_percent": sentiment.get("bearishPercent"),
                "articles_last_week": buzz.get("articlesInLastWeek"),
                "weekly_average": buzz.get("weeklyAverage"),
                "buzz_score": buzz.get("buzz"),
                "sectorAverageBullishPercent": json_data.get(
                    "sectorAverageBullishPercent"
                ),
                "word_cloud_count": len(json_data.get("wordCloud", [])),
            }

            # Parse the JSON data using the Pydantic model (strict validation)
            result = TipranksNewsSentimentModel(**transformed_data)
            return result

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            # HTTP errors are generally retriable
            query_info = ticker or query or "unknown"
            logger.warning(
                f"HTTP error fetching TipRanks news sentiment data for "
                f"{query_info}: {e}"
            )
            raise RetriableProviderException(
                f"HTTP error fetching Tipranks news sentiment data: {e}"
            ) from e
        except ValueError as e:
            # Non-retriable errors (e.g., empty data, invalid ticker)
            query_info = ticker or query or "unknown"
            logger.error(
                f"Non-retriable error in TipranksNewsSentimentProvider for "
                f"{query_info}: {e}"
            )
            raise NonRetriableProviderException(str(e)) from e
        except Exception as e:
            # Other errors retriable
            query_info = ticker or query or "unknown"
            logger.warning(
                f"Retriable error in TipranksNewsSentimentProvider for "
                f"{query_info}: {e}"
            )
            raise RetriableProviderException(str(e)) from e


# Factory function for easy provider creation
def create_tipranks_data_provider(
    timeout: float = 30.0,
    retries: int = 3,
) -> TipranksDataProvider:
    """
    Factory function to create a Tipranks provider with custom settings.

    Args:
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        Configured TipranksDataProvider instance
    """
    config = ProviderConfig(timeout=timeout, retries=retries)
    return TipranksDataProvider(config)


def create_tipranks_news_sentiment_provider(
    timeout: float = 30.0,
    retries: int = 3,
) -> TipranksNewsSentimentProvider:
    """
    Factory function to create a Tipranks news sentiment provider with custom settings.

    Args:
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        Configured TipranksNewsSentimentProvider instance
    """
    config = ProviderConfig(timeout=timeout, retries=retries)
    return TipranksNewsSentimentProvider(config)
