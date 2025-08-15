"""
Tipranks provider module
This module provides functionality to fetch data from Tipranks.
"""

import asyncio
import time

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
from .headers import get_random_user_agent
from ..lib.logger import logger


class TipranksDataModel(BaseModel):
    """Pydantic model for Tipranks data."""

    ticker: str
    company_name: str = Field(alias="companyName", default="")
    consensus_rating: int = Field(default=0)
    analyst_count: int = Field(alias="numOfAnalysts", default=0)
    buy_count: int = Field(default=0)
    hold_count: int = Field(default=0)
    sell_count: int = Field(default=0)
    price_target: float = Field(default=0.0)
    price_target_high: float = Field(default=0.0)
    price_target_low: float = Field(default=0.0)
    smart_score: int = Field(default=0)
    market_cap: int = Field(alias="marketCap", default=0)

    model_config = {"populate_by_name": True, "extra": "ignore"}


class TipranksNewsSentimentModel(BaseModel):
    """Pydantic model for Tipranks news sentiment data."""

    ticker: str
    company_name: str = Field(alias="companyName", default="")
    sentiment_score: float = Field(alias="score", default=0.0)
    bullish_percent: float = Field(default=0.0)
    bearish_percent: float = Field(default=0.0)
    articles_last_week: int = Field(default=0)
    weekly_average: float = Field(default=0.0)
    buzz_score: float = Field(default=0.0)
    sector_avg_bullish_percent: float = Field(
        alias="sectorAverageBullishPercent", default=0.0
    )
    word_cloud_count: int = Field(default=0)

    model_config = {"populate_by_name": True, "extra": "ignore"}


class TipranksDataProvider(BaseProvider[BaseModel]):
    """
    Provider for fetching analyst data from Tipranks.

    This provider fetches analyst ratings, price targets,
    and consensus information for stocks.
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.CUSTOM

    @cache
    async def _fetch_data(self, query: str | None, **kwargs) -> BaseModel:
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
        logger.debug(f"TipranksDataProvider._fetch_data called for query: {query}")
        ticker = None  # Initialize ticker to avoid UnboundLocalError
        try:
            # Validate query
            if query is None or query.strip() == "":
                logger.error("Query cannot be None or empty for TipranksDataProvider")
                raise ValueError("Query must be provided for TipranksDataProvider")
            ticker = query.upper().strip()
            logger.debug(f"Normalized ticker: {ticker}")

            # Tipranks API configuration
            base_url = "https://www.tipranks.com/api/stocks/"
            user_agent = get_random_user_agent()
            headers = {"User-Agent": user_agent}
            logger.debug(f"Using User-Agent: {user_agent}")

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
            logger.debug(f"Retrieved TipRanks data for {ticker}: {type(json_data)}")

            if not json_data or not isinstance(json_data, dict):
                logger.warning(f"No valid TipRanks data returned for {ticker}")
                raise ValueError(f"No Tipranks data found for query: {query}")

            # Extract and transform the data structure
            logger.debug(f"Extracting data fields for {ticker}")
            data_dict = {
                "ticker": ticker,
                "companyName": json_data.get("companyName", ""),
                "numOfAnalysts": json_data.get("numOfAnalysts", 0),
                "marketCap": json_data.get("marketCap", 0),
            }

            # Extract consensus data from latest consensus (isLatest=1)
            consensuses = json_data.get("consensuses", [])
            logger.debug(f"Found {len(consensuses)} consensus records for {ticker}")
            latest_consensus = next(
                (c for c in consensuses if c.get("isLatest") == 1), {}
            )
            if latest_consensus:
                logger.debug(f"Found latest consensus data for {ticker}")
                data_dict.update(
                    {
                        "consensus_rating": latest_consensus.get("rating", 0),
                        "buy_count": latest_consensus.get("nB", 0),
                        "hold_count": latest_consensus.get("nH", 0),
                        "sell_count": latest_consensus.get("nS", 0),
                    }
                )
            else:
                logger.debug(f"No latest consensus data found for {ticker}")

            # Extract price target data
            pt_consensus = json_data.get("ptConsensus", [])
            if pt_consensus and len(pt_consensus) > 0:
                pt_data = pt_consensus[0]
                data_dict.update(
                    {
                        "price_target": pt_data.get("priceTarget", 0.0),
                        "price_target_high": pt_data.get("high", 0.0),
                        "price_target_low": pt_data.get("low", 0.0),
                    }
                )

            # Extract smart score
            stock_score = json_data.get("tipranksStockScore", {})
            if stock_score:
                data_dict["smart_score"] = stock_score.get("score", 0)

            # Parse the JSON data using the Pydantic model
            result = TipranksDataModel(**data_dict)
            logger.debug(f"Successfully parsed TipRanks data for {ticker}")
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
            logger.warning(f"Retriable error in TipranksDataProvider for {query_info}: {e}")
            raise RetriableProviderException(str(e)) from e


class TipranksNewsSentimentProvider(BaseProvider[BaseModel]):
    """
    Provider for fetching news sentiment data from Tipranks.

    This provider fetches news sentiment, buzz metrics,
    and word cloud data for stocks.
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.CUSTOM

    @cache
    async def _fetch_data(self, query: str | None, **kwargs) -> BaseModel:
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
        logger.debug(
            f"TipranksNewsSentimentProvider._fetch_data called for query: {query}"
        )
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
            logger.debug(f"Normalized ticker: {ticker}")

            # Tipranks API configuration
            base_url = "https://www.tipranks.com/api/stocks/"
            user_agent = get_random_user_agent()
            headers = {"User-Agent": user_agent}
            logger.debug(f"Using User-Agent: {user_agent}")

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
            logger.debug(
                f"Retrieved TipRanks news sentiment data for {ticker}: "
                f"{type(json_data)}"
            )

            if not json_data or not isinstance(json_data, dict):
                logger.warning(
                    f"No valid TipRanks news sentiment data returned for {ticker}"
                )
                raise ValueError(
                    f"No Tipranks news sentiment data found for query: {query}"
                )

            # Extract and transform the data structure
            data_dict = {
                "ticker": ticker,
                "companyName": json_data.get("companyName", ""),
                "score": json_data.get("score", 0.0),
                "sectorAverageBullishPercent": json_data.get(
                    "sectorAverageBullishPercent", 0.0
                ),
            }

            # Extract sentiment data
            sentiment = json_data.get("sentiment", {})
            data_dict.update(
                {
                    "bullish_percent": sentiment.get("bullishPercent", 0.0),
                    "bearish_percent": sentiment.get("bearishPercent", 0.0),
                }
            )

            # Extract buzz data
            buzz = json_data.get("buzz", {})
            data_dict.update(
                {
                    "articles_last_week": buzz.get("articlesInLastWeek", 0),
                    "weekly_average": buzz.get("weeklyAverage", 0.0),
                    "buzz_score": buzz.get("buzz", 0.0),
                }
            )

            # Count word cloud items
            word_cloud = json_data.get("wordCloud", [])
            data_dict["word_cloud_count"] = len(word_cloud) if word_cloud else 0

            # Parse the JSON data using the Pydantic model
            result = TipranksNewsSentimentModel(**data_dict)
            logger.debug(
                f"Successfully parsed TipRanks news sentiment data for {ticker}"
            )
            return result

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            # HTTP errors are generally retriable
            query_info = ticker or query or "unknown"
            logger.warning(
                f"HTTP error fetching TipRanks news sentiment data for {query_info}: {e}"
            )
            raise RetriableProviderException(
                f"HTTP error fetching Tipranks news sentiment data: {e}"
            ) from e
        except ValueError as e:
            # Non-retriable errors (e.g., empty data, invalid ticker)
            query_info = ticker or query or "unknown"
            logger.error(
                f"Non-retriable error in TipranksNewsSentimentProvider for {query_info}: "
                f"{e}"
            )
            raise NonRetriableProviderException(str(e)) from e
        except Exception as e:
            # Other errors retriable
            query_info = ticker or query or "unknown"
            logger.warning(
                f"Retriable error in TipranksNewsSentimentProvider for {query_info}: {e}"
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
    logger.debug(
        f"Creating TipranksDataProvider: timeout={timeout}s, retries={retries}"
    )
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
    logger.debug(
        f"Creating TipranksNewsSentimentProvider: timeout={timeout}s, retries={retries}"
    )
    config = ProviderConfig(timeout=timeout, retries=retries)
    return TipranksNewsSentimentProvider(config)
