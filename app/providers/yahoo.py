"""
Yahoo provider module
This module provides functionality to fetch data from Yahoo Finance.
"""

import asyncio
import yfinance as yf
from pandas import DataFrame
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


class YahooInfoModel(BaseModel):
    """Pydantic model for Yahoo Finance info data."""

    # Important: Do not set default values for required fields.
    # This would prevent us from detecting API changes.
    # If the API changes and fields are missing,
    # Pydantic should raise a validation error.

    ticker: str = Field(alias="symbol")
    company_name: str = Field(alias="longName")
    price: float = Field(alias="regularMarketPrice")
    change: float = Field(alias="regularMarketChange")
    percent_change: float = Field(alias="regularMarketChangePercent")
    volume: int = Field(alias="regularMarketVolume")
    market_cap: int = Field(alias="marketCap")
    pe_ratio: float = Field(alias="trailingPE")
    dividend_yield: float = Field(alias="dividendYield")
    beta: float = Field(alias="beta")
    week_52_high: float = Field(alias="fiftyTwoWeekHigh")
    week_52_low: float = Field(alias="fiftyTwoWeekLow")
    currency: str = Field(alias="currency")
    exchange: str = Field(alias="exchange")
    sector: str = Field(alias="sector")
    industry: str = Field(alias="industry")

    @field_validator("volume", "market_cap", mode="before")
    @classmethod
    def convert_null_ints(cls, v):
        if v is None:
            raise ValueError("Required int field cannot be None")
        return int(v)

    @field_validator(
        "price",
        "change",
        "percent_change",
        "pe_ratio",
        "dividend_yield",
        "beta",
        "week_52_high",
        "week_52_low",
        mode="before",
    )
    @classmethod
    def convert_null_floats(cls, v):
        if v is None:
            raise ValueError("Required float field cannot be None")
        return float(v)

    model_config = {"populate_by_name": True, "extra": "ignore"}


class YahooHistoryProvider(BaseProvider[DataFrame]):
    """
    Provider for fetching historical price data from Yahoo Finance.

    This provider fetches OHLCV (Open, High, Low, Close, Volume) data
    for stocks, ETFs, and other financial instruments.
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.YAHOO_HISTORY

    @apply_provider_cache
    # @apply_provider_cache triggers pyrefly bad-override - no easy fix
    async def _fetch_data(
        self, query: str | None, *args, cache_date: str | None = None, **kwargs
    ) -> DataFrame:
        """
        Fetch historical price data for a ticker from yfinance.

        Args:
            query: Stock ticker symbol to fetch (must be non-null)
            **kwargs: Additional parameters:
                - period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y,
                  10y, ytd, max). Default: "max" for optimal caching.
                - interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m,
                  1h, 1d, 5d, 1wk, 1mo, 3mo)
                - start: Start date (YYYY-MM-DD)
                - end: End date (YYYY-MM-DD)

        Note:
            ✅ CORRECT PROVIDER PATTERN: Defaults to "max" period to ensure
            maximum historical data is cached. Workflows filter to shorter
            periods without re-fetching. This prevents the period limitation
            anti-pattern that breaks period selection.

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If no data is returned or ticker is invalid
            Exception: For other yfinance-related errors
        """
        try:
            # Extract parameters with defaults
            # Default to "max" period for optimal caching - workflows can filter
            period = kwargs.get("period", self.config.extra_config.get("period", "max"))
            interval = kwargs.get(
                "interval", self.config.extra_config.get("interval", "1d")
            )
            start = kwargs.get("start", self.config.extra_config.get("start"))
            end = kwargs.get("end", self.config.extra_config.get("end"))

            # Validate query
            if query is None:
                logger.error("Query cannot be None for YahooHistoryProvider")
                raise ValueError("Query must be provided for YahooHistoryProvider")
            ticker = query.upper().strip()

            # Run yfinance call in a separate thread to avoid blocking
            def fetch_history():
                yf_ticker = yf.Ticker(ticker)
                if start and end:
                    logger.info(
                        f"Calling yfinance.history for {ticker} from {start} to "
                        f"{end} with interval {interval}"
                    )
                    return yf_ticker.history(start=start, end=end, interval=interval)
                logger.info(
                    f"Calling yfinance.history for {ticker} with period "
                    f"{period} and interval {interval}"
                )
                return yf_ticker.history(period=period, interval=interval)

            data = await asyncio.to_thread(fetch_history)

            if data.empty:
                logger.warning(
                    f"No historical data returned from yfinance for ticker: {ticker}"
                )
                raise ValueError(f"No historical data found for query: {query}")

            # Clean up the data
            data.index.name = "Date"
            data = data.round(2)  # Round to 2 decimal places

            return data
        except ValueError as e:
            # Non-retriable errors (e.g., empty data)
            logger.error(
                f"Non-retriable error in YahooHistoryProvider for {query}: {e}"
            )
            raise NonRetriableProviderException(str(e)) from e
        except Exception as e:
            # Other errors retriable
            logger.warning(
                f"Retriable error in YahooHistoryProvider for {query}: "
                f"{type(e).__name__}: {e}"
            )
            raise RetriableProviderException(str(e)) from e


class YahooInfoProvider(BaseProvider[BaseModel]):
    """
    Provider for fetching fundamental and market data from Yahoo Finance.

    This provider fetches company information, financial metrics,
    and real-time market data.
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.YAHOO_INFO

    @apply_provider_cache
    # @apply_provider_cache triggers pyrefly bad-override - no easy fix
    async def _fetch_data(
        self, query: str | None, *args, cache_date: str | None = None, **kwargs
    ) -> BaseModel:
        """
        Get ticker info/fundamentals from yfinance.

        Args:
            query: Stock ticker symbol to fetch (must be non-null)
            **kwargs: Additional parameters (currently unused)

        Returns:
            Pydantic model instance with ticker info

        Raises:
            ValueError: If no info is returned or ticker is invalid
            Exception: For other yfinance-related errors
        """
        try:
            # Validate query
            if query is None:
                logger.error("Query cannot be None for YahooInfoProvider")
                raise ValueError("Query must be provided for YahooInfoProvider")
            ticker = query.upper().strip()
            # Run yfinance call in a separate thread to avoid blocking

            def fetch_info():
                logger.info(f"Calling yfinance.info for {ticker}")
                yf_ticker = yf.Ticker(ticker)
                return yf_ticker.info

            json_data = await asyncio.to_thread(fetch_info)

            if not json_data or not isinstance(json_data, dict):
                logger.warning(
                    f"No info data returned from yfinance for ticker: {ticker}"
                )
                raise ValueError(f"No info data found for query: {query}")

            # Parse the JSON data using the Pydantic model (strict validation)
            result = YahooInfoModel(**json_data)
            return result
        except ValueError as e:
            logger.error(f"Non-retriable error in YahooInfoProvider for {query}: {e}")
            raise NonRetriableProviderException(str(e)) from e
        except Exception as e:
            logger.warning(
                f"Retriable error in YahooInfoProvider for {query}: "
                f"{type(e).__name__}: {e}"
            )
            raise RetriableProviderException(str(e)) from e


# Factory functions for easy provider creation
def create_yahoo_history_provider(
    period: str = "max",  # Default to max for caching - workflows filter down
    interval: str = "1d",
    timeout: float = 30.0,
    retries: int = 3,
) -> YahooHistoryProvider:
    """
    Factory function to create a Yahoo History provider with custom settings.

    Args:
        period: Default period for data fetching (default: "max" for caching)
        interval: Default interval for data fetching
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        Configured YahooHistoryProvider instance

    Note:
        The default period is "max" to ensure maximum historical data is cached.
        Workflows can filter to shorter durations, but cannot bypass cache for
        longer periods.
    """
    config = ProviderConfig(
        timeout=timeout,
        retries=retries,
        extra_config={"period": period, "interval": interval},
    )
    return YahooHistoryProvider(config)


def create_yahoo_info_provider(
    timeout: float = 30.0,
    retries: int = 3,
) -> YahooInfoProvider:
    """
    Factory function to create a Yahoo Info provider with custom settings.

    Args:
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        Configured YahooInfoProvider instance
    """
    config = ProviderConfig(timeout=timeout, retries=retries)
    return YahooInfoProvider(config)
