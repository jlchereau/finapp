"""
Yahoo provider module
This module provides functionality to fetch data from Yahoo Finance.
"""

import asyncio
from pandas import DataFrame
from pydantic import BaseModel
import yfinance as yf
from .base import BaseProvider, ProviderType, ProviderConfig
from .parsers import PydanticJSONParser, ParserConfig


# Configuration for Yahoo Finance info data parsing
YAHOO_INFO_CONFIG = ParserConfig(
    name="YahooInfoModel",
    fields={
        "ticker": {"expr": "symbol", "default": None},
        "company_name": {"expr": "longName", "default": None},
        "price": {"expr": "regularMarketPrice", "default": None},
        "change": {"expr": "regularMarketChange", "default": None},
        "percent_change": {"expr": "regularMarketChangePercent", "default": None},
        "volume": {"expr": "regularMarketVolume", "default": None},
        "market_cap": {"expr": "marketCap", "default": None},
        "pe_ratio": {"expr": "trailingPE", "default": None},
        "dividend_yield": {"expr": "dividendYield", "default": None},
        "beta": {"expr": "beta", "default": None},
        "52_week_high": {"expr": "fiftyTwoWeekHigh", "default": None},
        "52_week_low": {"expr": "fiftyTwoWeekLow", "default": None},
        "currency": {"expr": "currency", "default": "USD"},
        "exchange": {"expr": "exchange", "default": None},
        "sector": {"expr": "sector", "default": None},
        "industry": {"expr": "industry", "default": None},
    },
    strict_mode=False,
    default_value=None,
)


class YahooHistoryProvider(BaseProvider[DataFrame]):
    """
    Provider for fetching historical price data from Yahoo Finance.

    This provider fetches OHLCV (Open, High, Low, Close, Volume) data
    for stocks, ETFs, and other financial instruments.
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.YAHOO_HISTORY

    async def _fetch_data(self, ticker: str, **kwargs) -> DataFrame:
        """
        Fetch historical price data for a ticker from yfinance.

        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters:
                - period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y,
                  10y, ytd, max)
                - interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m,
                  1h, 1d, 5d, 1wk, 1mo, 3mo)
                - start: Start date (YYYY-MM-DD)
                - end: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If no data is returned or ticker is invalid
            Exception: For other yfinance-related errors
        """
        # Extract parameters with defaults
        period = kwargs.get("period", self.config.extra_config.get("period", "1y"))
        interval = kwargs.get(
            "interval", self.config.extra_config.get("interval", "1d")
        )
        start = kwargs.get("start", self.config.extra_config.get("start"))
        end = kwargs.get("end", self.config.extra_config.get("end"))

        # Run yfinance call in a separate thread to avoid blocking
        def fetch_history():
            yf_ticker = yf.Ticker(ticker)
            if start and end:
                return yf_ticker.history(start=start, end=end, interval=interval)
            return yf_ticker.history(period=period, interval=interval)

        data = await asyncio.to_thread(fetch_history)

        if data.empty:
            raise ValueError(f"No historical data found for ticker: {ticker}")

        # Clean up the data
        data.index.name = "Date"
        data = data.round(2)  # Round to 2 decimal places

        return data


class YahooInfoProvider(BaseProvider[BaseModel]):
    """
    Provider for fetching fundamental and market data from Yahoo Finance.

    This provider fetches company information, financial metrics,
    and real-time market data.
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.YAHOO_INFO

    async def _fetch_data(self, ticker: str, **kwargs) -> BaseModel:
        """
        Get ticker info/fundamentals from yfinance.

        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters (currently unused)

        Returns:
            Pydantic model instance with ticker info

        Raises:
            ValueError: If no info is returned or ticker is invalid
            Exception: For other yfinance-related errors
        """

        # Run yfinance call in a separate thread to avoid blocking
        def fetch_info():
            yf_ticker = yf.Ticker(ticker)
            return yf_ticker.info

        json_data = await asyncio.to_thread(fetch_info)

        if not json_data or not isinstance(json_data, dict):
            raise ValueError(f"No info data found for ticker: {ticker}")

        # Parse the JSON data using our parser
        parser = PydanticJSONParser(YAHOO_INFO_CONFIG)
        result = await parser.parse_async(json_data)

        return result


# Factory functions for easy provider creation
def create_yahoo_history_provider(
    period: str = "1y",
    interval: str = "1d",
    timeout: float = 30.0,
    retries: int = 3,
) -> YahooHistoryProvider:
    """
    Factory function to create a Yahoo History provider with custom settings.

    Args:
        period: Default period for data fetching
        interval: Default interval for data fetching
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        Configured YahooHistoryProvider instance
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
