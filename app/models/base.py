"""
Abstract base class for all data providers.
All provider classes should inherit from this class.

Note: we call them providers instead of models to avoid confusion
with Pydantic models. Providers are responsible for fetching and
processing data from various sources.

Such sources are Yahoo Finance, Zacks, Interactive Brokers, etc.:
    - Some providers will use external APIs like yfinance and ibapi.
    - Others will use httpx to get the result of JSON Restful APIs.
    - Other will scrape data from web pages using BeautifulSoup
    or similar libraries.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic
from pandas import DataFrame
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# Type variables for generic typing
T = TypeVar("T", bound=DataFrame | BaseModel)


class ProviderType(str, Enum):
    """Enum for provider types."""

    YAHOO_HISTORY = "yahoo_history"
    YAHOO_INFO = "yahoo_info"
    ZACKS = "zacks"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX = "iex"
    CUSTOM = "custom"


class ProviderStatus(str, Enum):
    """Enum for provider execution status."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class ProviderResult(BaseModel, Generic[T]):
    """
    Standardized result wrapper for all provider operations.
    Provides consistent error handling and metadata across providers.
    """

    model_config = {"arbitrary_types_allowed": True}

    success: bool = Field(description="Whether the operation was successful")
    data: T | None = Field(
        default=None,
        description="The actual data returned",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if operation failed",
    )
    error_code: str | None = Field(
        default=None,
        description="Error code for categorization",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    execution_time: float | None = Field(
        default=None,
        description="Execution time in seconds",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the operation was executed",
    )
    provider_type: ProviderType = Field(
        description="Type of provider that generated this result"
    )
    ticker: str | None = Field(
        default=None,
        description="Ticker symbol if applicable",
    )


class ProviderConfig(BaseModel):
    """
    Configuration for data providers.
    Extensible for different provider-specific settings.
    """

    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds",
    )
    retries: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Whether to enable caching",
    )
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    rate_limit: float | None = Field(
        default=None,
        description="Rate limit in requests per second",
    )
    user_agent: str = Field(
        default="FinApp/1.0", description="User agent for HTTP requests"
    )

    # Provider-specific configs can be added here
    extra_config: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific configuration"
    )


class BaseProvider(ABC, Generic[T]):
    """
    Abstract base class for all providers.
    All provider classes should inherit from this class.

    Key improvements:
    - Async/await support for workflows
    - Standardized error handling
    - Future caching preparation
    - Thread-safe operations
    - Comprehensive logging
    """

    def __init__(self, config: ProviderConfig | None = None):
        """
        Initialize the provider with optional configuration.

        Args:
            config: Provider configuration. Uses defaults if not provided.
        """
        self.config = config or ProviderConfig()
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self.logger = logging.getLogger(logger_name)
        self.provider_type = self._get_provider_type()
        self._semaphore = asyncio.Semaphore(10)  # Limit concurrent operations

    @abstractmethod
    def _get_provider_type(self) -> ProviderType:
        """Return the provider type. Must be implemented by subclasses."""
        raise NotImplementedError()

    @abstractmethod
    async def _fetch_data(self, ticker: str, **kwargs) -> T:
        """
        Internal method to fetch data. Must be implemented by subclasses.
        This method should contain the actual data fetching logic.

        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters specific to the provider

        Returns:
            Raw data (DataFrame or Pydantic model)

        Raises:
            Exception: Any provider-specific exceptions
        """
        raise NotImplementedError()

    async def get_data(self, ticker: str, **kwargs) -> ProviderResult[T]:
        """
        Fetch data from the provider with comprehensive error handling.
        This is the main public interface that workflows should use.

        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters specific to the provider

        Returns:
            ProviderResult containing data or error information
        """
        start_time = asyncio.get_event_loop().time()

        async with self._semaphore:  # Limit concurrent operations
            try:
                self.logger.info("Fetching data for ticker: %s", ticker)

                # Validate inputs
                if not ticker or not isinstance(ticker, str):
                    raise ValueError("Ticker must be a non-empty string")

                ticker = ticker.upper().strip()

                # TODO: Check cache here when caching is implemented

                # Apply rate limiting if configured
                if self.config.rate_limit:
                    await asyncio.sleep(1.0 / self.config.rate_limit)

                # Retry logic
                last_exception = None
                for attempt in range(self.config.retries + 1):
                    try:
                        # Apply timeout
                        data = await asyncio.wait_for(
                            self._fetch_data(ticker, **kwargs),
                            timeout=self.config.timeout,
                        )

                        # Calculate execution time
                        loop = asyncio.get_event_loop()
                        execution_time = loop.time() - start_time

                        # TODO: Store in cache here when caching is implemented

                        result = ProviderResult[T](
                            success=True,
                            data=data,
                            execution_time=execution_time,
                            provider_type=self.provider_type,
                            ticker=ticker,
                            metadata={"attempt": attempt + 1},
                        )

                        # pylint: disable=logging-fstring-interpolation
                        self.logger.info(
                            "Successfully fetched data for %s in %.2f",
                            ticker,
                            execution_time,
                        )
                        return result

                    except asyncio.TimeoutError as e:
                        last_exception = e
                        self.logger.warning(
                            "Timeout for %s, attempt %d", ticker, attempt + 1
                        )
                        if attempt < self.config.retries:
                            delay = self.config.retry_delay * (attempt + 1)
                            await asyncio.sleep(delay)

                    # pylint: disable=broad-exception-caught
                    except Exception as e:
                        last_exception = e
                        self.logger.warning(
                            "Error fetching %s, attempt %d: %s",
                            ticker,
                            attempt + 1,
                            e,
                        )
                        if attempt < self.config.retries:
                            delay = self.config.retry_delay * (attempt + 1)
                            await asyncio.sleep(delay)

                # All retries failed
                # Calculate execution time
                loop = asyncio.get_event_loop()
                execution_time = loop.time() - start_time
                error_message = (
                    f"Failed after {self.config.retries + 1} attempts: "
                    f"{last_exception}"
                )

                # Determine error code
                if last_exception:
                    error_code = type(last_exception).__name__
                else:
                    error_code = "UnknownError"

                return ProviderResult[T](
                    success=False,
                    error_message=error_message,
                    error_code=error_code,
                    execution_time=execution_time,
                    provider_type=self.provider_type,
                    ticker=ticker,
                    metadata={"total_attempts": self.config.retries + 1},
                )

            # pylint: disable=broad-exception-caught
            except Exception as e:
                execution_time = asyncio.get_event_loop().time() - start_time
                self.logger.error("Unexpected error for %s: %s", ticker, e)

                return ProviderResult[T](
                    success=False,
                    error_message=f"Unexpected error: {str(e)}",
                    error_code=type(e).__name__,
                    execution_time=execution_time,
                    provider_type=self.provider_type,
                    ticker=ticker,
                )

    def get_data_sync(self, ticker: str, **kwargs) -> ProviderResult[T]:
        """
        Synchronous wrapper for get_data. Useful for non-async contexts.

        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters specific to the provider

        Returns:
            ProviderResult containing data or error information
        """
        return asyncio.run(self.get_data(ticker, **kwargs))
