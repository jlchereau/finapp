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
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic
from datetime import datetime
from enum import Enum
from pandas import DataFrame
from pydantic import BaseModel, Field

from ..lib.logger import logger


class NonRetriableProviderException(Exception):
    """Indicates a provider error that should not be retried."""


class RetriableProviderException(Exception):
    """Indicates a transient provider error that can be retried."""


# Type variables for generic typing
T = TypeVar("T", bound=DataFrame | BaseModel)


class ProviderType(str, Enum):
    """Enum for provider types."""

    YAHOO_HISTORY = "yahoo_history"
    YAHOO_INFO = "yahoo_info"
    ZACKS = "zacks"
    BLACKROCK_HOLDINGS = "blackrock_holdings"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX = "iex"
    FRED_SERIES = "fred_series"
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
    query: str | None = Field(
        default=None,
        description="A query, for example a stock ticker",
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
        self.provider_type = self._get_provider_type()
        self._semaphore = asyncio.Semaphore(10)  # Limit concurrent operations

        logger.debug(
            f"Initialized {self.__class__.__name__} provider: "
            f"timeout={self.config.timeout}s, retries={self.config.retries}"
        )

    @property
    def logger(self):
        """Provide access to the global logger instance."""
        return logger

    @abstractmethod
    def _get_provider_type(self) -> ProviderType:
        """Return the provider type. Must be implemented by subclasses."""
        raise NotImplementedError()

    @abstractmethod
    async def _fetch_data(self, query: str | None, **kwargs) -> T:
        """
        Internal method to fetch data. Must be implemented by subclasses.
        This method should contain the actual data fetching logic.

        Args:
            query: a query, for example a stock ticker or None if not applicable
            **kwargs: Additional parameters specific to the provider

        Returns:
            Raw data (DataFrame or Pydantic model)

        Raises:
            Exception: Any provider-specific exceptions
        """
        raise NotImplementedError()

    async def get_data(self, query: str | None, **kwargs) -> ProviderResult[T]:
        """
        Fetch data from the provider with comprehensive error handling.
        This is the main public interface that workflows should use.

        Args:
            query: a query, for example a stock ticker or None if not applicable
            **kwargs: Additional parameters specific to the provider

        Returns:
            ProviderResult containing data or error information
        """
        # Basic type validation
        if query is not None and not isinstance(query, str):
            logger.error(
                f"Invalid query type for {self.provider_type}: "
                f"expected string/None, got {type(query).__name__}"
            )
            return ProviderResult(
                success=False,
                error_message="Query must be a string or None",
                error_code="ValueError",
                provider_type=self.provider_type,
                query=None,
            )

        start_time = asyncio.get_event_loop().time()

        async with self._semaphore:  # Limit concurrent operations
            try:
                logger.info(
                    f"Starting data fetch for {self.provider_type} with query: {query}"
                )

                # Cache check placeholder (caching not yet implemented)

                # Apply rate limiting if configured
                if self.config.rate_limit:
                    delay = 1.0 / self.config.rate_limit
                    logger.debug(f"Applying rate limit: sleeping {delay:.3f}s")
                    await asyncio.sleep(delay)

                last_exception = None
                for attempt in range(self.config.retries + 1):
                    try:
                        # Fetch data with timeout
                        data = await asyncio.wait_for(
                            self._fetch_data(query, **kwargs),
                            timeout=self.config.timeout,
                        )
                        # Success
                        execution_time = asyncio.get_event_loop().time() - start_time
                        logger.info(
                            f"Successfully fetched data for {query} in "
                            f"{execution_time:.2f}s (attempt {attempt + 1})"
                        )
                        return ProviderResult(
                            success=True,
                            data=data,
                            execution_time=execution_time,
                            provider_type=self.provider_type,
                            query=query,
                            metadata={"attempt": attempt + 1},
                        )
                    except asyncio.TimeoutError as e:
                        # retry on timeout
                        last_exception = e
                        logger.warning(
                            f"Timeout for {query} after {self.config.timeout}s "
                            f"(attempt {attempt + 1}/{self.config.retries + 1})"
                        )
                        if attempt < self.config.retries:
                            retry_delay = self.config.retry_delay * (attempt + 1)
                            logger.debug(f"Retrying in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                    except NonRetriableProviderException as e:
                        # fail fast on non-retriable
                        logger.error(f"Non-retriable error fetching {query}: {e}")
                        return ProviderResult(
                            success=False,
                            error_message=str(e),
                            error_code=type(e).__name__,
                            execution_time=asyncio.get_event_loop().time() - start_time,
                            provider_type=self.provider_type,
                            query=query,
                        )
                    except RetriableProviderException as e:
                        # backoff and retry
                        last_exception = e
                        logger.warning(
                            f"Retriable error fetching {query} "
                            f"(attempt {attempt + 1}/{self.config.retries + 1}): {e}"
                        )
                        if attempt < self.config.retries:
                            retry_delay = self.config.retry_delay * (attempt + 1)
                            logger.debug(f"Retrying in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                        continue
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        # retry on unexpected errors
                        last_exception = e
                        logger.warning(
                            f"Unexpected error fetching {query} "
                            f"(attempt {attempt + 1}/{self.config.retries + 1}): "
                            f"{type(e).__name__}: {e}"
                        )
                        if attempt < self.config.retries:
                            retry_delay = self.config.retry_delay * (attempt + 1)
                            logger.debug(f"Retrying in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                        continue
                # All retries exhausted without success
                attempts = self.config.retries + 1
                msg = f"Failed after {attempts} attempts"
                if last_exception:
                    msg += f": {last_exception}"
                execution_time = asyncio.get_event_loop().time() - start_time
                logger.error(
                    f"All retries exhausted for {query} after "
                    f"{execution_time:.2f}s: {msg}"
                )
                # Prepare final error code
                error_code = type(last_exception).__name__ if last_exception else None
                return ProviderResult(
                    success=False,
                    error_message=msg,
                    error_code=error_code,
                    execution_time=execution_time,
                    provider_type=self.provider_type,
                    query=query,
                    metadata={"total_attempts": attempts},
                )

            except Exception as e:  # pylint: disable=broad-exception-caught
                execution_time = asyncio.get_event_loop().time() - start_time
                logger.error(
                    f"Unexpected outer exception for {query}: "
                    f"{type(e).__name__}: {e}"
                )

                return ProviderResult[T](
                    success=False,
                    error_message=f"Unexpected error: {str(e)}",
                    error_code=type(e).__name__,
                    execution_time=execution_time,
                    provider_type=self.provider_type,
                    query=query,
                )

    def get_data_sync(self, query: str | None, **kwargs) -> ProviderResult[T]:
        """
        Synchronous wrapper for get_data. Useful for non-async contexts.

        Args:
            query: a query, for example a stock ticker or None if not applicable
            **kwargs: Additional parameters specific to the provider

        Returns:
            ProviderResult containing data or error information
        """
        return asyncio.run(self.get_data(query, **kwargs))
