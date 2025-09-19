"""
TimeSeriesWorkflow for historical data collection and time series chart processing.

This workflow handles fetching historical price data for multiple tickers
using the YahooHistoryProvider with proper LlamaIndex event-driven patterns.
Used primarily by the plots tab for returns, volatility, volume, and RSI charts.
"""

import time
from typing import List, Dict
from datetime import datetime

from pandas import DataFrame
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from app.providers.yahoo import create_yahoo_history_provider
from app.flows.base import FlowResult
from app.lib.logger import logger
from app.lib.exceptions import FlowException
from app.lib.finance import get_close_prices
from app.lib.periods import ensure_minimum_data_points
from app.flows.cache import apply_flow_cache


class DispatchEvent(Event):
    """Event to coordinate parallel ticker data fetching."""


class FetchTickerDataEvent(Event):
    """Event to trigger data collection for a specific ticker."""

    ticker: str


class TickerDataResponseEvent(Event):
    """Event to return collected ticker data."""

    ticker: str
    data: DataFrame | None
    success: bool
    error_message: str | None = None


class TimeSeriesWorkflow(Workflow):
    """
    LlamaIndex workflow for time series data collection and normalization.

    This workflow:
    - Fetches historical OHLCV data for multiple tickers in parallel
    - Normalizes data to percentage returns from base date
    - Handles errors gracefully (partial success)
    - Returns data in format compatible with plotly charts
    - Follows LlamaIndex Pattern 3 for parallel execution with single provider
    """

    def __init__(self):
        """Initialize workflow with Yahoo history provider."""
        super().__init__()
        # Create provider with max period for comprehensive caching
        self.yahoo_history = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def dispatch(
        self, ctx: Context, ev: StartEvent
    ) -> FetchTickerDataEvent | DispatchEvent:
        """
        Entry step that coordinates parallel ticker data fetching.

        Args:
            ev.tickers: List of ticker symbols
            ev.base_date: Start date for data fetching

        Returns:
            DispatchEvent to coordinate collection
        """
        tickers = ev.tickers
        base_date = ev.base_date

        logger.debug(
            f"TimeSeriesWorkflow: Dispatching data fetch for {len(tickers)} tickers "
            f"from {base_date}"
        )

        # Store metadata for later steps
        await ctx.store.set("tickers", tickers)
        await ctx.store.set("base_date", base_date)
        await ctx.store.set("num_to_collect", len(tickers))
        await ctx.store.set("start_time", time.time())

        # Send parallel fetch events for each ticker
        for ticker in tickers:
            ctx.send_event(FetchTickerDataEvent(ticker=ticker))

        return DispatchEvent()

    @step(num_workers=3)  # Limit concurrent API calls
    async def fetch_ticker_data(
        self, ev: FetchTickerDataEvent
    ) -> TickerDataResponseEvent:
        """
        Fetch historical data for a single ticker.

        Args:
            ev.ticker: Ticker symbol to fetch

        Returns:
            TickerDataResponseEvent with fetched data or error
        """
        ticker = ev.ticker

        try:
            # Fetch maximum period data for better caching
            provider_result = await self.yahoo_history.get_data(ticker)

            if not provider_result.success:
                return TickerDataResponseEvent(
                    ticker=ticker,
                    data=None,
                    success=False,
                    error_message=f"Provider failed: {provider_result.error_message}",
                )

            data = provider_result.data
            if data is None or data.empty:
                return TickerDataResponseEvent(
                    ticker=ticker,
                    data=None,
                    success=False,
                    error_message="Empty data returned",
                )

            logger.debug(f"Fetched {len(data)} rows for {ticker}")

            return TickerDataResponseEvent(ticker=ticker, data=data, success=True)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.warning(f"Error fetching data for {ticker}: {e}")
            return TickerDataResponseEvent(
                ticker=ticker, data=None, success=False, error_message=str(e)
            )

    @step
    async def normalize_data(
        self, ctx: Context, ev: DispatchEvent | TickerDataResponseEvent
    ) -> StopEvent | None:
        """
        Collect all ticker data and normalize to percentage returns.

        Args:
            ev: DispatchEvent (initial) or TickerDataResponseEvent (data)

        Returns:
            StopEvent with FlowResult containing normalized data
        """
        # Continue waiting after receiving DispatchEvent
        if isinstance(ev, DispatchEvent):
            return None

        # Wait for all ticker data responses
        num_to_collect = await ctx.store.get("num_to_collect")
        events = ctx.collect_events(ev, [TickerDataResponseEvent] * num_to_collect)
        if not events:
            return None

        # All ticker data received - process normalization
        tickers = await ctx.store.get("tickers")
        base_date = await ctx.store.get("base_date")
        start_time = await ctx.store.get("start_time")

        logger.debug(f"TimeSeriesWorkflow: Normalizing data for {len(tickers)} tickers")

        # Process each ticker's data
        normalized_data = DataFrame()
        successful_tickers = []
        failed_tickers = []

        for event in events:
            ticker = event.ticker

            if not event.success:
                failed_tickers.append(ticker)
                logger.warning(
                    f"Skipping {ticker} due to fetch failure: " f"{event.error_message}"
                )
                continue

            try:
                data = event.data

                # Filter data to start from base_date with minimum data points
                filtered_data = ensure_minimum_data_points(
                    data=data, base_date=base_date, min_points=2
                )

                logger.debug(
                    f"{ticker}: {len(data)} -> {len(filtered_data)} rows after "
                    f"filtering"
                )

                if filtered_data.empty:
                    logger.warning(f"No data after {base_date} for {ticker}, skipping")
                    failed_tickers.append(ticker)
                    continue

                # Get close prices and normalize to percentage returns
                close_prices = get_close_prices(filtered_data, ticker)
                if close_prices is None or close_prices.empty:
                    logger.warning(f"No valid close prices for {ticker}, skipping")
                    failed_tickers.append(ticker)
                    continue

                # Calculate percentage returns from first value (after filtering)
                # Formula: ((current_price / first_price) - 1) * 100
                first_price = close_prices.iloc[0]
                percentage_returns = ((close_prices / first_price) - 1) * 100

                # Add to normalized data
                normalized_data[ticker] = percentage_returns
                successful_tickers.append(ticker)

            except (KeyError, IndexError, AttributeError) as e:
                logger.warning(f"Error normalizing data for {ticker}: {e}")
                failed_tickers.append(ticker)
                continue

        # Log summary
        execution_time = time.time() - start_time
        logger.info(
            f"TimeSeriesWorkflow completed: {len(successful_tickers)} successful, "
            f"{len(failed_tickers)} failed tickers"
        )

        if failed_tickers:
            logger.debug(f"Failed tickers: {failed_tickers}")

        # Check if we have any successful data
        if not successful_tickers:
            raise FlowException(
                workflow="TimeSeriesWorkflow",
                step="normalize_data",
                message=f"Failed to process data for all {len(tickers)} tickers",
                user_message=(
                    "Unable to fetch data for any of the selected tickers. "
                    "Please check the ticker symbols and try again."
                ),
                context={"tickers": tickers, "failed_count": len(failed_tickers)},
            )

        return StopEvent(
            result=FlowResult.success_result(
                data=normalized_data,
                base_date=base_date,
                execution_time=execution_time,
                successful_items=successful_tickers,
                failed_items=failed_tickers,
            )
        )


@apply_flow_cache
async def fetch_time_series_data(
    tickers: List[str], base_date: datetime
) -> Dict[str, DataFrame]:
    """
    Shared function to fetch raw OHLCV data with in-memory caching.

    This function eliminates redundant data fetching across multiple chart types
    by caching the raw data in memory for 5 minutes.

    Args:
        tickers: List of ticker symbols to fetch
        base_date: Start date for historical data (used for cache key)

    Returns:
        Dictionary mapping ticker -> raw OHLCV DataFrame

    Raises:
        FlowException: If data fetching fails for all tickers
    """
    if not tickers:
        logger.debug("fetch_time_series_data: No tickers provided")
        return {}

    try:
        logger.info(f"Fetching time series data for {len(tickers)} tickers")

        # Create provider to fetch raw OHLCV data with max period
        yahoo_history = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

        # Fetch data directly for caching purposes
        successful_data = {}
        failed_tickers = []

        for ticker in tickers:
            try:
                provider_result = await yahoo_history.get_data(ticker)

                if (
                    provider_result.success
                    and provider_result.data is not None
                    and not provider_result.data.empty
                ):
                    # Store raw data without filtering - let individual
                    # processors handle filtering
                    successful_data[ticker] = provider_result.data
                    logger.debug(
                        f"Cached raw data for {ticker}: "
                        f"{len(provider_result.data)} rows"
                    )
                else:
                    failed_tickers.append(ticker)
                    error_msg = provider_result.error_message or "Unknown error"
                    logger.warning(
                        f"Failed to fetch raw data for {ticker}: {error_msg}"
                    )

            except (ValueError, TypeError, KeyError, AttributeError) as e:
                failed_tickers.append(ticker)
                logger.warning(f"Exception fetching raw data for {ticker}: {e}")

        logger.info(
            f"Time series data fetch completed: {len(successful_data)} successful, "
            f"{len(failed_tickers)} failed"
        )

        if failed_tickers:
            logger.debug(f"Failed tickers: {failed_tickers}")

        # Check if we have any successful data
        if not successful_data:
            raise FlowException(
                workflow="fetch_time_series_data",
                step="data_validation",
                message=f"Failed to fetch data for all {len(tickers)} tickers",
                user_message=(
                    "Unable to fetch data for any of the selected tickers. "
                    "Please check the ticker symbols and try again."
                ),
                context={"tickers": tickers, "failed_count": len(failed_tickers)},
            )

        return successful_data

    except Exception as e:
        logger.error(f"Time series data fetch failed: {e}")
        # Re-raise as FlowException for better handling
        raise FlowException(
            workflow="fetch_time_series_data",
            step="data_processing",
            message=f"Time series data fetch workflow failed: {e}",
            user_message=(
                "Data fetching failed due to a system error. Please try again."
            ),
            context={"tickers": tickers, "base_date": str(base_date)},
        ) from e
