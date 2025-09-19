"""
VolatilityWorkflow for calculating rolling volatility from time series data.

This workflow follows the page → workflow → provider pattern and delegates
calculation logic to finance functions for consistency.
"""

import time
from typing import List, Dict, Any
from datetime import datetime

from pandas import DataFrame
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent

from app.flows.base import FlowResultEvent, FlowRunner
from app.lib.logger import logger
from app.lib.exceptions import FlowException
from app.lib.finance import calculate_volatility
from app.lib.periods import ensure_minimum_data_points
from .time_series import fetch_time_series_data


class VolatilityStartEvent(Event):
    """Start event for volatility calculation."""

    tickers: List[str]
    base_date: datetime
    window: int = 30
    annualize: bool = True


class FetchDataEvent(Event):
    """Event to trigger data fetching for volatility calculation."""

    tickers: List[str]
    base_date: datetime
    window: int
    annualize: bool


class VolatilityWorkflow(Workflow):
    """
    LlamaIndex workflow for calculating rolling volatility.

    This workflow:
    - Step 1: Fetches time series data using cached fetch_time_series_data()
    - Step 2: Delegates volatility calculation to finance.calculate_volatility()
    - Follows proper event-driven patterns
    - Maintains consistent architecture
    """

    @step
    async def fetch_data(self, ctx: Context, ev: StartEvent) -> FetchDataEvent:
        """
        Step 1: Fetch time series data for volatility calculation.

        Args:
            ev.tickers: List of ticker symbols
            ev.base_date: Start date for calculation
            ev.window: Rolling window size (default: 30)
            ev.annualize: Whether to annualize volatility (default: True)

        Returns:
            FetchDataEvent to proceed to processing
        """
        tickers = ev.tickers
        base_date = ev.base_date
        window = getattr(ev, "window", 30)
        annualize = getattr(ev, "annualize", True)

        logger.debug(f"VolatilityWorkflow: Fetching data for {len(tickers)} tickers")

        # Store metadata for later steps
        await ctx.store.set("tickers", tickers)
        await ctx.store.set("base_date", base_date)
        await ctx.store.set("window", window)
        await ctx.store.set("annualize", annualize)
        await ctx.store.set("start_time", time.time())

        return FetchDataEvent(
            tickers=tickers, base_date=base_date, window=window, annualize=annualize
        )

    @step
    async def process_data(self, ctx: Context, ev: FetchDataEvent) -> FlowResultEvent:
        """
        Step 2: Process raw data into volatility metrics.

        Args:
            ev.tickers: List of tickers to process
            ev.base_date: Base date for calculation
            ev.window: Rolling window size
            ev.annualize: Whether to annualize volatility

        Returns:
            FlowResultEvent containing volatility data
        """
        tickers = ev.tickers
        base_date = ev.base_date
        window = ev.window
        annualize = ev.annualize
        start_time = await ctx.store.get("start_time")

        logger.info(
            f"VolatilityWorkflow: Processing volatility for {len(tickers)} tickers"
        )

        # Get cached time series data
        raw_data = await fetch_time_series_data(tickers, base_date)

        if not raw_data:
            raise FlowException(
                workflow="VolatilityWorkflow",
                step="process_data",
                message="No time series data available from cache",
                user_message=(
                    "Unable to fetch data for any of the selected tickers. "
                    "Please check the ticker symbols and try again."
                ),
                context={"tickers": tickers, "base_date": str(base_date)},
            )

        # Process each ticker using finance function
        volatility_data = DataFrame()
        successful_tickers = []
        failed_tickers = []

        for ticker in tickers:
            try:
                if ticker not in raw_data:
                    failed_tickers.append(ticker)
                    logger.warning(f"VolatilityWorkflow: No data for {ticker}")
                    continue

                ticker_data = raw_data[ticker]
                if ticker_data.empty:
                    failed_tickers.append(ticker)
                    logger.warning(f"VolatilityWorkflow: Empty data for {ticker}")
                    continue

                # Calculate volatility on full historical data
                vol_result = calculate_volatility(
                    ticker_data, window=window, annualize=annualize
                )

                if vol_result.empty:
                    failed_tickers.append(ticker)
                    logger.warning(
                        f"VolatilityWorkflow: No volatility calculated for {ticker}"
                    )
                    continue

                # Filter calculated volatility to user's selected period
                filtered_volatility = ensure_minimum_data_points(
                    data=vol_result,
                    base_date=base_date,
                    min_points=2,
                )

                if filtered_volatility.empty:
                    failed_tickers.append(ticker)
                    logger.warning(
                        f"VolatilityWorkflow: No data after period filtering for "
                        f"{ticker}"
                    )
                    continue

                # Add to volatility data (volatility function returns DataFrame
                # with 'Volatility' column)
                volatility_data[ticker] = filtered_volatility["Volatility"]
                successful_tickers.append(ticker)

                logger.debug(
                    f"VolatilityWorkflow: Calculated and filtered volatility for "
                    f"{ticker}"
                )

            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.warning(f"VolatilityWorkflow: Error processing {ticker}: {e}")
                failed_tickers.append(ticker)
                continue

        # Log summary
        execution_time = time.time() - start_time
        logger.info(
            f"VolatilityWorkflow completed: {len(successful_tickers)} successful, "
            f"{len(failed_tickers)} failed tickers"
        )

        if failed_tickers:
            logger.debug(f"Failed tickers: {failed_tickers}")

        # Check if we have any successful data
        if not successful_tickers:
            raise FlowException(
                workflow="VolatilityWorkflow",
                step="process_data",
                message=(
                    f"Failed to calculate volatility for all {len(tickers)} tickers"
                ),
                user_message=(
                    "Unable to calculate volatility for any of the selected "
                    "tickers. Please check the ticker symbols and try again."
                ),
                context={"tickers": tickers, "failed_count": len(failed_tickers)},
            )

        return FlowResultEvent.success_result(
            data=volatility_data,
            base_date=base_date,
            metadata={"execution_time": execution_time},
            successful_items=successful_tickers,
            failed_items=failed_tickers,
        )


# FlowRunner wrapper function for graceful error handling
async def fetch_volatility_data(
    tickers: List[str], base_date: datetime, window: int = 30, annualize: bool = True
) -> Dict[str, Any]:
    """
    Fetch volatility data using FlowRunner for graceful error handling.

    Args:
        tickers: List of ticker symbols to process
        base_date: Start date for volatility calculation
        window: Rolling window size in days (default: 30)
        annualize: Whether to annualize volatility (default: True)

    Returns:
        Dictionary containing volatility data or error information
    """
    workflow = VolatilityWorkflow()
    runner = FlowRunner[DataFrame](workflow)

    result_event = await runner.run(
        tickers=tickers, base_date=base_date, window=window, annualize=annualize
    )

    return {
        "data": result_event.data if result_event.success else None,
        "base_date": result_event.base_date,
        "successful_items": result_event.successful_items,
        "failed_items": result_event.failed_items,
        "execution_time": result_event.metadata.get("execution_time"),
        "success": result_event.success,
        "error_message": result_event.error_message,
    }
