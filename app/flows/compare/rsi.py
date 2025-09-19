"""
RSIWorkflow for calculating Relative Strength Index from time series data.

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
from app.lib.finance import calculate_rsi
from app.lib.periods import ensure_minimum_data_points
from .time_series import fetch_time_series_data


class RSIStartEvent(Event):
    """Start event for RSI calculation."""

    tickers: List[str]
    base_date: datetime
    window: int = 14


class FetchDataEvent(Event):
    """Event to trigger data fetching for RSI calculation."""

    tickers: List[str]
    base_date: datetime
    window: int


class RSIWorkflow(Workflow):
    """
    LlamaIndex workflow for calculating Relative Strength Index (RSI).

    This workflow:
    - Step 1: Fetches time series data using cached fetch_time_series_data()
    - Step 2: Delegates RSI calculation to finance.calculate_rsi()
    - Follows proper event-driven patterns
    - Maintains consistent architecture
    """

    @step
    async def fetch_data(self, ctx: Context, ev: StartEvent) -> FetchDataEvent:
        """
        Step 1: Fetch time series data for RSI calculation.

        Args:
            ev.tickers: List of ticker symbols
            ev.base_date: Start date for calculation
            ev.window: RSI calculation window (default: 14)

        Returns:
            FetchDataEvent to proceed to processing
        """
        tickers = ev.tickers
        base_date = ev.base_date
        window = getattr(ev, "window", 14)

        logger.debug(f"RSIWorkflow: Fetching data for {len(tickers)} tickers")

        # Store metadata for later steps
        await ctx.store.set("tickers", tickers)
        await ctx.store.set("base_date", base_date)
        await ctx.store.set("window", window)
        await ctx.store.set("start_time", time.time())

        return FetchDataEvent(tickers=tickers, base_date=base_date, window=window)

    @step
    async def process_data(self, ctx: Context, ev: FetchDataEvent) -> FlowResultEvent:
        """
        Step 2: Process raw data into RSI values.

        Args:
            ev.tickers: List of tickers to process
            ev.base_date: Base date for calculation
            ev.window: RSI calculation window

        Returns:
            FlowResultEvent containing RSI data
        """
        tickers = ev.tickers
        base_date = ev.base_date
        window = ev.window
        start_time = await ctx.store.get("start_time")

        logger.info(f"RSIWorkflow: Processing RSI for {len(tickers)} tickers")

        # Get cached time series data
        raw_data = await fetch_time_series_data(tickers, base_date)

        if not raw_data:
            raise FlowException(
                workflow="RSIWorkflow",
                step="process_data",
                message="No time series data available from cache",
                user_message=(
                    "Unable to fetch data for any of the selected tickers. "
                    "Please check the ticker symbols and try again."
                ),
                context={"tickers": tickers, "base_date": str(base_date)},
            )

        # Process each ticker using finance function
        rsi_data = DataFrame()
        successful_tickers = []
        failed_tickers = []

        for ticker in tickers:
            try:
                if ticker not in raw_data:
                    failed_tickers.append(ticker)
                    logger.warning(f"RSIWorkflow: No data for {ticker}")
                    continue

                ticker_data = raw_data[ticker]
                if ticker_data.empty:
                    failed_tickers.append(ticker)
                    logger.warning(f"RSIWorkflow: Empty data for {ticker}")
                    continue

                # Calculate RSI on full historical data
                rsi_result = calculate_rsi(ticker_data, window=window)

                if rsi_result.empty:
                    failed_tickers.append(ticker)
                    logger.warning(f"RSIWorkflow: No RSI calculated for {ticker}")
                    continue

                # Filter calculated RSI to user's selected period
                filtered_rsi = ensure_minimum_data_points(
                    data=rsi_result,
                    base_date=base_date,
                    min_points=2,
                )

                if filtered_rsi.empty:
                    failed_tickers.append(ticker)
                    logger.warning(
                        f"RSIWorkflow: No data after period filtering for {ticker}"
                    )
                    continue

                # Add to RSI data (RSI function returns DataFrame with 'RSI' column)
                if "RSI" in filtered_rsi.columns:
                    rsi_data[ticker] = filtered_rsi["RSI"]
                    successful_tickers.append(ticker)
                    logger.debug(
                        f"RSIWorkflow: Calculated and filtered RSI for {ticker}"
                    )
                else:
                    failed_tickers.append(ticker)
                    logger.warning(f"RSIWorkflow: No RSI column in result for {ticker}")

            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.warning(f"RSIWorkflow: Error processing {ticker}: {e}")
                failed_tickers.append(ticker)
                continue

        # Log summary
        execution_time = time.time() - start_time
        logger.info(
            f"RSIWorkflow completed: {len(successful_tickers)} successful, "
            f"{len(failed_tickers)} failed tickers"
        )

        if failed_tickers:
            logger.debug(f"Failed tickers: {failed_tickers}")

        # Check if we have any successful data
        if not successful_tickers:
            raise FlowException(
                workflow="RSIWorkflow",
                step="process_data",
                message=f"Failed to calculate RSI for all {len(tickers)} tickers",
                user_message=(
                    "Unable to calculate RSI for any of the selected tickers. "
                    "Please check the ticker symbols and try again."
                ),
                context={"tickers": tickers, "failed_count": len(failed_tickers)},
            )

        return FlowResultEvent.success_result(
            data=rsi_data,
            base_date=base_date,
            metadata={"execution_time": execution_time},
            successful_items=successful_tickers,
            failed_items=failed_tickers,
        )


# FlowRunner wrapper function for graceful error handling
async def fetch_rsi_data(
    tickers: List[str], base_date: datetime, window: int = 14
) -> Dict[str, Any]:
    """
    Fetch RSI data using FlowRunner for graceful error handling.

    Args:
        tickers: List of ticker symbols to process
        base_date: Start date for RSI calculation
        window: RSI calculation window (default: 14)

    Returns:
        Dictionary containing RSI data or error information
    """
    workflow = RSIWorkflow()
    runner = FlowRunner[DataFrame](workflow)

    result_event = await runner.run(tickers=tickers, base_date=base_date, window=window)

    return {
        "data": result_event.data if result_event.success else None,
        "base_date": result_event.base_date,
        "successful_items": result_event.successful_items,
        "failed_items": result_event.failed_items,
        "execution_time": result_event.metadata.get("execution_time"),
        "success": result_event.success,
        "error_message": result_event.error_message,
    }
