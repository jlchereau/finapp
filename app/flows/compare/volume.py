"""
VolumeWorkflow for extracting and processing volume data from time series data.

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
from app.lib.finance import calculate_volume_metrics
from app.lib.periods import ensure_minimum_data_points
from .time_series import fetch_time_series_data


class VolumeStartEvent(Event):
    """Start event for volume processing."""

    tickers: List[str]
    base_date: datetime
    ma_window: int = 20


class FetchDataEvent(Event):
    """Event to trigger data fetching for volume processing."""

    tickers: List[str]
    base_date: datetime
    ma_window: int


class VolumeWorkflow(Workflow):
    """
    LlamaIndex workflow for extracting and processing volume data.

    This workflow:
    - Step 1: Fetches time series data using cached fetch_time_series_data()
    - Step 2: Delegates volume processing to finance.calculate_volume_metrics()
    - Follows proper event-driven patterns
    - Maintains consistent architecture
    """

    @step
    async def fetch_data(self, ctx: Context, ev: StartEvent) -> FetchDataEvent:
        """
        Step 1: Fetch time series data for volume processing.

        Args:
            ev.tickers: List of ticker symbols
            ev.base_date: Start date for calculation
            ev.ma_window: Moving average window (default: 20)

        Returns:
            FetchDataEvent to proceed to processing
        """
        tickers = ev.tickers
        base_date = ev.base_date
        ma_window = getattr(ev, "ma_window", 20)

        logger.debug(f"VolumeWorkflow: Fetching data for {len(tickers)} tickers")

        # Store metadata for later steps
        await ctx.store.set("tickers", tickers)
        await ctx.store.set("base_date", base_date)
        await ctx.store.set("ma_window", ma_window)
        await ctx.store.set("start_time", time.time())

        return FetchDataEvent(tickers=tickers, base_date=base_date, ma_window=ma_window)

    @step
    async def process_data(self, ctx: Context, ev: FetchDataEvent) -> FlowResultEvent:
        """
        Step 2: Process raw data into volume metrics.

        Args:
            ev.tickers: List of tickers to process
            ev.base_date: Base date for calculation
            ev.ma_window: Moving average window

        Returns:
            FlowResultEvent containing volume data
        """
        tickers = ev.tickers
        base_date = ev.base_date
        ma_window = ev.ma_window
        start_time = await ctx.store.get("start_time")

        logger.info(f"VolumeWorkflow: Processing volume for {len(tickers)} tickers")

        # Get cached time series data
        raw_data = await fetch_time_series_data(tickers, base_date)

        if not raw_data:
            raise FlowException(
                workflow="VolumeWorkflow",
                step="process_data",
                message="No time series data available from cache",
                user_message=(
                    "Unable to fetch data for any of the selected tickers. "
                    "Please check the ticker symbols and try again."
                ),
                context={"tickers": tickers, "base_date": str(base_date)},
            )

        # Process each ticker using finance function
        volume_data = DataFrame()
        successful_tickers = []
        failed_tickers = []

        for ticker in tickers:
            try:
                if ticker not in raw_data:
                    failed_tickers.append(ticker)
                    logger.warning(f"VolumeWorkflow: No data for {ticker}")
                    continue

                ticker_data = raw_data[ticker]
                if ticker_data.empty:
                    failed_tickers.append(ticker)
                    logger.warning(f"VolumeWorkflow: Empty data for {ticker}")
                    continue

                # Check for Volume column
                if "Volume" not in ticker_data.columns:
                    failed_tickers.append(ticker)
                    logger.warning(f"VolumeWorkflow: No Volume column for {ticker}")
                    continue

                # Calculate volume metrics on full historical data
                vol_result = calculate_volume_metrics(ticker_data, ma_window=ma_window)

                if vol_result.empty:
                    failed_tickers.append(ticker)
                    logger.warning(
                        f"VolumeWorkflow: No volume metrics calculated for {ticker}"
                    )
                    continue

                # Filter calculated volume metrics to user's selected period
                filtered_volume = ensure_minimum_data_points(
                    data=vol_result,
                    base_date=base_date,
                    min_points=2,
                )

                if filtered_volume.empty:
                    failed_tickers.append(ticker)
                    logger.warning(
                        f"VolumeWorkflow: No data after period filtering for {ticker}"
                    )
                    continue

                # Extract just the Volume column for comparison charts
                if "Volume" in filtered_volume.columns:
                    volume_data[ticker] = filtered_volume["Volume"]
                    successful_tickers.append(ticker)
                    logger.debug(
                        f"VolumeWorkflow: Calculated and filtered volume for {ticker}"
                    )
                else:
                    failed_tickers.append(ticker)
                    logger.warning(f"VolumeWorkflow: No Volume in result for {ticker}")

            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.warning(f"VolumeWorkflow: Error processing {ticker}: {e}")
                failed_tickers.append(ticker)
                continue

        # Log summary
        execution_time = time.time() - start_time
        logger.info(
            f"VolumeWorkflow completed: {len(successful_tickers)} successful, "
            f"{len(failed_tickers)} failed tickers"
        )

        if failed_tickers:
            logger.debug(f"Failed tickers: {failed_tickers}")

        # Check if we have any successful data
        if not successful_tickers:
            raise FlowException(
                workflow="VolumeWorkflow",
                step="process_data",
                message=(f"Failed to process volume for all {len(tickers)} tickers"),
                user_message=(
                    "Unable to process volume data for any of the selected "
                    "tickers. Please check the ticker symbols and try again."
                ),
                context={"tickers": tickers, "failed_count": len(failed_tickers)},
            )

        return FlowResultEvent.success_result(
            data=volume_data,
            base_date=base_date,
            metadata={"execution_time": execution_time},
            successful_items=successful_tickers,
            failed_items=failed_tickers,
        )


# FlowRunner wrapper function for graceful error handling
async def fetch_volume_data(
    tickers: List[str], base_date: datetime, ma_window: int = 20
) -> Dict[str, Any]:
    """
    Fetch volume data using FlowRunner for graceful error handling.

    Args:
        tickers: List of ticker symbols to process
        base_date: Start date for volume calculation
        ma_window: Moving average window size (default: 20)

    Returns:
        Dictionary containing volume data or error information
    """
    workflow = VolumeWorkflow()
    runner = FlowRunner[DataFrame](workflow)

    result_event = await runner.run(tickers=tickers, base_date=base_date, ma_window=ma_window)

    return {
        "data": result_event.data if result_event.success else None,
        "base_date": result_event.base_date,
        "successful_items": result_event.successful_items,
        "failed_items": result_event.failed_items,
        "execution_time": result_event.metadata.get("execution_time"),
        "success": result_event.success,
        "error_message": result_event.error_message,
    }
