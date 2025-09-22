"""
ReturnsWorkflow for calculating normalized percentage returns from time series data.

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
from app.lib.finance import calculate_returns
from app.lib.periods import ensure_minimum_data_points
from .time_series import fetch_time_series_data


class ReturnsStartEvent(Event):
    """Start event for returns calculation."""

    tickers: List[str]
    base_date: datetime


class FetchDataEvent(Event):
    """Event to trigger data fetching for returns calculation."""

    tickers: List[str]
    base_date: datetime


class ReturnsWorkflow(Workflow):
    """
    LlamaIndex workflow for calculating normalized percentage returns.

    This workflow:
    - Step 1: Fetches time series data using cached fetch_time_series_data()
    - Step 2: Delegates returns calculation to finance.calculate_returns()
    - Follows proper event-driven patterns
    - Maintains consistent architecture
    """

    @step
    async def fetch_data(self, ctx: Context, ev: StartEvent) -> FetchDataEvent:
        """
        Step 1: Fetch time series data for returns calculation.

        Args:
            ev.tickers: List of ticker symbols
            ev.base_date: Start date for calculation

        Returns:
            FetchDataEvent to proceed to processing
        """
        tickers = ev.tickers
        base_date = ev.base_date

        logger.debug(f"ReturnsWorkflow: Fetching data for {len(tickers)} tickers")

        # Store metadata for later steps
        await ctx.store.set("tickers", tickers)
        await ctx.store.set("base_date", base_date)
        await ctx.store.set("start_time", time.time())

        return FetchDataEvent(tickers=tickers, base_date=base_date)

    @step
    async def process_data(self, ctx: Context, ev: FetchDataEvent) -> FlowResultEvent:
        """
        Step 2: Process raw data into normalized returns.

        Args:
            ev.tickers: List of tickers to process
            ev.base_date: Base date for calculation

        Returns:
            StopEvent with FlowResult containing returns data
        """
        tickers = ev.tickers
        base_date = ev.base_date
        start_time = await ctx.store.get("start_time")

        logger.info(f"ReturnsWorkflow: Processing returns for {len(tickers)} tickers")

        # Get cached time series data
        raw_data = await fetch_time_series_data(tickers, base_date)

        if not raw_data:
            raise FlowException(
                workflow="ReturnsWorkflow",
                step="process_data",
                message="No time series data available from cache",
                user_message=(
                    "Unable to fetch data for any of the selected tickers. "
                    "Please check the ticker symbols and try again."
                ),
                context={"tickers": tickers, "base_date": str(base_date)},
            )

        # Process each ticker using finance function
        normalized_data = DataFrame()
        successful_tickers = []
        failed_tickers = []

        for ticker in tickers:
            try:
                if ticker not in raw_data:
                    failed_tickers.append(ticker)
                    logger.warning(f"ReturnsWorkflow: No data for {ticker}")
                    continue

                ticker_data = raw_data[ticker]
                if ticker_data.empty:
                    failed_tickers.append(ticker)
                    logger.warning(f"ReturnsWorkflow: Empty data for {ticker}")
                    continue

                # Calculate returns on full historical data (no base_date filtering)
                returns_result = calculate_returns(ticker_data, base_date=None)

                if returns_result.empty:
                    failed_tickers.append(ticker)
                    logger.warning(
                        f"ReturnsWorkflow: No returns calculated for {ticker}"
                    )
                    continue

                # Filter calculated returns to user's selected period
                filtered_returns = ensure_minimum_data_points(
                    data=returns_result,
                    base_date=base_date,
                    min_points=2,
                )

                if filtered_returns.empty:
                    failed_tickers.append(ticker)
                    logger.warning(
                        f"ReturnsWorkflow: No data after period filtering for {ticker}"
                    )
                    continue

                # Add to normalized data (returns function returns DataFrame
                # with 'Returns' column)
                normalized_data[ticker] = filtered_returns["Returns"]
                successful_tickers.append(ticker)

                logger.debug(
                    f"ReturnsWorkflow: Calculated and filtered returns for {ticker}"
                )

            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.warning(f"ReturnsWorkflow: Error processing {ticker}: {e}")
                failed_tickers.append(ticker)
                continue

        # Log summary
        execution_time = time.time() - start_time
        logger.info(
            f"ReturnsWorkflow completed: {len(successful_tickers)} successful, "
            f"{len(failed_tickers)} failed tickers"
        )

        if failed_tickers:
            logger.debug(f"Failed tickers: {failed_tickers}")

        # Check if we have any successful data
        if not successful_tickers:
            raise FlowException(
                workflow="ReturnsWorkflow",
                step="process_data",
                message=(f"Failed to calculate returns for all {len(tickers)} tickers"),
                user_message=(
                    "Unable to calculate returns for any of the selected "
                    "tickers. Please check the ticker symbols and try again."
                ),
                context={"tickers": tickers, "failed_count": len(failed_tickers)},
            )

        return FlowResultEvent.success_result(
            data=normalized_data,
            base_date=base_date,
            metadata={"execution_time": execution_time},
            successful_items=successful_tickers,
            failed_items=failed_tickers,
        )


# FlowRunner wrapper function for graceful error handling
async def fetch_returns_data(tickers: List[str], base_date: datetime) -> Dict[str, Any]:
    """
    Fetch returns data using FlowRunner for graceful error handling.

    Args:
        tickers: List of ticker symbols to process
        base_date: Start date for percentage return calculation

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with returns data or None if failed
        - base_date: The base date used
        - successful_items: List of successfully processed tickers
        - failed_items: List of failed tickers
        - execution_time: Time taken for execution
    """
    workflow = ReturnsWorkflow()
    runner = FlowRunner[DataFrame](workflow)

    result_event = await runner.run(tickers=tickers, base_date=base_date)

    return {
        "data": result_event.data if result_event.success else None,
        "base_date": result_event.base_date,
        "successful_items": result_event.successful_items,
        "failed_items": result_event.failed_items,
        "execution_time": result_event.metadata.get("execution_time"),
        "success": result_event.success,
        "error_message": result_event.error_message,
    }
