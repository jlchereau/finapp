"""
LlamaIndex workflow for compare page data collection.

This workflow handles fetching historical price data for multiple tickers
using the YahooHistoryProvider with proper error handling and parallel processing.
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime

import pandas as pd
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from ..models.yahoo import create_yahoo_history_provider
from ..lib.logger import logger


class DataFetchedEvent(Event):
    """Event emitted when all ticker data is fetched."""

    tickers: List[str]
    base_date: datetime
    results: Dict[str, Any]


class CompareDataWorkflow(Workflow):
    """
    Workflow that fetches historical price data for multiple tickers and
    normalizes them for comparison charts.

    This workflow:
    - Fetches data for multiple tickers in parallel using YahooHistoryProvider
    - Normalizes data to percentage returns from base date
    - Handles errors gracefully (skips failed tickers)
    - Returns data in format compatible with plotly charts
    """

    def __init__(self):
        """Initialize workflow with Yahoo history provider."""
        super().__init__()
        # Create provider with max period for comprehensive caching
        self.yahoo_history = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def fetch_ticker_data(self, ev: StartEvent) -> DataFetchedEvent:
        """
        Fetch historical data for all tickers in parallel.

        Args:
            ev.tickers: List of ticker symbols
            ev.base_date: Start date for data fetching

        Returns:
            DataFetchedEvent with fetched data results
        """
        tickers = ev.tickers
        base_date = ev.base_date

        logger.debug(
            f"CompareDataWorkflow: Fetching data for {len(tickers)} tickers "
            f"from {base_date}"
        )

        # Create tasks for parallel execution
        # Use period="max" to get comprehensive cached data, then filter in normalize step
        tasks = {}

        logger.debug(f"Fetching max period data, will filter from {base_date}")

        for ticker in tickers:
            # Fetch max period data for better caching
            task = self.yahoo_history.get_data(ticker)
            tasks[ticker] = task

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Map results back to tickers
        ticker_results = {}
        for ticker, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch data for {ticker}: {result}")
                ticker_results[ticker] = {"success": False, "error": str(result)}
            else:
                # result is a ProviderResult from the provider
                if hasattr(result, "success") and result.success:
                    ticker_results[ticker] = {
                        "success": True,
                        "data": result.data,
                        "execution_time": getattr(result, "execution_time", None),
                    }
                    logger.debug(
                        f"Successfully fetched {len(result.data)} rows for {ticker}"
                    )
                else:
                    error_msg = getattr(
                        result, "error_message", "Unknown provider error"
                    )
                    logger.warning(f"Provider failed for {ticker}: {error_msg}")
                    ticker_results[ticker] = {"success": False, "error": error_msg}

        return DataFetchedEvent(
            tickers=tickers, base_date=base_date, results=ticker_results
        )

    @step
    async def normalize_data(self, ev: DataFetchedEvent) -> StopEvent:
        """
        Normalize the fetched data to percentage returns for comparison.

        This step:
        - Processes successful ticker data
        - Normalizes prices to percentage returns from base date
        - Creates a combined DataFrame for plotting
        - Skips failed tickers gracefully

        Args:
            ev: DataFetchedEvent with raw ticker data

        Returns:
            StopEvent with normalized DataFrame ready for plotly
        """
        tickers = ev.tickers
        base_date = ev.base_date
        results = ev.results

        logger.debug(
            f"CompareDataWorkflow: Normalizing data for {len(tickers)} tickers"
        )

        # Create normalized DataFrame
        normalized_data = pd.DataFrame()
        successful_tickers = []
        failed_tickers = []

        for ticker in tickers:
            result = results.get(ticker, {})

            if not result.get("success", False):
                failed_tickers.append(ticker)
                logger.warning(f"Skipping {ticker} due to fetch failure")
                continue

            try:
                data = result["data"]

                if data.empty:
                    logger.warning(f"Empty data for {ticker}, skipping")
                    failed_tickers.append(ticker)
                    continue

                # Filter data to start from base_date
                logger.debug(f"Raw data for {ticker}: {len(data)} rows, index range: {data.index.min()} to {data.index.max()}")
                logger.debug(f"Base date for filtering: {base_date}")
                
                # Convert base_date to pandas datetime for comparison
                # Handle timezone-aware indexes by making base_date timezone-naive
                base_date_pd = pd.to_datetime(base_date.date())
                
                # Make sure both dates are timezone-naive for comparison
                if data.index.tz is not None:
                    data_index = data.index.tz_localize(None)
                    filtered_data = data[data_index >= base_date_pd]
                else:
                    filtered_data = data[data.index >= base_date_pd]
                
                logger.debug(f"Filtered data for {ticker}: {len(filtered_data)} rows")

                if filtered_data.empty:
                    logger.warning(f"No data after {base_date} for {ticker}, skipping")
                    failed_tickers.append(ticker)
                    continue

                # Get close prices and normalize to percentage returns
                if "Close" in filtered_data.columns:
                    close_prices = filtered_data["Close"].dropna()
                elif "Adj Close" in filtered_data.columns:
                    close_prices = filtered_data["Adj Close"].dropna()
                else:
                    logger.warning(f"No Close price data for {ticker}, skipping")
                    failed_tickers.append(ticker)
                    continue

                if close_prices.empty:
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

                logger.debug(
                    f"Normalized {ticker}: {len(percentage_returns)} data points"
                )

            except (KeyError, IndexError, AttributeError) as e:
                logger.warning(f"Error normalizing data for {ticker}: {e}")
                failed_tickers.append(ticker)
                continue

        # Log summary
        logger.info(
            f"CompareDataWorkflow completed: {len(successful_tickers)} successful, "
            f"{len(failed_tickers)} failed tickers"
        )

        if failed_tickers:
            logger.debug(f"Failed tickers: {failed_tickers}")

        return StopEvent(
            result={
                "data": normalized_data,
                "successful_tickers": successful_tickers,
                "failed_tickers": failed_tickers,
                "base_date": base_date,
            }
        )


async def fetch_compare_data(tickers: List[str], base_date: datetime) -> Dict[str, Any]:
    """
    Main function to fetch and normalize comparison data for multiple tickers.

    This function should be called from Reflex background events.

    Args:
        tickers: List of ticker symbols to fetch
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with normalized percentage returns
        - successful_tickers: List of tickers that were processed successfully
        - failed_tickers: List of tickers that failed to process
        - base_date: The base date used for normalization
    """
    if not tickers:
        logger.debug("fetch_compare_data: No tickers provided")
        return {
            "data": pd.DataFrame(),
            "successful_tickers": [],
            "failed_tickers": [],
            "base_date": base_date,
        }

    try:
        logger.info(f"Starting compare data workflow for {len(tickers)} tickers")

        workflow = CompareDataWorkflow()
        # Add tickers and base_date as attributes to the StartEvent
        start_event = StartEvent()
        start_event.tickers = tickers
        start_event.base_date = base_date
        handler = workflow.run(start_event=start_event)
        result = await handler

        # Extract result data
        if hasattr(result, "result"):
            return result.result
        else:
            return result

    except Exception as e:
        logger.error(f"Compare workflow execution failed: {e}")
        return {
            "data": pd.DataFrame(),
            "successful_tickers": [],
            "failed_tickers": tickers,
            "base_date": base_date,
            "error": str(e),
        }
