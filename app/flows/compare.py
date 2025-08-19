"""
LlamaIndex workflow for compare page data collection.

This workflow handles fetching historical price data for multiple tickers
using the YahooHistoryProvider with proper error handling and parallel processing.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from app.providers.yahoo import create_yahoo_history_provider
from app.lib.logger import logger
from app.lib.finance import calculate_volatility, calculate_rsi
from app.lib.exceptions import DataFetchException, WorkflowException
from app.flows.cache import apply_flow_cache


def _filter_data_by_date(data: pd.DataFrame, base_date: datetime) -> pd.DataFrame:
    """
    Common date filtering logic used across all workflows.

    Args:
        data: DataFrame with datetime index to filter
        base_date: Start date for filtering

    Returns:
        Filtered DataFrame containing data from base_date onwards
    """
    base_date_pd = pd.to_datetime(base_date.date())

    # Handle timezone-aware indexes
    if data.index.tz is not None:
        data_index = data.index.tz_localize(None)
        return data[data_index >= base_date_pd]
    else:
        return data[data.index >= base_date_pd]


def _process_ticker_result(
    ticker: str, raw_data: Dict[str, pd.DataFrame], failed_tickers: List[str]
) -> Optional[pd.DataFrame]:
    """
    Standard ticker validation and data extraction.

    Args:
        ticker: Ticker symbol to process
        raw_data: Dictionary mapping ticker to raw data
        failed_tickers: List to append failed tickers to

    Returns:
        DataFrame if successful, None if failed (ticker added to failed_tickers)
    """
    if ticker not in raw_data:
        failed_tickers.append(ticker)
        return None

    data = raw_data[ticker]
    if data.empty:
        failed_tickers.append(ticker)
        return None

    return data


def _get_close_prices(data: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    """
    Extract close prices from OHLCV data with fallback to Adj Close.

    Args:
        data: OHLCV DataFrame
        ticker: Ticker symbol for logging

    Returns:
        Close price Series or None if not available
    """
    if "Close" in data.columns:
        return data["Close"].dropna()
    elif "Adj Close" in data.columns:
        return data["Adj Close"].dropna()
    else:
        logger.warning(f"No Close price data for {ticker}")
        return None


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
        # Use period="max" to get comprehensive cached data,
        # then filter in normalize step
        tasks = {}

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
                filtered_data = _filter_data_by_date(data, base_date)

                logger.debug(
                    f"{ticker}: {len(data)} -> {len(filtered_data)} rows after "
                    f"filtering"
                )

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


@apply_flow_cache
async def fetch_raw_ticker_data(
    tickers: List[str], base_date: datetime
) -> Dict[str, pd.DataFrame]:
    """
    Shared function to fetch raw OHLCV data with in-memory caching.

    This function eliminates redundant data fetching across multiple chart types
    by caching the raw data in memory for 5 minutes.

    Args:
        tickers: List of ticker symbols to fetch
        base_date: Start date for historical data

    Returns:
        Dictionary mapping ticker -> raw OHLCV DataFrame

    Raises:
        Exception: If data fetching fails for all tickers
    """
    if not tickers:
        logger.debug("fetch_raw_ticker_data: No tickers provided")
        return {}

    try:
        logger.info(f"Fetching raw data for {len(tickers)} tickers")

        # Create provider to fetch raw OHLCV data
        yahoo_history = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

        # Fetch data for all tickers in parallel
        tasks = {}
        for ticker in tickers:
            task = yahoo_history.get_data(ticker)
            tasks[ticker] = task

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Process results
        successful_data = {}
        failed_tickers = []

        for ticker, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch raw data for {ticker}: {result}")
                failed_tickers.append(ticker)
                continue

            if not (hasattr(result, "success") and result.success):
                logger.warning(f"Provider failed for {ticker}")
                failed_tickers.append(ticker)
                continue

            try:
                data = result.data
                if data.empty:
                    logger.warning(f"Empty raw data for {ticker}")
                    failed_tickers.append(ticker)
                    continue

                # Store raw data without filtering - let individual
                # processors handle filtering
                successful_data[ticker] = data
                logger.debug(f"Cached raw data for {ticker}: {len(data)} rows")

            except Exception as e:
                logger.warning(f"Error processing raw data for {ticker}: {e}")
                failed_tickers.append(ticker)

        logger.info(
            f"Raw data fetch completed: {len(successful_data)} successful, "
            f"{len(failed_tickers)} failed"
        )

        if failed_tickers:
            logger.debug(f"Failed tickers: {failed_tickers}")

        # Check if we have any successful data
        if not successful_data:
            raise DataFetchException(
                provider="yahoo",
                query=f"tickers={tickers}",
                message=f"Failed to fetch data for all {len(tickers)} tickers",
                user_message=(
                    "Unable to fetch data for any of the selected tickers. "
                    "Please check the ticker symbols and try again."
                ),
                context={"tickers": tickers, "failed_count": len(failed_tickers)},
            )

        return successful_data

    except Exception as e:
        logger.error(f"Raw data fetch failed: {e}")
        # Re-raise as WorkflowException for better handling
        raise WorkflowException(
            workflow="fetch_raw_ticker_data",
            step="data_processing",
            message=f"Raw data fetch workflow failed: {e}",
            user_message=(
                "Data fetching failed due to a system error. Please try again."
            ),
            context={"tickers": tickers, "base_date": str(base_date)},
        ) from e


async def fetch_returns_data(tickers: List[str], base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and normalize returns data for multiple tickers using shared data source.

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
        logger.debug("fetch_returns_data: No tickers provided")
        return {
            "data": pd.DataFrame(),
            "successful_tickers": [],
            "failed_tickers": [],
            "base_date": base_date,
        }

    try:
        logger.info(f"Starting returns data processing for {len(tickers)} tickers")

        # Get raw data from shared cache
        raw_data = await fetch_raw_ticker_data(tickers, base_date)

        if not raw_data:
            logger.warning("No raw data available from shared cache")
            return {
                "data": pd.DataFrame(),
                "successful_tickers": [],
                "failed_tickers": tickers,
                "base_date": base_date,
            }

        # Process raw data into normalized returns
        normalized_data = pd.DataFrame()
        successful_tickers = []
        failed_tickers = []

        for ticker in tickers:
            try:
                # Use helper function for standard ticker processing
                data = _process_ticker_result(ticker, raw_data, failed_tickers)
                if data is None:
                    continue

                # Filter data from base_date
                # (returns need filtering first)
                filtered_data = _filter_data_by_date(data, base_date)

                if filtered_data.empty:
                    logger.warning(f"No data after {base_date} for {ticker}, skipping")
                    failed_tickers.append(ticker)
                    continue

                # Get close prices using helper function
                close_prices = _get_close_prices(filtered_data, ticker)
                if close_prices is None or close_prices.empty:
                    logger.warning(f"No valid close prices for {ticker}, skipping")
                    failed_tickers.append(ticker)
                    continue

                # Calculate percentage returns from first value (after filtering)
                first_price = close_prices.iloc[0]
                percentage_returns = ((close_prices / first_price) - 1) * 100

                # Add to normalized data
                normalized_data[ticker] = percentage_returns
                successful_tickers.append(ticker)

            except Exception as e:
                logger.warning(f"Error processing returns for {ticker}: {e}")
                failed_tickers.append(ticker)
                continue

        logger.info(
            f"Returns processing completed: {len(successful_tickers)} successful, "
            f"{len(failed_tickers)} failed"
        )

        # Check if we have any successful data
        if not successful_tickers:
            raise DataFetchException(
                provider="yahoo",
                query=f"returns_data for {tickers}",
                message=(
                    f"Failed to process returns data for all {len(tickers)} tickers"
                ),
                user_message=(
                    "Unable to calculate returns for any of the selected tickers. "
                    "Please check the ticker symbols and try again."
                ),
                context={"tickers": tickers, "failed_count": len(failed_tickers)},
            )

        return {
            "data": normalized_data,
            "successful_tickers": successful_tickers,
            "failed_tickers": failed_tickers,
            "base_date": base_date,
        }

    except Exception as e:
        logger.error(f"Returns data processing failed: {e}")
        # Re-raise as WorkflowException for better handling
        raise WorkflowException(
            workflow="fetch_returns_data",
            step="returns_processing",
            message=f"Returns data processing workflow failed: {e}",
            user_message=(
                "Returns calculation failed due to a system error. Please try again."
            ),
            context={"tickers": tickers, "base_date": str(base_date)},
        ) from e


async def fetch_volatility_data(
    tickers: List[str], base_date: datetime
) -> Dict[str, Any]:
    """
    Calculate volatility data for multiple tickers using shared data source.

    Args:
        tickers: List of ticker symbols to fetch
        base_date: Start date for historical data

    Returns:
        Dictionary containing volatility data for comparison charts
    """
    if not tickers:
        logger.debug("fetch_volatility_data: No tickers provided")
        return {
            "data": pd.DataFrame(),
            "successful_tickers": [],
            "failed_tickers": [],
            "base_date": base_date,
        }

    try:
        logger.info(f"Starting volatility data processing for {len(tickers)} tickers")

        # Get raw data from shared cache
        raw_data = await fetch_raw_ticker_data(tickers, base_date)

        if not raw_data:
            logger.warning("No raw data available from shared cache")
            return {
                "data": pd.DataFrame(),
                "successful_tickers": [],
                "failed_tickers": tickers,
                "base_date": base_date,
            }

        # Process raw data into volatility
        successful_tickers = []
        failed_tickers = []
        volatility_data = pd.DataFrame()

        for ticker in tickers:
            try:
                # Use helper function for standard ticker processing
                data = _process_ticker_result(ticker, raw_data, failed_tickers)
                if data is None:
                    continue

                # Calculate volatility on FULL dataset first
                # (avoid truncation)
                vol_data = calculate_volatility(data, window=30, annualize=True)

                if vol_data.empty:
                    logger.warning(f"No volatility calculated for {ticker}")
                    failed_tickers.append(ticker)
                    continue

                # Filter calculated volatility results to display period
                filtered_vol_data = _filter_data_by_date(vol_data, base_date)

                if filtered_vol_data.empty:
                    logger.warning(f"No volatility data after {base_date} for {ticker}")
                    failed_tickers.append(ticker)
                    continue

                # Store filtered volatility data
                volatility_data[ticker] = filtered_vol_data["Volatility"]
                successful_tickers.append(ticker)

            except Exception as e:
                logger.warning(f"Error calculating volatility for {ticker}: {e}")
                failed_tickers.append(ticker)

        logger.info(
            f"Volatility processing completed: {len(successful_tickers)} successful, "
            f"{len(failed_tickers)} failed"
        )

        return {
            "data": volatility_data,
            "successful_tickers": successful_tickers,
            "failed_tickers": failed_tickers,
            "base_date": base_date,
        }

    except Exception as e:
        logger.error(f"Volatility data processing failed: {e}")
        return {
            "data": pd.DataFrame(),
            "successful_tickers": [],
            "failed_tickers": tickers,
            "base_date": base_date,
            "error": str(e),
        }


async def fetch_volume_data(tickers: List[str], base_date: datetime) -> Dict[str, Any]:
    """
    Extract volume data for multiple tickers using shared data source.

    Args:
        tickers: List of ticker symbols to fetch
        base_date: Start date for historical data

    Returns:
        Dictionary containing volume data for comparison charts
    """
    if not tickers:
        logger.debug("fetch_volume_data: No tickers provided")
        return {
            "data": pd.DataFrame(),
            "successful_tickers": [],
            "failed_tickers": [],
            "base_date": base_date,
        }

    try:
        logger.info(f"Starting volume data processing for {len(tickers)} tickers")

        # Get raw data from shared cache
        raw_data = await fetch_raw_ticker_data(tickers, base_date)

        if not raw_data:
            logger.warning("No raw data available from shared cache")
            return {
                "data": pd.DataFrame(),
                "successful_tickers": [],
                "failed_tickers": tickers,
                "base_date": base_date,
            }

        # Process raw data into volume
        successful_tickers = []
        failed_tickers = []
        volume_data = pd.DataFrame()

        for ticker in tickers:
            try:
                # Use helper function for standard ticker processing
                data = _process_ticker_result(ticker, raw_data, failed_tickers)
                if data is None:
                    continue

                # Filter data from base_date
                # (volume is direct extraction)
                filtered_data = _filter_data_by_date(data, base_date)

                if filtered_data.empty:
                    logger.warning(f"No data after {base_date} for {ticker}")
                    failed_tickers.append(ticker)
                    continue

                # Extract volume data
                if "Volume" in filtered_data.columns:
                    volume_data[ticker] = filtered_data["Volume"]
                    successful_tickers.append(ticker)
                else:
                    logger.warning(f"No Volume data for {ticker}")
                    failed_tickers.append(ticker)

            except Exception as e:
                logger.warning(f"Error extracting volume for {ticker}: {e}")
                failed_tickers.append(ticker)

        logger.info(
            f"Volume processing completed: {len(successful_tickers)} successful, "
            f"{len(failed_tickers)} failed"
        )

        return {
            "data": volume_data,
            "successful_tickers": successful_tickers,
            "failed_tickers": failed_tickers,
            "base_date": base_date,
        }

    except Exception as e:
        logger.error(f"Volume data processing failed: {e}")
        return {
            "data": pd.DataFrame(),
            "successful_tickers": [],
            "failed_tickers": tickers,
            "base_date": base_date,
            "error": str(e),
        }


async def fetch_rsi_data(tickers: List[str], base_date: datetime) -> Dict[str, Any]:
    """
    Calculate RSI data for multiple tickers using shared data source.

    Args:
        tickers: List of ticker symbols to fetch
        base_date: Start date for historical data

    Returns:
        Dictionary containing RSI data for comparison charts
    """
    if not tickers:
        logger.debug("fetch_rsi_data: No tickers provided")
        return {
            "data": pd.DataFrame(),
            "successful_tickers": [],
            "failed_tickers": [],
            "base_date": base_date,
        }

    try:
        logger.info(f"Starting RSI data processing for {len(tickers)} tickers")

        # Get raw data from shared cache
        raw_data = await fetch_raw_ticker_data(tickers, base_date)

        if not raw_data:
            logger.warning("No raw data available from shared cache")
            return {
                "data": pd.DataFrame(),
                "successful_tickers": [],
                "failed_tickers": tickers,
                "base_date": base_date,
            }

        # Process raw data into RSI
        successful_tickers = []
        failed_tickers = []
        rsi_data = pd.DataFrame()

        for ticker in tickers:
            try:
                # Use helper function for standard ticker processing
                data = _process_ticker_result(ticker, raw_data, failed_tickers)
                if data is None:
                    continue

                # Calculate RSI on FULL dataset first (avoid truncation)
                rsi_result = calculate_rsi(data, window=14)

                if rsi_result.empty:
                    logger.warning(f"No RSI calculated for {ticker}")
                    failed_tickers.append(ticker)
                    continue

                # Filter calculated RSI results to display period
                filtered_rsi_data = _filter_data_by_date(rsi_result, base_date)

                if filtered_rsi_data.empty:
                    logger.warning(f"No RSI data after {base_date} for {ticker}")
                    failed_tickers.append(ticker)
                    continue

                # Store filtered RSI data
                rsi_data[ticker] = filtered_rsi_data["RSI"]
                successful_tickers.append(ticker)

            except Exception as e:
                logger.warning(f"Error calculating RSI for {ticker}: {e}")
                failed_tickers.append(ticker)

        logger.info(
            f"RSI processing completed: {len(successful_tickers)} successful, "
            f"{len(failed_tickers)} failed"
        )

        return {
            "data": rsi_data,
            "successful_tickers": successful_tickers,
            "failed_tickers": failed_tickers,
            "base_date": base_date,
        }

    except Exception as e:
        logger.error(f"RSI data processing failed: {e}")
        return {
            "data": pd.DataFrame(),
            "successful_tickers": [],
            "failed_tickers": tickers,
            "base_date": base_date,
            "error": str(e),
        }
