"""
MetricsWorkflow for fundamental data collection and metrics comparison.

This workflow handles fetching fundamental data for multiple tickers
using the YahooInfoProvider with proper LlamaIndex event-driven patterns.
Used primarily by the metrics tab for valuation and company comparison.
"""

import time
from typing import Dict
from datetime import datetime

from pandas import DataFrame
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from app.providers.yahoo import create_yahoo_info_provider, YahooInfoModel
from app.flows.base import FlowResult
from app.lib.logger import logger
from app.lib.exceptions import FlowException


class DispatchEvent(Event):
    """Event to coordinate parallel ticker metrics fetching."""


class FetchMetricsEvent(Event):
    """Event to trigger metrics collection for a specific ticker."""

    ticker: str


class MetricsResponseEvent(Event):
    """Event to return collected ticker metrics."""

    ticker: str
    data: YahooInfoModel | None
    success: bool
    error_message: str | None = None


def _convert_metrics_to_dataframe(metrics_data: Dict[str, YahooInfoModel]) -> DataFrame:
    """
    Convert metrics data to DataFrame for comparison display.

    Args:
        metrics_data: Dictionary mapping ticker to YahooInfoModel

    Returns:
        DataFrame with tickers as columns and metrics as rows
    """
    if not metrics_data:
        return DataFrame()

    # Extract metrics from all tickers
    metrics_dict = {}

    for ticker, model in metrics_data.items():
        metrics_dict[ticker] = {
            "Company Name": model.company_name,
            "Price": model.price,
            "Change": model.change,
            "Change %": model.percent_change,
            "Volume": model.volume,
            "Market Cap": model.market_cap,
            "P/E Ratio": model.pe_ratio,
            "Dividend Yield": model.dividend_yield,
            "Beta": model.beta,
            "52W High": model.week_52_high,
            "52W Low": model.week_52_low,
            "Currency": model.currency,
            "Exchange": model.exchange,
            "Sector": model.sector,
            "Industry": model.industry,
        }

    # Convert to DataFrame with metrics as rows, tickers as columns
    df = DataFrame.from_dict(metrics_dict, orient="index").T
    return df


class MetricsWorkflow(Workflow):
    """
    LlamaIndex workflow for fundamental metrics data collection and comparison.

    This workflow:
    - Fetches fundamental data for multiple tickers in parallel
    - Collects valuation metrics, market data, and company information
    - Handles errors gracefully (partial success)
    - Returns data in format compatible with comparison tables
    - Follows LlamaIndex Pattern 3 for parallel execution with single provider
    """

    def __init__(self):
        """Initialize workflow with Yahoo info provider."""
        super().__init__()
        # Create provider for fundamental data
        self.yahoo_info = create_yahoo_info_provider(timeout=30.0, retries=2)

    @step
    async def dispatch(
        self, ctx: Context, ev: StartEvent
    ) -> FetchMetricsEvent | DispatchEvent:
        """
        Entry step that coordinates parallel ticker metrics fetching.

        Args:
            ev.tickers: List of ticker symbols
            ev.base_date: Not used for metrics but kept for consistency

        Returns:
            DispatchEvent to coordinate collection
        """
        tickers = ev.tickers

        logger.debug(
            f"MetricsWorkflow: Dispatching metrics fetch for {len(tickers)} tickers"
        )

        # Store metadata for later steps
        await ctx.store.set("tickers", tickers)
        await ctx.store.set("num_to_collect", len(tickers))
        await ctx.store.set("start_time", time.time())

        # Send parallel fetch events for each ticker
        for ticker in tickers:
            ctx.send_event(FetchMetricsEvent(ticker=ticker))

        return DispatchEvent()

    @step(num_workers=3)  # Limit concurrent API calls
    async def fetch_ticker_metrics(self, ev: FetchMetricsEvent) -> MetricsResponseEvent:
        """
        Fetch fundamental metrics for a single ticker.

        Args:
            ev.ticker: Ticker symbol to fetch

        Returns:
            MetricsResponseEvent with fetched data or error
        """
        ticker = ev.ticker

        try:
            # Fetch fundamental data
            provider_result = await self.yahoo_info.get_data(ticker)

            if not provider_result.success:
                return MetricsResponseEvent(
                    ticker=ticker,
                    data=None,
                    success=False,
                    error_message=f"Provider failed: {provider_result.error_message}",
                )

            metrics = provider_result.data
            if not isinstance(metrics, YahooInfoModel):
                return MetricsResponseEvent(
                    ticker=ticker,
                    data=None,
                    success=False,
                    error_message="Invalid data format returned",
                )

            logger.debug(f"Fetched metrics for {ticker}: {metrics.company_name}")

            return MetricsResponseEvent(ticker=ticker, data=metrics, success=True)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.warning(f"Error fetching metrics for {ticker}: {e}")
            return MetricsResponseEvent(
                ticker=ticker, data=None, success=False, error_message=str(e)
            )

    @step
    async def compile_metrics(
        self, ctx: Context, ev: DispatchEvent | MetricsResponseEvent
    ) -> StopEvent | None:
        """
        Collect all ticker metrics and compile into comparison format.

        Args:
            ev: DispatchEvent (initial) or MetricsResponseEvent (data)

        Returns:
            StopEvent with FlowResult containing metrics comparison data
        """
        # Continue waiting after receiving DispatchEvent
        if isinstance(ev, DispatchEvent):
            return None

        # Wait for all ticker metrics responses
        num_to_collect = await ctx.store.get("num_to_collect")
        events = ctx.collect_events(ev, [MetricsResponseEvent] * num_to_collect)
        if not events:
            return None

        # All ticker metrics received - process compilation
        tickers = await ctx.store.get("tickers")
        start_time = await ctx.store.get("start_time")

        logger.debug(f"MetricsWorkflow: Compiling metrics for {len(tickers)} tickers")

        # Process each ticker's metrics
        metrics_data = {}
        successful_tickers = []
        failed_tickers = []

        for event in events:
            ticker = event.ticker

            if not event.success:
                failed_tickers.append(ticker)
                logger.warning(
                    f"Skipping {ticker} due to fetch failure: {event.error_message}"
                )
                continue

            try:
                metrics = event.data
                if metrics is None:
                    failed_tickers.append(ticker)
                    logger.warning(f"No metrics data for {ticker}")
                    continue

                # Store metrics data
                metrics_data[ticker] = metrics
                successful_tickers.append(ticker)

            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.warning(f"Error processing metrics for {ticker}: {e}")
                failed_tickers.append(ticker)
                continue

        # Convert to DataFrame for comparison display
        try:
            comparison_df = _convert_metrics_to_dataframe(metrics_data)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(f"Error converting metrics to DataFrame: {e}")
            comparison_df = DataFrame()

        # Log summary
        execution_time = time.time() - start_time
        logger.info(
            f"MetricsWorkflow completed: {len(successful_tickers)} successful, "
            f"{len(failed_tickers)} failed tickers"
        )

        if failed_tickers:
            logger.debug(f"Failed tickers: {failed_tickers}")

        # Check if we have any successful data
        if not successful_tickers:
            raise FlowException(
                workflow="MetricsWorkflow",
                step="compile_metrics",
                message=f"Failed to fetch metrics for all {len(tickers)} tickers",
                user_message=(
                    "Unable to fetch fundamental data for any of the selected tickers. "
                    "Please check the ticker symbols and try again."
                ),
                context={"tickers": tickers, "failed_count": len(failed_tickers)},
            )

        # Return both raw metrics data and comparison DataFrame
        result_data = {"comparison_df": comparison_df, "raw_metrics": metrics_data}

        return StopEvent(
            result=FlowResult.success_result(
                data=result_data,
                base_date=datetime.now(),  # Metrics are current, not historical
                execution_time=execution_time,
                successful_items=successful_tickers,
                failed_items=failed_tickers,
            )
        )
