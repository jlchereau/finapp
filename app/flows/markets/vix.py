"""
LlamaIndex workflow for VIX data collection.

This workflow handles fetching VIX volatility index data
using YahooHistoryProvider with FlowRunner architecture.
"""

from typing import Dict, Any
from datetime import datetime

import pandas as pd
from workflows import Workflow, step
from workflows.events import StartEvent

from app.providers.yahoo import create_yahoo_history_provider
from app.flows.base import FlowRunner, FlowResultEvent
from app.lib.logger import logger
from app.lib.exceptions import FlowException
from app.flows.cache import apply_flow_cache


class VIXWorkflow(Workflow):
    """
    Workflow that fetches VIX (volatility index) data.

    The VIX is a measure of market volatility, often called the "fear gauge".
    We use the ^VIX ticker from Yahoo Finance to get historical VIX data.

    This workflow follows Pattern 1 from flows.patterns.ipynb:
    - Single step that fetches and processes VIX data
    - Handles errors within the workflow logic
    - Returns SuccessStopEvent or ErrorStopEvent directly
    """

    def __init__(self):
        """Initialize workflow with Yahoo provider."""
        super().__init__()
        # Create provider
        self.yahoo_provider = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def fetch_and_process_vix_data(self, ev: StartEvent) -> FlowResultEvent:
        """
        Fetch and process VIX data from Yahoo Finance.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            SuccessStopEvent with processed VIX data or ErrorStopEvent on failure
        """
        base_date = ev.base_date

        logger.debug(f"VIXWorkflow: Fetching VIX data from {base_date}")

        # Fetch VIX data using provider - let exceptions bubble to FlowRunner
        vix_result = await self.yahoo_provider.get_data("^VIX")

        # Validate provider result - raise exceptions that FlowRunner will catch
        if not vix_result.success:
            error_msg = getattr(vix_result, "error_message", "Unknown VIX fetch error")
            raise FlowException(
                workflow="VIXWorkflow",
                step="fetch_and_process_vix_data",
                message=f"VIX data fetch failed: {error_msg}",
            )
        if not (
            hasattr(vix_result, "data") and isinstance(vix_result.data, pd.DataFrame)
        ):
            raise FlowException(
                workflow="VIXWorkflow",
                step="fetch_and_process_vix_data",
                message="Invalid VIX data format returned",
            )

        vix_data = vix_result.data
        if vix_data.empty:
            raise FlowException(
                workflow="VIXWorkflow",
                step="fetch_and_process_vix_data",
                message="No VIX data available",
            )

        logger.debug("VIXWorkflow: Processing VIX data")

        # Extract close prices from VIX data
        if "Close" in vix_data.columns:
            vix_close = vix_data["Close"].dropna()
        elif "Adj Close" in vix_data.columns:
            vix_close = vix_data["Adj Close"].dropna()
        else:
            raise FlowException(
                workflow="VIXWorkflow",
                step="fetch_and_process_vix_data",
                message="No Close price data available for VIX",
            )

        if vix_close.empty:
            raise FlowException(
                workflow="VIXWorkflow",
                step="fetch_and_process_vix_data",
                message="No VIX close price data available",
            )

        # Normalize timezone to ensure proper filtering
        if hasattr(vix_close.index, "tz") and vix_close.index.tz is not None:
            vix_close_naive = vix_close.tz_localize(None)
        else:
            vix_close_naive = vix_close

        # Calculate historical mean from full dataset
        historical_mean = float(vix_close_naive.mean())

        # Calculate 50-day moving average on full dataset
        moving_avg_50 = vix_close_naive.rolling(window=50, min_periods=1).mean()

        # Create complete result DataFrame
        result_df = pd.DataFrame(
            {
                "VIX": vix_close_naive,
                "VIX_MA50": moving_avg_50,
            },
            index=vix_close_naive.index,
        )

        # Filter by base_date for display purposes
        base_date_pd = pd.to_datetime(base_date.date())
        display_data = result_df.loc[result_df.index >= base_date_pd]

        if display_data.empty:
            logger.warning(f"No VIX data after base_date {base_date} for display")
            # Return empty result but don't error - this is just a display filter
            display_data = pd.DataFrame(columns=pd.Index(["VIX", "VIX_MA50"]))

        logger.info(
            f"VIX processing completed: {len(display_data)} data points "
            f"from {base_date}, historical mean: {historical_mean:.2f}"
        )

        return FlowResultEvent.success_result(
            data=display_data,
            base_date=base_date,
            metadata={
                "historical_mean": historical_mean,
                "latest_value": (
                    float(display_data["VIX"].iloc[-1])
                    if not display_data.empty
                    else None
                ),
                "data_points": len(display_data),
            },
        )


@apply_flow_cache
async def fetch_vix_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process VIX data using FlowRunner.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with VIX values
        - base_date: The base date used
        - historical_mean: Historical mean VIX value
        - latest_value: Most recent VIX value
        - data_points: Number of data points

    Raises:
        Exception: If workflow fails (for backward compatibility)
    """
    logger.info(f"Starting VIX data fetch from {base_date}")

    # Create workflow and FlowRunner
    workflow = VIXWorkflow()
    runner = FlowRunner[pd.DataFrame](workflow)

    # Run workflow using FlowRunner
    result_event = await runner.run(base_date=base_date)

    if result_event.success:
        logger.info("VIX workflow completed successfully")

        # Convert FlowResultEvent back to dictionary format for backward compatibility
        metadata = result_event.metadata or {}

        return {
            "data": result_event.data,
            "base_date": result_event.base_date,
            "historical_mean": metadata.get("historical_mean", 20.0),
            "latest_value": metadata.get("latest_value"),
            "data_points": metadata.get("data_points", 0),
        }

    else:
        # Handle error case - raise exception for backward compatibility
        error_msg = result_event.error_message or "Unknown error"

        logger.error(f"VIX workflow failed: {error_msg}")

        # Raise a standard exception for backward compatibility
        raise FlowException(
            workflow="VIXWorkflow",
            step="fetch_vix_data",
            message=f"VIX workflow failed: {error_msg}",
        )
