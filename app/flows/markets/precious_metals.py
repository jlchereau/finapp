"""
LlamaIndex workflow for precious metals (Gold Futures) data collection.

This workflow handles fetching gold futures data from Yahoo Finance
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


class PreciousMetalsWorkflow(Workflow):
    """
    Workflow that fetches precious metals (Gold Futures) data.

    This workflow fetches gold futures data from Yahoo Finance using the GC=F ticker
    (COMEX Gold Futures). Gold is a key commodity and safe-haven asset.

    The workflow:
    - Fetches daily gold futures data from Yahoo Finance
    - Calculates 50-day and 200-day moving averages for trend analysis
    - Applies base_date filtering for display

    This workflow follows Pattern 1 from flows.patterns.ipynb:
    - Single step that fetches and processes gold data
    - Handles errors within the workflow logic
    - Returns FlowResultEvent directly
    """

    def __init__(self):
        """Initialize workflow with Yahoo provider."""
        super().__init__()
        # Create provider
        self.yahoo_provider = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def fetch_and_process_gold_data(self, ev: StartEvent) -> FlowResultEvent:
        """
        Fetch and process gold futures data from Yahoo Finance.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            FlowResultEvent with processed gold data and statistics
        """
        base_date = ev.base_date

        logger.debug(f"PreciousMetalsWorkflow: Fetching gold data from {base_date}")

        # Fetch Gold Futures data (COMEX) - let exceptions bubble to FlowRunner
        gold_result = await self.yahoo_provider.get_data("GC=F")

        # Validate provider result - raise exceptions that FlowRunner will catch
        if not gold_result.success:
            error_msg = getattr(
                gold_result, "error_message", "Unknown gold fetch error"
            )
            raise FlowException(
                workflow="PreciousMetalsWorkflow",
                step="fetch_and_process_gold_data",
                message=f"Gold data fetch failed: {error_msg}",
            )
        if not (
            hasattr(gold_result, "data") and isinstance(gold_result.data, pd.DataFrame)
        ):
            raise FlowException(
                workflow="PreciousMetalsWorkflow",
                step="fetch_and_process_gold_data",
                message="Invalid gold data format returned",
            )

        gold_data = gold_result.data
        if gold_data.empty:
            raise FlowException(
                workflow="PreciousMetalsWorkflow",
                step="fetch_and_process_gold_data",
                message="No gold data available",
            )

        logger.debug("PreciousMetalsWorkflow: Processing gold data")

        # Extract close prices from gold data
        if "Close" in gold_data.columns:
            gold_close = gold_data["Close"].dropna()
        elif "Adj Close" in gold_data.columns:
            gold_close = gold_data["Adj Close"].dropna()
        else:
            raise FlowException(
                workflow="PreciousMetalsWorkflow",
                step="fetch_and_process_gold_data",
                message="No Close price data available for gold",
            )

        if gold_close.empty:
            raise FlowException(
                workflow="PreciousMetalsWorkflow",
                step="fetch_and_process_gold_data",
                message="No gold close price data available",
            )

        # Normalize timezone to ensure proper filtering
        if hasattr(gold_close.index, "tz") and gold_close.index.tz is not None:
            gold_close_naive = gold_close.tz_localize(None)
        else:
            gold_close_naive = gold_close

        # Calculate moving averages on full dataset
        moving_avg_50 = gold_close_naive.rolling(window=50, min_periods=50).mean()
        moving_avg_200 = gold_close_naive.rolling(window=200, min_periods=200).mean()

        # Create complete result DataFrame
        result_df = pd.DataFrame(
            {
                "Gold": gold_close_naive,
                "Gold_MA50": moving_avg_50,
                "Gold_MA200": moving_avg_200,
            },
            index=gold_close_naive.index,
        )

        # Filter by base_date for display purposes
        base_date_pd = pd.to_datetime(base_date.date())
        display_data = result_df[result_df.index >= base_date_pd]

        if display_data.empty:
            logger.warning(f"No gold data after base_date {base_date} for display")
            # Return empty result but don't error - this is just a display filter
            display_data = pd.DataFrame(
                columns=pd.Index(["Gold", "Gold_MA50", "Gold_MA200"])
            )

        logger.info(
            f"Gold processing completed: {len(display_data)} data points "
            f"from {base_date}"
        )

        return FlowResultEvent.success_result(
            data=display_data,
            base_date=base_date,
            metadata={
                "latest_value": (
                    display_data["Gold"].iloc[-1]
                    if not display_data.empty and isinstance(display_data, pd.DataFrame)
                    else None
                ),
                "data_points": len(display_data),
            },
        )


@apply_flow_cache
async def fetch_precious_metals_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process precious metals (Gold Futures) data using FlowRunner.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with gold values and moving average
        - base_date: The base date used
        - latest_value: Most recent gold price
        - data_points: Number of data points

    Raises:
        Exception: If workflow fails (for backward compatibility)
    """
    logger.info(f"Starting precious metals data fetch from {base_date}")

    # Create workflow and FlowRunner
    workflow = PreciousMetalsWorkflow()
    runner = FlowRunner[pd.DataFrame](workflow)

    # Run workflow using FlowRunner
    result_event = await runner.run(base_date=base_date)

    if result_event.success:
        logger.info("Precious metals workflow completed successfully")

        # Convert FlowResultEvent back to dictionary format for backward compatibility
        metadata = result_event.metadata or {}

        return {
            "data": result_event.data,
            "base_date": result_event.base_date,
            "latest_value": metadata.get("latest_value"),
            "data_points": metadata.get("data_points", 0),
        }

    else:
        # Handle error case - raise exception for backward compatibility
        error_msg = result_event.error_message or "Unknown error"

        logger.error(f"Precious metals workflow failed: {error_msg}")

        # Raise a standard exception for backward compatibility
        raise FlowException(
            workflow="PreciousMetalsWorkflow",
            step="fetch_precious_metals_data",
            message=f"Precious metals workflow failed: {error_msg}",
        )
