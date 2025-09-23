"""
LlamaIndex workflow for Bloomberg Commodity Index (^BCOM) data collection.

This workflow handles fetching Bloomberg Commodity Index data from Yahoo Finance
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


class BloombergCommodityWorkflow(Workflow):
    """
    Workflow that fetches Bloomberg Commodity Index (^BCOM) data.

    This workflow fetches the Bloomberg Commodity Index from Yahoo Finance using
    the ^BCOM ticker. The index tracks the performance of a diversified basket of
    commodity futures contracts.

    The workflow:
    - Fetches daily ^BCOM data from Yahoo Finance
    - Calculates 50-day and 200-day moving averages on full dataset
    - Applies base_date filtering for display

    This workflow follows Pattern 1 from flows.patterns.ipynb:
    - Single step that fetches and processes Bloomberg Commodity Index data
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
    async def fetch_and_process_bcom_data(self, ev: StartEvent) -> FlowResultEvent:
        """
        Fetch and process Bloomberg Commodity Index data from Yahoo Finance.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            FlowResultEvent with processed ^BCOM data and moving averages
        """
        base_date = ev.base_date

        logger.debug(
            f"BloombergCommodityWorkflow: Fetching ^BCOM data from {base_date}"
        )

        # Fetch Bloomberg Commodity Index data - let exceptions bubble to FlowRunner
        bcom_result = await self.yahoo_provider.get_data("^BCOM")

        # Validate provider result - raise exceptions that FlowRunner will catch
        if not bcom_result.success:
            error_msg = getattr(
                bcom_result, "error_message", "Unknown ^BCOM fetch error"
            )
            raise FlowException(
                workflow="BloombergCommodityWorkflow",
                step="fetch_and_process_bcom_data",
                message=f"^BCOM data fetch failed: {error_msg}",
            )
        if not (
            hasattr(bcom_result, "data") and isinstance(bcom_result.data, pd.DataFrame)
        ):
            raise FlowException(
                workflow="BloombergCommodityWorkflow",
                step="fetch_and_process_bcom_data",
                message="Invalid ^BCOM data format returned",
            )

        bcom_data = bcom_result.data
        if bcom_data.empty:
            raise FlowException(
                workflow="BloombergCommodityWorkflow",
                step="fetch_and_process_bcom_data",
                message="No ^BCOM data available",
            )

        logger.debug("BloombergCommodityWorkflow: Processing ^BCOM data")

        # Extract close prices from ^BCOM data
        if "Close" in bcom_data.columns:
            bcom_close = bcom_data["Close"].dropna()
        elif "Adj Close" in bcom_data.columns:
            bcom_close = bcom_data["Adj Close"].dropna()
        else:
            raise FlowException(
                workflow="BloombergCommodityWorkflow",
                step="fetch_and_process_bcom_data",
                message="No Close price data available for ^BCOM",
            )

        if bcom_close.empty:
            raise FlowException(
                workflow="BloombergCommodityWorkflow",
                step="fetch_and_process_bcom_data",
                message="No ^BCOM close price data available",
            )

        # Normalize timezone to ensure proper filtering
        if hasattr(bcom_close.index, "tz") and bcom_close.index.tz is not None:
            bcom_close_naive = bcom_close.tz_localize(None)
        else:
            bcom_close_naive = bcom_close

        # Calculate moving averages on full dataset
        moving_avg_50 = bcom_close_naive.rolling(window=50, min_periods=50).mean()
        moving_avg_200 = bcom_close_naive.rolling(window=200, min_periods=200).mean()

        # Create complete result DataFrame
        result_df = pd.DataFrame(
            {
                "BCOM": bcom_close_naive,
                "BCOM_MA50": moving_avg_50,
                "BCOM_MA200": moving_avg_200,
            },
            index=bcom_close_naive.index,
        )

        # Filter by base_date for display purposes
        base_date_pd = pd.to_datetime(base_date.date())
        display_data = result_df[result_df.index >= base_date_pd]

        if display_data.empty:
            logger.warning(f"No ^BCOM data after base_date {base_date} for display")
            # Return empty result but don't error - this is just a display filter
            display_data = pd.DataFrame(
                columns=pd.Index(["BCOM", "BCOM_MA50", "BCOM_MA200"])
            )

        logger.info(
            f"Bloomberg Commodity processing completed: {len(display_data)} "
            f"data points from {base_date}"
        )

        return FlowResultEvent.success_result(
            data=display_data,
            base_date=base_date,
            metadata={
                "latest_value": (
                    display_data["BCOM"].iloc[-1]
                    if not display_data.empty and isinstance(display_data, pd.DataFrame)
                    else None
                ),
                "data_points": len(display_data),
            },
        )


@apply_flow_cache
async def fetch_bloomberg_commodity_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process Bloomberg Commodity Index (^BCOM) data using FlowRunner.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with BCOM values and moving averages
        - base_date: The base date used
        - latest_value: Most recent BCOM value
        - data_points: Number of data points

    Raises:
        Exception: If workflow fails (for backward compatibility)
    """
    logger.info(f"Starting Bloomberg Commodity data fetch from {base_date}")

    # Create workflow and FlowRunner
    workflow = BloombergCommodityWorkflow()
    runner = FlowRunner[pd.DataFrame](workflow)

    # Run workflow using FlowRunner
    result_event = await runner.run(base_date=base_date)

    if result_event.success:
        logger.info("Bloomberg Commodity workflow completed successfully")

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

        logger.error(f"Bloomberg Commodity workflow failed: {error_msg}")

        # Raise a standard exception for backward compatibility
        raise FlowException(
            workflow="BloombergCommodityWorkflow",
            step="fetch_bloomberg_commodity_data",
            message=f"Bloomberg Commodity workflow failed: {error_msg}",
        )
