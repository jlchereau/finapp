"""
LlamaIndex workflow for MSCI World Index data collection.

This workflow handles fetching MSCI World Index data from Yahoo Finance
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


class MSCIWorldWorkflow(Workflow):
    """
    Workflow that fetches MSCI World Index data.

    This workflow fetches MSCI World Index data from Yahoo Finance using
    the ^990100-USD-STRD ticker. The MSCI World Index is a broad global
    equity index that captures large and mid cap representation across
    23 Developed Markets countries.

    The workflow:
    - Fetches daily MSCI World Index data from Yahoo Finance
    - Calculates 50-day and 200-day moving averages on full dataset
    - Calculates Bollinger Bands (20-day MA ± 2 std dev)
    - Applies base_date filtering for display

    This workflow follows Pattern 1 from flows.patterns.ipynb:
    - Single step that fetches and processes MSCI World data
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
    async def fetch_and_process_msci_data(self, ev: StartEvent) -> FlowResultEvent:
        """
        Fetch and process MSCI World Index data from Yahoo Finance.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            FlowResultEvent with processed MSCI World data and indicators
        """
        base_date = ev.base_date

        logger.debug(f"MSCIWorldWorkflow: Fetching MSCI World data from {base_date}")

        # Fetch MSCI World Index data - let exceptions bubble to FlowRunner
        msci_result = await self.yahoo_provider.get_data("^990100-USD-STRD")

        # Validate provider result - raise exceptions that FlowRunner will catch
        if not msci_result.success:
            error_msg = getattr(
                msci_result, "error_message", "Unknown MSCI World fetch error"
            )
            raise FlowException(
                workflow="MSCIWorldWorkflow",
                step="fetch_and_process_msci_data",
                message=f"MSCI World data fetch failed: {error_msg}",
            )
        if not (
            hasattr(msci_result, "data") and isinstance(msci_result.data, pd.DataFrame)
        ):
            raise FlowException(
                workflow="MSCIWorldWorkflow",
                step="fetch_and_process_msci_data",
                message="Invalid MSCI World data format returned",
            )

        msci_data = msci_result.data
        if msci_data.empty:
            raise FlowException(
                workflow="MSCIWorldWorkflow",
                step="fetch_and_process_msci_data",
                message="No MSCI World data available",
            )

        logger.debug("MSCIWorldWorkflow: Processing MSCI World data")

        # Extract close prices from MSCI World data
        if "Close" in msci_data.columns:
            msci_close = msci_data["Close"].dropna()
        elif "Adj Close" in msci_data.columns:
            msci_close = msci_data["Adj Close"].dropna()
        else:
            raise FlowException(
                workflow="MSCIWorldWorkflow",
                step="fetch_and_process_msci_data",
                message="No Close price data available for MSCI World",
            )

        if msci_close.empty:
            raise FlowException(
                workflow="MSCIWorldWorkflow",
                step="fetch_and_process_msci_data",
                message="No MSCI World close price data available",
            )

        # Normalize timezone to ensure proper filtering
        if hasattr(msci_close.index, "tz") and msci_close.index.tz is not None:
            msci_close_naive = msci_close.tz_localize(None)
        else:
            msci_close_naive = msci_close

        # Calculate moving averages on full dataset
        moving_avg_50 = msci_close_naive.rolling(window=50, min_periods=50).mean()
        moving_avg_200 = msci_close_naive.rolling(window=200, min_periods=200).mean()

        # Calculate Bollinger Bands (20-day MA ± 2 standard deviations)
        rolling_mean = msci_close_naive.rolling(window=20, min_periods=20).mean()
        rolling_std = msci_close_naive.rolling(window=20, min_periods=20).std()
        bollinger_upper = rolling_mean + (rolling_std * 2)
        bollinger_lower = rolling_mean - (rolling_std * 2)

        # Create complete result DataFrame
        result_df = pd.DataFrame(
            {
                "MSCI_World": msci_close_naive,
                "MSCI_MA50": moving_avg_50,
                "MSCI_MA200": moving_avg_200,
                "MSCI_BB_Upper": bollinger_upper,
                "MSCI_BB_Lower": bollinger_lower,
                "MSCI_BB_Mid": rolling_mean,
            },
            index=msci_close_naive.index,
        )

        # Filter by base_date for display purposes
        base_date_pd = pd.to_datetime(base_date.date())
        display_data = result_df[result_df.index >= base_date_pd]

        if display_data.empty:
            logger.warning(
                f"No MSCI World data after base_date {base_date} for display"
            )
            # Return empty result but don't error - this is just a display filter
            display_data = pd.DataFrame(
                columns=pd.Index(
                    [
                        "MSCI_World",
                        "MSCI_MA50",
                        "MSCI_MA200",
                        "MSCI_BB_Upper",
                        "MSCI_BB_Lower",
                        "MSCI_BB_Mid",
                    ]
                )
            )

        logger.info(
            f"MSCI World processing completed: {len(display_data)} data points "
            f"from {base_date}"
        )

        return FlowResultEvent.success_result(
            data=display_data,
            base_date=base_date,
            metadata={
                "latest_value": (
                    display_data["MSCI_World"].iloc[-1]
                    if not display_data.empty and isinstance(display_data, pd.DataFrame)
                    else None
                ),
                "data_points": len(display_data),
            },
        )


@apply_flow_cache
async def fetch_msci_world_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process MSCI World Index data using FlowRunner.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with MSCI World values, moving averages,
          and Bollinger bands
        - base_date: The base date used
        - latest_value: Most recent MSCI World value
        - data_points: Number of data points

    Raises:
        Exception: If workflow fails (for backward compatibility)
    """
    logger.info(f"Starting MSCI World data fetch from {base_date}")

    # Create workflow and FlowRunner
    workflow = MSCIWorldWorkflow()
    runner = FlowRunner[pd.DataFrame](workflow)

    # Run workflow using FlowRunner
    result_event = await runner.run(base_date=base_date)

    if result_event.success:
        logger.info("MSCI World workflow completed successfully")

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

        logger.error(f"MSCI World workflow failed: {error_msg}")

        # Raise a standard exception for backward compatibility
        raise FlowException(
            workflow="MSCIWorldWorkflow",
            step="fetch_msci_world_data",
            message=f"MSCI World workflow failed: {error_msg}",
        )
