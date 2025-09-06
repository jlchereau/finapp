"""
LlamaIndex workflow for VIX data collection.

This workflow handles fetching VIX volatility index data
using YahooHistoryProvider with FlowRunner architecture.
"""

from typing import Dict, Any
from datetime import datetime

import pandas as pd
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from app.providers.yahoo import create_yahoo_history_provider
from app.flows.base import FlowRunner, FlowResult
from app.lib.logger import logger
from app.lib.exceptions import WorkflowException
from app.flows.cache import apply_flow_cache


class VIXEvent(Event):
    """Event emitted when VIX data is fetched."""

    vix_data: pd.DataFrame
    base_date: datetime


class VIXWorkflow(Workflow):
    """
    Workflow that fetches VIX (volatility index) data.

    The VIX is a measure of market volatility, often called the "fear gauge".
    We use the ^VIX ticker from Yahoo Finance to get historical VIX data.

    This workflow:
    - Fetches daily VIX data from Yahoo Finance
    - Calculates historical mean for reference
    - Filters data based on base_date for display
    """

    def __init__(self):
        """Initialize workflow with Yahoo provider."""
        super().__init__()
        # Create provider
        self.yahoo_provider = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def fetch_vix_data(self, ev: StartEvent) -> VIXEvent:
        """
        Fetch VIX data from Yahoo Finance.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            VIXEvent with VIX data
        """
        base_date = ev.base_date

        logger.debug(f"VIXWorkflow: Fetching VIX data from {base_date}")

        # Fetch VIX data using helper function
        vix_result = await self.yahoo_provider.get_data("^VIX")

        # Validate the provider result
        if isinstance(vix_result, Exception):
            raise vix_result
        if not (hasattr(vix_result, "success") and vix_result.success):
            error_msg = getattr(vix_result, "error_message", "Unknown VIX fetch error")
            raise WorkflowException(
                workflow="VIXWorkflow",
                step="fetch_vix_data",
                message=f"VIX data fetch failed: {error_msg}",
            )
        if not (
            hasattr(vix_result, "data") and isinstance(vix_result.data, pd.DataFrame)
        ):
            error_msg = getattr(vix_result, "error_message", "Unknown VIX data error")
            raise WorkflowException(
                workflow="VIXWorkflow",
                step="fetch_vix_data",
                message=f"VIX data fetch failed: {error_msg}",
            )

        vix_data = vix_result.data
        if vix_data.empty:
            raise WorkflowException(
                workflow="VIXWorkflow",
                step="fetch_vix_data",
                message="No vix data available",
            )

        return VIXEvent(vix_data=vix_data, base_date=base_date)

    @step
    async def process_vix_data(self, ev: VIXEvent) -> StopEvent:
        """
        Process VIX data and calculate statistics.

        Args:
            ev: VIXEvent with VIX data

        Returns:
            StopEvent with FlowResult containing processed VIX data and statistics
        """
        vix_data = ev.vix_data
        base_date = ev.base_date

        logger.debug("VIXWorkflow: Processing VIX data")

        try:
            # Extract close prices from VIX data
            if "Close" in vix_data.columns:
                vix_close = vix_data["Close"].dropna()
            elif "Adj Close" in vix_data.columns:
                vix_close = vix_data["Adj Close"].dropna()
            else:
                logger.error("No Close price data in VIX data")
                raise WorkflowException(
                    workflow="VIXWorkflow",
                    step="process_vix_data",
                    message="No Close price data available for VIX",
                )

            if vix_close.empty:
                logger.error("No VIX close price data available")
                raise WorkflowException(
                    workflow="VIXWorkflow",
                    step="process_vix_data",
                    message="No VIX data available",
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

            return StopEvent(
                result=FlowResult.success_result(
                    data=display_data,
                    base_date=base_date,
                    execution_time=None,  # Will be set by FlowRunner
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
            )

        except WorkflowException:
            # Re-raise WorkflowException without wrapping
            raise
        except Exception as e:
            logger.error(f"Error processing VIX data: {e}")
            # Re-raise as WorkflowException for better handling
            raise WorkflowException(
                workflow="VIXWorkflow",
                step="process_vix_data",
                message=f"VIX data processing failed: {e}",
                user_message="Failed to process VIX data. Please try again later.",
                context={"base_date": str(base_date)},
            ) from e


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
    """
    try:
        logger.info(f"Starting VIX data fetch from {base_date}")

        # Create workflow and FlowRunner
        workflow = VIXWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        # Run workflow using FlowRunner
        flow_result = await runner.run(base_date=base_date)

        logger.info("VIX workflow completed successfully")

        # Convert FlowResult back to dictionary format for backward compatibility
        if flow_result.success and flow_result.data is not None:
            # Extract metadata fields
            metadata = flow_result.metadata or {}

            return {
                "data": flow_result.data,
                "base_date": flow_result.base_date,
                "historical_mean": metadata.get("historical_mean", 20.0),
                "latest_value": metadata.get("latest_value"),
                "data_points": metadata.get("data_points", 0),
            }
        else:
            # Handle error case
            error_message = flow_result.error_message or "Unknown workflow error"
            raise WorkflowException(
                workflow="fetch_vix_data",
                step="process_result",
                message=f"Workflow returned error: {error_message}",
                user_message=(
                    "Failed to fetch VIX data due to a system error. "
                    "Please try again."
                ),
                context={"base_date": str(base_date)},
            )

    except WorkflowException:
        # Re-raise WorkflowException without wrapping
        raise
    except Exception as e:
        logger.error(f"VIX workflow failed: {e}")
        # Re-raise as WorkflowException for better handling
        raise WorkflowException(
            workflow="fetch_vix_data",
            step="workflow_execution",
            message=f"VIX workflow execution failed: {e}",
            user_message=(
                "Failed to fetch VIX data due to a system error. Please try again."
            ),
            context={"base_date": str(base_date)},
        ) from e
