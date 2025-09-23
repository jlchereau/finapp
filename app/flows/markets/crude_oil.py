"""
LlamaIndex workflow for crude oil (WTI and Brent) price data collection.

This workflow handles fetching crude oil price data using
YahooHistoryProvider with FlowRunner architecture.
"""

from typing import Dict, Any
from datetime import datetime

import pandas as pd
from workflows import Workflow, step, Context
from workflows.events import Event, StartEvent

from app.providers.yahoo import create_yahoo_history_provider
from app.flows.base import FlowRunner, FlowResultEvent
from app.lib.logger import logger
from app.lib.exceptions import FlowException
from app.flows.cache import apply_flow_cache


class DispatchEvent(Event):
    """Event to initiate parallel fetching of crude oil data."""


class FetchWTIEvent(Event):
    """Event to initiate fetching of WTI crude oil data from Yahoo."""

    base_date: datetime


class FetchBrentEvent(Event):
    """Event to initiate fetching of Brent crude oil data from Yahoo."""

    base_date: datetime


class WTIResultEvent(Event):
    """Event containing result of WTI crude oil data fetch."""

    data: pd.DataFrame | None
    error: str | None


class BrentResultEvent(Event):
    """Event containing result of Brent crude oil data fetch."""

    data: pd.DataFrame | None
    error: str | None


class CrudeOilWorkflow(Workflow):
    """
    Workflow that fetches crude oil (WTI and Brent) price data.
    This workflow fetches crude oil price data from Yahoo Finance using
    CL=F and BZ=F tickers. These are the major crude oil benchmarks.
    The workflow:
        - Fetches daily WTI and Brent data from Yahoo Finance in parallel
    - Applies base_date filtering for display
    - No moving averages or trend lines per user requirements
    """

    def __init__(self):
        """Initialize workflow with Yahoo provider."""
        super().__init__()
        # Create provider
        self.yahoo_provider = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def initiate_crude_oil_fetch(
        self, ctx: Context, ev: StartEvent
    ) -> FetchWTIEvent | FetchBrentEvent:
        """
        Step 1: Dispatch - Send parallel fetch events for WTI and Brent data.
        LlamaIndex automatically executes the sent events in parallel.

        Args:
        ctx: Workflow context for storing state and sending events
        ev.base_date: Start date for data fetching

        Returns:
        FetchWTIEvent (dummy event for LlamaIndex validation)
        """
        base_date = ev.base_date

        logger.debug(
            f"CrudeOilWorkflow: Dispatching parallel crude oil data fetch "
            f"from {base_date}"
        )

        # Store shared state for later steps
        await ctx.store.set("base_date", base_date)

        # Send events for parallel execution - LlamaIndex handles the parallelism
        ctx.send_event(FetchWTIEvent(base_date=base_date))
        ctx.send_event(FetchBrentEvent(base_date=base_date))

        # Return dummy event to satisfy return type - LlamaIndex processes sent events
        return FetchWTIEvent(base_date=base_date)

    @step
    async def fetch_wti_data(self, ev: FetchWTIEvent) -> WTIResultEvent:
        """
        Step 2: WTI Processing - Fetch WTI crude oil data from Yahoo Finance.
        This step runs in parallel with Brent data fetching automatically via
        LlamaIndex.

        Args:
        ev: FetchWTIEvent with fetch details

        Returns:
        WTIResultEvent with fetch result
        """
        # Mark event parameter as intentionally unused
        _ = ev
        logger.debug("Fetching WTI crude oil data from Yahoo Finance")

        # Use standard provider interface - fetch full historical data
        provider_result = await self.yahoo_provider.get_data(
            query="CL=F",  # WTI Crude Oil futures
        )

        # Handle ProviderResult properly
        if provider_result.success and provider_result.data is not None:
            data_df = provider_result.data
            if isinstance(data_df, pd.DataFrame) and not data_df.empty:
                logger.debug(f"Successfully fetched {len(data_df)} WTI observations")
                return WTIResultEvent(
                    data=data_df,
                    error=None,
                )
            else:
                logger.warning("Empty WTI data returned")
                return WTIResultEvent(
                    data=None,
                    error="Empty data returned",
                )
        else:
            # Provider failed
            error_msg = provider_result.error_message or "Provider failed"
            logger.warning(f"WTI provider failed: {error_msg}")
            return WTIResultEvent(
                data=None,
                error=error_msg,
            )

    @step
    async def fetch_brent_data(self, ev: FetchBrentEvent) -> BrentResultEvent:
        """
        Step 3: Brent Processing - Fetch Brent crude oil data from Yahoo Finance.
        This step runs in parallel with WTI data fetching automatically via
        LlamaIndex.

        Args:
        ev: FetchBrentEvent with fetch details

        Returns:
        BrentResultEvent with fetch result
        """
        # Mark event parameter as intentionally unused
        _ = ev
        logger.debug("Fetching Brent crude oil data from Yahoo Finance")

        # Use standard provider interface - fetch full historical data
        provider_result = await self.yahoo_provider.get_data(
            query="BZ=F",  # Brent Crude Oil futures
        )

        # Handle ProviderResult properly
        if provider_result.success and provider_result.data is not None:
            data_df = provider_result.data
            if isinstance(data_df, pd.DataFrame) and not data_df.empty:
                logger.debug(f"Successfully fetched {len(data_df)} Brent observations")
                return BrentResultEvent(
                    data=data_df,
                    error=None,
                )
            else:
                logger.warning("Empty Brent data returned")
                return BrentResultEvent(
                    data=None,
                    error="Empty data returned",
                )
        else:
            # Provider failed
            error_msg = provider_result.error_message or "Provider failed"
            logger.warning(f"Brent provider failed: {error_msg}")
            return BrentResultEvent(
                data=None,
                error=error_msg,
            )

    @step
    async def combine_crude_oil_data(
        self, ctx: Context, ev: WTIResultEvent | BrentResultEvent
    ) -> FlowResultEvent | None:
        """
        Step 4: Combine - Combine all parallel results and create crude oil
        comparison data.
        Uses collector pattern to wait for both WTI and Brent results.

        Args:
        ctx: Workflow context for collecting events and accessing stored state
        ev: Either WTIResultEvent or BrentResultEvent from parallel fetches

        Returns:
        FlowResultEvent with processed crude oil data when both results
        collected, None otherwise
        """
        # Get the stored state from the dispatch step
        base_date = await ctx.store.get("base_date")

        # Collect events until we have both results (WTI + Brent = 2 events)
        results = ctx.collect_events(ev, [WTIResultEvent, BrentResultEvent])
        if results is None:
            # Not all results collected yet
            return None

        logger.debug(
            f"CrudeOilWorkflow: Combining {len(results)} crude oil data source results"
        )

        # Separate and validate the results
        wti_result = None
        brent_result = None

        for result in results:
            if isinstance(result, WTIResultEvent):
                wti_result = result
            elif isinstance(result, BrentResultEvent):
                brent_result = result

        # Check that we have both results - raise exceptions that FlowRunner will catch
        if wti_result is None:
            raise FlowException(
                workflow="CrudeOilWorkflow",
                step="combine_crude_oil_data",
                message="WTI result not received",
            )
        if brent_result is None:
            raise FlowException(
                workflow="CrudeOilWorkflow",
                step="combine_crude_oil_data",
                message="Brent result not received",
            )

        # Check for errors in the results - both are required for crude oil comparison
        if wti_result.error is not None:
            raise FlowException(
                workflow="CrudeOilWorkflow",
                step="combine_crude_oil_data",
                message=f"WTI data fetch failed: {wti_result.error}",
            )
        if brent_result.error is not None:
            raise FlowException(
                workflow="CrudeOilWorkflow",
                step="combine_crude_oil_data",
                message=f"Brent data fetch failed: {brent_result.error}",
            )

        # Extract the data
        wti_data = wti_result.data
        brent_data = brent_result.data

        logger.debug("CrudeOilWorkflow: Processing crude oil data")

        # Validate data is not None first - let exceptions bubble up
        if wti_data is None:
            raise FlowException(
                workflow="CrudeOilWorkflow",
                step="combine_crude_oil_data",
                message="WTI data is missing",
            )
        if brent_data is None:
            raise FlowException(
                workflow="CrudeOilWorkflow",
                step="combine_crude_oil_data",
                message="Brent data is missing",
            )

        logger.debug(
            f"Processing WTI data: {len(wti_data)} rows, "
            f"Brent data: {len(brent_data)} rows"
        )

        # Extract close prices from WTI data
        if "Close" in wti_data.columns:
            wti_close = wti_data["Close"].dropna()
        elif "Adj Close" in wti_data.columns:
            wti_close = wti_data["Adj Close"].dropna()
        else:
            raise FlowException(
                workflow="CrudeOilWorkflow",
                step="combine_crude_oil_data",
                message="No Close price data available for WTI",
            )

        if wti_close.empty:
            raise FlowException(
                workflow="CrudeOilWorkflow",
                step="combine_crude_oil_data",
                message="No WTI close price data available",
            )

        # Extract close prices from Brent data
        if "Close" in brent_data.columns:
            brent_close = brent_data["Close"].dropna()
        elif "Adj Close" in brent_data.columns:
            brent_close = brent_data["Adj Close"].dropna()
        else:
            raise FlowException(
                workflow="CrudeOilWorkflow",
                step="combine_crude_oil_data",
                message="No Close price data available for Brent",
            )

        if brent_close.empty:
            raise FlowException(
                workflow="CrudeOilWorkflow",
                step="combine_crude_oil_data",
                message="No Brent close price data available",
            )

        # Normalize timezone to ensure proper filtering
        if hasattr(wti_close.index, "tz") and wti_close.index.tz is not None:
            wti_close_naive = wti_close.tz_localize(None)
        else:
            wti_close_naive = wti_close

        if hasattr(brent_close.index, "tz") and brent_close.index.tz is not None:
            brent_close_naive = brent_close.tz_localize(None)
        else:
            brent_close_naive = brent_close

        # Combine data into single DataFrame for comparison chart
        # Align data to common dates
        common_dates = wti_close_naive.index.intersection(brent_close_naive.index)

        if common_dates.empty:
            logger.warning("No common dates between WTI and Brent data")
            result_df = pd.DataFrame(columns=pd.Index(["WTI", "Brent"]))
        else:
            result_df = pd.DataFrame(
                {
                    "WTI": wti_close_naive.reindex(common_dates),
                    "Brent": brent_close_naive.reindex(common_dates),
                },
                index=common_dates,
            )

        # Filter by base_date for display purposes
        base_date_pd = pd.to_datetime(base_date.date())
        display_data = result_df[result_df.index >= base_date_pd]

        if display_data.empty:
            logger.warning(f"No crude oil data after base_date {base_date} for display")
            # Return empty result but don't error - this is just a display filter
            display_data = pd.DataFrame(columns=pd.Index(["WTI", "Brent"]))

        logger.info(
            f"Crude oil processing completed: {len(display_data)} data points "
            f"from {base_date}"
        )

        return FlowResultEvent.success_result(
            data=display_data,
            base_date=base_date,
            metadata={
                "latest_wti": (
                    display_data["WTI"].iloc[-1]
                    if not display_data.empty and isinstance(display_data, pd.DataFrame)
                    else None
                ),
                "latest_brent": (
                    display_data["Brent"].iloc[-1]
                    if not display_data.empty and isinstance(display_data, pd.DataFrame)
                    else None
                ),
                "data_points": len(display_data),
            },
        )


@apply_flow_cache
async def fetch_crude_oil_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process crude oil (WTI and Brent) data using FlowRunner.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with WTI and Brent values
        - base_date: The base date used
        - latest_wti: Most recent WTI price
        - latest_brent: Most recent Brent price
        - data_points: Number of data points

    Raises:
        Exception: If workflow fails (for backward compatibility)
    """
    logger.info(f"Starting crude oil data fetch from {base_date}")

    # Create workflow and FlowRunner
    workflow = CrudeOilWorkflow()
    runner = FlowRunner[pd.DataFrame](workflow)

    # Run workflow using FlowRunner
    result_event = await runner.run(base_date=base_date)

    if result_event.success:
        logger.info("Crude oil workflow completed successfully")

        # Convert FlowResultEvent back to dictionary format for backward compatibility
        metadata = result_event.metadata or {}

        return {
            "data": result_event.data,
            "base_date": result_event.base_date,
            "latest_wti": metadata.get("latest_wti"),
            "latest_brent": metadata.get("latest_brent"),
            "data_points": metadata.get("data_points", 0),
        }

    else:
        # Handle error case - raise exception for backward compatibility
        error_msg = result_event.error_message or "Unknown error"

        logger.error(f"Crude oil workflow failed: {error_msg}")

        # Raise a standard exception for backward compatibility
        raise FlowException(
            workflow="CrudeOilWorkflow",
            step="fetch_crude_oil_data",
            message=f"Crude oil workflow failed: {error_msg}",
        )
