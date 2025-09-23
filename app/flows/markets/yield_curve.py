"""
LlamaIndex workflow for US Treasury yield curve data collection.

This workflow handles fetching yield curve data from FRED API using
FlowRunner architecture with parallel execution. The yield curve shows
interest rates across different maturities, from 1-month to 30-year Treasury securities.
"""

from typing import Dict, Any
from datetime import datetime

import pandas as pd
from workflows import Workflow, step, Context
from workflows.events import Event, StartEvent

from app.providers.fred import create_fred_series_provider
from app.flows.base import FlowRunner, FlowResultEvent
from app.lib.logger import logger
from app.lib.exceptions import FlowException
from app.flows.cache import apply_flow_cache

# Yield curve series mapping (FRED series ID to maturity label)
YIELD_CURVE_SERIES = {
    "DGS1MO": "1M",  # 1-Month Treasury Constant Maturity Rate
    "DGS3MO": "3M",  # 3-Month Treasury Constant Maturity Rate
    "DGS6MO": "6M",  # 6-Month Treasury Constant Maturity Rate
    "DGS1": "1Y",  # 1-Year Treasury Constant Maturity Rate
    "DGS2": "2Y",  # 2-Year Treasury Constant Maturity Rate
    "DGS5": "5Y",  # 5-Year Treasury Constant Maturity Rate
    "DGS10": "10Y",  # 10-Year Treasury Constant Maturity Rate
    "DGS30": "30Y",  # 30-Year Treasury Constant Maturity Rate
}


class FetchSeriesEvent(Event):
    """Event to initiate fetching of individual Treasury series."""

    series_id: str
    maturity_label: str
    base_date: datetime
    observation_start: str | None


class SeriesResultEvent(Event):
    """Event containing result of individual series fetch."""

    series_id: str
    maturity_label: str
    data: pd.DataFrame | None
    error: str | None


class YieldCurveWorkflow(Workflow):
    """
    ParallelWorkflow that fetches US Treasury yield curve data.

    The yield curve shows interest rates across different maturities,
    from 1-month to 30-year Treasury securities. This workflow follows
    the proper LlamaIndex ParallelWorkflow pattern:
    1. Dispatch step: Creates parallel FetchSeriesEvent for each Treasury series
    2. Processing step: Fetches individual series data (runs in parallel)
    3. Combine step: Combines all results into final DataFrame
    """

    def __init__(self):
        """Initialize workflow with FRED provider."""
        super().__init__()
        # Create provider
        self.fred_provider = create_fred_series_provider(timeout=30.0, retries=2)

    @step
    async def initiate_parallel_fetch(
        self, ctx: Context, ev: StartEvent
    ) -> FetchSeriesEvent:
        """
        Step 1: Dispatch - Send parallel fetch events for all Treasury series.
        LlamaIndex automatically executes the sent events in parallel.

        Args:
            ctx: Workflow context for storing state and sending events
            ev.base_date: Start date for data fetching

        Returns:
            None (events are sent via ctx.send_event)
        """
        base_date = ev.base_date
        observation_start = base_date.strftime("%Y-%m-%d") if base_date else None

        logger.debug(
            f"YieldCurveWorkflow: Dispatching parallel fetch of "
            f"{len(YIELD_CURVE_SERIES)} Treasury series from {base_date}"
        )

        # Store the number of series to collect and base_date for later
        await ctx.store.set("num_to_collect", len(YIELD_CURVE_SERIES))
        await ctx.store.set("base_date", base_date)

        # Send events for parallel execution - LlamaIndex handles the parallelism
        for series_id, maturity_label in YIELD_CURVE_SERIES.items():
            ctx.send_event(
                FetchSeriesEvent(
                    series_id=series_id,
                    maturity_label=maturity_label,
                    base_date=base_date,
                    observation_start=observation_start,
                )
            )

        # Return a dummy event to satisfy the return type - LlamaIndex should
        # process sent events
        return FetchSeriesEvent(
            series_id="dummy",
            maturity_label="dummy",
            base_date=base_date,
            observation_start=observation_start,
        )

    @step
    async def fetch_individual_series(self, ev: FetchSeriesEvent) -> SeriesResultEvent:
        """
        Step 2: Single-Series Processing - Fetch individual Treasury series data.
        This step runs in parallel for each series automatically via LlamaIndex.

        Args:
            ev: FetchSeriesEvent with series details

        Returns:
            SeriesResultEvent with fetch result
        """
        # Skip dummy events used for workflow validation
        if ev.series_id == "dummy":
            return SeriesResultEvent(
                series_id="dummy",
                maturity_label="dummy",
                data=None,
                error="dummy_event",
            )

        logger.debug(f"Fetching Treasury series {ev.series_id} ({ev.maturity_label})")

        # Use standard provider interface
        provider_result = await self.fred_provider.get_data(
            query=ev.series_id,
            observation_start=ev.observation_start,
        )

        # Handle ProviderResult properly
        if provider_result.success and provider_result.data is not None:
            data_df = provider_result.data
            if isinstance(data_df, pd.DataFrame) and not data_df.empty:
                logger.debug(
                    f"Successfully fetched {len(data_df)} observations for "
                    f"{ev.maturity_label}"
                )
                return SeriesResultEvent(
                    series_id=ev.series_id,
                    maturity_label=ev.maturity_label,
                    data=data_df,
                    error=None,
                )
            else:
                logger.warning(f"Empty data returned for series {ev.series_id}")
                return SeriesResultEvent(
                    series_id=ev.series_id,
                    maturity_label=ev.maturity_label,
                    data=None,
                    error="Empty data returned",
                )
        else:
            # Provider failed
            error_msg = provider_result.error_message or "Provider failed"
            logger.warning(f"Provider failed for series {ev.series_id}: {error_msg}")
            return SeriesResultEvent(
                series_id=ev.series_id,
                maturity_label=ev.maturity_label,
                data=None,
                error=error_msg,
            )

    @step
    async def combine_yield_data(
        self, ctx: Context, ev: SeriesResultEvent
    ) -> FlowResultEvent | None:
        """
        Step 3: Combine - Combine all parallel results into final yield curve DataFrame.
        Uses collector pattern to wait for all parallel results.

        Args:
            ctx: Workflow context for collecting events and accessing stored state
            ev: SeriesResultEvent from a single parallel fetch

        Returns:
            StopEvent with FlowResult when all results collected, None otherwise
        """
        # Get the number of results we're expecting and base_date
        num_to_collect = await ctx.store.get("num_to_collect")
        base_date = await ctx.store.get("base_date")

        # Collect events until we have all results
        results = ctx.collect_events(ev, [SeriesResultEvent] * num_to_collect)
        if results is None:
            # Not all results collected yet
            return None

        logger.debug(
            f"YieldCurveWorkflow: Combining {len(results)} Treasury series results"
        )

        # Process results and combine into single DataFrame
        combined_data = {}
        successful_series = 0

        for result in results:
            # Skip dummy results used for workflow validation
            if result.series_id == "dummy":
                continue

            if result.error is not None:
                logger.warning(
                    f"Failed to fetch series {result.series_id}: {result.error}"
                )
                continue

            if result.data is not None and not result.data.empty:
                # Add series data with maturity label as column name
                combined_data[result.maturity_label] = result.data["value"]
                successful_series += 1
                logger.debug(
                    f"Successfully processed {len(result.data)} observations "
                    f"for {result.maturity_label}"
                )
            else:
                logger.warning(f"Empty data returned for series {result.series_id}")

        if not combined_data:
            raise FlowException(
                workflow="YieldCurveWorkflow",
                step="combine_yield_data",
                message="No yield curve data retrieved from any series",
            )

        # Create combined DataFrame
        yield_data = pd.DataFrame(combined_data)
        yield_data.index.name = "date"

        # Sort columns by maturity order (1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y)
        maturity_order = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
        available_maturities = [m for m in maturity_order if m in yield_data.columns]
        yield_data = yield_data[available_maturities]

        # Sort by date
        yield_data = yield_data.sort_index()

        # Fill forward missing values (common for treasury data)
        yield_data = yield_data.ffill()

        logger.info(
            f"Successfully combined yield curve data: {len(yield_data)} dates, "
            f"{len(yield_data.columns)} maturities ({successful_series} series)"
        )

        # Apply date filtering if base_date is provided
        if base_date:
            base_date_pd = pd.to_datetime(base_date.date())
            display_data = yield_data.loc[yield_data.index >= base_date_pd]
        else:
            display_data = yield_data

        if display_data.empty:
            logger.warning(f"No yield curve data after base_date {base_date}")
            # Return empty result but don't error
            display_data = pd.DataFrame()

        logger.info(
            f"YieldCurve completed: {len(display_data)} trading days "
            f"from {base_date}"
        )

        # Return FlowResultEvent
        return FlowResultEvent.success_result(
            data=display_data,
            base_date=base_date,
            metadata={
                "latest_date": (
                    display_data.index[-1] if not display_data.empty else None
                ),
                "latest_yields": (
                    display_data.iloc[-1].to_dict()
                    if not display_data.empty and isinstance(display_data, pd.DataFrame)
                    else {}
                ),
                "data_points": len(display_data),
                "maturities": (
                    list(display_data.columns) if not display_data.empty else []
                ),
            },
        )


@apply_flow_cache
async def fetch_yield_curve_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process US Treasury yield curve data using FlowRunner.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with yield curve data
        - base_date: The base date used
        - latest_date: Date of most recent data
        - maturities: List of available maturities
        - data_points: Number of data points
    """
    logger.info(f"Starting yield curve data fetch from {base_date}")

    # Create workflow and FlowRunner
    workflow = YieldCurveWorkflow()
    runner = FlowRunner[pd.DataFrame](workflow)

    # Run workflow using FlowRunner
    result_event = await runner.run(base_date=base_date)

    if result_event.success:
        logger.info("Yield curve workflow completed successfully")

        # Convert FlowResultEvent back to dictionary format for backward compatibility
        metadata = result_event.metadata or {}

        return {
            "data": result_event.data,
            "base_date": result_event.base_date,
            "latest_date": metadata.get("latest_date"),
            "latest_yields": metadata.get("latest_yields", {}),
            "data_points": metadata.get("data_points", 0),
            "maturities": metadata.get("maturities", []),
        }

    else:
        # Handle error case - raise exception for backward compatibility
        error_msg = result_event.error_message or "Unknown error"

        logger.error(f"Yield curve workflow failed: {error_msg}")

        # Raise a standard exception for backward compatibility
        raise FlowException(
            workflow="YieldCurveWorkflow",
            step="fetch_yield_curve_data",
            message=f"Yield curve workflow failed: {error_msg}",
        )
