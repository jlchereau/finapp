"""
LlamaIndex workflow for Buffet Indicator data collection.

This workflow handles fetching economic data for the Buffet Indicator calculation
using FredSeriesProvider and YahooHistoryProvider with FlowRunner architecture.
"""

from typing import Dict, Any
from datetime import datetime

import pandas as pd
from workflows import Workflow, step, Context
from workflows.events import Event, StartEvent

from app.providers.fred import create_fred_series_provider
from app.providers.yahoo import create_yahoo_history_provider
from app.flows.base import FlowRunner, FlowResultEvent
from app.lib.logger import logger
from app.lib.exceptions import FlowException
from app.lib.periods import (
    ensure_minimum_data_points,
    filter_trend_data_to_period,
)
from app.lib.finance import calculate_exponential_trend
from app.flows.cache import apply_flow_cache


class DispatchEvent(Event):
    """Event to initiate fetching of GDP data from FRED."""


class FetchGDPEvent(Event):
    """Event to initiate fetching of GDP data from FRED."""

    base_date: datetime
    observation_start: str | None


class FetchWilshireEvent(Event):
    """Event to initiate fetching of Wilshire 5000 data from Yahoo."""

    base_date: datetime
    period: str


class GDPResultEvent(Event):
    """Event containing result of GDP data fetch."""

    data: pd.DataFrame | None
    error: str | None


class WilshireResultEvent(Event):
    """Event containing result of Wilshire 5000 data fetch."""

    data: pd.DataFrame | None
    error: str | None


class BuffetIndicatorWorkflow(Workflow):
    """
    Workflow that fetches data for the Buffet Indicator calculation.

    The Buffet Indicator is the ratio of total market cap to GDP.
    We use:
    - Nominal GDP data from FRED (series: GDP - Gross Domestic Product)
    - Wilshire 5000 Total Market Index from Yahoo Finance (^FTW5000)

    Note: We use nominal GDP (not real/inflation-adjusted) to match the
    current dollar terms of the Wilshire 5000 market cap data.

    This workflow:
    - Fetches quarterly nominal GDP data from FRED
    - Fetches daily Wilshire 5000 data from Yahoo Finance
    - Aligns the data on common dates
    - Calculates the Buffet Indicator ratio
    """

    def __init__(self):
        """Initialize workflow with FRED and Yahoo providers."""
        super().__init__()
        # Create providers
        self.fred_provider = create_fred_series_provider(timeout=30.0, retries=2)
        self.yahoo_provider = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def initiate_data_fetch(
        self, ctx: Context, ev: StartEvent
    ) -> FetchGDPEvent | FetchWilshireEvent:
        """
        Step 1: Dispatch - Send parallel fetch events for GDP and Wilshire data.
        LlamaIndex automatically executes the sent events in parallel.

        Args:
            ctx: Workflow context for storing state and sending events
            ev.base_date: Start date for data fetching
            ev.original_period: Original period selected by user (e.g., "2M")

        Returns:
            FetchGDPEvent (dummy event for LlamaIndex validation)
        """
        base_date = ev.base_date
        original_period = ev.original_period

        logger.debug(
            f"BuffetIndicatorWorkflow: Dispatching parallel data fetch from {base_date}"
        )

        # Store shared state for later steps
        await ctx.store.set("base_date", base_date)
        await ctx.store.set("original_period", original_period)

        # Send events for parallel execution - LlamaIndex handles the parallelism
        ctx.send_event(
            FetchGDPEvent(
                base_date=base_date,
                observation_start=None,  # No longer limit data collection
            )
        )

        ctx.send_event(
            FetchWilshireEvent(
                base_date=base_date,
                period="max",  # Yahoo provider uses "max" for historical data
            )
        )

        # Return dummy event to satisfy return type - LlamaIndex processes sent events
        return FetchGDPEvent(
            base_date=base_date,
            observation_start="dummy",
        )

    @step
    async def fetch_gdp_data(self, ev: FetchGDPEvent) -> GDPResultEvent:
        """
        Step 2: GDP Processing - Fetch GDP data from FRED.
        This step runs in parallel with Wilshire data fetching automatically via
        LlamaIndex.

        Args:
            ev: FetchGDPEvent (not used since we fetch max data)

        Returns:
            GDPResultEvent with fetch result
        """
        # Note: ev parameter required by LlamaIndex workflow, but not used
        # since we always fetch maximum data to prevent period limitation
        _ = ev  # Mark as intentionally unused
        logger.debug("Fetching GDP data from FRED")

        # Use standard provider interface - fetch full historical data
        provider_result = await self.fred_provider.get_data(
            query="GDP",  # Nominal GDP, Quarterly
        )

        # Handle ProviderResult properly
        if provider_result.success and provider_result.data is not None:
            data_df = provider_result.data
            if isinstance(data_df, pd.DataFrame) and not data_df.empty:
                logger.debug(f"Successfully fetched {len(data_df)} GDP observations")
                return GDPResultEvent(
                    data=data_df,
                    error=None,
                )
            else:
                logger.warning("Empty GDP data returned")
                return GDPResultEvent(
                    data=None,
                    error="Empty data returned",
                )
        else:
            # Provider failed
            error_msg = provider_result.error_message or "Provider failed"
            logger.warning(f"GDP provider failed: {error_msg}")
            return GDPResultEvent(
                data=None,
                error=error_msg,
            )

    @step
    async def fetch_wilshire_data(self, ev: FetchWilshireEvent) -> WilshireResultEvent:
        """
        Step 3: Wilshire Processing - Fetch Wilshire 5000 data from Yahoo Finance.
        This step runs in parallel with GDP data fetching automatically via LlamaIndex.

        Args:
            ev: FetchWilshireEvent with Wilshire fetch details

        Returns:
            WilshireResultEvent with fetch result
        """
        # Skip dummy events used for workflow validation
        if ev.period == "dummy":
            return WilshireResultEvent(
                data=None,
                error="dummy_event",
            )

        logger.debug("Fetching Wilshire 5000 data from Yahoo Finance")

        # Use standard provider interface
        provider_result = await self.yahoo_provider.get_data(
            query="^FTW5000",  # Wilshire 5000 Total Market Index
        )

        # Handle ProviderResult properly
        if provider_result.success and provider_result.data is not None:
            data_df = provider_result.data
            if isinstance(data_df, pd.DataFrame) and not data_df.empty:
                logger.debug(
                    f"Successfully fetched {len(data_df)} Wilshire observations"
                )
                return WilshireResultEvent(
                    data=data_df,
                    error=None,
                )
            else:
                logger.warning("Empty Wilshire data returned")
                return WilshireResultEvent(
                    data=None,
                    error="Empty data returned",
                )
        else:
            # Provider failed
            error_msg = provider_result.error_message or "Provider failed"
            logger.warning(f"Wilshire provider failed: {error_msg}")
            return WilshireResultEvent(
                data=None,
                error=error_msg,
            )

    @step
    async def calculate_buffet_indicator(
        self, ctx: Context, ev: GDPResultEvent | WilshireResultEvent
    ) -> FlowResultEvent | None:
        """
        Step 4: Combine - Combine all parallel results and calculate Buffet Indicator.
        Uses collector pattern to wait for both GDP and Wilshire results.

        The Buffet Indicator is calculated as:
        (Total Market Cap / GDP) * 100

        We use the Wilshire 5000 index as a proxy for total market cap.

        Args:
            ctx: Workflow context for collecting events and accessing stored state
            ev: Either GDPResultEvent or WilshireResultEvent from parallel fetches

        Returns:
            StopEvent with FlowResult when both results collected, None otherwise
        """
        # Get the stored state from the dispatch step
        base_date = await ctx.store.get("base_date")
        original_period = await ctx.store.get("original_period")

        # Collect events until we have both results (GDP + Wilshire = 2 events)
        # Note: We expect 2 real events + 1 dummy event = 3 total, but dummy
        # events are filtered out
        results = ctx.collect_events(ev, [GDPResultEvent, WilshireResultEvent])
        if results is None:
            # Not all results collected yet
            return None

        logger.debug(
            f"BuffetIndicatorWorkflow: Combining {len(results)} data source results"
        )

        # Separate and validate the results
        gdp_result = None
        wilshire_result = None

        for result in results:
            # Skip dummy results used for workflow validation
            if hasattr(result, "error") and result.error == "dummy_event":
                continue

            if isinstance(result, GDPResultEvent):
                gdp_result = result
            elif isinstance(result, WilshireResultEvent):
                wilshire_result = result

        # Check that we have both results - raise exceptions that FlowRunner will catch
        if gdp_result is None:
            raise FlowException(
                workflow="BuffetIndicatorWorkflow",
                step="calculate_buffet_indicator",
                message="GDP result not received",
            )
        if wilshire_result is None:
            raise FlowException(
                workflow="BuffetIndicatorWorkflow",
                step="calculate_buffet_indicator",
                message="Wilshire result not received",
            )

        # Check for errors in the results - both are required for Buffet Indicator
        if gdp_result.error is not None:
            raise FlowException(
                workflow="BuffetIndicatorWorkflow",
                step="calculate_buffet_indicator",
                message=f"GDP data fetch failed: {gdp_result.error}",
            )
        if wilshire_result.error is not None:
            raise FlowException(
                workflow="BuffetIndicatorWorkflow",
                step="calculate_buffet_indicator",
                message=f"Wilshire data fetch failed: {wilshire_result.error}",
            )

        # Extract the data
        gdp_data = gdp_result.data
        wilshire_data = wilshire_result.data

        logger.debug("BuffetIndicatorWorkflow: Calculating Buffet Indicator")

        # Validate data is not None first - let exceptions bubble up
        if gdp_data is None:
            raise FlowException(
                workflow="BuffetIndicatorWorkflow",
                step="calculate_buffet_indicator",
                message="GDP data is missing",
            )
        if wilshire_data is None:
            raise FlowException(
                workflow="BuffetIndicatorWorkflow",
                step="calculate_buffet_indicator",
                message="Wilshire data is missing",
            )

        logger.debug(
            f"Processing GDP data: {len(gdp_data)} rows, "
            f"Wilshire data: {len(wilshire_data)} rows"
        )

        # STEP 1: Extract and process data on FULL datasets (no filtering yet)

        # Get GDP values from full dataset
        gdp_values = gdp_data["value"].dropna()
        if gdp_values.empty:
            raise FlowException(
                workflow="BuffetIndicatorWorkflow",
                step="calculate_buffet_indicator",
                message="No GDP data available",
            )

        # Get close prices from Wilshire data (full dataset)
        if "Close" in wilshire_data.columns:
            wilshire_close = wilshire_data["Close"].dropna()
        elif "Adj Close" in wilshire_data.columns:
            wilshire_close = wilshire_data["Adj Close"].dropna()
        else:
            raise FlowException(
                workflow="BuffetIndicatorWorkflow",
                step="calculate_buffet_indicator",
                message="No Close price data available for Wilshire 5000",
            )

        if wilshire_close.empty:
            raise FlowException(
                workflow="BuffetIndicatorWorkflow",
                step="calculate_buffet_indicator",
                message="No Wilshire close price data available",
            )

        # STEP 2: Normalize timezones before resampling
        # CRITICAL: Convert both datasets to timezone-naive to ensure alignment

        # Normalize GDP data timezone (FRED data might be timezone-naive already)
        if hasattr(gdp_values.index, "tz") and gdp_values.index.tz is not None:
            gdp_values_naive = gdp_values.tz_localize(None)
        else:
            gdp_values_naive = gdp_values

        # Normalize Wilshire data timezone (Yahoo data is usually timezone-aware)
        if hasattr(wilshire_close.index, "tz") and wilshire_close.index.tz is not None:
            wilshire_close_naive = wilshire_close.tz_localize(None)
        else:
            wilshire_close_naive = wilshire_close

        # STEP 3: Resample BOTH datasets to Quarter Start for consistent alignment
        # This follows the proven working pattern from the code snippet

        # Resample GDP to Quarter Start (even though it's already quarterly)
        # This ensures consistent quarterly periods
        gdp_quarterly = gdp_values_naive.resample("QS").first().dropna()

        # Resample Wilshire daily data to Quarter Start using FULL dataset
        wilshire_quarterly = wilshire_close_naive.resample("QS").first().dropna()

        logger.debug(
            f"Quarterly resampling: GDP {len(gdp_quarterly)} quarters, "
            f"Wilshire {len(wilshire_quarterly)} quarters"
        )

        # STEP 4: Simple alignment using pandas - both datasets now have QS periods
        # This follows the working pattern: aligned_data = data1 / data2

        # Check for overlapping dates before calculation
        common_quarters = gdp_quarterly.index.intersection(wilshire_quarterly.index)

        if common_quarters.empty:
            raise FlowException(
                workflow="BuffetIndicatorWorkflow",
                step="calculate_buffet_indicator",
                message="No overlapping dates between GDP and Wilshire data",
            )

        logger.debug(f"Found {len(common_quarters)} overlapping quarters")

        # Align the data using pandas division
        # (automatically aligns on common index)
        buffet_indicator_full = (wilshire_quarterly / gdp_quarterly) * 100
        buffet_indicator_full = buffet_indicator_full.dropna()

        # Get aligned data for result DataFrame
        common_dates = buffet_indicator_full.index
        aligned_gdp = gdp_quarterly.loc[common_dates]
        aligned_wilshire = wilshire_quarterly.loc[common_dates]

        logger.debug(f"Buffet Indicator calculated for {len(common_dates)} quarters")

        # STEP 5: Calculate Buffet Indicator on aligned data
        # Note: This is a simplified calculation. The actual indicator
        # uses total market cap, but Wilshire 5000 is a good proxy
        buffet_indicator = (aligned_wilshire / aligned_gdp) * 100

        # Create complete result DataFrame with all aligned data
        result_df = pd.DataFrame(
            {
                "GDP": aligned_gdp,
                "Wilshire_5000": aligned_wilshire,
                "Buffet_Indicator": buffet_indicator,
            },
            index=common_dates,  # Use timezone-naive common dates
        )

        # STEP 6: Calculate exponential trend on FULL dataset (before filtering)
        # This ensures trend lines represent full historical context
        full_trend_data = calculate_exponential_trend(result_df, "Buffet_Indicator")

        logger.debug(
            f"Calculated exponential trend on full dataset: "
            f"{len(result_df)} quarters"
        )

        # STEP 7: Apply smart filtering with minimum data points guarantee
        display_data = ensure_minimum_data_points(
            data=result_df,
            base_date=base_date,
            min_points=2,
            reference_date=datetime.now(),
        )

        # STEP 8: Filter trend data to match display period
        # This preserves statistical integrity while showing relevant time range
        display_trend_data = filter_trend_data_to_period(full_trend_data, display_data)

        # Log the completion
        logger.info(
            f"BuffetIndicator completed: {len(display_data)} quarters "
            f"for {original_period} from {base_date}"
        )

        # Return FlowResultEvent with processed data
        return FlowResultEvent.success_result(
            data=display_data,
            base_date=base_date,
            metadata={
                "trend_data": display_trend_data,
                "original_period": original_period,
                "latest_value": (
                    display_data["Buffet_Indicator"].iloc[-1]
                    if not display_data.empty and isinstance(display_data, pd.DataFrame)
                    else None
                ),
                "data_points": len(display_data),
            },
        )


@apply_flow_cache
async def fetch_buffet_indicator_data(
    base_date: datetime, original_period: str = "1Y"
) -> Dict[str, Any]:
    """
    Fetch and calculate Buffet Indicator data using FlowRunner.

    Args:
        base_date: Start date for historical data
        original_period: Original period selected by user (for smart filtering)

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with GDP, Wilshire 5000, and Buffet Indicator
        - trend_data: Pre-calculated trend data for display
        - base_date: The base date used
        - original_period: Original period selected by user
        - actual_period: Period actually used after adjustment
        - was_adjusted: Whether period was adjusted for minimum data points
        - latest_value: Most recent Buffet Indicator value
        - data_points: Number of data points
    """
    logger.info(
        f"Starting Buffet Indicator data fetch from {base_date} "
        f"(period: {original_period})"
    )

    # Create workflow and FlowRunner
    workflow = BuffetIndicatorWorkflow()
    runner = FlowRunner[pd.DataFrame](workflow)

    # Run workflow using FlowRunner
    result_event = await runner.run(
        base_date=base_date, original_period=original_period
    )

    if result_event.success:
        logger.info("Buffet Indicator workflow completed successfully")

        # Convert FlowResultEvent back to dictionary format for backward compatibility
        metadata = result_event.metadata or {}

        return {
            "data": result_event.data,
            "trend_data": metadata.get("trend_data"),
            "base_date": result_event.base_date,
            "original_period": metadata.get("original_period", original_period),
            "actual_period": metadata.get("actual_period", original_period),
            "was_adjusted": metadata.get("was_adjusted", False),
            "latest_value": metadata.get("latest_value"),
            "data_points": metadata.get("data_points", 0),
        }

    else:
        # Handle error case - raise exception for backward compatibility
        error_msg = result_event.error_message or "Unknown error"

        logger.error(f"Buffet Indicator workflow failed: {error_msg}")

        # Raise a standard exception for backward compatibility
        raise FlowException(
            workflow="BuffetIndicatorWorkflow",
            step="fetch_buffet_indicator_data",
            message=f"Buffet Indicator workflow failed: {error_msg}",
        )
