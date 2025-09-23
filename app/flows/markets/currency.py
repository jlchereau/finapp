"""
LlamaIndex workflow for currency exchange rate data collection.

This workflow handles fetching currency exchange rate data for major pairs
using YahooHistoryProvider with FlowRunner architecture.
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
    """Event to initiate parallel fetching of currency data."""


class FetchUSDEUREvent(Event):
    """Event to initiate fetching of USD/EUR data from Yahoo."""

    base_date: datetime


class FetchGBPEUREvent(Event):
    """Event to initiate fetching of GBP/EUR data from Yahoo."""

    base_date: datetime


class USDEURResultEvent(Event):
    """Event containing result of USD/EUR data fetch."""

    data: pd.DataFrame | None
    error: str | None


class GBPEURResultEvent(Event):
    """Event containing result of GBP/EUR data fetch."""

    data: pd.DataFrame | None
    error: str | None


class CurrencyWorkflow(Workflow):
    """
    Workflow that fetches currency exchange rate data.

    This workflow fetches EUR/USD and EUR/GBP exchange rates from Yahoo Finance
    to display major currency pairs against the Euro.

    The workflow:
        - Fetches EURUSD=X and EURGBP=X data from Yahoo Finance in parallel
    - Normalizes and aligns the data on common dates
    - Applies base_date filtering for display
    """

    def __init__(self):
        """Initialize workflow with Yahoo provider."""
        super().__init__()
        # Create provider
        self.yahoo_provider = create_yahoo_history_provider(
            period="max", timeout=30.0, retries=2
        )

    @step
    async def initiate_currency_fetch(
        self, ctx: Context, ev: StartEvent
    ) -> FetchUSDEUREvent | FetchGBPEUREvent:
        """
        Step 1: Dispatch - Send parallel fetch events for USD/EUR and GBP/EUR data.
        LlamaIndex automatically executes the sent events in parallel.

        Args:
        ctx: Workflow context for storing state and sending events
        ev.base_date: Start date for data fetching

        Returns:
        FetchUSDEUREvent (dummy event for LlamaIndex validation)
        """
        base_date = ev.base_date

        logger.debug(
            f"CurrencyWorkflow: Dispatching parallel currency data fetch "
            f"from {base_date}"
        )

        # Store shared state for later steps
        await ctx.store.set("base_date", base_date)

        # Send events for parallel execution - LlamaIndex handles the parallelism
        ctx.send_event(FetchUSDEUREvent(base_date=base_date))
        ctx.send_event(FetchGBPEUREvent(base_date=base_date))

        # Return dummy event to satisfy return type - LlamaIndex processes sent events
        return FetchUSDEUREvent(base_date=base_date)

    @step
    async def fetch_usdeur_data(self, ev: FetchUSDEUREvent) -> USDEURResultEvent:
        """
        Step 2: USD/EUR Processing - Fetch USD/EUR data from Yahoo Finance.
        This step runs in parallel with GBP/EUR data fetching automatically via
        LlamaIndex.

        Args:
        ev: FetchUSDEUREvent with fetch details

        Returns:
        USDEURResultEvent with fetch result
        """
        # Mark event parameter as intentionally unused
        _ = ev
        logger.debug("Fetching USD/EUR data from Yahoo Finance")

        # Use standard provider interface - fetch full historical data
        provider_result = await self.yahoo_provider.get_data(
            query="EUR=X",  # EUR/USD exchange rate
        )

        # Handle ProviderResult properly
        if provider_result.success and provider_result.data is not None:
            data_df = provider_result.data
            if isinstance(data_df, pd.DataFrame) and not data_df.empty:
                logger.debug(
                    f"Successfully fetched {len(data_df)} USD/EUR observations"
                )
                return USDEURResultEvent(
                    data=data_df,
                    error=None,
                )
            else:
                logger.warning("Empty USD/EUR data returned")
                return USDEURResultEvent(
                    data=None,
                    error="Empty data returned",
                )
        else:
            # Provider failed
            error_msg = provider_result.error_message or "Provider failed"
            logger.warning(f"USD/EUR provider failed: {error_msg}")
            return USDEURResultEvent(
                data=None,
                error=error_msg,
            )

    @step
    async def fetch_gbpeur_data(self, ev: FetchGBPEUREvent) -> GBPEURResultEvent:
        """
        Step 3: GBP/EUR Processing - Fetch GBP/EUR data from Yahoo Finance.
        This step runs in parallel with USD/EUR data fetching automatically via
        LlamaIndex.

        Args:
        ev: FetchGBPEUREvent with fetch details

        Returns:
        GBPEURResultEvent with fetch result
        """
        # Mark event parameter as intentionally unused
        _ = ev
        logger.debug("Fetching GBP/EUR data from Yahoo Finance")

        # Use standard provider interface - fetch full historical data
        provider_result = await self.yahoo_provider.get_data(
            query="GBPEUR=X",  # GBP/EUR exchange rate
        )

        # Handle ProviderResult properly
        if provider_result.success and provider_result.data is not None:
            data_df = provider_result.data
            if isinstance(data_df, pd.DataFrame) and not data_df.empty:
                logger.debug(
                    f"Successfully fetched {len(data_df)} GBP/EUR observations"
                )
                return GBPEURResultEvent(
                    data=data_df,
                    error=None,
                )
            else:
                logger.warning("Empty GBP/EUR data returned")
                return GBPEURResultEvent(
                    data=None,
                    error="Empty data returned",
                )
        else:
            # Provider failed
            error_msg = provider_result.error_message or "Provider failed"
            logger.warning(f"GBP/EUR provider failed: {error_msg}")
            return GBPEURResultEvent(
                data=None,
                error=error_msg,
            )

    @step
    async def combine_currency_data(
        self, ctx: Context, ev: USDEURResultEvent | GBPEURResultEvent
    ) -> FlowResultEvent | None:
        """
        Step 4: Combine - Combine all parallel results and create currency pairs data.
        Uses collector pattern to wait for both USD/EUR and GBP/EUR results.

        Args:
        ctx: Workflow context for collecting events and accessing stored state
        ev: Either USDEURResultEvent or GBPEURResultEvent from parallel fetches

        Returns:
        FlowResultEvent with processed currency data when both results
        collected, None otherwise
        """
        # Get the stored state from the dispatch step
        base_date = await ctx.store.get("base_date")

        # Collect events until we have both results (USD/EUR + GBP/EUR = 2 events)
        results = ctx.collect_events(ev, [USDEURResultEvent, GBPEURResultEvent])
        if results is None:
            # Not all results collected yet
            return None

        logger.debug(
            f"CurrencyWorkflow: Combining {len(results)} currency data source results"
        )

        # Separate and validate the results
        usdeur_result = None
        gbpeur_result = None

        for result in results:
            if isinstance(result, USDEURResultEvent):
                usdeur_result = result
            elif isinstance(result, GBPEURResultEvent):
                gbpeur_result = result

        # Check that we have both results - raise exceptions that FlowRunner will catch
        if usdeur_result is None:
            raise FlowException(
                workflow="CurrencyWorkflow",
                step="combine_currency_data",
                message="USD/EUR result not received",
            )
        if gbpeur_result is None:
            raise FlowException(
                workflow="CurrencyWorkflow",
                step="combine_currency_data",
                message="GBP/EUR result not received",
            )

        # Check for errors in the results - both are required for currency comparison
        if usdeur_result.error is not None:
            raise FlowException(
                workflow="CurrencyWorkflow",
                step="combine_currency_data",
                message=f"USD/EUR data fetch failed: {usdeur_result.error}",
            )
        if gbpeur_result.error is not None:
            raise FlowException(
                workflow="CurrencyWorkflow",
                step="combine_currency_data",
                message=f"GBP/EUR data fetch failed: {gbpeur_result.error}",
            )

        # Extract the data
        usdeur_data = usdeur_result.data
        gbpeur_data = gbpeur_result.data

        logger.debug("CurrencyWorkflow: Processing currency exchange rate data")

        # Validate data is not None first - let exceptions bubble up
        if usdeur_data is None:
            raise FlowException(
                workflow="CurrencyWorkflow",
                step="combine_currency_data",
                message="USD/EUR data is missing",
            )
        if gbpeur_data is None:
            raise FlowException(
                workflow="CurrencyWorkflow",
                step="combine_currency_data",
                message="GBP/EUR data is missing",
            )

        logger.debug(
            f"Processing USD/EUR data: {len(usdeur_data)} rows, "
            f"GBP/EUR data: {len(gbpeur_data)} rows"
        )

        # Extract close prices from USD/EUR data
        if "Close" in usdeur_data.columns:
            usdeur_close = usdeur_data["Close"].dropna()
        elif "Adj Close" in usdeur_data.columns:
            usdeur_close = usdeur_data["Adj Close"].dropna()
        else:
            raise FlowException(
                workflow="CurrencyWorkflow",
                step="combine_currency_data",
                message="No Close price data in USD/EUR data",
            )

        if usdeur_close.empty:
            raise FlowException(
                workflow="CurrencyWorkflow",
                step="combine_currency_data",
                message="No USD/EUR close price data available",
            )

        # Extract close prices from GBP/EUR data
        if "Close" in gbpeur_data.columns:
            gbpeur_close = gbpeur_data["Close"].dropna()
        elif "Adj Close" in gbpeur_data.columns:
            gbpeur_close = gbpeur_data["Adj Close"].dropna()
        else:
            raise FlowException(
                workflow="CurrencyWorkflow",
                step="combine_currency_data",
                message="No Close price data in GBP/EUR data",
            )

        if gbpeur_close.empty:
            raise FlowException(
                workflow="CurrencyWorkflow",
                step="combine_currency_data",
                message="No GBP/EUR close price data available",
            )

        # Normalize timezones to ensure proper alignment
        if hasattr(usdeur_close.index, "tz") and usdeur_close.index.tz is not None:
            usdeur_close_naive = usdeur_close.tz_localize(None)
        else:
            usdeur_close_naive = usdeur_close

        if hasattr(gbpeur_close.index, "tz") and gbpeur_close.index.tz is not None:
            gbpeur_close_naive = gbpeur_close.tz_localize(None)
        else:
            gbpeur_close_naive = gbpeur_close

        # Create combined DataFrame with aligned dates
        # Use outer join to get all available dates from both series
        result_df = pd.DataFrame(
            {
                "USD_EUR": usdeur_close_naive,
                "GBP_EUR": gbpeur_close_naive,
            }
        )

        # Forward fill missing values for alignment
        # (currency markets may have different holidays)
        result_df = result_df.ffill().dropna()

        if result_df.empty:
            raise FlowException(
                workflow="CurrencyWorkflow",
                step="combine_currency_data",
                message="No overlapping currency data after alignment",
            )

        # Filter by base_date for display purposes
        base_date_pd = pd.to_datetime(base_date.date())
        display_data = result_df.loc[result_df.index >= base_date_pd]

        if display_data.empty:
            logger.warning(f"No currency data after base_date {base_date} for display")
            # Return empty result but don't error - this is just a display filter
            display_data: pd.DataFrame = pd.DataFrame(
                columns=pd.Index(["USD_EUR", "GBP_EUR"])
            )

        logger.info(
            f"Currency processing completed: {len(display_data)} data points "
            f"from {base_date}"
        )

        return FlowResultEvent.success_result(
            data=display_data,
            base_date=base_date,
            metadata={
                "latest_usdeur": (
                    display_data["USD_EUR"].iloc[-1]
                    if not display_data.empty and isinstance(display_data, pd.DataFrame)
                    else None
                ),
                "latest_gbpeur": (
                    display_data["GBP_EUR"].iloc[-1]
                    if not display_data.empty and isinstance(display_data, pd.DataFrame)
                    else None
                ),
                "data_points": len(display_data),
            },
        )


@apply_flow_cache
async def fetch_currency_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process currency exchange rate data using FlowRunner.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with USD/EUR and GBP/EUR rates
        - base_date: The base date used
        - latest_usdeur: Most recent USD/EUR rate
        - latest_gbpeur: Most recent GBP/EUR rate
        - data_points: Number of data points

    Raises:
        Exception: If workflow fails (for backward compatibility)
    """
    logger.info(f"Starting currency data fetch from {base_date}")

    # Create workflow and FlowRunner
    workflow = CurrencyWorkflow()
    runner = FlowRunner[pd.DataFrame](workflow)

    # Run workflow using FlowRunner
    result_event = await runner.run(base_date=base_date)

    if result_event.success:
        logger.info("Currency workflow completed successfully")

        # Convert FlowResultEvent back to dictionary format for backward compatibility
        metadata = result_event.metadata or {}

        return {
            "data": result_event.data,
            "base_date": result_event.base_date,
            "latest_usdeur": metadata.get("latest_usdeur"),
            "latest_gbpeur": metadata.get("latest_gbpeur"),
            "data_points": metadata.get("data_points", 0),
        }

    else:
        # Handle error case - raise exception for backward compatibility
        error_msg = result_event.error_message or "Unknown error"

        logger.error(f"Currency workflow failed: {error_msg}")

        # Raise a standard exception for backward compatibility
        raise FlowException(
            workflow="CurrencyWorkflow",
            step="fetch_currency_data",
            message=f"Currency workflow failed: {error_msg}",
        )
