"""
LlamaIndex workflow for market page data collection.

This workflow handles fetching economic data for market indicators
like the Buffet Indicator using FredSeriesProvider and YahooHistoryProvider.
"""

import asyncio
from typing import Dict, Any
from datetime import datetime

import pandas as pd
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from ..providers.fred import create_fred_series_provider
from ..providers.yahoo import create_yahoo_history_provider
from ..lib.logger import logger
from .cache import apply_flow_cache


class BuffetIndicatorEvent(Event):
    """Event emitted when GDP and Wilshire 5000 data is fetched."""

    gdp_data: pd.DataFrame
    wilshire_data: pd.DataFrame
    base_date: datetime


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
    async def fetch_economic_data(self, ev: StartEvent) -> BuffetIndicatorEvent:
        """
        Fetch GDP data from FRED and Wilshire 5000 data from Yahoo Finance.

        Args:
            ev.base_date: Start date for data fetching

        Returns:
            BuffetIndicatorEvent with GDP and Wilshire data
        """
        base_date = ev.base_date

        logger.debug(f"BuffetIndicatorWorkflow: Fetching data from {base_date}")

        # Create tasks for parallel execution
        gdp_task = self.fred_provider.get_data("GDP")  # Nominal GDP, Quarterly
        wilshire_task = self.yahoo_provider.get_data("^FTW5000")  # Wilshire 5000

        # Execute tasks in parallel
        results = await asyncio.gather(gdp_task, wilshire_task, return_exceptions=True)

        gdp_result, wilshire_result = results

        # Process GDP data
        if isinstance(gdp_result, Exception):
            logger.error(f"Failed to fetch GDP data: {gdp_result}")
            raise gdp_result

        if not (hasattr(gdp_result, "success") and gdp_result.success):
            error_msg = getattr(gdp_result, "error_message", "Unknown GDP fetch error")
            logger.error(f"GDP provider failed: {error_msg}")
            raise Exception(f"GDP data fetch failed: {error_msg}")

        gdp_data = gdp_result.data
        if gdp_data.empty:
            logger.error("Empty GDP data returned")
            raise Exception("No GDP data available")

        logger.debug(
            f"GDP data: {len(gdp_data)} rows, "
            f"range: {gdp_data.index.min()} to {gdp_data.index.max()}"
        )

        # Process Wilshire data
        if isinstance(wilshire_result, Exception):
            logger.error(f"Failed to fetch Wilshire data: {wilshire_result}")
            raise wilshire_result

        if not (hasattr(wilshire_result, "success") and wilshire_result.success):
            error_msg = getattr(
                wilshire_result, "error_message", "Unknown Wilshire fetch error"
            )
            logger.error(f"Wilshire provider failed: {error_msg}")
            raise Exception(f"Wilshire data fetch failed: {error_msg}")

        wilshire_data = wilshire_result.data
        if wilshire_data.empty:
            logger.error("Empty Wilshire data returned")
            raise Exception("No Wilshire data available")

        logger.debug(
            f"Wilshire data: {len(wilshire_data)} rows, "
            f"range: {wilshire_data.index.min()} to {wilshire_data.index.max()}"
        )

        return BuffetIndicatorEvent(
            gdp_data=gdp_data, wilshire_data=wilshire_data, base_date=base_date
        )

    @step
    async def calculate_buffet_indicator(self, ev: BuffetIndicatorEvent) -> StopEvent:
        """
        Calculate the Buffet Indicator from GDP and Wilshire data.

        The Buffet Indicator is calculated as:
        (Total Market Cap / GDP) * 100

        We use the Wilshire 5000 index as a proxy for total market cap.

        Args:
            ev: BuffetIndicatorEvent with GDP and Wilshire data

        Returns:
            StopEvent with calculated indicator data
        """
        gdp_data = ev.gdp_data
        wilshire_data = ev.wilshire_data
        base_date = ev.base_date

        logger.debug("BuffetIndicatorWorkflow: Calculating Buffet Indicator")

        try:
            logger.debug(
                f"Processing GDP data: {len(gdp_data)} rows, "
                f"Wilshire data: {len(wilshire_data)} rows"
            )

            # STEP 1: Extract and process data on FULL datasets (no filtering yet)

            # Get GDP values from full dataset
            gdp_values = gdp_data["value"].dropna()
            if gdp_values.empty:
                logger.error("No GDP data available in full dataset")
                raise Exception("No GDP data available")

            # Get close prices from Wilshire data (full dataset)
            if "Close" in wilshire_data.columns:
                wilshire_close = wilshire_data["Close"].dropna()
            elif "Adj Close" in wilshire_data.columns:
                wilshire_close = wilshire_data["Adj Close"].dropna()
            else:
                logger.error("No Close price data in Wilshire data")
                raise Exception("No Close price data available for Wilshire 5000")

            if wilshire_close.empty:
                logger.error("No Wilshire close price data available")
                raise Exception("No Wilshire data available")

            # STEP 2: Normalize timezones before resampling
            # CRITICAL: Convert both datasets to timezone-naive to ensure alignment

            # Normalize GDP data timezone (FRED data might be timezone-naive already)
            if gdp_values.index.tz is not None:
                gdp_values_naive = gdp_values.tz_localize(None)
            else:
                gdp_values_naive = gdp_values

            # Normalize Wilshire data timezone (Yahoo data is usually timezone-aware)
            if wilshire_close.index.tz is not None:
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
                logger.error("No overlapping quarters between GDP and Wilshire data")
                logger.debug(
                    f"GDP range: {gdp_quarterly.index.min()} to {gdp_quarterly.index.max()}, "
                    f"Wilshire range: {wilshire_quarterly.index.min()} to {wilshire_quarterly.index.max()}"
                )
                raise Exception("No overlapping dates between GDP and Wilshire data")

            logger.debug(f"Found {len(common_quarters)} overlapping quarters")

            # Align the data using pandas division
            # (automatically aligns on common index)
            buffet_indicator_full = (wilshire_quarterly / gdp_quarterly) * 100
            buffet_indicator_full = buffet_indicator_full.dropna()

            # Get aligned data for result DataFrame
            common_dates = buffet_indicator_full.index
            aligned_gdp = gdp_quarterly.loc[common_dates]
            aligned_wilshire = wilshire_quarterly.loc[common_dates]

            logger.debug(
                f"Buffet Indicator calculated for {len(common_dates)} quarters"
            )

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

            # STEP 6: ONLY NOW filter by base_date for display purposes
            base_date_pd = pd.to_datetime(base_date.date())
            display_data = result_df[result_df.index >= base_date_pd]

            if display_data.empty:
                logger.warning(f"No data after base_date {base_date} for display")
                # Return empty result but don't error - this is just a display filter
                display_data = pd.DataFrame(
                    columns=["GDP", "Wilshire_5000", "Buffet_Indicator"]
                )

            logger.info(
                f"BuffetIndicator completed: {len(display_data)} quarters from {base_date}"
            )

            return StopEvent(
                result={
                    "data": display_data,
                    "base_date": base_date,
                    "latest_value": (
                        display_data["Buffet_Indicator"].iloc[-1]
                        if not display_data.empty
                        else None
                    ),
                    "data_points": len(display_data),
                }
            )

        except Exception as e:
            logger.error(f"Error calculating Buffet Indicator: {e}")
            return StopEvent(
                result={"data": pd.DataFrame(), "base_date": base_date, "error": str(e)}
            )


@apply_flow_cache
async def fetch_buffet_indicator_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and calculate Buffet Indicator data.

    Args:
        base_date: Start date for historical data

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with GDP, Wilshire 5000, and Buffet Indicator
        - base_date: The base date used
        - latest_value: Most recent Buffet Indicator value
        - data_points: Number of data points
    """
    try:
        logger.info(f"Starting Buffet Indicator data fetch from {base_date}")

        # Create and run workflow
        workflow = BuffetIndicatorWorkflow()
        result = await workflow.run(base_date=base_date)

        logger.info("Buffet Indicator workflow completed successfully")

        # Extract result data from workflow result (similar to test.py pattern)
        if hasattr(result, "result"):
            return result.result
        else:
            logger.warning(
                "Workflow result missing .result attribute, returning directly"
            )
            return result

    except Exception as e:
        logger.error(f"Buffet Indicator workflow failed: {e}")
        return {"data": pd.DataFrame(), "base_date": base_date, "error": str(e)}
