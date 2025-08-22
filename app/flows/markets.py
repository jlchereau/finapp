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

from app.providers.fred import create_fred_series_provider
from app.providers.yahoo import create_yahoo_history_provider
from app.lib.logger import logger
from app.lib.exceptions import WorkflowException
from app.lib.periods import (
    ensure_minimum_data_points,
    format_period_adjustment_message,
    filter_trend_data_to_period,
)
from app.lib.finance import calculate_exponential_trend
from app.flows.cache import apply_flow_cache


class BuffetIndicatorEvent(Event):
    """Event emitted when GDP and Wilshire 5000 data is fetched."""

    gdp_data: pd.DataFrame
    wilshire_data: pd.DataFrame
    base_date: datetime
    original_period: str


class VIXEvent(Event):
    """Event emitted when VIX data is fetched."""

    vix_data: pd.DataFrame
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
            ev.original_period: Original period selected by user (e.g., "2M")

        Returns:
            BuffetIndicatorEvent with GDP and Wilshire data
        """
        base_date = ev.base_date
        original_period = getattr(ev, "original_period", "1Y")  # Default fallback

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
            gdp_data=gdp_data,
            wilshire_data=wilshire_data,
            base_date=base_date,
            original_period=original_period,
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
        original_period = ev.original_period

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
                    f"GDP range: {gdp_quarterly.index.min()} to "
                    f"{gdp_quarterly.index.max()}, Wilshire range: "
                    f"{wilshire_quarterly.index.min()} to "
                    f"{wilshire_quarterly.index.max()}"
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

            # STEP 6: Calculate exponential trend on FULL dataset (before filtering)
            # This ensures trend lines represent full historical context
            full_trend_data = calculate_exponential_trend(result_df, "Buffet_Indicator")

            logger.debug(
                f"Calculated exponential trend on full dataset: "
                f"{len(result_df)} quarters"
            )

            # STEP 7: Apply smart filtering with minimum data points guarantee
            display_data, actual_period, was_adjusted = ensure_minimum_data_points(
                data=result_df,
                original_period=original_period,
                base_date=base_date,
                min_points=2,
                data_frequency="quarterly",
                reference_date=datetime.now(),
            )

            # STEP 8: Filter trend data to match display period
            # This preserves statistical integrity while showing relevant time range
            display_trend_data = filter_trend_data_to_period(
                full_trend_data, display_data
            )

            # Log the filtering results
            if was_adjusted:
                adjustment_msg = format_period_adjustment_message(
                    original_period, actual_period, len(display_data)
                )
                logger.info(f"BuffetIndicator period adjusted: {adjustment_msg}")
            else:
                logger.info(
                    f"BuffetIndicator completed: {len(display_data)} quarters "
                    f"for {original_period} from {base_date}"
                )

            return StopEvent(
                result={
                    "data": display_data,
                    "trend_data": display_trend_data,
                    "base_date": base_date,
                    "original_period": original_period,
                    "actual_period": actual_period,
                    "was_adjusted": was_adjusted,
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
            # Re-raise as WorkflowException for better handling
            raise WorkflowException(
                workflow="BuffetIndicatorWorkflow",
                step="calculate_indicator",
                message=f"Buffet Indicator calculation failed: {e}",
                user_message=(
                    "Failed to calculate Buffet Indicator. Please try again later."
                ),
                context={"base_date": str(base_date)},
            ) from e


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

        # Fetch VIX data
        vix_result = await self.yahoo_provider.get_data("^VIX")

        # Process VIX data
        if isinstance(vix_result, Exception):
            logger.error(f"Failed to fetch VIX data: {vix_result}")
            raise vix_result

        if not (hasattr(vix_result, "success") and vix_result.success):
            error_msg = getattr(vix_result, "error_message", "Unknown VIX fetch error")
            logger.error(f"VIX provider failed: {error_msg}")
            raise Exception(f"VIX data fetch failed: {error_msg}")

        vix_data = vix_result.data
        if vix_data.empty:
            logger.error("Empty VIX data returned")
            raise Exception("No VIX data available")

        logger.debug(
            f"VIX data: {len(vix_data)} rows, "
            f"range: {vix_data.index.min()} to {vix_data.index.max()}"
        )

        return VIXEvent(vix_data=vix_data, base_date=base_date)

    @step
    async def process_vix_data(self, ev: VIXEvent) -> StopEvent:
        """
        Process VIX data and calculate statistics.

        Args:
            ev: VIXEvent with VIX data

        Returns:
            StopEvent with processed VIX data and statistics
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
                raise Exception("No Close price data available for VIX")

            if vix_close.empty:
                logger.error("No VIX close price data available")
                raise Exception("No VIX data available")

            # Normalize timezone to ensure proper filtering
            if vix_close.index.tz is not None:
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
            display_data = result_df[result_df.index >= base_date_pd]

            if display_data.empty:
                logger.warning(f"No VIX data after base_date {base_date} for display")
                # Return empty result but don't error - this is just a display filter
                display_data = pd.DataFrame(columns=["VIX", "VIX_MA50"])

            logger.info(
                f"VIX processing completed: {len(display_data)} data points "
                f"from {base_date}, historical mean: {historical_mean:.2f}"
            )

            return StopEvent(
                result={
                    "data": display_data,
                    "base_date": base_date,
                    "historical_mean": historical_mean,
                    "latest_value": (
                        display_data["VIX"].iloc[-1] if not display_data.empty else None
                    ),
                    "data_points": len(display_data),
                }
            )

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
async def fetch_buffet_indicator_data(
    base_date: datetime, original_period: str = "1Y"
) -> Dict[str, Any]:
    """
    Fetch and calculate Buffet Indicator data.

    Args:
        base_date: Start date for historical data
        original_period: Original period selected by user (for smart filtering)

    Returns:
        Dictionary containing:
        - data: pandas DataFrame with GDP, Wilshire 5000, and Buffet Indicator
        - base_date: The base date used
        - original_period: Original period selected by user
        - actual_period: Period actually used after adjustment
        - was_adjusted: Whether period was adjusted for minimum data points
        - latest_value: Most recent Buffet Indicator value
        - data_points: Number of data points
    """
    try:
        logger.info(
            f"Starting Buffet Indicator data fetch from {base_date} "
            f"(period: {original_period})"
        )

        # Create and run workflow
        workflow = BuffetIndicatorWorkflow()
        result = await workflow.run(
            base_date=base_date, original_period=original_period
        )

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
        # Re-raise as WorkflowException for better handling
        raise WorkflowException(
            workflow="fetch_buffet_indicator_data",
            step="workflow_execution",
            message=f"Buffet Indicator workflow execution failed: {e}",
            user_message=(
                "Failed to fetch Buffet Indicator data due to a system error. "
                "Please try again."
            ),
            context={"base_date": str(base_date)},
        ) from e


@apply_flow_cache
async def fetch_vix_data(base_date: datetime) -> Dict[str, Any]:
    """
    Fetch and process VIX data.

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

        # Create and run workflow
        workflow = VIXWorkflow()
        result = await workflow.run(base_date=base_date)

        logger.info("VIX workflow completed successfully")

        # Extract result data from workflow result
        if hasattr(result, "result"):
            return result.result
        else:
            logger.warning(
                "VIX workflow result missing .result attribute, returning directly"
            )
            return result

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
