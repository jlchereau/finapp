"""
LlamaIndex workflow for Buffet Indicator data collection.

This workflow handles fetching economic data for the Buffet Indicator calculation
using FredSeriesProvider and YahooHistoryProvider with FlowRunner architecture.
"""

from typing import Dict, Any
from datetime import datetime

import pandas as pd
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from app.providers.fred import create_fred_series_provider
from app.providers.yahoo import create_yahoo_history_provider
from app.flows.base import FlowRunner, FlowResult
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
        tasks = {
            "GDP": self.fred_provider.get_data("GDP"),  # Nominal GDP, Quarterly
            "Wilshire": self.yahoo_provider.get_data("^FTW5000"),  # Wilshire 5000
        }

        # Use FlowRunner static method for provider task processing
        results = await FlowRunner.process_provider_tasks(tasks)

        # Check for failures and extract data
        if not results["GDP"]["success"]:
            raise WorkflowException(
                workflow="BuffetIndicatorWorkflow",
                step="fetch_economic_data",
                message=f"GDP data fetch failed: {results['GDP']['error']}",
            )
        if not results["Wilshire"]["success"]:
            raise WorkflowException(
                workflow="BuffetIndicatorWorkflow",
                step="fetch_economic_data",
                message=f"Wilshire data fetch failed: {results['Wilshire']['error']}",
            )

        gdp_data = results["GDP"]["data"]
        wilshire_data = results["Wilshire"]["data"]

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
            StopEvent with FlowResult containing calculated indicator data
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
                raise WorkflowException(
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
                logger.error("No Close price data in Wilshire data")
                raise WorkflowException(
                    workflow="BuffetIndicatorWorkflow",
                    step="calculate_buffet_indicator",
                    message="No Close price data available for Wilshire 5000",
                )

            if wilshire_close.empty:
                logger.error("No Wilshire close price data available")
                raise WorkflowException(
                    workflow="BuffetIndicatorWorkflow",
                    step="calculate_buffet_indicator",
                    message="No Wilshire data available",
                )

            # STEP 2: Normalize timezones before resampling
            # CRITICAL: Convert both datasets to timezone-naive to ensure alignment

            # Normalize GDP data timezone (FRED data might be timezone-naive already)
            if hasattr(gdp_values.index, "tz") and gdp_values.index.tz is not None:
                gdp_values_naive = gdp_values.tz_localize(None)
            else:
                gdp_values_naive = gdp_values

            # Normalize Wilshire data timezone (Yahoo data is usually timezone-aware)
            if (
                hasattr(wilshire_close.index, "tz")
                and wilshire_close.index.tz is not None
            ):
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
                raise WorkflowException(
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

            # Return FlowResult instead of dictionary
            return StopEvent(
                result=FlowResult.success_result(
                    data=display_data,
                    base_date=base_date,
                    execution_time=None,  # Will be set by FlowRunner
                    metadata={
                        "trend_data": display_trend_data,
                        "original_period": original_period,
                        "actual_period": actual_period,
                        "was_adjusted": was_adjusted,
                        "latest_value": (
                            display_data["Buffet_Indicator"].iloc[-1]
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
    try:
        logger.info(
            f"Starting Buffet Indicator data fetch from {base_date} "
            f"(period: {original_period})"
        )

        # Create workflow and FlowRunner
        workflow = BuffetIndicatorWorkflow()
        runner = FlowRunner[pd.DataFrame](workflow)

        # Run workflow using FlowRunner
        flow_result = await runner.run(
            base_date=base_date, original_period=original_period
        )

        logger.info("Buffet Indicator workflow completed successfully")

        # Convert FlowResult back to dictionary format for backward compatibility
        if flow_result.success and flow_result.data is not None:
            # Extract metadata fields
            metadata = flow_result.metadata or {}

            return {
                "data": flow_result.data,
                "trend_data": metadata.get("trend_data"),
                "base_date": flow_result.base_date,
                "original_period": metadata.get("original_period", original_period),
                "actual_period": metadata.get("actual_period", original_period),
                "was_adjusted": metadata.get("was_adjusted", False),
                "latest_value": metadata.get("latest_value"),
                "data_points": metadata.get("data_points", 0),
            }
        else:
            # Handle error case
            error_message = flow_result.error_message or "Unknown workflow error"
            raise WorkflowException(
                workflow="fetch_buffet_indicator_data",
                step="process_result",
                message=f"Workflow returned error: {error_message}",
                user_message=(
                    "Failed to fetch Buffet Indicator data due to a system error. "
                    "Please try again."
                ),
                context={"base_date": str(base_date)},
            )

    except WorkflowException:
        # Re-raise WorkflowException without wrapping
        raise
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
