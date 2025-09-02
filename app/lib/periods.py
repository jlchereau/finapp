"""
Period handling utilities for financial data filtering.

This module provides centralized period definitions and date calculations
for consistent time period handling across the application.
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import List, Tuple
import pandas as pd
import numpy as np


class PeriodOption(str, Enum):
    """Available time period options for charts and data filtering."""

    ONE_WEEK = "1W"
    TWO_WEEKS = "2W"
    ONE_MONTH = "1M"
    TWO_MONTHS = "2M"
    ONE_QUARTER = "1Q"
    TWO_QUARTERS = "2Q"
    THREE_QUARTERS = "3Q"
    ONE_YEAR = "1Y"
    TWO_YEARS = "2Y"
    THREE_YEARS = "3Y"
    FOUR_YEARS = "4Y"
    FIVE_YEARS = "5Y"
    TEN_YEARS = "10Y"
    TWENTY_YEARS = "20Y"
    YEAR_TO_DATE = "YTD"
    MAXIMUM = "MAX"


def get_period_options() -> List[str]:
    """
    Get list of all available period options.

    Returns:
        List of period option strings in display order
    """
    return [
        PeriodOption.ONE_WEEK,
        PeriodOption.TWO_WEEKS,
        PeriodOption.ONE_MONTH,
        PeriodOption.TWO_MONTHS,
        PeriodOption.ONE_QUARTER,
        PeriodOption.TWO_QUARTERS,
        PeriodOption.THREE_QUARTERS,
        PeriodOption.ONE_YEAR,
        PeriodOption.TWO_YEARS,
        PeriodOption.THREE_YEARS,
        PeriodOption.FOUR_YEARS,
        PeriodOption.FIVE_YEARS,
        PeriodOption.TEN_YEARS,
        PeriodOption.TWENTY_YEARS,
        PeriodOption.YEAR_TO_DATE,
        PeriodOption.MAXIMUM,
    ]


def get_period_default() -> str:
    """
    Get default period option.

    Returns:
        Default period option string
    """
    return PeriodOption.ONE_YEAR


def calculate_base_date(
    period: str, reference_date: datetime | None = None
) -> datetime | None:
    """
    Convert period option to base datetime.

    Args:
        period: Period option string (e.g., "1Y", "YTD", "MAX")
        reference_date: Reference date for calculations (defaults to now)

    Returns:
        Base datetime for the period, or None for MAX option

    Raises:
        ValueError: If period is not recognized
    """
    if reference_date is None:
        reference_date = datetime.now()

    # Handle MAX option
    if period == PeriodOption.MAXIMUM:
        return None

    # Handle YTD option
    if period == PeriodOption.YEAR_TO_DATE:
        return datetime(reference_date.year, 1, 1)

    # Week-based periods
    if period == PeriodOption.ONE_WEEK:
        return reference_date - timedelta(weeks=1)
    elif period == PeriodOption.TWO_WEEKS:
        return reference_date - timedelta(weeks=2)

    # Month-based periods (approximate)
    elif period == PeriodOption.ONE_MONTH:
        return reference_date - timedelta(days=30)
    elif period == PeriodOption.TWO_MONTHS:
        return reference_date - timedelta(days=60)

    # Quarter-based periods
    elif period == PeriodOption.ONE_QUARTER:
        return reference_date - timedelta(days=90)
    elif period == PeriodOption.TWO_QUARTERS:
        return reference_date - timedelta(days=180)
    elif period == PeriodOption.THREE_QUARTERS:
        return reference_date - timedelta(days=270)

    # Year-based periods
    elif period == PeriodOption.ONE_YEAR:
        return reference_date - timedelta(days=365)
    elif period == PeriodOption.TWO_YEARS:
        return reference_date - timedelta(days=730)
    elif period == PeriodOption.THREE_YEARS:
        return reference_date - timedelta(days=1095)
    elif period == PeriodOption.FOUR_YEARS:
        return reference_date - timedelta(days=1460)
    elif period == PeriodOption.FIVE_YEARS:
        return reference_date - timedelta(days=1825)
    elif period == PeriodOption.TEN_YEARS:
        return reference_date - timedelta(days=3650)
    elif period == PeriodOption.TWENTY_YEARS:
        return reference_date - timedelta(days=7300)

    else:
        raise ValueError(f"Unknown period option: {period}")


def get_period_description(period: str) -> str:
    """
    Get human-readable description of period.

    Args:
        period: Period option string

    Returns:
        Human-readable description

    Raises:
        ValueError: If period is not recognized
    """
    descriptions = {
        PeriodOption.ONE_WEEK.value: "1 Week",
        PeriodOption.TWO_WEEKS.value: "2 Weeks",
        PeriodOption.ONE_MONTH.value: "1 Month",
        PeriodOption.TWO_MONTHS.value: "2 Months",
        PeriodOption.ONE_QUARTER.value: "1 Quarter",
        PeriodOption.TWO_QUARTERS.value: "2 Quarters",
        PeriodOption.THREE_QUARTERS.value: "3 Quarters",
        PeriodOption.ONE_YEAR.value: "1 Year",
        PeriodOption.TWO_YEARS.value: "2 Years",
        PeriodOption.THREE_YEARS.value: "3 Years",
        PeriodOption.FOUR_YEARS.value: "4 Years",
        PeriodOption.FIVE_YEARS.value: "5 Years",
        PeriodOption.TEN_YEARS.value: "10 Years",
        PeriodOption.TWENTY_YEARS.value: "20 Years",
        PeriodOption.YEAR_TO_DATE.value: "Year to Date",
        PeriodOption.MAXIMUM.value: "Maximum Available",
    }

    if period not in descriptions:
        raise ValueError(f"Unknown period option: {period}")

    return descriptions[period]


def get_max_fallback_date(data_type: str = "default") -> datetime:
    """
    Get appropriate fallback date for MAX option based on data type.

    Args:
        data_type: Type of data ("vix", "markets", "stocks", or "default")

    Returns:
        Appropriate fallback datetime for the data type
    """
    fallbacks = {
        "vix": datetime(1990, 1, 1),  # VIX started in 1990
        "markets": datetime(1970, 1, 1),  # General market data
        "stocks": datetime(2000, 1, 1),  # Stock comparison data
        "default": datetime(1970, 1, 1),  # Conservative default
    }

    return fallbacks.get(data_type, fallbacks["default"])


def format_date_range_message(period: str, base_date: datetime | None) -> str:
    """
    Format user-friendly message for date range selection.

    Args:
        period: Period option string
        base_date: Calculated base date (None for MAX)

    Returns:
        Formatted message for user feedback
    """
    if base_date is None:
        return "Loading maximum available data..."

    if period == PeriodOption.YEAR_TO_DATE:
        return f"Loading data from {get_period_description(period)}"

    date_str = base_date.strftime("%Y-%m-%d")
    return f"Loading data from {get_period_description(period)} ({date_str})"


def ensure_minimum_data_points(
    data: pd.DataFrame,
    original_period: str,
    base_date: datetime,
    min_points: int = 2,
    data_frequency: str = "quarterly",
    reference_date: datetime | None = None,
) -> Tuple[pd.DataFrame, str, bool]:
    """
    Ensure filtered data has minimum number of points by extending period if needed.

    This function addresses the issue where economic indicators with quarterly data
    fail to show any points when users select short periods (e.g., 2M) that fall
    between quarterly data points.

    Args:
        data: DataFrame with datetime index containing the data to filter
        original_period: Original period selected by user (e.g., "2M", "1Q")
        base_date: Calculated base date for the original period
        min_points: Minimum number of data points required (default: 2)
        data_frequency: Frequency of the data ("quarterly", "monthly", "daily")
        reference_date: Reference date for calculations (defaults to now)

    Returns:
        Tuple of (filtered_data, actual_period_used, was_adjusted)
        - filtered_data: DataFrame filtered to show adequate data points
        - actual_period_used: Period that was actually used (may differ from original)
        - was_adjusted: Boolean indicating if period was extended from original

    Example:
        >>> # Quarterly GDP data with 2M period that falls between quarters
        >>> filtered_data, period_used, adjusted = ensure_minimum_data_points(
        ...     data=gdp_data,
        ...     original_period="2M",
        ...     base_date=datetime(2024, 1, 15),  # Falls between Q4 and Q1
        ...     min_points=2,
        ...     data_frequency="quarterly"
        ... )
        >>> # Result: period_used="1Q", adjusted=True, shows Q1 data
    """
    if reference_date is None:
        reference_date = datetime.now()

    # Define fallback sequence based on data frequency
    if data_frequency == "quarterly":
        # For quarterly data, ensure minimum periods that work with quarters
        fallback_sequence = ["1Q", "2Q", "3Q", "1Y", "2Y", "5Y", "MAX"]
    elif data_frequency == "monthly":
        # For monthly data, start with reasonable monthly periods
        fallback_sequence = ["1M", "2M", "1Q", "2Q", "1Y", "2Y", "MAX"]
    else:  # daily or other high-frequency data
        # For daily data, keep original short periods
        fallback_sequence = ["1W", "1M", "1Q", "2Q", "1Y", "2Y", "MAX"]

    # Start with the original period
    current_period = original_period
    base_date_pd = pd.to_datetime(base_date.date())

    # Handle MAX period immediately - return all data
    if original_period == "MAX":
        return data, current_period, False

    # Try filtering with original period first (handle empty data)
    if len(data) == 0:
        return data, current_period, False  # Empty data, nothing to filter

    filtered_data = data[data.index >= base_date_pd]

    # Ensure we return a proper DataFrame
    if not isinstance(filtered_data, pd.DataFrame):
        filtered_data = pd.DataFrame(filtered_data)

    # If we have enough data points, return as-is
    if len(filtered_data) >= min_points:
        return filtered_data, current_period, False

    # If original period is already in fallback sequence, start from there
    # Otherwise start from beginning of fallback sequence
    if current_period in fallback_sequence:
        start_idx = fallback_sequence.index(current_period) + 1
    else:
        start_idx = 0

    # Try progressively longer periods until we get enough data
    for period in fallback_sequence[start_idx:]:
        try:
            # Calculate new base date for this period
            if period == "MAX":
                # For MAX, use all available data
                filtered_data = data.copy()
                current_period = period
                break
            else:
                new_base_date = calculate_base_date(period, reference_date)
                if new_base_date is None:
                    # This should only happen for MAX, which we handle above
                    continue

                new_base_date_pd = pd.to_datetime(new_base_date.date())
                filtered_data = data[data.index >= new_base_date_pd]

                # Ensure we return a proper DataFrame
                if not isinstance(filtered_data, pd.DataFrame):
                    filtered_data = pd.DataFrame(filtered_data)

                # Check if this period gives us enough data points
                if len(filtered_data) >= min_points:
                    current_period = period
                    break

        except ValueError:
            # Skip invalid period options
            continue

    # Return results
    was_adjusted = current_period != original_period
    return filtered_data, current_period, was_adjusted


def format_period_adjustment_message(
    original_period: str, actual_period: str, data_points: int
) -> str:
    """
    Format user-friendly message when period gets adjusted for minimum data points.

    Args:
        original_period: Period originally selected by user
        actual_period: Period actually used after adjustment
        data_points: Number of data points in result

    Returns:
        Formatted message explaining the adjustment
    """
    if original_period == actual_period:
        return (
            f"Loaded {data_points} data points for "
            f"{get_period_description(original_period)}"
        )

    original_desc = get_period_description(original_period)
    actual_desc = get_period_description(actual_period)

    return (
        f"Extended period from {original_desc} to {actual_desc} "
        f"to show sufficient data ({data_points} points)"
    )


def filter_trend_data_to_period(trend_data, filtered_data: pd.DataFrame):
    """
    Filter trend data to match the time period of filtered display data.

    Args:
        trend_data: Full trend data dictionary from calculate_exponential_trend
        filtered_data: Filtered DataFrame showing the display period

    Returns:
        Filtered trend data dictionary with same structure but matching time range
    """
    if trend_data is None or filtered_data.empty:
        return trend_data

    # Get date range from filtered data
    start_date = filtered_data.index.min()
    end_date = filtered_data.index.max()

    # Convert trend dates to DatetimeIndex for consistent operations
    trend_dates = pd.DatetimeIndex(trend_data["dates"])

    # Create boolean mask - this will work with any comparable datetime types
    date_mask = (trend_dates >= start_date) & (trend_dates <= end_date)

    # Filter all trend data arrays using the same mask
    filtered_trend = {}
    for key, values in trend_data.items():
        if key == "dates":
            # Preserve the filtered DatetimeIndex
            filtered_trend[key] = trend_dates[date_mask]
        else:
            # Convert to numpy array for consistent indexing
            values_array = np.array(values)
            mask_array = np.array(date_mask)
            filtered_trend[key] = values_array[mask_array].tolist()

    return filtered_trend
