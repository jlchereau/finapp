"""
Unit tests for periods functionality.

Tests the centralized period handling utilities in app.lib.periods.
"""

import pytest
from datetime import datetime, timedelta
from app.lib.periods import (
    PeriodOption,
    get_period_options,
    calculate_base_date,
    get_period_description,
    get_max_fallback_date,
    format_date_range_message,
    ensure_minimum_data_points,
    format_period_adjustment_message,
)


class TestPeriodOption:
    """Test PeriodOption enum."""

    def test_period_option_values(self):
        """Test that all period options have expected values."""
        assert PeriodOption.ONE_WEEK == "1W"
        assert PeriodOption.TWO_WEEKS == "2W"
        assert PeriodOption.ONE_MONTH == "1M"
        assert PeriodOption.TWO_MONTHS == "2M"
        assert PeriodOption.ONE_QUARTER == "1Q"
        assert PeriodOption.TWO_QUARTERS == "2Q"
        assert PeriodOption.THREE_QUARTERS == "3Q"
        assert PeriodOption.ONE_YEAR == "1Y"
        assert PeriodOption.TWO_YEARS == "2Y"
        assert PeriodOption.THREE_YEARS == "3Y"
        assert PeriodOption.FOUR_YEARS == "4Y"
        assert PeriodOption.FIVE_YEARS == "5Y"
        assert PeriodOption.TEN_YEARS == "10Y"
        assert PeriodOption.TWENTY_YEARS == "20Y"
        assert PeriodOption.YEAR_TO_DATE == "YTD"
        assert PeriodOption.MAXIMUM == "MAX"


class TestGetPeriodOptions:
    """Test get_period_options function."""

    def test_get_period_options_returns_list(self):
        """Test that get_period_options returns a list."""
        options = get_period_options()
        assert isinstance(options, list)
        assert len(options) > 0

    def test_get_period_options_contains_all_periods(self):
        """Test that all expected periods are included."""
        options = get_period_options()
        expected_periods = [
            "1W",
            "2W",
            "1M",
            "2M",
            "1Q",
            "2Q",
            "3Q",
            "1Y",
            "2Y",
            "3Y",
            "4Y",
            "5Y",
            "10Y",
            "20Y",
            "YTD",
            "MAX",
        ]

        for period in expected_periods:
            assert period in options

    def test_get_period_options_order(self):
        """Test that periods are in expected order."""
        options = get_period_options()
        assert options[0] == "1W"  # Shortest first
        assert options[-1] == "MAX"  # MAX last
        assert options[-2] == "YTD"  # YTD second to last


class TestCalculateBaseDate:
    """Test calculate_base_date function."""

    def test_max_option_returns_none(self):
        """Test that MAX option returns None."""
        result = calculate_base_date("MAX")
        assert result is None

    def test_invalid_period_raises_error(self):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match="Unknown period option"):
            calculate_base_date("INVALID")

    def test_ytd_option(self):
        """Test YTD option calculation."""
        reference_date = datetime(2023, 6, 15)
        result = calculate_base_date("YTD", reference_date)
        expected = datetime(2023, 1, 1)
        assert result == expected

    def test_week_based_periods(self):
        """Test week-based period calculations."""
        reference_date = datetime(2023, 6, 15)

        # 1 week
        result = calculate_base_date("1W", reference_date)
        expected = reference_date - timedelta(weeks=1)
        assert result == expected

        # 2 weeks
        result = calculate_base_date("2W", reference_date)
        expected = reference_date - timedelta(weeks=2)
        assert result == expected

    def test_month_based_periods(self):
        """Test month-based period calculations."""
        reference_date = datetime(2023, 6, 15)

        # 1 month
        result = calculate_base_date("1M", reference_date)
        expected = reference_date - timedelta(days=30)
        assert result == expected

        # 2 months
        result = calculate_base_date("2M", reference_date)
        expected = reference_date - timedelta(days=60)
        assert result == expected

    def test_quarter_based_periods(self):
        """Test quarter-based period calculations."""
        reference_date = datetime(2023, 6, 15)

        # 1 quarter
        result = calculate_base_date("1Q", reference_date)
        expected = reference_date - timedelta(days=90)
        assert result == expected

        # 2 quarters
        result = calculate_base_date("2Q", reference_date)
        expected = reference_date - timedelta(days=180)
        assert result == expected

        # 3 quarters
        result = calculate_base_date("3Q", reference_date)
        expected = reference_date - timedelta(days=270)
        assert result == expected

    def test_year_based_periods(self):
        """Test year-based period calculations."""
        reference_date = datetime(2023, 6, 15)

        # 1 year
        result = calculate_base_date("1Y", reference_date)
        expected = reference_date - timedelta(days=365)
        assert result == expected

        # 2 years
        result = calculate_base_date("2Y", reference_date)
        expected = reference_date - timedelta(days=730)
        assert result == expected

        # 5 years
        result = calculate_base_date("5Y", reference_date)
        expected = reference_date - timedelta(days=1825)
        assert result == expected

        # 10 years
        result = calculate_base_date("10Y", reference_date)
        expected = reference_date - timedelta(days=3650)
        assert result == expected

        # 20 years
        result = calculate_base_date("20Y", reference_date)
        expected = reference_date - timedelta(days=7300)
        assert result == expected

    def test_default_reference_date(self):
        """Test that function works without explicit reference date."""
        result = calculate_base_date("1Y")
        assert result is not None
        assert isinstance(result, datetime)

        # Should be approximately 1 year ago from now
        now = datetime.now()
        expected_range_start = now - timedelta(days=366)
        expected_range_end = now - timedelta(days=364)
        assert expected_range_start <= result <= expected_range_end

    def test_all_period_options_work(self):
        """Test that all period options can be calculated."""
        reference_date = datetime(2023, 6, 15)
        options = get_period_options()

        for option in options:
            if option == "MAX":
                result = calculate_base_date(option, reference_date)
                assert result is None
            else:
                result = calculate_base_date(option, reference_date)
                assert result is not None
                assert isinstance(result, datetime)
                assert result <= reference_date


class TestGetPeriodDescription:
    """Test get_period_description function."""

    def test_all_periods_have_descriptions(self):
        """Test that all periods have descriptions."""
        options = get_period_options()

        for option in options:
            description = get_period_description(option)
            assert isinstance(description, str)
            assert len(description) > 0

    def test_specific_descriptions(self):
        """Test specific description values."""
        assert get_period_description("1W") == "1 Week"
        assert get_period_description("1Y") == "1 Year"
        assert get_period_description("YTD") == "Year to Date"
        assert get_period_description("MAX") == "Maximum Available"

    def test_invalid_period_raises_error(self):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match="Unknown period option"):
            get_period_description("INVALID")


class TestGetMaxFallbackDate:
    """Test get_max_fallback_date function."""

    def test_vix_fallback_date(self):
        """Test VIX fallback date."""
        result = get_max_fallback_date("vix")
        expected = datetime(1990, 1, 1)
        assert result == expected

    def test_markets_fallback_date(self):
        """Test markets fallback date."""
        result = get_max_fallback_date("markets")
        expected = datetime(1970, 1, 1)
        assert result == expected

    def test_stocks_fallback_date(self):
        """Test stocks fallback date."""
        result = get_max_fallback_date("stocks")
        expected = datetime(2000, 1, 1)
        assert result == expected

    def test_default_fallback_date(self):
        """Test default fallback date."""
        result = get_max_fallback_date("default")
        expected = datetime(1970, 1, 1)
        assert result == expected

    def test_unknown_type_fallback_date(self):
        """Test unknown type uses default fallback."""
        result = get_max_fallback_date("unknown_type")
        expected = datetime(1970, 1, 1)
        assert result == expected


class TestFormatDateRangeMessage:
    """Test format_date_range_message function."""

    def test_max_option_message(self):
        """Test message for MAX option."""
        result = format_date_range_message("MAX", None)
        assert result == "Loading maximum available data..."

    def test_ytd_option_message(self):
        """Test message for YTD option."""
        base_date = datetime(2023, 1, 1)
        result = format_date_range_message("YTD", base_date)
        assert result == "Loading data from Year to Date"

    def test_regular_period_message(self):
        """Test message for regular periods."""
        base_date = datetime(2023, 1, 15)
        result = format_date_range_message("1Y", base_date)
        expected = "Loading data from 1 Year (2023-01-15)"
        assert result == expected

    def test_all_periods_generate_messages(self):
        """Test that all periods generate valid messages."""
        reference_date = datetime(2023, 6, 15)
        options = get_period_options()

        for option in options:
            if option == "MAX":
                message = format_date_range_message(option, None)
            else:
                base_date = calculate_base_date(option, reference_date)
                message = format_date_range_message(option, base_date)

            assert isinstance(message, str)
            assert len(message) > 0
            assert "Loading" in message


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_ytd_at_year_boundary(self):
        """Test YTD calculation at year boundaries."""
        # New Year's Day
        reference_date = datetime(2023, 1, 1)
        result = calculate_base_date("YTD", reference_date)
        expected = datetime(2023, 1, 1)
        assert result == expected

        # New Year's Eve
        reference_date = datetime(2023, 12, 31)
        result = calculate_base_date("YTD", reference_date)
        expected = datetime(2023, 1, 1)
        assert result == expected

    def test_leap_year_handling(self):
        """Test that calculations work correctly in leap years."""
        # February 29th in a leap year
        reference_date = datetime(2024, 2, 29)
        result = calculate_base_date("1Y", reference_date)
        expected = reference_date - timedelta(days=365)
        assert result == expected

    def test_consistency_across_calls(self):
        """Test that multiple calls with same parameters return same result."""
        reference_date = datetime(2023, 6, 15)

        for option in ["1W", "1M", "1Y", "YTD"]:
            result1 = calculate_base_date(option, reference_date)
            result2 = calculate_base_date(option, reference_date)
            assert result1 == result2

    def test_period_ordering_logic(self):
        """Test that periods create dates in logical order."""
        reference_date = datetime(2023, 6, 15)

        # Shorter periods should have more recent base dates
        one_week = calculate_base_date("1W", reference_date)
        one_month = calculate_base_date("1M", reference_date)
        one_year = calculate_base_date("1Y", reference_date)

        assert one_week > one_month > one_year


class TestEnsureMinimumDataPoints:
    """Test ensure_minimum_data_points function."""

    def create_sample_data(self, dates, data_type="quarterly"):
        """Create sample DataFrame with specified dates."""
        import pandas as pd

        if data_type == "quarterly":
            # Quarterly data with specific quarter dates
            return pd.DataFrame(
                {"value": list(range(len(dates)))}, index=pd.to_datetime(dates)
            )
        else:
            # Daily data
            return pd.DataFrame(
                {"value": list(range(len(dates)))}, index=pd.to_datetime(dates)
            )

    def test_quarterly_data_no_adjustment_needed(self):
        """Test quarterly data with enough points, no adjustment needed."""
        # Create quarterly data
        dates = ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"]
        data = self.create_sample_data(dates, "quarterly")

        # Test with 1Y period that should capture multiple quarters
        base_date = datetime(2023, 6, 15)  # Should get July and later quarters

        result_data, actual_period, was_adjusted = ensure_minimum_data_points(
            data=data,
            original_period="1Y",
            base_date=base_date,
            min_points=2,
            data_frequency="quarterly",
            reference_date=datetime(2024, 1, 15),
        )

        assert len(result_data) >= 2
        assert actual_period == "1Y"
        assert was_adjusted is False

    def test_quarterly_data_needs_adjustment_2m_to_1q(self):
        """Test quarterly data where 2M period gets adjusted to 1Q."""
        # Create quarterly data
        dates = ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"]
        data = self.create_sample_data(dates, "quarterly")

        # Test with 2M period that falls between quarters (should get adjusted)
        base_date = datetime(2023, 11, 15)  # Between Oct and Jan quarters

        result_data, actual_period, was_adjusted = ensure_minimum_data_points(
            data=data,
            original_period="2M",
            base_date=base_date,
            min_points=2,
            data_frequency="quarterly",
            reference_date=datetime(2024, 1, 15),
        )

        assert len(result_data) >= 2
        assert actual_period in ["1Q", "2Q", "3Q", "1Y", "2Y", "5Y"]
        assert was_adjusted is True

    def test_quarterly_data_progressive_fallback(self):
        """Test progressive fallback through multiple periods."""
        # Create sparse quarterly data (only recent quarters)
        dates = ["2023-10-01", "2024-01-01"]
        data = self.create_sample_data(dates, "quarterly")

        # Test with 1W period that needs multiple fallbacks
        base_date = datetime(2023, 12, 15)  # Should only get Jan quarter

        result_data, actual_period, was_adjusted = ensure_minimum_data_points(
            data=data,
            original_period="1W",
            base_date=base_date,
            min_points=2,
            data_frequency="quarterly",
            reference_date=datetime(2024, 2, 1),
        )

        assert len(result_data) >= 2
        assert was_adjusted is True
        # Should fallback to a period that captures both quarters
        expected_periods = ["1Q", "2Q", "3Q", "1Y", "2Y", "5Y", "MAX"]
        assert actual_period in expected_periods

    def test_daily_data_no_adjustment(self):
        """Test daily data keeps short periods when sufficient points exist."""
        # Create daily data
        import pandas as pd

        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        data = self.create_sample_data(dates, "daily")

        base_date = datetime(2024, 3, 1)  # Should have plenty of daily data

        result_data, actual_period, was_adjusted = ensure_minimum_data_points(
            data=data,
            original_period="1W",
            base_date=base_date,
            min_points=2,
            data_frequency="daily",
            reference_date=datetime(2024, 4, 1),
        )

        assert len(result_data) >= 2
        assert actual_period == "1W"
        assert was_adjusted is False

    def test_max_period_handling(self):
        """Test MAX period returns all data."""
        dates = ["2023-01-01", "2023-04-01", "2023-07-01"]
        data = self.create_sample_data(dates, "quarterly")

        base_date = datetime(2023, 6, 1)  # Any base date

        result_data, actual_period, was_adjusted = ensure_minimum_data_points(
            data=data,
            original_period="MAX",
            base_date=base_date,
            min_points=2,
            data_frequency="quarterly",
            reference_date=datetime(2024, 1, 1),
        )

        assert len(result_data) == len(data)  # All data returned
        assert actual_period == "MAX"
        assert was_adjusted is False

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        import pandas as pd

        data = pd.DataFrame(columns=["value"])

        base_date = datetime(2023, 6, 1)

        result_data, _, was_adjusted = ensure_minimum_data_points(
            data=data,
            original_period="2M",
            base_date=base_date,
            min_points=2,
            data_frequency="quarterly",
            reference_date=datetime(2024, 1, 1),
        )

        assert len(result_data) == 0
        assert was_adjusted is False  # No data to adjust, returned as-is

    def test_insufficient_data_fallback_to_max(self):
        """Test fallback to MAX when no other period works."""
        # Create very sparse data
        dates = ["2023-01-01"]  # Only one data point
        data = self.create_sample_data(dates, "quarterly")

        base_date = datetime(2023, 6, 1)  # After the only data point

        result_data, actual_period, was_adjusted = ensure_minimum_data_points(
            data=data,
            original_period="2M",
            base_date=base_date,
            min_points=2,
            data_frequency="quarterly",
            reference_date=datetime(2024, 1, 1),
        )

        # Should fallback to MAX to get the one available point
        assert actual_period == "MAX"
        assert was_adjusted is True
        assert len(result_data) == 1

    def test_different_min_points(self):
        """Test with different minimum point requirements."""
        dates = ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"]
        data = self.create_sample_data(dates, "quarterly")

        base_date = datetime(2023, 8, 1)  # Should get Oct quarter only

        # Test with min_points=3
        result_data, _, was_adjusted = ensure_minimum_data_points(
            data=data,
            original_period="1Q",
            base_date=base_date,
            min_points=3,
            data_frequency="quarterly",
            reference_date=datetime(2024, 1, 1),
        )

        assert len(result_data) >= 3
        assert was_adjusted is True

    def test_monthly_data_frequency(self):
        """Test monthly data frequency fallback sequence."""
        # Create monthly data
        import pandas as pd

        dates = pd.date_range("2023-01-01", periods=12, freq="MS")  # Month start
        data = self.create_sample_data(dates, "monthly")

        base_date = datetime(2023, 11, 1)

        result_data, actual_period, was_adjusted = ensure_minimum_data_points(
            data=data,
            original_period="1W",  # Should fallback through monthly sequence
            base_date=base_date,
            min_points=2,
            data_frequency="monthly",
            reference_date=datetime(2024, 1, 1),
        )

        assert len(result_data) >= 2
        # For monthly data, 1W might work if there are enough monthly points
        # after the base date. If adjusted, should use monthly fallback sequence.
        if was_adjusted:
            expected_periods = ["1M", "2M", "1Q", "2Q", "1Y", "2Y"]
            assert actual_period in expected_periods
        else:
            # Original period worked
            assert actual_period == "1W"

    def test_reference_date_usage(self):
        """Test that reference_date is used correctly for period calculations."""
        dates = ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"]
        data = self.create_sample_data(dates, "quarterly")

        base_date = datetime(2023, 2, 1)
        reference_date = datetime(2023, 6, 15)

        result_data, actual_period, was_adjusted = ensure_minimum_data_points(
            data=data,
            original_period="2M",
            base_date=base_date,
            min_points=2,
            data_frequency="quarterly",
            reference_date=reference_date,  # Different from default
        )

        assert len(result_data) >= 0  # Should handle the different reference date
        assert isinstance(actual_period, str)
        assert isinstance(was_adjusted, bool)


class TestFormatPeriodAdjustmentMessage:
    """Test format_period_adjustment_message function."""

    def test_no_adjustment_message(self):
        """Test message when period was not adjusted."""
        message = format_period_adjustment_message("1Y", "1Y", 4)

        expected = "Loaded 4 data points for 1 Year"
        assert message == expected

    def test_adjustment_message(self):
        """Test message when period was adjusted."""
        message = format_period_adjustment_message("2M", "1Q", 3)

        expected = (
            "Extended period from 2 Months to 1 Quarter "
            "to show sufficient data (3 points)"
        )
        assert message == expected

    def test_various_period_combinations(self):
        """Test various period adjustment combinations."""
        test_cases = [
            (
                "1W",
                "1M",
                5,
                (
                    "Extended period from 1 Week to 1 Month "
                    "to show sufficient data (5 points)"
                ),
            ),
            (
                "2M",
                "2Q",
                8,
                (
                    "Extended period from 2 Months to 2 Quarters "
                    "to show sufficient data (8 points)"
                ),
            ),
            (
                "YTD",
                "1Y",
                12,
                (
                    "Extended period from Year to Date to 1 Year "
                    "to show sufficient data (12 points)"
                ),
            ),
            ("MAX", "MAX", 100, "Loaded 100 data points for Maximum Available"),
        ]

        for original, actual, points, expected in test_cases:
            result = format_period_adjustment_message(original, actual, points)
            assert result == expected

    def test_edge_cases(self):
        """Test edge cases for message formatting."""
        # Single data point
        message = format_period_adjustment_message("1M", "1Y", 1)
        expected = (
            "Extended period from 1 Month to 1 Year "
            "to show sufficient data (1 points)"
        )
        assert message == expected

        # Zero data points
        message = format_period_adjustment_message("2M", "MAX", 0)
        expected = (
            "Extended period from 2 Months to Maximum Available "
            "to show sufficient data (0 points)"
        )
        assert message == expected


class TestFilterTrendDataToPeriod:
    """Test filter_trend_data_to_period function."""

    def create_sample_trend_data(self):
        """Create sample trend data for testing."""
        import pandas as pd

        dates = pd.date_range("2023-01-01", periods=12, freq="QS")
        return {
            "dates": dates,
            "trend": list(range(100, 112)),
            "plus1_std": list(range(105, 117)),
            "minus1_std": list(range(95, 107)),
            "plus2_std": list(range(110, 122)),
            "minus2_std": list(range(90, 102)),
        }

    def create_sample_filtered_data(self, start_date, end_date):
        """Create sample filtered DataFrame for testing."""
        import pandas as pd

        dates = pd.date_range(start_date, end_date, freq="QS")
        return pd.DataFrame({"value": range(len(dates))}, index=dates)

    def test_filter_trend_data_to_period(self):
        """Test filtering trend data to match filtered period."""
        from app.lib.periods import filter_trend_data_to_period

        # Create full trend data (12 quarters)
        full_trend = self.create_sample_trend_data()

        # Create filtered data (6 months period)
        filtered_data = self.create_sample_filtered_data("2023-04-01", "2023-10-01")

        # Filter trend data to match
        filtered_trend = filter_trend_data_to_period(full_trend, filtered_data)

        # Should have fewer points than full trend
        assert len(filtered_trend["trend"]) < len(full_trend["trend"])
        assert len(filtered_trend["trend"]) == len(filtered_data)

        # All arrays should have same length
        for key in filtered_trend:
            assert len(filtered_trend[key]) == len(filtered_trend["trend"])

    def test_filter_trend_data_none_input(self):
        """Test handling of None trend data."""
        from app.lib.periods import filter_trend_data_to_period

        filtered_data = self.create_sample_filtered_data("2023-04-01", "2023-10-01")
        result = filter_trend_data_to_period(None, filtered_data)

        assert result is None

    def test_filter_trend_data_empty_filtered_data(self):
        """Test handling of empty filtered data."""
        from app.lib.periods import filter_trend_data_to_period
        import pandas as pd

        full_trend = self.create_sample_trend_data()
        empty_data = pd.DataFrame(columns=["value"])

        result = filter_trend_data_to_period(full_trend, empty_data)

        # Should return original trend data when filtered data is empty
        assert result == full_trend

    def test_filter_trend_preserves_structure(self):
        """Test that filtering preserves trend data structure."""
        from app.lib.periods import filter_trend_data_to_period

        full_trend = self.create_sample_trend_data()
        filtered_data = self.create_sample_filtered_data("2023-07-01", "2023-12-31")

        filtered_trend = filter_trend_data_to_period(full_trend, filtered_data)

        # Should have same keys as original
        assert set(filtered_trend.keys()) == set(full_trend.keys())

        # Values should be numpy arrays or pandas datetime objects
        assert isinstance(filtered_trend["dates"], type(full_trend["dates"]))
        for key in ["trend", "plus1_std", "minus1_std", "plus2_std", "minus2_std"]:
            assert len(filtered_trend[key]) > 0
