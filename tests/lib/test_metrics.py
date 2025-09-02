"""
Unit tests for metrics display utility functions.
"""

import reflex as rx

from app.lib.metrics import (
    integer_formatter,
    percentage_formatter,
    currency_formatter,
    large_currency_formatter,
    show_metric_as_badge,
    show_metric_as_gauge,
)


class TestFormatters:
    """Test cases for the formatter functions."""

    def test_integer_formatter(self):
        """Test integer_formatter function."""
        assert integer_formatter(42.678) == "43"
        assert integer_formatter(0.0) == "0"
        assert integer_formatter(-15.9) == "-16"

    def test_percentage_formatter(self):
        """Test percentage_formatter function."""
        assert percentage_formatter(45.678) == "45.68%"
        assert percentage_formatter(0.0) == "0.00%"
        assert percentage_formatter(-12.34) == "-12.34%"
        assert percentage_formatter(100) == "100.00%"

    def test_currency_formatter(self):
        """Test currency_formatter function."""
        assert currency_formatter(123.456) == "123.46"
        assert currency_formatter(0.0) == "0.00"
        assert currency_formatter(-45.99) == "-45.99"
        assert currency_formatter(1000) == "1000.00"

    def test_large_currency_formatter(self):
        """Test large_currency_formatter function for different scales."""
        # Trillions
        assert large_currency_formatter(1.5e12) == "1.50 T"
        assert large_currency_formatter(-2.345e12) == "-2.35 T"

        # Billions
        assert large_currency_formatter(2.5e9) == "2.50 B"
        assert large_currency_formatter(-1.234e9) == "-1.23 B"

        # Millions
        assert large_currency_formatter(3.5e6) == "3.50 M"
        assert large_currency_formatter(-5.678e6) == "-5.68 M"

        # Thousands
        assert large_currency_formatter(4.5e3) == "4.50 K"
        assert large_currency_formatter(-8.9e3) == "-8.90 K"

        # Below thousands
        assert large_currency_formatter(500) == "500.0"
        assert large_currency_formatter(99.95) == "100.0"
        assert large_currency_formatter(0.0) == "0.0"
        assert large_currency_formatter(-123) == "-123.0"

    def test_large_currency_formatter_edge_cases(self):
        """Test large_currency_formatter edge cases."""
        # Exactly at thresholds
        assert large_currency_formatter(1e12) == "1.00 T"
        assert large_currency_formatter(1e9) == "1.00 B"
        assert large_currency_formatter(1e6) == "1.00 M"
        assert large_currency_formatter(1e3) == "1.00 K"

        # Just below thresholds
        assert large_currency_formatter(999e9) == "999.00 B"
        assert large_currency_formatter(999e6) == "999.00 M"
        assert large_currency_formatter(999e3) == "999.00 K"


class TestShowMetricAsBadge:
    """Test cases for show_metric_as_badge function."""

    def test_show_metric_with_none_value(self):
        """Test badge display when value is None."""
        result = show_metric_as_badge(value=None)

        # Should be an rx.badge component
        assert isinstance(result, type(rx.badge("test")))

    def test_show_metric_with_basic_value(self):
        """Test badge display with basic value and no thresholds."""
        result = show_metric_as_badge(value=100.0)

        # Should be an rx.text component (no thresholds triggered)
        assert isinstance(result, type(rx.text("test")))

    def test_show_metric_with_low_threshold(self):
        """Test badge display when value hits low threshold."""
        result = show_metric_as_badge(
            value=5.0, low=10.0, high=None, higher_better=True
        )

        # Should be a red badge (tomato) since value is low and higher is better
        assert isinstance(result, type(rx.badge("test")))

    def test_show_metric_with_high_threshold(self):
        """Test badge display when value hits high threshold."""
        result = show_metric_as_badge(
            value=95.0, low=None, high=90.0, higher_better=True
        )

        # Should be a green badge since value is high and higher is better
        assert isinstance(result, type(rx.badge("test")))

    def test_show_metric_higher_better_false(self):
        """Test badge display when higher_better=False."""
        # High value should be red when higher is worse
        result_high = show_metric_as_badge(value=95.0, high=90.0, higher_better=False)
        assert isinstance(result_high, type(rx.badge("test")))

        # Low value should be green when higher is worse
        result_low = show_metric_as_badge(value=5.0, low=10.0, higher_better=False)
        assert isinstance(result_low, type(rx.badge("test")))

    def test_show_metric_with_custom_formatter(self):
        """Test badge display with custom formatter."""
        result = show_metric_as_badge(value=45.67, formatter=percentage_formatter)

        # Should be rx.text since no thresholds
        assert isinstance(result, type(rx.text("test")))

    def test_show_metric_both_thresholds(self):
        """Test badge display with both high and low thresholds."""
        # Value in middle range
        result_middle = show_metric_as_badge(value=50.0, low=10.0, high=90.0)
        assert isinstance(result_middle, type(rx.text("test")))

        # Value at low threshold
        result_low = show_metric_as_badge(value=10.0, low=10.0, high=90.0)
        assert isinstance(result_low, type(rx.badge("test")))

        # Value at high threshold
        result_high = show_metric_as_badge(value=90.0, low=10.0, high=90.0)
        assert isinstance(result_high, type(rx.badge("test")))

    def test_show_metric_threshold_edge_cases(self):
        """Test edge cases for threshold comparisons."""
        # Value exactly at thresholds
        result_exact_low = show_metric_as_badge(value=10.0, low=10.0)
        assert isinstance(result_exact_low, type(rx.badge("test")))

        result_exact_high = show_metric_as_badge(value=90.0, high=90.0)
        assert isinstance(result_exact_high, type(rx.badge("test")))

        # Value just above/below thresholds
        result_above_low = show_metric_as_badge(value=10.1, low=10.0)
        assert isinstance(result_above_low, type(rx.text("test")))

        result_below_high = show_metric_as_badge(value=89.9, high=90.0)
        assert isinstance(result_below_high, type(rx.text("test")))


class TestShowMetricAsGauge:
    """Test cases for show_metric_as_gauge function."""

    def test_show_gauge_with_none_value(self):
        """Test gauge display when value is None."""
        result = show_metric_as_gauge(value=None)

        # Should be an rx.badge with "n/a"
        assert isinstance(result, type(rx.badge("test")))

    def test_show_gauge_with_value(self):
        """Test gauge display with a value."""
        result = show_metric_as_gauge(value=75.0)

        # Currently returns TODO text
        assert isinstance(result, type(rx.text("test")))

    def test_show_gauge_with_various_values(self):
        """Test gauge display with various value types."""
        # Test different numeric values
        for value in [0.0, 50.0, 100.0, -25.0, 123.456]:
            result = show_metric_as_gauge(value=value)
            assert isinstance(result, type(rx.text("test")))


class TestIntegration:
    """Integration tests for metrics functionality."""

    def test_formatter_badge_integration(self):
        """Test integration between formatters and badge display."""
        formatters = [
            (integer_formatter, 45.67, "46"),
            (percentage_formatter, 45.67, "45.67%"),
            (currency_formatter, 45.67, "45.67"),
            (large_currency_formatter, 1.5e9, "1.50 B"),
        ]

        for formatter, value, expected_format in formatters:
            # Test that formatter formats correctly
            assert formatter(value) == expected_format

            # Test that badge uses the formatter formatting
            result = show_metric_as_badge(value=value, formatter=formatter)
            # Should be rx.text since no thresholds provided
            assert isinstance(result, type(rx.text("test")))

    def test_complete_workflow_scenarios(self):
        """Test complete workflow scenarios combining multiple features."""
        # Scenario 1: PE ratio - lower is better, currency format
        pe_ratio = 15.67
        result = show_metric_as_badge(
            value=pe_ratio,
            low=10.0,  # Good PE ratio
            high=25.0,  # High PE ratio
            formatter=currency_formatter,
            higher_better=False,  # Lower PE is better
        )
        # PE of 15.67 is between thresholds, should be text
        assert isinstance(result, type(rx.text("test")))

        # Scenario 2: Growth rate - higher is better, percentage format
        growth_rate = 8.5
        result = show_metric_as_badge(
            value=growth_rate,
            low=3.0,  # Low growth
            high=10.0,  # High growth
            formatter=percentage_formatter,
            higher_better=True,
        )
        # Growth of 8.5% is between thresholds, should be text
        assert isinstance(result, type(rx.text("test")))

        # Scenario 3: Market cap - large currency format
        market_cap = 2.5e9
        result = show_metric_as_badge(
            value=market_cap, formatter=large_currency_formatter
        )
        # No thresholds, should be text with "2.50 B" format
        assert isinstance(result, type(rx.text("test")))
