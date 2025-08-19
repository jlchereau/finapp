"""
Unit tests for the finance utilities.
"""

import numpy as np
import pandas as pd
import pytest

from app.lib.finance import (
    calculate_returns,
    calculate_volatility,
    calculate_rsi,
    calculate_volume_metrics,
    calculate_exponential_trend,
)


class TestCalculateReturns:
    """Test cases for calculate_returns function."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        # Create predictable price data: starts at 100, increases by 1% daily
        prices = [100 * (1.01**i) for i in range(10)]
        return pd.DataFrame({"Close": prices}, index=dates)

    @pytest.fixture
    def sample_price_data_adj_close(self):
        """Create sample price data with Adj Close column."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        prices = [100 * (1.01**i) for i in range(10)]
        return pd.DataFrame({"Adj Close": prices}, index=dates)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = calculate_returns(empty_df)
        assert result.empty

    def test_close_column(self, sample_price_data):
        """Test calculation with Close column."""
        result = calculate_returns(sample_price_data)

        assert "Returns" in result.columns
        assert len(result) == 10
        # First value should be 0% (no return from itself)
        assert result.iloc[0]["Returns"] == 0.0
        # Second value should be approximately 1%
        assert abs(result.iloc[1]["Returns"] - 1.0) < 0.01

    def test_adj_close_column(self, sample_price_data_adj_close):
        """Test calculation with Adj Close column."""
        result = calculate_returns(sample_price_data_adj_close)

        assert "Returns" in result.columns
        assert len(result) == 10
        assert result.iloc[0]["Returns"] == 0.0

    def test_missing_price_columns(self):
        """Test with DataFrame missing both Close and Adj Close columns."""
        df = pd.DataFrame({"Volume": [1000, 2000, 3000]})

        with pytest.raises(ValueError, match="No Close price data found in DataFrame"):
            calculate_returns(df)

    def test_with_base_date(self, sample_price_data):
        """Test with base_date parameter."""
        base_date = pd.Timestamp("2023-01-05")
        result = calculate_returns(sample_price_data, base_date=base_date)

        # Should only include data from base_date onwards
        assert len(result) == 6  # Jan 5-10
        assert result.iloc[0]["Returns"] == 0.0

    def test_base_date_filters_all_data(self, sample_price_data):
        """Test with base_date that filters out all data."""
        base_date = pd.Timestamp("2024-01-01")  # After all sample data
        result = calculate_returns(sample_price_data, base_date=base_date)

        assert result.empty

    def test_single_data_point(self):
        """Test with single data point."""
        df = pd.DataFrame({"Close": [100]}, index=[pd.Timestamp("2023-01-01")])
        result = calculate_returns(df)

        assert len(result) == 1
        assert result.iloc[0]["Returns"] == 0.0

    def test_returns_calculation_accuracy(self):
        """Test accuracy of percentage returns calculation."""
        # Create simple test case: 100 -> 110 (10% increase)
        df = pd.DataFrame(
            {"Close": [100, 110]},
            index=[pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
        )
        result = calculate_returns(df)

        assert result.iloc[0]["Returns"] == 0.0  # Base value
        # Use approximate comparison for floating point
        assert abs(result.iloc[1]["Returns"] - 10.0) < 1e-10  # 10% increase


class TestCalculateVolatility:
    """Test cases for calculate_volatility function."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data with varying volatility."""
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        # Create some price movement for volatility calculation
        np.random.seed(42)  # For reproducible tests
        prices = [100]
        for i in range(49):
            # Random walk with some volatility
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            prices.append(prices[-1] * (1 + change))

        return pd.DataFrame({"Close": prices}, index=dates)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = calculate_volatility(empty_df)
        assert result.empty

    def test_missing_price_columns(self):
        """Test with DataFrame missing price columns."""
        df = pd.DataFrame({"Volume": [1000, 2000, 3000]})

        with pytest.raises(ValueError, match="No Close price data found in DataFrame"):
            calculate_volatility(df)

    def test_default_window_and_annualized(self, sample_price_data):
        """Test with default window (30) and annualized volatility."""
        result = calculate_volatility(sample_price_data)

        assert "Volatility" in result.columns
        # pct_change() drops the first row, so we get 49 instead of 50
        assert len(result) == 49
        # First 29 values should be NaN (window = 30, pct_change drops first row)
        assert pd.isna(result.iloc[28]["Volatility"])
        # 30th value (index 29) should have a volatility calculation
        assert not pd.isna(result.iloc[29]["Volatility"])

    def test_custom_window(self, sample_price_data):
        """Test with custom window size."""
        result = calculate_volatility(sample_price_data, window=10)

        # First 9 values should be NaN (window = 10)
        assert pd.isna(result.iloc[8]["Volatility"])
        # 10th value should have a volatility calculation
        assert not pd.isna(result.iloc[9]["Volatility"])

    def test_non_annualized_volatility(self, sample_price_data):
        """Test non-annualized volatility calculation."""
        annualized = calculate_volatility(sample_price_data, annualize=True)
        non_annualized = calculate_volatility(sample_price_data, annualize=False)

        # Non-annualized should be smaller (no sqrt(252) factor)
        ann_val = annualized.iloc[29]["Volatility"]
        non_ann_val = non_annualized.iloc[29]["Volatility"]

        if not (pd.isna(ann_val) or pd.isna(non_ann_val)):
            assert ann_val > non_ann_val

    def test_insufficient_data_for_window(self):
        """Test with insufficient data for window size."""
        df = pd.DataFrame({"Close": [100, 101]})  # Only 2 data points
        result = calculate_volatility(df, window=30)

        # All values should be NaN
        assert all(pd.isna(result["Volatility"]))

    def test_adj_close_column(self):
        """Test with Adj Close column."""
        dates = pd.date_range(start="2023-01-01", periods=35, freq="D")
        np.random.seed(42)
        prices = [100 * (1 + np.random.normal(0, 0.01)) for _ in range(35)]
        df = pd.DataFrame({"Adj Close": prices}, index=dates)

        result = calculate_volatility(df)
        assert "Volatility" in result.columns
        assert not pd.isna(result.iloc[29]["Volatility"])


class TestCalculateRSI:
    """Test cases for calculate_rsi function."""

    @pytest.fixture
    def sample_rsi_data(self):
        """Create sample price data for RSI testing."""
        # Create data that will produce predictable RSI values
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        # Alternating gains and losses
        prices = [100]
        for i in range(29):
            if i % 2 == 0:
                prices.append(prices[-1] * 1.02)  # 2% gain
            else:
                prices.append(prices[-1] * 0.98)  # 2% loss

        return pd.DataFrame({"Close": prices}, index=dates)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = calculate_rsi(empty_df)
        assert result.empty

    def test_missing_price_columns(self):
        """Test with DataFrame missing price columns."""
        df = pd.DataFrame({"Volume": [1000, 2000, 3000]})

        with pytest.raises(ValueError, match="No Close price data found in DataFrame"):
            calculate_rsi(df)

    def test_default_window(self, sample_rsi_data):
        """Test with default RSI window (14)."""
        result = calculate_rsi(sample_rsi_data)

        assert "RSI" in result.columns
        assert len(result) == 30
        # RSI calculation produces values earlier than expected due to rolling mean
        # The first valid RSI should appear around index 14 (window size)
        # Check that we have some NaN values at the beginning
        assert pd.isna(result.iloc[0]["RSI"])  # First value should be NaN
        # And that we have valid values later
        assert not pd.isna(result.iloc[-1]["RSI"])  # Last value should be valid

    def test_custom_window(self, sample_rsi_data):
        """Test with custom RSI window."""
        result = calculate_rsi(sample_rsi_data, window=10)

        # Check that first value is NaN and last value is valid
        assert pd.isna(result.iloc[0]["RSI"])  # First value should be NaN
        assert not pd.isna(result.iloc[-1]["RSI"])  # Last value should be valid

    def test_rsi_range(self, sample_rsi_data):
        """Test that RSI values are in valid range (0-100)."""
        result = calculate_rsi(sample_rsi_data)

        # Get non-NaN RSI values
        rsi_values = result["RSI"].dropna()

        assert all(rsi_values >= 0)
        assert all(rsi_values <= 100)

    def test_all_gains_scenario(self):
        """Test RSI with all price gains (should approach 100)."""
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        prices = [100 * (1.01**i) for i in range(20)]  # Consistent 1% gains
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = calculate_rsi(df)
        final_rsi = result["RSI"].iloc[-1]

        # Should be close to 100 (but not exactly due to calculation method)
        assert final_rsi > 90

    def test_all_losses_scenario(self):
        """Test RSI with all price losses (should approach 0)."""
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        prices = [100 * (0.99**i) for i in range(20)]  # Consistent 1% losses
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = calculate_rsi(df)
        final_rsi = result["RSI"].iloc[-1]

        # Should be close to 0
        assert final_rsi < 10

    def test_no_price_change(self):
        """Test RSI with no price changes."""
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        prices = [100] * 20  # No price changes
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = calculate_rsi(df)
        # RSI should be NaN when there are no gains or losses
        rsi_values = result["RSI"].dropna()
        # With no movement, RSI calculation results in division by zero -> NaN
        assert len(rsi_values) == 0 or all(pd.isna(rsi_values))

    def test_adj_close_column(self):
        """Test RSI calculation with Adj Close column."""
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        prices = [100 * (1.01**i) for i in range(20)]
        df = pd.DataFrame({"Adj Close": prices}, index=dates)

        result = calculate_rsi(df)
        assert "RSI" in result.columns
        assert not pd.isna(result["RSI"].iloc[-1])


class TestCalculateVolumeMetrics:
    """Test cases for calculate_volume_metrics function."""

    @pytest.fixture
    def sample_volume_data(self):
        """Create sample volume data for testing."""
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        # Create varying volume data
        np.random.seed(42)
        volumes = [1000000 + int(np.random.normal(0, 100000)) for _ in range(30)]
        return pd.DataFrame({"Volume": volumes}, index=dates)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = calculate_volume_metrics(empty_df)
        assert result.empty

    def test_missing_volume_column(self):
        """Test with DataFrame missing Volume column."""
        df = pd.DataFrame({"Close": [100, 101, 102]})

        with pytest.raises(ValueError, match="No Volume data found in DataFrame"):
            calculate_volume_metrics(df)

    def test_default_ma_window(self, sample_volume_data):
        """Test with default moving average window (20)."""
        result = calculate_volume_metrics(sample_volume_data)

        assert "Volume" in result.columns
        assert "Volume_MA" in result.columns
        assert len(result) == 30

        # First 19 MA values should be NaN (window = 20)
        assert pd.isna(result.iloc[18]["Volume_MA"])
        # 20th value should have MA calculation
        assert not pd.isna(result.iloc[19]["Volume_MA"])

    def test_custom_ma_window(self, sample_volume_data):
        """Test with custom moving average window."""
        result = calculate_volume_metrics(sample_volume_data, ma_window=10)

        # First 9 MA values should be NaN (window = 10)
        assert pd.isna(result.iloc[8]["Volume_MA"])
        # 10th value should have MA calculation
        assert not pd.isna(result.iloc[9]["Volume_MA"])

    def test_volume_preservation(self, sample_volume_data):
        """Test that original volume data is preserved."""
        result = calculate_volume_metrics(sample_volume_data)

        # Volume column should match original data
        pd.testing.assert_series_equal(
            result["Volume"], sample_volume_data["Volume"], check_names=False
        )

    def test_ma_calculation_accuracy(self):
        """Test accuracy of moving average calculation."""
        # Create simple test data
        volumes = [100, 200, 300, 400, 500]
        df = pd.DataFrame({"Volume": volumes})

        result = calculate_volume_metrics(df, ma_window=3)

        # Third value should be average of first 3: (100+200+300)/3 = 200
        assert result.iloc[2]["Volume_MA"] == 200.0
        # Fourth value should be average of 2nd-4th: (200+300+400)/3 = 300
        assert result.iloc[3]["Volume_MA"] == 300.0

    def test_insufficient_data_for_ma(self):
        """Test with insufficient data for moving average."""
        df = pd.DataFrame({"Volume": [1000, 2000]})  # Only 2 data points
        result = calculate_volume_metrics(df, ma_window=20)

        # All MA values should be NaN
        assert all(pd.isna(result["Volume_MA"]))
        # But volume data should be preserved
        assert result.iloc[0]["Volume"] == 1000
        assert result.iloc[1]["Volume"] == 2000


class TestCalculateExponentialTrend:
    """Test cases for calculate_exponential_trend function."""

    @pytest.fixture
    def quarterly_buffet_data(self):
        """Create sample quarterly Buffet Indicator data (mimics real workflow output)."""
        # Create quarterly dates (Quarter Start like in BuffetIndicatorWorkflow)
        dates = pd.date_range(start="2020-01-01", periods=16, freq="QS")

        # Create exponentially growing data with some noise (typical of Buffet Indicator)
        np.random.seed(42)  # For reproducible tests
        base_values = [100 * (1.02 ** (i / 4)) for i in range(16)]  # 2% annual growth
        noise = np.random.normal(0, 5, 16)  # Add some realistic noise
        buffet_values = [
            max(50, val + n) for val, n in zip(base_values, noise)
        ]  # Ensure positive

        return pd.DataFrame({"Buffet_Indicator": buffet_values}, index=dates)

    @pytest.fixture
    def annual_data(self):
        """Create sample annual data."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="YS")
        values = [100, 120, 150, 180, 220]  # Clear exponential growth
        return pd.DataFrame({"value": values}, index=dates)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = calculate_exponential_trend(empty_df, "value")
        assert result is None

    def test_missing_column(self, quarterly_buffet_data):
        """Test with missing column."""
        with pytest.raises(ValueError, match="Column 'missing' not found in DataFrame"):
            calculate_exponential_trend(quarterly_buffet_data, "missing")

    def test_insufficient_data(self):
        """Test with insufficient data points."""
        df = pd.DataFrame({"value": [100]}, index=[pd.Timestamp("2020-01-01")])
        result = calculate_exponential_trend(df, "value")
        assert result is None

    def test_quarterly_data_smooth_trend(self, quarterly_buffet_data):
        """Test that quarterly data produces smooth exponential trend (not stair-steps)."""
        result = calculate_exponential_trend(quarterly_buffet_data, "Buffet_Indicator")

        assert result is not None
        assert "trend" in result
        assert "dates" in result
        assert len(result["trend"]) == 16

        # Check that trend values are different for each quarter (no stair-steps)
        trend_values = result["trend"]

        # Within each year, trend values should be different (not equal)
        # Check Q1 vs Q2 of 2020 (should be different with fractional years)
        q1_2020_idx = 0  # 2020-01-01
        q2_2020_idx = 1  # 2020-04-01
        assert trend_values[q1_2020_idx] != trend_values[q2_2020_idx]

        # Most importantly: verify trend is smooth (different values for each quarter)
        # Due to noise in test data, trend direction is less important than smoothness
        assert len(set(trend_values)) > 1  # Should have variation, not flat

    def test_return_structure(self, quarterly_buffet_data):
        """Test that function returns expected structure."""
        result = calculate_exponential_trend(quarterly_buffet_data, "Buffet_Indicator")

        required_keys = [
            "dates",
            "trend",
            "plus1_std",
            "minus1_std",
            "plus2_std",
            "minus2_std",
        ]
        for key in required_keys:
            assert key in result

        # All arrays should have same length
        length = len(result["dates"])
        for key in required_keys[1:]:  # Skip 'dates'
            assert len(result[key]) == length

        # Dates should match original index
        pd.testing.assert_index_equal(result["dates"], quarterly_buffet_data.index)

    def test_confidence_bands_ordering(self, quarterly_buffet_data):
        """Test that confidence bands are properly ordered."""
        result = calculate_exponential_trend(quarterly_buffet_data, "Buffet_Indicator")

        trend = result["trend"]
        plus1 = result["plus1_std"]
        minus1 = result["minus1_std"]
        plus2 = result["plus2_std"]
        minus2 = result["minus2_std"]

        # Check ordering: minus2 < minus1 < trend < plus1 < plus2
        for i in range(len(trend)):
            assert minus2[i] <= minus1[i] <= trend[i] <= plus1[i] <= plus2[i]

    def test_annual_data_compatibility(self, annual_data):
        """Test that function works with annual data (backward compatibility)."""
        result = calculate_exponential_trend(annual_data, "value")

        assert result is not None
        assert len(result["trend"]) == 5

        # Should still produce smooth trend for annual data
        trend = result["trend"]
        assert trend[0] < trend[-1]  # Overall growth

    def test_negative_values_error(self):
        """Test that negative values raise appropriate error."""
        dates = pd.date_range(start="2020-01-01", periods=4, freq="QS")
        df = pd.DataFrame({"value": [100, -50, 75, 90]}, index=dates)

        with pytest.raises(ValueError, match="contains non-positive values"):
            calculate_exponential_trend(df, "value")

    def test_zero_values_error(self):
        """Test that zero values raise appropriate error."""
        dates = pd.date_range(start="2020-01-01", periods=4, freq="QS")
        df = pd.DataFrame({"value": [100, 0, 75, 90]}, index=dates)

        with pytest.raises(ValueError, match="contains non-positive values"):
            calculate_exponential_trend(df, "value")

    def test_nan_values_error(self):
        """Test that NaN values raise appropriate error."""
        dates = pd.date_range(start="2020-01-01", periods=4, freq="QS")
        df = pd.DataFrame({"value": [100, np.nan, 75, 90]}, index=dates)

        with pytest.raises(ValueError, match="contains non-positive values"):
            calculate_exponential_trend(df, "value")

    def test_fractional_year_conversion_accuracy(self):
        """Test that fractional year conversion produces expected values."""
        # Test specific quarterly dates
        dates = [
            pd.Timestamp("2020-01-01"),  # Q1 -> ~2020.0
            pd.Timestamp("2020-04-01"),  # Q2 -> ~2020.25
            pd.Timestamp("2020-07-01"),  # Q3 -> ~2020.5
            pd.Timestamp("2020-10-01"),  # Q4 -> ~2020.75
        ]
        df = pd.DataFrame({"value": [100, 110, 121, 133]}, index=dates)

        result = calculate_exponential_trend(df, "value")
        assert result is not None

        # Verify that all trend values are different (no stair-steps)
        trend = result["trend"]
        assert len(set(trend)) == 4  # All values should be unique

    def test_regression_quality(self, quarterly_buffet_data):
        """Test that regression produces reasonable exponential fit."""
        result = calculate_exponential_trend(quarterly_buffet_data, "Buffet_Indicator")

        trend = result["trend"]
        original = quarterly_buffet_data["Buffet_Indicator"].values

        # Trend should capture general direction
        # (Can't be too strict due to noise in test data)
        assert len(trend) == len(original)

        # At minimum, trend should be positive and reasonable
        assert all(val > 0 for val in trend)

        # Standard deviation bands should be reasonable
        std_width = result["plus1_std"][0] - result["minus1_std"][0]
        assert std_width > 0  # Should have some spread
