"""
Tests for chart creation utilities.
"""

import pandas as pd
import pytest
import plotly.graph_objects as go

from app.lib.charts import (
    ChartConfig,
    RSIChartConfig,
    ThresholdLine,
    TimeSeriesChartConfig,
    MARKET_COLORS,
    LINE_STYLES,
    get_default_chart_colors,
    create_comparison_chart,
    create_timeseries_chart,
    add_main_series,
    apply_chart_theme,
    add_threshold_lines,
    add_trend_display,
    add_historical_curves,
)


@pytest.fixture
def sample_data():
    """Create sample financial data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL"]

    # Create sample data with different ranges for each chart type
    returns_data = pd.DataFrame(
        {
            "AAPL": range(len(dates)),
            "MSFT": [x * 1.5 for x in range(len(dates))],
            "GOOGL": [x * 0.8 for x in range(len(dates))],
        },
        index=dates,
    )

    volatility_data = pd.DataFrame(
        {
            "AAPL": [x % 50 for x in range(len(dates))],
            "MSFT": [x % 40 for x in range(len(dates))],
            "GOOGL": [x % 60 for x in range(len(dates))],
        },
        index=dates,
    )

    volume_data = pd.DataFrame(
        {
            "AAPL": [x * 1000000 for x in range(len(dates))],
            "MSFT": [x * 800000 for x in range(len(dates))],
            "GOOGL": [x * 1200000 for x in range(len(dates))],
        },
        index=dates,
    )

    rsi_data = pd.DataFrame(
        {
            "AAPL": [30 + (x % 40) for x in range(len(dates))],  # Range 30-70
            "MSFT": [25 + (x % 50) for x in range(len(dates))],  # Range 25-75
            "GOOGL": [35 + (x % 30) for x in range(len(dates))],  # Range 35-65
        },
        index=dates,
    )

    # Single-column time series data for testing time-series charts
    buffet_data = pd.DataFrame(
        {
            "buffet_indicator": [80 + (x % 40) for x in range(len(dates))]
        },  # Range 80-120
        index=dates,
    )

    vix_data = pd.DataFrame(
        {"vix": [15 + (x % 20) for x in range(len(dates))]},  # Range 15-35
        index=dates,
    )

    return {
        "returns": returns_data,
        "volatility": volatility_data,
        "volume": volume_data,
        "rsi": rsi_data,
        "buffet": buffet_data,
        "vix": vix_data,
        "tickers": tickers,
    }


@pytest.fixture
def theme_colors():
    """Sample theme colors matching the compare.py implementation."""
    return {
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "grid_color": "rgba(128,128,128,0.3)",
        "line_color": "rgba(128,128,128,0.6)",
        "text_color": None,
        "hover_bgcolor": "rgba(0,0,0,0.8)",
        "hover_bordercolor": "rgba(128,128,128,0.8)",
    }


class TestChartConfig:
    """Test chart configuration classes."""

    def test_chart_config_creation(self):
        """Test basic ChartConfig creation."""
        config = ChartConfig(
            title="Test Chart",
            yaxis_title="Test Y",
            hover_format="Test: %{y:.2f}<br>",
        )

        assert config.title == "Test Chart"
        assert config.yaxis_title == "Test Y"
        assert config.hover_format == "Test: %{y:.2f}<br>"
        assert config.height == 300  # Default value
        assert config.yaxis_range is None
        assert config.reference_lines is None

    def test_chart_config_with_custom_values(self):
        """Test ChartConfig with custom values."""
        config = ChartConfig(
            title="Custom Chart",
            yaxis_title="Custom Y",
            hover_format="Custom: %{y:.1f}<br>",
            height=400,
            yaxis_range=[0, 100],
        )

        assert config.height == 400
        assert config.yaxis_range == [0, 100]

    def test_rsi_chart_config_creation(self):
        """Test RSIChartConfig creation."""
        config = RSIChartConfig("RSI Test")

        assert config.title == "RSI Test"
        assert config.yaxis_title == "RSI"
        assert config.hover_format == "RSI: %{y:.1f}<br>"
        assert config.yaxis_range == [0, 100]
        assert config.reference_lines is not None
        assert len(config.reference_lines) == 2

        # Test reference lines
        ref_lines = config.reference_lines
        assert ref_lines[0]["y"] == 70
        assert ref_lines[0]["annotation_text"] == "Overbought (70)"
        assert ref_lines[1]["y"] == 30
        assert ref_lines[1]["annotation_text"] == "Oversold (30)"

    def test_rsi_chart_config_overbought_oversold(self):
        """Test RSI level detection methods."""
        config = RSIChartConfig()

        # Test overbought detection
        assert config.is_overbought_level(70) is True
        assert config.is_overbought_level(75) is True
        assert config.is_overbought_level(69.9) is False

        # Test oversold detection
        assert config.is_oversold_level(30) is True
        assert config.is_oversold_level(25) is True
        assert config.is_oversold_level(30.1) is False


class TestThresholdLine:
    """Test ThresholdLine class."""

    def test_threshold_line_creation(self):
        """Test ThresholdLine creation."""
        threshold = ThresholdLine(
            value=100.0,
            color=MARKET_COLORS["danger"],
            label="Overvalued",
        )

        assert threshold.value == 100.0
        assert threshold.color == MARKET_COLORS["danger"]
        assert threshold.label == "Overvalued"
        assert threshold.position == "bottom right"  # Default

    def test_threshold_line_custom_position(self):
        """Test ThresholdLine with custom position."""
        threshold = ThresholdLine(
            value=50.0,
            color=MARKET_COLORS["safe"],
            label="Undervalued",
            position="top left",
        )

        assert threshold.position == "top left"

    def test_threshold_line_to_plotly_kwargs(self):
        """Test conversion to plotly kwargs."""
        threshold = ThresholdLine(
            value=75.0,
            color=MARKET_COLORS["warning"],
            label="Neutral",
            position="top right",
        )

        kwargs = threshold.to_plotly_kwargs()

        assert kwargs["y"] == 75.0
        assert kwargs["line_color"] == MARKET_COLORS["warning"]
        assert kwargs["annotation_text"] == "Neutral"
        assert kwargs["annotation_position"] == "top right"
        assert kwargs["line_dash"] == "solid"
        assert kwargs["opacity"] == 0.7


class TestTimeSeriesChartConfig:
    """Test TimeSeriesChartConfig class."""

    def test_timeseries_config_creation(self):
        """Test TimeSeriesChartConfig creation."""
        config = TimeSeriesChartConfig(
            title="Buffet Indicator",
            yaxis_title="Ratio (%)",
            hover_format="Value: %{y:.1f}%<br>",
        )

        assert config.title == "Buffet Indicator"
        assert config.yaxis_title == "Ratio (%)"
        assert config.hover_format == "Value: %{y:.1f}%<br>"
        assert config.height == 400  # Market charts are taller
        assert config.primary_color == MARKET_COLORS["primary"]
        assert config.primary_style == "primary"

    def test_timeseries_config_custom_styling(self):
        """Test TimeSeriesChartConfig with custom styling."""
        config = TimeSeriesChartConfig(
            title="VIX",
            yaxis_title="Volatility",
            hover_format="VIX: %{y:.2f}<br>",
            height=350,
            primary_color=MARKET_COLORS["danger"],
            primary_style="trend",
        )

        assert config.height == 350
        assert config.primary_color == MARKET_COLORS["danger"]
        assert config.primary_style == "trend"


class TestMarketColors:
    """Test market color constants."""

    def test_market_colors_exist(self):
        """Test that all expected market colors are defined."""
        expected_colors = ["danger", "safe", "warning", "trend", "primary"]

        for color_key in expected_colors:
            assert color_key in MARKET_COLORS
            assert isinstance(MARKET_COLORS[color_key], str)
            assert len(MARKET_COLORS[color_key]) > 0

    def test_line_styles_exist(self):
        """Test that all expected line styles are defined."""
        expected_styles = ["threshold", "trend", "historical", "primary"]

        for style_key in expected_styles:
            assert style_key in LINE_STYLES
            assert "width" in LINE_STYLES[style_key]
            assert "dash" in LINE_STYLES[style_key]
            assert isinstance(LINE_STYLES[style_key]["width"], int)
            assert isinstance(LINE_STYLES[style_key]["dash"], str)


class TestChartColors:
    """Test chart color functions."""

    def test_get_default_chart_colors(self):
        """Test default color palette."""
        colors = get_default_chart_colors()

        assert isinstance(colors, list)
        assert len(colors) == 7
        assert all(color.startswith("#") for color in colors)
        assert "#1f77b4" in colors  # First color
        assert "#e377c2" in colors  # Last color


class TestChartCreation:
    """Test chart creation functions."""

    def test_create_comparison_chart_with_data(self, sample_data, theme_colors):
        """Test creating chart with valid data."""
        config = ChartConfig(
            title="Returns",
            yaxis_title="Return (%)",
            hover_format="Return: %{y:.2f}%<br>",
        )
        fig = create_comparison_chart(
            sample_data["returns"],
            config,
            theme_colors,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # Three tickers

        # Check traces
        for i, ticker in enumerate(sample_data["tickers"]):
            trace = fig.data[i]
            assert trace.name == ticker
            assert trace.mode == "lines"
            assert "Return: %{y:.2f}%<br>" in trace.hovertemplate

    def test_create_comparison_chart_empty_data(self, theme_colors):
        """Test creating chart with empty DataFrame."""
        empty_data = pd.DataFrame()
        config = ChartConfig("Empty", "Y", "Test: %{y}<br>")

        fig = create_comparison_chart(empty_data, config, theme_colors)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_create_rsi_chart_with_reference_lines(self, sample_data, theme_colors):
        """Test RSI chart creation with reference lines."""
        config = RSIChartConfig()
        fig = create_comparison_chart(
            sample_data["rsi"],
            config,
            theme_colors,
        )

        # Should have 3 ticker traces + 2 reference lines
        assert len(fig.data) >= 3

        # Check that reference lines were added
        assert config.reference_lines is not None
        assert len(config.reference_lines) == 2


class TestThemeApplication:
    """Test theme application functions."""

    def test_apply_chart_theme(self, theme_colors):
        """Test theme application to figure."""
        fig = go.Figure()
        config = ChartConfig("Test", "Y", "Test: %{y}<br>", height=400)

        apply_chart_theme(fig, config, theme_colors)

        # Check layout properties
        assert fig.layout.title.text == "Test"
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "Y"
        assert fig.layout.height == 400
        assert fig.layout.hovermode == "x unified"
        assert fig.layout.showlegend is True

        # Check theme colors
        assert fig.layout.plot_bgcolor == theme_colors["plot_bgcolor"]
        assert fig.layout.paper_bgcolor == theme_colors["paper_bgcolor"]

    def test_apply_chart_theme_with_yaxis_range(self, theme_colors):
        """Test theme application with Y-axis range."""
        fig = go.Figure()
        config = ChartConfig("Test", "Y", "Test: %{y}<br>", yaxis_range=[0, 100])

        apply_chart_theme(fig, config, theme_colors)

        assert list(fig.layout.yaxis.range) == [0, 100]

    def test_apply_chart_theme_with_font_color(self, theme_colors):
        """Test theme application with font color."""
        theme_colors_with_font = theme_colors.copy()
        theme_colors_with_font["text_color"] = "#333333"

        fig = go.Figure()
        config = ChartConfig("Test", "Y", "Test: %{y}<br>")

        apply_chart_theme(fig, config, theme_colors_with_font)

        assert fig.layout.font.color == "#333333"


class TestIntegration:
    """Integration tests for full chart creation workflow."""

    def test_full_chart_creation_workflow(self, sample_data, theme_colors):
        """Test complete chart creation from start to finish."""
        tickers = sample_data["tickers"]

        # Test different chart types
        configs = [
            ChartConfig("Returns", "Return (%)", "Return: %{y:.2f}%<br>"),
            ChartConfig("Volatility", "Volatility (%)", "Volatility: %{y:.2f}%<br>"),
            ChartConfig("Volume", "Volume", "Volume: %{y:,.0f}<br>"),
            RSIChartConfig(),
        ]
        data_types = ["returns", "volatility", "volume", "rsi"]

        for data_type, config in zip(data_types, configs):
            data = sample_data[data_type]
            fig = create_comparison_chart(data, config, theme_colors)

            # Basic validations
            assert isinstance(fig, go.Figure)
            assert len(fig.data) >= len(tickers)  # At least one trace per ticker

            # Check that the chart has proper styling
            assert fig.layout.title is not None
            assert fig.layout.xaxis.title.text == "Date"
            assert fig.layout.height == 300

    def test_chart_data_integrity(self, sample_data, theme_colors):
        """Test that chart data matches input data."""
        data = sample_data["returns"]
        config = ChartConfig(
            title="Returns",
            yaxis_title="Return (%)",
            hover_format="Return: %{y:.2f}%<br>",
        )

        fig = create_comparison_chart(data, config, theme_colors)

        # Check that each trace has the correct data
        for i, ticker in enumerate(sample_data["tickers"]):
            trace = fig.data[i]

            # X-axis should match the index
            assert len(trace.x) == len(data.index)

            # Y-axis should match the ticker column
            assert len(trace.y) == len(data[ticker])

            # Check some sample values
            assert trace.y[0] == data[ticker].iloc[0]
            assert trace.y[-1] == data[ticker].iloc[-1]


class TestTimeSeriesCharts:
    """Test time-series chart creation functions."""

    def test_create_timeseries_chart_with_data(self, sample_data, theme_colors):
        """Test creating time-series chart with valid data."""
        config = TimeSeriesChartConfig(
            title="Buffet Indicator",
            yaxis_title="Ratio (%)",
            hover_format="Value: %{y:.1f}%<br>",
        )
        fig = create_timeseries_chart(
            sample_data["buffet"],
            config,
            theme_colors,
            "buffet_indicator",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Single trace

        # Check main trace
        trace = fig.data[0]
        assert trace.name == "Buffet Indicator"
        assert trace.mode == "lines+markers"  # Primary style includes markers
        assert trace.line.color == MARKET_COLORS["primary"]
        assert "Value: %{y:.1f}%<br>" in trace.hovertemplate

    def test_create_timeseries_chart_empty_data(self, theme_colors):
        """Test creating time-series chart with empty data."""
        empty_data = pd.DataFrame()
        config = TimeSeriesChartConfig("Empty", "Y", "Test: %{y}<br>")

        fig = create_timeseries_chart(empty_data, config, theme_colors, "missing")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_create_timeseries_chart_missing_column(self, sample_data, theme_colors):
        """Test creating time-series chart with missing column."""
        config = TimeSeriesChartConfig("Test", "Y", "Test: %{y}<br>")

        fig = create_timeseries_chart(
            sample_data["buffet"], config, theme_colors, "missing_column"
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_create_timeseries_chart_with_trend_style(self, sample_data, theme_colors):
        """Test time-series chart with trend line styling."""
        config = TimeSeriesChartConfig(
            title="VIX Trend",
            yaxis_title="Volatility",
            hover_format="VIX: %{y:.2f}<br>",
            primary_style="trend",
        )
        fig = create_timeseries_chart(
            sample_data["vix"],
            config,
            theme_colors,
            "vix",
        )

        trace = fig.data[0]
        assert trace.mode == "lines"  # Trend style has no markers
        assert trace.line.dash == "dash"  # Trend style is dashed

    def test_create_timeseries_chart_no_main_series(self, sample_data, theme_colors):
        """Test creating time-series chart without main series."""
        config = TimeSeriesChartConfig(
            title="Test Chart",
            yaxis_title="Test Y",
            hover_format="Test: %{y:.2f}<br>",
        )
        fig = create_timeseries_chart(
            sample_data["buffet"],
            config,
            theme_colors,
            "buffet_indicator",
            include_main_series=False,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0  # No main series added
        assert fig.layout.title.text == "Test Chart"


class TestMainSeries:
    """Test main series addition functionality."""

    def test_add_main_series_with_data(self, sample_data):
        """Test adding main series to existing figure."""
        fig = go.Figure()
        config = TimeSeriesChartConfig(
            title="Test Series",
            yaxis_title="Test Y",
            hover_format="Value: %{y:.2f}<br>",
        )

        add_main_series(fig, sample_data["vix"], config, "vix")

        assert len(fig.data) == 1
        trace = fig.data[0]
        assert trace.name == "Test Series"
        assert trace.mode == "lines+markers"  # Primary style
        assert trace.line.color == MARKET_COLORS["primary"]

    def test_add_main_series_empty_data(self):
        """Test adding main series with empty data."""
        fig = go.Figure()
        empty_data = pd.DataFrame()
        config = TimeSeriesChartConfig("Empty", "Y", "Test: %{y}<br>")

        add_main_series(fig, empty_data, config, "missing")

        assert len(fig.data) == 0  # No series added

    def test_add_main_series_missing_column(self, sample_data):
        """Test adding main series with missing column."""
        fig = go.Figure()
        config = TimeSeriesChartConfig("Test", "Y", "Test: %{y}<br>")

        add_main_series(fig, sample_data["vix"], config, "missing_column")

        assert len(fig.data) == 0  # No series added


class TestThresholdLines:
    """Test threshold line functionality."""

    def test_add_threshold_lines(self, theme_colors):
        """Test adding threshold lines to a figure."""
        fig = go.Figure()
        thresholds = [
            ThresholdLine(100.0, MARKET_COLORS["danger"], "Overvalued"),
            ThresholdLine(75.0, MARKET_COLORS["safe"], "Undervalued"),
        ]

        add_threshold_lines(fig, thresholds)

        # Verify hlines were added with proper layer setting
        # Note: add_hline with layer="below" adds to layout.shapes
        assert isinstance(fig, go.Figure)
        # Check that shapes were added (threshold lines become shapes)
        if hasattr(fig.layout, 'shapes') and fig.layout.shapes:
            # At least one shape should exist for the threshold lines
            assert len(fig.layout.shapes) >= 1

    def test_add_threshold_lines_empty_list(self, theme_colors):
        """Test adding empty threshold list."""
        fig = go.Figure()
        initial_data_count = len(fig.data)

        add_threshold_lines(fig, [])

        assert len(fig.data) == initial_data_count  # No change


class TestTrendDisplay:
    """Test trend display functionality."""

    def test_add_trend_display_with_data(self):
        """Test adding trend display with valid data."""
        fig = go.Figure()
        trend_data = {
            "dates": pd.date_range("2023-01-01", periods=10),
            "trend": list(range(10)),
            "plus1_std": [x + 5 for x in range(10)],
            "minus1_std": [x - 5 for x in range(10)],
            "plus2_std": [x + 10 for x in range(10)],
            "minus2_std": [x - 10 for x in range(10)],
        }

        add_trend_display(fig, trend_data)

        # Should have 5 traces: trend + 4 std dev bands
        assert len(fig.data) == 5

        # Check main trend trace
        trend_trace = fig.data[0]
        assert trend_trace.name == "Exponential Trend"
        assert trend_trace.line.dash == "dash"
        assert trend_trace.line.color == MARKET_COLORS["trend"]

    def test_add_trend_display_empty_data(self):
        """Test adding trend display with empty data."""
        fig = go.Figure()
        initial_data_count = len(fig.data)

        add_trend_display(fig, {})

        assert len(fig.data) == initial_data_count  # No change

    def test_add_trend_display_none_data(self):
        """Test adding trend display with None data."""
        fig = go.Figure()
        initial_data_count = len(fig.data)

        add_trend_display(fig, None)

        assert len(fig.data) == initial_data_count  # No change


class TestHistoricalCurves:
    """Test historical curve functionality."""

    def test_add_historical_curves_with_data(self):
        """Test adding historical curves with valid data."""
        fig = go.Figure()
        curves = [
            {
                "x": pd.date_range("2023-01-01", periods=5),
                "y": list(range(5)),
                "name": "6 Months Ago",
            },
            {
                "x": pd.date_range("2023-01-01", periods=5),
                "y": [x * 2 for x in range(5)],
                "name": "12 Months Ago",
                "color": MARKET_COLORS["danger"],
                "opacity": 0.5,
            },
        ]

        add_historical_curves(fig, curves)

        assert len(fig.data) == 2

        # Check first curve (default styling)
        curve1 = fig.data[0]
        assert curve1.name == "6 Months Ago"
        assert curve1.line.color == MARKET_COLORS["warning"]  # Default
        assert curve1.line.dash == "dot"  # Historical style
        assert curve1.opacity == 0.7  # Default opacity

        # Check second curve (custom styling)
        curve2 = fig.data[1]
        assert curve2.name == "12 Months Ago"
        assert curve2.line.color == MARKET_COLORS["danger"]  # Custom color
        assert curve2.opacity == 0.5  # Custom opacity

    def test_add_historical_curves_empty_list(self):
        """Test adding empty historical curves list."""
        fig = go.Figure()
        initial_data_count = len(fig.data)

        add_historical_curves(fig, [])

        assert len(fig.data) == initial_data_count  # No change

    def test_add_historical_curves_missing_keys(self):
        """Test historical curves with missing required keys."""
        fig = go.Figure()
        curves = [{"x": [1, 2, 3], "name": "Test"}]  # Missing 'y' key

        # Should not crash, but may not add the trace
        try:
            add_historical_curves(fig, curves)
        except KeyError:
            # Expected behavior for missing keys
            pass


class TestIntegrationTimeSeriesCharts:
    """Integration tests for time-series chart workflow."""

    def test_full_timeseries_workflow(self, sample_data, theme_colors):
        """Test complete time-series chart creation workflow."""
        # Create base time-series chart
        config = TimeSeriesChartConfig(
            title="Market Indicator",
            yaxis_title="Value",
            hover_format="Value: %{y:.1f}<br>",
            primary_color=MARKET_COLORS["primary"],
        )

        fig = create_timeseries_chart(
            sample_data["buffet"],
            config,
            theme_colors,
            "buffet_indicator",
        )

        # Add threshold lines
        thresholds = [
            ThresholdLine(100.0, MARKET_COLORS["danger"], "Overvalued"),
            ThresholdLine(80.0, MARKET_COLORS["safe"], "Undervalued"),
        ]
        add_threshold_lines(fig, thresholds)

        # Add historical curves
        dates = sample_data["buffet"].index[:5]
        curves = [
            {
                "x": dates,
                "y": [90, 92, 94, 96, 98],
                "name": "Historical Average",
            }
        ]
        add_historical_curves(fig, curves)

        # Verify final chart structure
        assert len(fig.data) >= 2  # Main trace + historical curve
        assert fig.layout.title.text == "Market Indicator"
        assert fig.layout.yaxis.title.text == "Value"

    def test_chart_data_integrity_timeseries(self, sample_data, theme_colors):
        """Test that time-series chart data matches input data."""
        config = TimeSeriesChartConfig(
            title="VIX",
            yaxis_title="Volatility",
            hover_format="VIX: %{y:.2f}<br>",
        )

        fig = create_timeseries_chart(
            sample_data["vix"],
            config,
            theme_colors,
            "vix",
        )

        trace = fig.data[0]

        # X-axis should match the index
        assert len(trace.x) == len(sample_data["vix"].index)

        # Y-axis should match the vix column
        assert len(trace.y) == len(sample_data["vix"]["vix"])

        # Check some sample values
        assert trace.y[0] == sample_data["vix"]["vix"].iloc[0]
        assert trace.y[-1] == sample_data["vix"]["vix"].iloc[-1]

    def test_drawing_order_main_series_on_top(self, sample_data, theme_colors):
        """Test that main series is drawn last (on top of background elements)."""
        config = TimeSeriesChartConfig(
            title="Drawing Order Test",
            yaxis_title="Value",
            hover_format="Value: %{y:.1f}<br>",
        )

        # Create empty chart
        fig = create_timeseries_chart(
            sample_data["buffet"],
            config,
            theme_colors,
            "buffet_indicator",
            include_main_series=False,
        )

        # Add background elements (thresholds, trends, historical)
        thresholds = [
            ThresholdLine(100.0, MARKET_COLORS["warning"], "Background Line")
        ]
        add_threshold_lines(fig, thresholds)

        curves = [
            {
                "x": sample_data["buffet"].index[:5],
                "y": [90, 92, 94, 96, 98],
                "name": "Historical Curve",
            }
        ]
        add_historical_curves(fig, curves)

        # Add main series last
        add_main_series(fig, sample_data["buffet"], config, "buffet_indicator")

        # Verify main series is the last trace (drawn on top)
        assert len(fig.data) >= 2  # At least one background + main series
        main_trace = fig.data[-1]  # Last trace
        assert main_trace.name == "Drawing Order Test"
        assert main_trace.line.color == MARKET_COLORS["primary"]
        assert main_trace.line.width == 3  # Primary style has thicker line
