"""
Tests for chart creation utilities.
"""

import pandas as pd
import pytest
import plotly.graph_objects as go

from app.lib.charts import (
    ChartConfig,
    RSIChartConfig,
    get_default_chart_colors,
    create_comparison_chart,
    apply_chart_theme,
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

    return {
        "returns": returns_data,
        "volatility": volatility_data,
        "volume": volume_data,
        "rsi": rsi_data,
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
