"""
Chart creation utilities for financial data visualization.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd
import plotly.graph_objects as go


# Semantic colors for market indicators
MARKET_COLORS = {
    "danger": "#dc2626",  # Red for overvalued/high risk
    "safe": "#16a34a",  # Green for undervalued/low risk
    "warning": "#f59e0b",  # Orange for neutral/historical
    "trend": "rgba(128,128,128,0.7)",  # Light grey for broad trends
    "primary": "#2563eb",  # Blue for main data series
}

# Line styles for different indicator types
LINE_STYLES = {
    "threshold": {"width": 1, "dash": "solid"},  # Thin solid for thresholds
    "trend": {"width": 1, "dash": "dash"},  # Dashed for trend lines
    "historical": {"width": 2, "dash": "dot"},  # Dotted for historical curves
    "primary": {"width": 2, "dash": "solid"},  # Thick solid for main metric
    "solid": {"width": 2, "dash": "solid"},  # Solid line without markers
}


@dataclass
class ChartConfig:
    """Configuration for chart creation."""

    title: str
    yaxis_title: str
    hover_format: str
    height: int = 300
    yaxis_range: Optional[List[float]] = None
    reference_lines: Optional[List[Dict[str, Any]]] = None


@dataclass
class ThresholdLine:
    """Configuration for horizontal threshold lines."""

    value: float
    color: str
    label: str
    position: str = "bottom right"  # Annotation position

    def to_plotly_kwargs(self) -> Dict[str, Any]:
        """Convert to plotly add_hline kwargs."""
        style = LINE_STYLES["threshold"]
        return {
            "y": self.value,
            "line_dash": style["dash"],
            "line_color": self.color,
            "opacity": 0.7,
            "annotation_text": self.label,
            "annotation_position": self.position,
        }


@dataclass
class TimeSeriesChartConfig(ChartConfig):
    """Configuration for single time-series charts with market indicators."""

    def __init__(
        self,
        title: str,
        yaxis_title: str,
        hover_format: str,
        height: int = 400,  # Markets charts are typically taller
        primary_color: str = MARKET_COLORS["primary"],
        primary_style: str = "primary",
        yaxis_range: Optional[List[float]] = None,
    ):
        super().__init__(
            title=title,
            yaxis_title=yaxis_title,
            hover_format=hover_format,
            height=height,
            yaxis_range=yaxis_range,
        )
        self.primary_color = primary_color
        self.primary_style = primary_style


def get_default_chart_colors() -> List[str]:
    """Get the standard color palette used across charts."""
    return [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]


def get_default_theme_colors() -> Dict[str, Any]:
    """
    Get neutral theme colors that work well in both light and dark themes.

    Returns:
        Dictionary with plot styling colors optimized for theme compatibility:
        - Transparent backgrounds that inherit from page theme
        - Semi-transparent grays for grid lines and borders
        - None for text_color to let Plotly use theme-appropriate defaults
    """
    return {
        "plot_bgcolor": "rgba(0,0,0,0)",  # Transparent - inherits page background
        "paper_bgcolor": "rgba(0,0,0,0)",  # Transparent - inherits page background
        "grid_color": "rgba(128,128,128,0.3)",  # Semi-transparent gray
        "line_color": "rgba(128,128,128,0.6)",  # Semi-transparent gray
        "text_color": None,  # Let Plotly use default which respects theme
        "hover_bgcolor": "rgba(0,0,0,0.8)",  # Semi-transparent dark background
        "hover_bordercolor": "rgba(128,128,128,0.8)",  # Semi-transparent border
    }


def create_comparison_chart(
    data: pd.DataFrame,
    config: ChartConfig,
    theme_colors: Dict[str, Any],
) -> go.Figure:
    """
    Create a comparison chart from financial data.

    Args:
        data: DataFrame with tickers as columns and dates as index
        config: Chart configuration object
        theme_colors: Theme color dictionary

    Returns:
        Configured Plotly Figure object
    """
    if data.empty:
        return go.Figure()

    fig = go.Figure()
    colors = get_default_chart_colors()

    # Add traces for each ticker
    for i, ticker in enumerate(data.columns):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[ticker],
                mode="lines",
                name=ticker,
                line={"color": color, "width": 2},
                hovertemplate=f"<b>{ticker}</b><br>"
                + config.hover_format
                + "<extra></extra>",
            )
        )

    # Add reference lines if specified (for RSI charts)
    if config.reference_lines:
        for ref_line in config.reference_lines:
            fig.add_hline(**ref_line)

    # Apply theme and layout
    apply_chart_theme(fig, config, theme_colors)

    return fig


def apply_chart_theme(
    fig: go.Figure,
    config: ChartConfig,
    theme_colors: Dict[str, Any],
) -> None:
    """
    Apply theme colors and layout to a chart figure.

    Args:
        fig: Plotly figure to modify
        config: Chart configuration
        theme_colors: Theme color dictionary
    """
    # Build layout properties
    layout_props = {
        "title": config.title,
        "xaxis_title": "Date",
        "yaxis_title": config.yaxis_title,
        "hovermode": "x unified",
        "showlegend": True,
        "height": config.height,
        "margin": {"l": 50, "r": 50, "t": 80, "b": 50},
        "plot_bgcolor": theme_colors["plot_bgcolor"],
        "paper_bgcolor": theme_colors["paper_bgcolor"],
        "hoverlabel": {
            "bgcolor": theme_colors["hover_bgcolor"],
            "bordercolor": theme_colors["hover_bordercolor"],
            "font_size": 14,
            "font_color": "white",
        },
    }

    # Add Y-axis range if specified
    if config.yaxis_range:
        layout_props["yaxis"] = {"range": config.yaxis_range}

    # Only add font_color if it's not None
    if theme_colors["text_color"] is not None:
        layout_props["font_color"] = theme_colors["text_color"]

    fig.update_layout(**layout_props)

    # Update axes styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=theme_colors["grid_color"],
        showline=True,
        linewidth=1,
        linecolor=theme_colors["line_color"],
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=theme_colors["grid_color"],
        showline=True,
        linewidth=1,
        linecolor=theme_colors["line_color"],
    )


def create_timeseries_chart(
    data: pd.DataFrame,
    config: TimeSeriesChartConfig,
    theme_colors: Dict[str, Any],
    column_name: str,
    include_main_series: bool = True,
) -> go.Figure:
    """
    Create a time-series chart for single-metric market data.

    Args:
        data: DataFrame with date index and metric column
        config: Time-series chart configuration
        theme_colors: Theme color dictionary
        column_name: Name of the column to plot
        include_main_series: Whether to add the main data series (default True)

    Returns:
        Configured Plotly Figure with optional main data series
    """
    fig = go.Figure()

    # Add main data series if requested and data is valid
    if include_main_series and not data.empty and column_name in data.columns:
        # Get primary line style
        style = LINE_STYLES[config.primary_style]

        # Add main data series
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[column_name],
                mode="lines+markers" if config.primary_style == "primary" else "lines",
                name=config.title,
                line={
                    "color": config.primary_color,
                    "width": style["width"],
                    "dash": style["dash"],
                },
                marker={"size": 4} if config.primary_style == "primary" else None,
                hovertemplate=f"<b>{config.title}</b><br>"
                + config.hover_format
                + "<extra></extra>",
            )
        )

    # Apply theme
    apply_chart_theme(fig, config, theme_colors)

    return fig


def add_main_series(
    fig: go.Figure,
    data: pd.DataFrame,
    config: TimeSeriesChartConfig,
    column_name: str,
) -> None:
    """
    Add the main data series to a chart (should be called last for top layer).

    Args:
        fig: Plotly figure to modify
        data: DataFrame with date index and metric column
        config: Time-series chart configuration
        column_name: Name of the column to plot
    """
    if data.empty or column_name not in data.columns:
        return

    # Get primary line style
    style = LINE_STYLES[config.primary_style]

    # Add main data series
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[column_name],
            mode="lines+markers" if config.primary_style == "primary" else "lines",
            name=config.title,
            line={
                "color": config.primary_color,
                "width": style["width"],
                "dash": style["dash"],
            },
            marker={"size": 4} if config.primary_style == "primary" else None,
            hovertemplate=f"<b>{config.title}</b><br>"
            + config.hover_format
            + "<extra></extra>",
        )
    )


def add_threshold_lines(
    fig: go.Figure,
    thresholds: List[ThresholdLine],
) -> None:
    """
    Add horizontal threshold lines to a chart using shapes with below layer.
    This ensures they appear behind traces.

    Args:
        fig: Plotly figure to modify
        thresholds: List of threshold line configurations
    """
    for threshold in thresholds:
        style = LINE_STYLES["threshold"]
        fig.add_hline(
            y=threshold.value,
            line_dash=style["dash"],
            line_color=threshold.color,
            opacity=0.7,
            annotation_text=threshold.label,
            annotation_position=threshold.position,
            layer="below",  # This ensures the line appears behind traces
        )


def add_trend_display(
    fig: go.Figure,
    trend_data: Dict[str, Any],
) -> None:
    """
    Display pre-calculated trend lines and deviation bands.

    Args:
        fig: Plotly figure to modify
        trend_data: Dictionary containing pre-calculated trend data with keys:
                   - dates: x-axis dates
                   - trend: main trend line
                   - plus1_std, minus1_std: ±1 std dev bands
                   - plus2_std, minus2_std: ±2 std dev bands
    """
    if not trend_data:
        return

    style = LINE_STYLES["trend"]
    color = MARKET_COLORS["trend"]

    # Main trend line
    fig.add_trace(
        go.Scatter(
            x=trend_data["dates"],
            y=trend_data["trend"],
            mode="lines",
            name="Exponential Trend",
            line={"color": color, "width": style["width"], "dash": style["dash"]},
            hovertemplate="<b>Exponential Trend</b><br>"
            + "Value: %{y:.1f}<br><extra></extra>",
        )
    )

    # ±2 std dev bands (outer, lighter)
    fig.add_trace(
        go.Scatter(
            x=trend_data["dates"],
            y=trend_data["plus2_std"],
            mode="lines",
            name="+2 Std Dev",
            line={"color": "rgba(128,128,128,0.5)", "width": 1, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend_data["dates"],
            y=trend_data["minus2_std"],
            mode="lines",
            name="-2 Std Dev",
            line={"color": "rgba(128,128,128,0.5)", "width": 1, "dash": "dash"},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # ±1 std dev bands (inner, darker)
    fig.add_trace(
        go.Scatter(
            x=trend_data["dates"],
            y=trend_data["plus1_std"],
            mode="lines",
            name="+1 Std Dev",
            line={"color": "rgba(128,128,128,0.6)", "width": 1, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend_data["dates"],
            y=trend_data["minus1_std"],
            mode="lines",
            name="-1 Std Dev",
            line={"color": "rgba(128,128,128,0.6)", "width": 1, "dash": "dash"},
            showlegend=False,
            hoverinfo="skip",
        )
    )


def add_historical_curves(
    fig: go.Figure,
    curves: List[Dict[str, Any]],
) -> None:
    """
    Add historical curves (moving averages, historical reference lines).

    Args:
        fig: Plotly figure to modify
        curves: List of curve data dictionaries, each containing:
               - x: x-axis data
               - y: y-axis data
               - name: curve name
               - color: optional color override
               - opacity: optional opacity (default 0.7)
    """
    style = LINE_STYLES["historical"]
    default_color = MARKET_COLORS["warning"]

    for curve in curves:
        color = curve.get("color", default_color)
        opacity = curve.get("opacity", 0.7)

        fig.add_trace(
            go.Scatter(
                x=curve["x"],
                y=curve["y"],
                mode="lines",
                name=curve["name"],
                line={"color": color, "width": style["width"], "dash": style["dash"]},
                opacity=opacity,
                hovertemplate=f"<b>{curve['name']}</b><br>"
                + "Value: %{y:.2f}<br><extra></extra>",
            )
        )
