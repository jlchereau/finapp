"""
Chart creation utilities for financial data visualization.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd
import plotly.graph_objects as go


@dataclass
class ChartConfig:
    """Configuration for chart creation."""

    title: str
    yaxis_title: str
    hover_format: str
    height: int = 300
    yaxis_range: Optional[List[float]] = None
    reference_lines: Optional[List[Dict[str, Any]]] = None


class RSIChartConfig(ChartConfig):
    """Configuration for RSI charts with reference lines."""

    def __init__(self, title: str = "RSI"):
        super().__init__(
            title=title,
            yaxis_title="RSI",
            hover_format="RSI: %{y:.1f}<br>",
            yaxis_range=[0, 100],
            reference_lines=[
                {
                    "y": 70,
                    "line_dash": "dash",
                    "line_color": "red",
                    "opacity": 0.7,
                    "annotation_text": "Overbought (70)",
                },
                {
                    "y": 30,
                    "line_dash": "dash",
                    "line_color": "green",
                    "opacity": 0.7,
                    "annotation_text": "Oversold (30)",
                },
            ],
        )

    def is_overbought_level(self, value: float) -> bool:
        """Check if RSI value indicates overbought condition (>= 70)."""
        return value >= 70

    def is_oversold_level(self, value: float) -> bool:
        """Check if RSI value indicates oversold condition (<= 30)."""
        return value <= 30


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
