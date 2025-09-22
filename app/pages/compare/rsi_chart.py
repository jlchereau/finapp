"""RSI chart component."""

from datetime import datetime
from typing import List

import reflex as rx
import plotly.graph_objects as go

from app.flows.compare import fetch_rsi_data
from app.lib.periods import fix_datetime
from app.lib.charts import (
    TimeSeriesChartConfig,
    ThresholdLine,
    MARKET_COLORS,
    create_comparison_chart,
    add_threshold_lines,
    get_default_theme_colors,
)
from app.lib.exceptions import PageOutputException


class RSIChartState(rx.State):  # pylint: disable=inherit-non-class
    """State for the RSI chart component."""

    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    @rx.event
    async def update_rsi_chart_data(self, tickers: List[str], base_date: datetime):
        """Update the RSI chart using workflow data."""
        if not tickers:
            self.chart_figure = go.Figure()
            return

        self.loading = True

        try:
            # Get RSI data directly from workflow
            result = await fetch_rsi_data(tickers=tickers, base_date=base_date)
            rsi_data = result.get("data")

            if rsi_data is None or rsi_data.empty:
                self.chart_figure = go.Figure()
                return

            # Create chart using reusable function with RSI-specific configuration
            config = TimeSeriesChartConfig(
                title="RSI",
                yaxis_title="RSI",
                hover_format="RSI: %{y:.1f}<br>",
                height=300,  # Comparison charts are shorter
                yaxis_range=[0, 100],  # RSI is bounded 0-100
            )
            theme_colors = self.get_theme_colors()
            fig = create_comparison_chart(rsi_data, config, theme_colors)

            # Add RSI threshold lines for overbought/oversold levels
            thresholds = [
                ThresholdLine(
                    value=70,
                    color=MARKET_COLORS["danger"],
                    label="Overbought (70)",
                    position="top right",
                ),
                ThresholdLine(
                    value=30,
                    color=MARKET_COLORS["safe"],
                    label="Oversold (30)",
                    position="bottom right",
                ),
            ]
            add_threshold_lines(fig, thresholds)

            self.chart_figure = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="RSI chart",
                message=f"Failed to generate RSI chart: {e}",
                user_message=(
                    "Failed to generate RSI chart. Please try refreshing the data."
                ),
                context={"tickers": tickers, "error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_rsi_chart(
    state: RSIChartState, tickers: List[str], base_date: datetime
):
    """
    Decentralized event handler to update RSI chart.
    Called from the main page when tickers or period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_rsi_chart_data(tickers, base_date)


def rsi_chart() -> rx.Component:
    """RSI chart component."""
    return rx.cond(
        RSIChartState.loading,
        rx.center(rx.spinner(), height="300px"),
        rx.plotly(
            data=RSIChartState.chart_figure,
            width="100%",
            height="300px",
        ),
    )
