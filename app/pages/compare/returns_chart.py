"""Returns chart component."""

from datetime import datetime
from typing import List

import reflex as rx
import plotly.graph_objects as go

from app.flows.compare import fetch_returns_data
from app.lib.periods import fix_datetime
from app.lib.charts import (
    ChartConfig,
    create_comparison_chart,
    get_default_theme_colors,
)
from app.lib.exceptions import PageOutputException


# pylint: disable=inherit-non-class
class ReturnsChartState(rx.State):
    """State for the returns chart component."""

    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    @rx.event
    async def update_returns_chart_data(self, tickers: List[str], base_date: datetime):
        """Update the returns chart using workflow data."""
        if not tickers:
            self.chart_figure = go.Figure()
            return

        self.loading = True

        try:
            # Get returns data directly from workflow
            result = await fetch_returns_data(tickers=tickers, base_date=base_date)
            returns_data = result.get("data")

            if returns_data is None or returns_data.empty:
                self.chart_figure = go.Figure()
                return

            # Create chart using reusable function
            config = ChartConfig(
                title="Returns",
                yaxis_title="Return (%)",
                hover_format="Return: %{y:.2f}%<br>",
            )
            theme_colors = self.get_theme_colors()
            fig = create_comparison_chart(returns_data, config, theme_colors)

            self.chart_figure = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="returns chart",
                message=f"Failed to generate returns chart: {e}",
                user_message=(
                    "Failed to generate returns chart. Please try refreshing the data."
                ),
                context={"tickers": tickers, "error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_returns_chart(
    state: ReturnsChartState, tickers: List[str], base_date: datetime
):
    """
    Decentralized event handler to update returns chart.
    Called from the main page when tickers or period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_returns_chart_data(tickers, base_date)


def returns_chart() -> rx.Component:
    """Returns chart component."""
    return rx.cond(
        ReturnsChartState.loading,
        rx.center(rx.spinner(), height="300px"),
        rx.plotly(
            data=ReturnsChartState.chart_figure,
            width="100%",
            height="300px",
        ),
    )
