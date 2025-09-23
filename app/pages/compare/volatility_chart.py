"""Volatility chart component."""

from datetime import datetime
from typing import List

import reflex as rx
import plotly.graph_objects as go

from app.flows.compare import fetch_volatility_data
from app.lib.periods import fix_datetime
from app.lib.charts import (
    ChartConfig,
    create_comparison_chart,
    get_default_theme_colors,
)
from app.lib.exceptions import PageOutputException


# pylint: disable=inherit-non-class
class VolatilityChartState(rx.State):
    """State for the volatility chart component."""

    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    @rx.event
    async def update_volatility_chart_data(
        self, tickers: List[str], base_date: datetime
    ):
        """Update the volatility chart using workflow data."""
        if not tickers:
            self.chart_figure = go.Figure()
            return

        self.loading = True

        try:
            # Get volatility data directly from workflow
            result = await fetch_volatility_data(tickers=tickers, base_date=base_date)
            volatility_data = result.get("data")

            if volatility_data is None or volatility_data.empty:
                self.chart_figure = go.Figure()
                return

            # Create chart using reusable function
            config = ChartConfig(
                title="Volatility",
                yaxis_title="Volatility (%)",
                hover_format="Volatility: %{y:.2f}%<br>",
            )
            theme_colors = self.get_theme_colors()
            fig = create_comparison_chart(volatility_data, config, theme_colors)

            self.chart_figure = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="volatility chart",
                message=f"Failed to generate volatility chart: {e}",
                user_message=(
                    "Failed to generate volatility chart. "
                    "Please try refreshing the data."
                ),
                context={"tickers": tickers, "error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_volatility_chart(
    state: VolatilityChartState, tickers: List[str], base_date: datetime
):
    """
    Decentralized event handler to update volatility chart.
    Called from the main page when tickers or period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_volatility_chart_data(tickers, base_date)


def volatility_chart() -> rx.Component:
    """Volatility chart component."""
    return rx.cond(
        VolatilityChartState.loading,
        rx.center(rx.spinner(), height="300px"),
        rx.plotly(
            data=VolatilityChartState.chart_figure,
            width="100%",
            height="300px",
        ),
    )
