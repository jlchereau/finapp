"""Volume chart component."""

from datetime import datetime
from typing import List

import reflex as rx
import plotly.graph_objects as go

from app.flows.compare import fetch_volume_data
from app.lib.periods import fix_datetime
from app.lib.charts import (
    ChartConfig,
    create_comparison_chart,
    get_default_theme_colors,
)
from app.lib.exceptions import PageOutputException


# pylint: disable=inherit-non-class
class VolumeChartState(rx.State):
    """State for the volume chart component."""

    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    @rx.event
    async def update_volume_chart_data(self, tickers: List[str], base_date: datetime):
        """Update the volume chart using workflow data."""
        if not tickers:
            self.chart_figure = go.Figure()
            return

        self.loading = True

        try:
            # Get volume data directly from workflow
            result = await fetch_volume_data(tickers=tickers, base_date=base_date)
            volume_data = result.get("data")

            if volume_data is None or volume_data.empty:
                self.chart_figure = go.Figure()
                return

            # Create chart using reusable function
            config = ChartConfig(
                title="Volume",
                yaxis_title="Volume",
                hover_format="Volume: %{y:,.0f}<br>",
            )
            theme_colors = self.get_theme_colors()
            fig = create_comparison_chart(volume_data, config, theme_colors)

            self.chart_figure = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="volume chart",
                message=f"Failed to generate volume chart: {e}",
                user_message=(
                    "Failed to generate volume chart. Please try refreshing the data."
                ),
                context={"tickers": tickers, "error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_volume_chart(
    state: VolumeChartState, tickers: List[str], base_date: datetime
):
    """
    Decentralized event handler to update volume chart.
    Called from the main page when tickers or period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_volume_chart_data(tickers, base_date)


def volume_chart() -> rx.Component:
    """Volume chart component."""
    return rx.cond(
        VolumeChartState.loading,
        rx.center(rx.spinner(), height="300px"),
        rx.plotly(
            data=VolumeChartState.chart_figure,
            width="100%",
            height="300px",
        ),
    )
