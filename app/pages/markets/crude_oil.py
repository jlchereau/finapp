"""Crude oil chart component."""

from datetime import datetime

import reflex as rx
import plotly.graph_objects as go

from app.lib.charts import (
    ChartConfig,
    create_comparison_chart,
    get_default_theme_colors,
)
from app.lib.periods import fix_datetime
from app.flows.markets.crude_oil import fetch_crude_oil_data
from app.lib.exceptions import PageOutputException


# pylint: disable=inherit-non-class
class CrudeOilState(rx.State):
    """State for the crude oil chart component."""

    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    @rx.event
    async def update_crude_oil_chart_data(self, base_date: datetime):
        """Update the crude oil (WTI and Brent) chart using workflow data."""
        self.loading = True

        try:
            # Get crude oil data
            crude_oil_result = await fetch_crude_oil_data(base_date)
            crude_oil_data = crude_oil_result.get("data")

            if crude_oil_data is None or crude_oil_data.empty:
                self.chart_figure = go.Figure()
                return

            # Create comparison chart configuration
            config = ChartConfig(
                title="Crude Oil",
                yaxis_title="Price (USD/bbl)",
                hover_format="Price: %{y:.2f}<br>",
                height=400,
            )

            # Get theme colors
            theme_colors = self.get_theme_colors()

            # Create the chart
            fig = create_comparison_chart(crude_oil_data, config, theme_colors)

            self.chart_figure = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="crude oil chart",
                message=f"Failed to generate crude oil chart: {e}",
                user_message=(
                    "Failed to generate crude oil chart. "
                    "Please try refreshing the data."
                ),
                context={"error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_crude_oil(state: CrudeOilState, base_date: datetime):
    """
    Decentralized event handler to update crude oil chart.
    Called from the main page when period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_crude_oil_chart_data(base_date)


def crude_oil() -> rx.Component:
    """Crude oil prices chart component (WTI and Brent)."""
    return rx.cond(
        CrudeOilState.loading,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=CrudeOilState.chart_figure,
            width="100%",
            height="400px",
        ),
    )
