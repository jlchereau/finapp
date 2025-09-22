"""MSCI World Index chart component."""

from datetime import datetime

import reflex as rx
import plotly.graph_objects as go

from app.lib.charts import (
    TimeSeriesChartConfig,
    create_timeseries_chart,
    add_main_series,
    get_default_theme_colors,
)
from app.lib.periods import fix_datetime
from app.flows.markets.msci_world import fetch_msci_world_data
from app.lib.exceptions import PageOutputException


class MsciWorldState(rx.State):  # pylint: disable=inherit-non-class
    """State for the MSCI World Index chart component."""

    # base_date: rx.Field[datetime] = rx.field(default_factory=datetime.now)
    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    @rx.event
    async def update_msci_chart(self, base_date: datetime):
        """Update the MSCI World chart using workflow data."""
        self.loading = True

        try:
            # Get MSCI World data directly from workflow
            result = await fetch_msci_world_data(base_date)
            msci_data = result.get("data")

            # No need to test this, let it raise an exception when creating the chart
            # if msci_data is None or msci_data.empty:
            #     self.chart_figure = go.Figure()
            #     # yield breaks the event and
            #     # rx.toast does not display (even without yield)
            #     yield rx.toast.warning(
            #         "No MSCI World data available for the selected period"
            #     )
            #     return

            # Create chart configuration
            config = TimeSeriesChartConfig(
                title="MSCI World",
                yaxis_title="Index Value",
                hover_format="%{y:.2f}",
                height=400,
            )

            # Create the base chart
            fig = create_timeseries_chart(
                data=msci_data,
                config=config,
                theme_colors=self.get_theme_colors(),
                column_name="MSCI_World",
                include_main_series=False,  # We'll add it manually
            )

            # Add main MSCI World series
            add_main_series(fig, msci_data, config, "MSCI_World")

            self.chart_figure = fig
            # yield breaks the event and
            # rx.toast does not display (even without yield)
            # besides, that is a stupid message:
            # the chart displaying is proof enough of success
            # yield rx.toast.success("MSCI World chart updated successfully")

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="MSCI World chart",
                message=f"Failed to generate MSCI World chart: {e}",
                user_message=(
                    "Failed to generate MSCI World chart. Please try "
                    "refreshing the data."
                ),
                context={"error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_msci_world(state: MsciWorldState, base_date: datetime):
    """
    Decentralized event handler to update MSCI World chart.
    Called from the main page when period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_msci_chart(base_date)


def msci_world() -> rx.Component:
    """MSCI World Index chart component with moving averages and Bollinger bands."""
    return rx.cond(
        MsciWorldState.loading,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=MsciWorldState.chart_figure,
            width="100%",
            height="400px",
        ),
    )
