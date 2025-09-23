"""Precious metals chart component."""

from datetime import datetime

import reflex as rx
import plotly.graph_objects as go

from app.lib.charts import (
    TimeSeriesChartConfig,
    MARKET_COLORS,
    create_timeseries_chart,
    add_main_series,
    add_historical_curves,
    get_default_theme_colors,
)
from app.lib.periods import fix_datetime
from app.flows.markets.precious_metals import fetch_precious_metals_data
from app.lib.exceptions import PageOutputException


# pylint: disable=inherit-non-class
class PreciousMetalsState(rx.State):
    """State for the precious metals chart component."""

    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    @rx.event
    async def update_precious_metals_chart_data(self, base_date: datetime):
        """Update the precious metals (Gold Futures) chart using workflow data."""
        self.loading = True

        try:
            # Get precious metals data directly from workflow
            precious_metals_result = await fetch_precious_metals_data(base_date)
            precious_metals_data = precious_metals_result.get("data")

            # Create chart configuration
            config = TimeSeriesChartConfig(
                title="Gold Futures",
                yaxis_title="Price (USD/oz)",
                hover_format="%{y:.2f}",
                height=400,
                primary_color=MARKET_COLORS["primary"],
                primary_style="solid",
            )

            # Create the base chart
            fig = create_timeseries_chart(
                data=precious_metals_data,
                config=config,
                theme_colors=self.get_theme_colors(),
                column_name="Gold",
                include_main_series=False,  # We'll add it manually
            )

            # Add moving averages if available
            curves = []
            if (
                "Gold_MA50" in precious_metals_data.columns
                and not precious_metals_data["Gold_MA50"].isna().all()
            ):
                curves.append(
                    {
                        "x": precious_metals_data.index,
                        "y": precious_metals_data["Gold_MA50"],
                        "name": "50-Day Moving Average",
                        "color": MARKET_COLORS["warning"],
                        "opacity": 0.8,
                    }
                )

            if (
                "Gold_MA200" in precious_metals_data.columns
                and not precious_metals_data["Gold_MA200"].isna().all()
            ):
                curves.append(
                    {
                        "x": precious_metals_data.index,
                        "y": precious_metals_data["Gold_MA200"],
                        "name": "200-Day Moving Average",
                        "color": MARKET_COLORS["safe"],
                        "opacity": 0.8,
                    }
                )

            if curves:
                add_historical_curves(fig, curves)

            # Add main gold data series on top
            add_main_series(fig, precious_metals_data, config, "Gold")

            self.chart_figure = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="precious metals chart",
                message=f"Failed to generate precious metals chart: {e}",
                user_message=(
                    "Failed to generate precious metals chart. Please try "
                    "refreshing the data."
                ),
                context={"error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_precious_metals(state: PreciousMetalsState, base_date: datetime):
    """
    Decentralized event handler to update precious metals chart.
    Called from the main page when period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_precious_metals_chart_data(base_date)


def precious_metals() -> rx.Component:
    """Precious Metals chart component (Gold Futures)."""
    return rx.cond(
        PreciousMetalsState.loading,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=PreciousMetalsState.chart_figure,
            width="100%",
            height="400px",
        ),
    )
