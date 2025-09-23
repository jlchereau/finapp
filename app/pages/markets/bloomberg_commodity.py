"""Bloomberg Commodity Index chart component."""

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
from app.flows.markets.bloomberg_commodity import fetch_bloomberg_commodity_data
from app.lib.exceptions import PageOutputException


# pylint: disable=inherit-non-class
class BloombergCommodityState(rx.State):
    """State for the Bloomberg Commodity Index chart component."""

    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    @rx.event
    async def update_bloomberg_commodity_chart_data(self, base_date: datetime):
        """Update the Bloomberg Commodity Index chart using workflow data."""
        self.loading = True

        try:
            # Get Bloomberg Commodity Index data directly from workflow
            bloomberg_result = await fetch_bloomberg_commodity_data(base_date)
            bloomberg_data = bloomberg_result.get("data")

            # Create chart configuration
            config = TimeSeriesChartConfig(
                title="Commodity Index",
                yaxis_title="Index Value",
                hover_format="%{y:.2f}",
                height=400,
                primary_color=MARKET_COLORS["primary"],
                primary_style="solid",
            )

            # Create the base chart
            fig = create_timeseries_chart(
                data=bloomberg_data,
                config=config,
                theme_colors=self.get_theme_colors(),
                column_name="BCOM",
                include_main_series=False,  # We'll add it manually
            )

            # Add moving averages if available
            curves = []
            if (
                "BCOM_MA50" in bloomberg_data.columns
                and not bloomberg_data["BCOM_MA50"].isna().all()
            ):
                curves.append(
                    {
                        "x": bloomberg_data.index,
                        "y": bloomberg_data["BCOM_MA50"],
                        "name": "50-Day Moving Average",
                        "color": MARKET_COLORS["warning"],
                        "opacity": 0.8,
                    }
                )

            if (
                "BCOM_MA200" in bloomberg_data.columns
                and not bloomberg_data["BCOM_MA200"].isna().all()
            ):
                curves.append(
                    {
                        "x": bloomberg_data.index,
                        "y": bloomberg_data["BCOM_MA200"],
                        "name": "200-Day Moving Average",
                        "color": MARKET_COLORS["safe"],
                        "opacity": 0.8,
                    }
                )

            if curves:
                add_historical_curves(fig, curves)

            # Add main Bloomberg Commodity Index data series on top
            add_main_series(fig, bloomberg_data, config, "BCOM")

            self.chart_figure = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="Bloomberg Commodity Index chart",
                message=f"Failed to generate Bloomberg Commodity Index chart: {e}",
                user_message=(
                    "Failed to generate Bloomberg Commodity Index chart. Please try "
                    "refreshing the data."
                ),
                context={"error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_bloomberg_commodity(
    state: BloombergCommodityState, base_date: datetime
):
    """
    Decentralized event handler to update Bloomberg Commodity Index chart.
    Called from the main page when period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_bloomberg_commodity_chart_data(base_date)


def bloomberg_commodity() -> rx.Component:
    """Bloomberg Commodity Index chart component."""
    return rx.cond(
        BloombergCommodityState.loading,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=BloombergCommodityState.chart_figure,
            width="100%",
            height="400px",
        ),
    )
