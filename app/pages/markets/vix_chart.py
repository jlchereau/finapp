"""VIX chart component."""

from datetime import datetime

import reflex as rx
import plotly.graph_objects as go

from app.lib.charts import (
    TimeSeriesChartConfig,
    ThresholdLine,
    MARKET_COLORS,
    create_timeseries_chart,
    add_main_series,
    add_threshold_lines,
    add_historical_curves,
    get_default_theme_colors,
)
from app.lib.periods import fix_datetime
from app.flows.markets.vix import fetch_vix_data
from app.lib.exceptions import PageOutputException


class VixChartState(rx.State):  # pylint: disable=inherit-non-class
    """State for the VIX chart component."""

    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    @rx.event
    async def update_vix_chart_data(self, base_date: datetime):
        """Update the VIX chart using workflow data."""
        self.loading = True

        try:
            # Get VIX data directly from workflow
            vix_result = await fetch_vix_data(base_date)
            vix_data = vix_result.get("data")
            historical_mean = vix_result.get("historical_mean", 20.0)

            # Create chart configuration
            config = TimeSeriesChartConfig(
                title="VIX Volatility",
                yaxis_title="VIX Level",
                hover_format="%{y:.2f}",
                height=400,
                primary_color=MARKET_COLORS["primary"],
                primary_style="solid",
            )

            # Create the base chart
            fig = create_timeseries_chart(
                data=vix_data,
                config=config,
                theme_colors=self.get_theme_colors(),
                column_name="VIX",
                include_main_series=False,  # We'll add it manually
            )

            # Add 50-day moving average if available
            if "VIX_MA50" in vix_data.columns and not vix_data["VIX_MA50"].isna().all():
                curves = [
                    {
                        "x": vix_data.index,
                        "y": vix_data["VIX_MA50"],
                        "name": "50-Day Moving Average",
                        "color": MARKET_COLORS["warning"],
                        "opacity": 0.8,
                    }
                ]
                add_historical_curves(fig, curves)

            # Add threshold lines with semantic colors
            thresholds = [
                ThresholdLine(
                    value=30,
                    color=MARKET_COLORS["danger"],
                    label="High Volatility (30)",
                    position="top right",
                ),
                ThresholdLine(
                    value=historical_mean,
                    color=MARKET_COLORS["warning"],
                    label=f"Historical Mean (~{historical_mean:.1f})",
                    position="bottom right",
                ),
                ThresholdLine(
                    value=10,
                    color=MARKET_COLORS["safe"],
                    label="Low Volatility (10)",
                    position="top left",
                ),
            ]
            add_threshold_lines(fig, thresholds)

            # Add main VIX data series on top
            add_main_series(fig, vix_data, config, "VIX")

            self.chart_figure = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="VIX chart",
                message=f"Failed to generate VIX chart: {e}",
                user_message=(
                    "Failed to generate VIX chart. Please try refreshing the data."
                ),
                context={"error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_vix_chart(state: VixChartState, base_date: datetime):
    """
    Decentralized event handler to update VIX chart.
    Called from the main page when period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_vix_chart_data(base_date)


def vix_chart() -> rx.Component:
    """VIX index chart component."""
    return rx.cond(
        VixChartState.loading,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=VixChartState.chart_figure,
            width="100%",
            height="400px",
        ),
    )
