"""
Buffet Indicator chart component.
See https://www.currentmarketvaluation.com/models/buffett-indicator.php
"""

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
    add_trend_display,
    get_default_theme_colors,
)
from app.lib.periods import fix_datetime
from app.flows.markets.buffet import fetch_buffet_indicator_data
from app.lib.exceptions import PageOutputException


class BuffetIndicatorState(rx.State):  # pylint: disable=inherit-non-class
    """State for the Buffet Indicator chart component."""

    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    @rx.event
    async def update_buffet_chart(self, base_date: datetime):
        """Update the Buffet Indicator chart using workflow data."""
        self.loading = True

        try:
            # Get Buffet Indicator data directly from workflow
            result = await fetch_buffet_indicator_data(base_date, "1Y")
            buffet_data = result.get("data")
            trend_data = result.get("trend_data")

            # Create chart configuration
            config = TimeSeriesChartConfig(
                title="Buffet Indicator",
                yaxis_title="Buffet Indicator (%)",
                hover_format="%{y:.1f}",
                height=400,
                primary_color=MARKET_COLORS["primary"],
            )

            # Create the base chart
            fig = create_timeseries_chart(
                data=buffet_data,
                config=config,
                theme_colors=self.get_theme_colors(),
                column_name="Buffet_Indicator",
                include_main_series=False,  # We'll add it manually
            )

            # Add trend data if available
            if trend_data is not None:
                add_trend_display(fig, trend_data)

            # Add threshold lines with semantic colors
            thresholds = [
                ThresholdLine(
                    value=130,
                    color=MARKET_COLORS["danger"],
                    label="Overvalued (>130)",
                    position="top right",
                ),
                ThresholdLine(
                    value=100,
                    color=MARKET_COLORS["warning"],
                    label="Historical Average (~100)",
                    position="bottom right",
                ),
            ]
            add_threshold_lines(fig, thresholds)

            # Add main data series on top
            add_main_series(fig, buffet_data, config, "Buffet_Indicator")

            self.chart_figure = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="Buffet Indicator chart",
                message=f"Failed to generate Buffet Indicator chart: {e}",
                user_message=(
                    "Failed to generate Buffet Indicator chart. Please try "
                    "refreshing the data."
                ),
                context={"error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_buffet_indicator(state: BuffetIndicatorState, base_date: datetime):
    """
    Decentralized event handler to update Buffet Indicator chart.
    Called from the main page when period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_buffet_chart(base_date)


def buffet_indicator() -> rx.Component:
    """
    Buffet indicator chart component.
    See https://www.currentmarketvaluation.com/models/buffett-indicator.php
    """
    return rx.cond(
        BuffetIndicatorState.loading,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=BuffetIndicatorState.chart_figure,
            width="100%",
            height="400px",
        ),
    )
