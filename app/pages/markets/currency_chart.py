"""Currency exchange rates chart component."""

from datetime import datetime

import reflex as rx
import plotly.graph_objects as go

from app.lib.charts import (
    TimeSeriesChartConfig,
    MARKET_COLORS,
    create_timeseries_chart,
    get_default_theme_colors,
)
from app.lib.periods import fix_datetime
from app.flows.markets.currency import fetch_currency_data
from app.lib.exceptions import PageOutputException


# pylint: disable=inherit-non-class
class CurrencyChartState(rx.State):
    """State for the currency chart component."""

    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    @rx.event
    async def update_currency_chart_data(self, base_date: datetime):
        """Update the currency exchange rate chart using workflow data."""
        self.loading = True

        try:
            # Get currency data
            currency_result = await fetch_currency_data(base_date)
            currency_data = currency_result.get("data")

            if currency_data is None or currency_data.empty:
                self.chart_figure = go.Figure()
                return

            # Create time-series chart using utility functions
            config = TimeSeriesChartConfig(
                title="Currency Rates",
                yaxis_title="Exchange Rate",
                hover_format="Rate: %{y:.4f}<br>",
                height=400,
                primary_color=MARKET_COLORS["primary"],
                primary_style="solid",  # Solid line without markers
            )

            # Get theme colors
            theme_colors = self.get_theme_colors()

            # Create empty chart with theme (no main series yet)
            fig = create_timeseries_chart(
                currency_data,
                config,
                theme_colors,
                "USD_EUR",
                include_main_series=False,
            )

            # Add USD/EUR series (main currency pair)
            if (
                "USD_EUR" in currency_data.columns
                and not currency_data["USD_EUR"].isna().all()
            ):
                fig.add_trace(
                    go.Scatter(
                        x=currency_data.index,
                        y=currency_data["USD_EUR"],
                        mode="lines",
                        name="USD/EUR",
                        line={"color": MARKET_COLORS["primary"], "width": 2},
                        hovertemplate="<b>USD/EUR</b><br>"
                        + "Rate: %{y:.4f}<br>"
                        + "<extra></extra>",
                    )
                )

            # Add GBP/EUR series (secondary currency pair)
            if (
                "GBP_EUR" in currency_data.columns
                and not currency_data["GBP_EUR"].isna().all()
            ):
                fig.add_trace(
                    go.Scatter(
                        x=currency_data.index,
                        y=currency_data["GBP_EUR"],
                        mode="lines",
                        name="GBP/EUR",
                        line={"color": MARKET_COLORS["warning"], "width": 2},
                        hovertemplate="<b>GBP/EUR</b><br>"
                        + "Rate: %{y:.4f}<br>"
                        + "<extra></extra>",
                    )
                )

            self.chart_figure = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="currency chart",
                message=f"Failed to generate currency chart: {e}",
                user_message=(
                    "Failed to generate currency chart. Please try refreshing the data."
                ),
                context={"error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_currency_chart(state: CurrencyChartState, base_date: datetime):
    """
    Decentralized event handler to update currency chart.
    Called from the main page when period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_currency_chart_data(base_date)


def currency_chart() -> rx.Component:
    """Currency exchange rates chart component (USD/EUR and GBP/EUR)."""
    return rx.cond(
        CurrencyChartState.loading,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=CurrencyChartState.chart_figure,
            width="100%",
            height="400px",
        ),
    )
