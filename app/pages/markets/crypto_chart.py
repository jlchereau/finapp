"""Cryptocurrency chart component."""

from datetime import datetime

import reflex as rx
import plotly.graph_objects as go

from app.lib.charts import (
    ChartConfig,
    create_comparison_chart,
    get_default_theme_colors,
)
from app.lib.periods import fix_datetime
from app.flows.markets.crypto import fetch_crypto_data
from app.lib.exceptions import PageOutputException


class CryptoChartState(rx.State):  # pylint: disable=inherit-non-class
    """State for the cryptocurrency chart component."""

    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    @rx.event
    async def update_crypto_chart_data(self, base_date: datetime):
        """Update the cryptocurrency chart using workflow data."""
        self.loading = True

        try:
            # Get cryptocurrency data
            crypto_result = await fetch_crypto_data(base_date)
            crypto_data = crypto_result.get("data")

            if crypto_data is None or crypto_data.empty:
                self.chart_figure = go.Figure()
                return

            # Create comparison chart configuration
            config = ChartConfig(
                title="Crypto Prices",
                yaxis_title="Price (USD)",
                hover_format="Price: %{y:.2f}<br>",
                height=400,
            )

            # Get theme colors
            theme_colors = self.get_theme_colors()

            # Create comparison chart for Bitcoin vs Ethereum
            fig = create_comparison_chart(crypto_data, config, theme_colors)

            self.chart_figure = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="cryptocurrency chart",
                message=f"Failed to generate cryptocurrency chart: {e}",
                user_message=(
                    "Failed to generate cryptocurrency chart. "
                    "Please try refreshing the data."
                ),
                context={"error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_crypto_chart(state: CryptoChartState, base_date: datetime):
    """
    Decentralized event handler to update crypto chart.
    Called from the main page when period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_crypto_chart_data(base_date)


def crypto_chart() -> rx.Component:
    """Cryptocurrency prices chart component (Bitcoin and Ethereum)."""
    return rx.cond(
        CryptoChartState.loading,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=CryptoChartState.chart_figure,
            width="100%",
            height="400px",
        ),
    )
