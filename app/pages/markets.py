"""
Markets page
"""

from typing import List, Optional
from datetime import datetime, timedelta

import pandas as pd
import reflex as rx
import plotly.graph_objects as go

from app.flows.markets import fetch_buffet_indicator_data
from app.templates.template import template


class MarketState(rx.State):  # pylint: disable=inherit-non-class
    """The app state."""

    active_tab: str = "summary"

    # Buffet Indicator state
    base_date_option: str = "10Y"
    base_date_options: List[str] = [
        "1Y",
        "2Y",
        "3Y",
        "5Y",
        "10Y",
        "20Y",
        "MAX",
    ]

    # Chart data
    chart_figure_buffet: go.Figure = go.Figure()

    # Loading state
    loading_buffet: bool = False

    def set_active_tab(self, tab: str):
        """Switch between metrics and plot tabs."""
        self.active_tab = tab

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        # Use transparent backgrounds and neutral colors that adapt to theme
        return {
            "plot_bgcolor": "rgba(0,0,0,0)",  # Transparent - inherits page background
            "paper_bgcolor": "rgba(0,0,0,0)",  # Transparent - inherits page background
            "grid_color": "rgba(128,128,128,0.3)",  # Semi-transparent gray
            "line_color": "rgba(128,128,128,0.6)",  # Semi-transparent gray
            "text_color": None,  # Let Plotly use default which respects theme
            "hover_bgcolor": "rgba(0,0,0,0.8)",  # Semi-transparent dark background
            "hover_bordercolor": "rgba(128,128,128,0.8)",  # Semi-transparent border
        }

    def set_base_date(self, option: str):
        """Set base date option and update Buffet Indicator chart."""
        self.base_date_option = option
        yield rx.toast.info(f"Changed time period to {option}")
        yield MarketState.update_buffet_chart

    async def get_buffet_data(self, base_date: datetime) -> pd.DataFrame:
        """Get Buffet Indicator data using workflow."""
        try:
            # Use the markets workflow to fetch and calculate data
            result = await fetch_buffet_indicator_data(base_date)

            # Extract the DataFrame
            buffet_data = result.get("data")

            if buffet_data is None or buffet_data.empty:
                return pd.DataFrame()

            return buffet_data

        except Exception:
            return pd.DataFrame()

    def _get_base_date(self) -> Optional[str]:
        """Convert base date option to actual date string."""
        today = datetime.now()

        if self.base_date_option == "1Y":
            base_date = today - timedelta(days=365)
        elif self.base_date_option == "2Y":
            base_date = today - timedelta(days=730)
        elif self.base_date_option == "3Y":
            base_date = today - timedelta(days=1095)
        elif self.base_date_option == "5Y":
            base_date = today - timedelta(days=1825)
        elif self.base_date_option == "10Y":
            base_date = today - timedelta(days=3650)
        elif self.base_date_option == "20Y":
            base_date = today - timedelta(days=7300)
        else:  # MAX
            return None

        return base_date.strftime("%Y-%m-%d")

    @rx.event(background=True)  # pylint: disable=not-callable
    async def update_buffet_chart(self):
        """Update the Buffet Indicator chart using background processing."""
        async with self:
            self.loading_buffet = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                # For MAX option, use a very old date
                base_date = datetime(1970, 1, 1)
                async with self:
                    yield rx.toast.info("Loading maximum available data...")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")
                async with self:
                    period = self.base_date_option
                    date_str = base_date.strftime("%Y-%m-%d")
                    yield rx.toast.info(f"Loading data from {period} ({date_str})")
            # Get Buffet Indicator data
            buffet_data = await self.get_buffet_data(base_date)

            if buffet_data is None or buffet_data.empty:
                async with self:
                    self.chart_figure_buffet = go.Figure()
                    yield rx.toast.warning(
                        "No Buffet Indicator data available for selected date range"
                    )
                return

            # Log successful data fetch
            async with self:
                yield rx.toast.success(
                    f"Loaded Buffet Indicator data: {buffet_data.shape[0]} quarters"
                )

            # Create plotly chart
            fig = go.Figure()

            # Plot Buffet Indicator
            fig.add_trace(
                go.Scatter(
                    x=buffet_data.index,
                    y=buffet_data["Buffet_Indicator"],
                    mode="lines+markers",
                    name="Buffet Indicator",
                    line=dict(color="#2563eb", width=3),
                    marker=dict(size=4),
                    hovertemplate="<b>Buffet Indicator</b><br>"
                    + "Date: %{x}<br>"
                    + "Value: %{y:.1f}<br>"
                    + "<extra></extra>",
                )
            )

            # Add reference lines
            fig.add_hline(
                y=100,
                line_dash="dash",
                line_color="orange",
                opacity=0.7,
                annotation_text="Historical Average (~100)",
                annotation_position="bottom right",
            )
            fig.add_hline(
                y=130,
                line_dash="dash",
                line_color="red",
                opacity=0.7,
                annotation_text="Overvalued (>130)",
                annotation_position="top right",
            )

            # Get theme-appropriate colors
            theme_colors = self.get_theme_colors()

            # Update layout
            title = "Buffet Indicator (Market Cap / GDP)"
            layout_props = {
                "title": title,
                "xaxis_title": "Date",
                "yaxis_title": "Buffet Indicator (%)",
                "hovermode": "x unified",
                "showlegend": True,
                "height": 400,
                "margin": dict(l=50, r=50, t=80, b=50),
                "plot_bgcolor": theme_colors["plot_bgcolor"],
                "paper_bgcolor": theme_colors["paper_bgcolor"],
                "hoverlabel": dict(
                    bgcolor=theme_colors["hover_bgcolor"],
                    bordercolor=theme_colors["hover_bordercolor"],
                    font_size=14,
                    font_color="white",
                ),
            }

            # Only add font_color if it's not None
            if theme_colors["text_color"] is not None:
                layout_props["font_color"] = theme_colors["text_color"]

            fig.update_layout(**layout_props)

            # Update axes
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=theme_colors["grid_color"],
                showline=True,
                linewidth=1,
                linecolor=theme_colors["line_color"],
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=theme_colors["grid_color"],
                showline=True,
                linewidth=1,
                linecolor=theme_colors["line_color"],
            )

            async with self:
                self.chart_figure_buffet = fig
                yield rx.toast.success("Buffet Indicator chart updated successfully")

        except Exception:
            async with self:
                self.chart_figure_buffet = go.Figure()
                yield rx.toast.error("Buffet Indicator chart update failed")
        finally:
            async with self:
                self.loading_buffet = False

    @rx.event(background=True)  # pylint: disable=not-callable
    async def run_workflows(self):
        """Load initial Buffet Indicator data."""
        yield MarketState.update_buffet_chart


def plots_fear_and_greed_index() -> rx.Component:
    """Fear and Greed index plot."""
    return rx.box(rx.text("Fear and Greed index plot"))


def plots_buffet_indicator() -> rx.Component:
    """
    Buffet indicator plot.
    See https://www.currentmarketvaluation.com/models/buffett-indicator.php
    """
    return rx.vstack(
        rx.hstack(
            rx.text("Period:", font_weight="bold"),
            rx.select(
                MarketState.base_date_options,
                value=MarketState.base_date_option,
                on_change=MarketState.set_base_date,
            ),
            spacing="2",
            align="center",
            justify="start",
        ),
        rx.cond(
            MarketState.loading_buffet,
            rx.center(rx.spinner(), height="400px"),
            rx.plotly(
                data=MarketState.chart_figure_buffet,
                width="100%",
                height="400px",
            ),
        ),
        spacing="3",
        width="100%",
    )


def plots_yield_curve() -> rx.Component:
    """Yield curve plot."""
    return rx.box(rx.text("Yield curve plot"))


def tabs_summary() -> rx.Component:
    """Summary tab content."""
    return rx.hstack(
        rx.grid(
            rx.card("Card 1"),
            rx.card("Card 2"),
            rx.card("Card 3"),
            rx.card("Card 4"),
            columns="2",
            spacing="3",
            width="70%",
        ),
        rx.card(rx.text("News feed"), width="30%"),
    )


def tabs_us() -> rx.Component:
    """US tab content."""
    return rx.grid(
        rx.card(plots_fear_and_greed_index()),
        rx.card(plots_buffet_indicator()),
        rx.card(plots_yield_curve()),
        rx.card("Card 4"),
        columns="2",
        spacing="3",
        width="100%",
    )


def tabs_eu() -> rx.Component:
    """Europe tab content."""
    return rx.grid(
        rx.card("Card 1"),
        rx.card("Card 2"),
        rx.card("Card 3"),
        rx.card("Card 4"),
        columns="2",
        spacing="3",
        width="100%",
    )


# pylint: disable=not-callable
@rx.page(
    route="/markets",
    on_load=MarketState.run_workflows,  # pyright: ignore[reportArgumentType]
)
@template
def page():
    """The markets page."""
    return rx.vstack(
        rx.heading("Markets", size="6", margin_bottom="1rem"),
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("Summary", value="summary"),
                rx.tabs.trigger("US", value="us"),
                rx.tabs.trigger("Europe", value="eu"),
                # Emerging markets
            ),
            rx.tabs.content(tabs_summary(), value="summary", padding_top="1rem"),
            rx.tabs.content(tabs_us(), value="us", padding_top="1rem"),
            rx.tabs.content(tabs_eu(), value="eu", padding_top="1rem"),
            value=MarketState.active_tab,
            on_change=MarketState.set_active_tab,
            width="100%",
        ),
        # height="100%",
        spacing="0",
    )
