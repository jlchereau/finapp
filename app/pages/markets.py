"""
Markets page
"""

from typing import List, Optional
from datetime import datetime

import pandas as pd
import reflex as rx
import plotly.graph_objects as go

from app.flows.markets import (
    fetch_buffet_indicator_data,
    fetch_vix_data,
    fetch_yield_curve_data,
)
from app.lib.exceptions import DataProcessingException, ChartException
from app.lib.periods import (
    get_period_options,
    calculate_base_date,
    get_max_fallback_date,
    format_date_range_message,
)
from app.templates.template import template


class MarketState(rx.State):  # pylint: disable=inherit-non-class
    """The app state."""

    active_tab: str = "summary"

    # Chart settings
    base_date_option: str = "10Y"
    base_date_options: List[str] = get_period_options()

    # Chart data
    chart_figure_buffet: go.Figure = go.Figure()
    chart_figure_vix: go.Figure = go.Figure()
    chart_figure_yield: go.Figure = go.Figure()

    # Loading state
    loading_buffet: bool = False
    loading_vix: bool = False
    loading_yield: bool = False

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
        """Set base date option and update charts."""
        self.base_date_option = option
        yield rx.toast.info(f"Changed time period to {option}")
        yield MarketState.update_buffet_chart
        yield MarketState.update_vix_chart
        yield MarketState.update_yield_chart

    async def get_buffet_data(
        self, base_date: datetime
    ) -> tuple[pd.DataFrame, dict | None, dict]:
        """Get Buffet Indicator data using workflow."""
        # Use the markets workflow to fetch and calculate data
        result = await fetch_buffet_indicator_data(base_date, self.base_date_option)

        # Extract the DataFrame, trend data, and adjustment info
        buffet_data = result.get("data")
        trend_data = result.get("trend_data")
        adjustment_info = {
            "was_adjusted": result.get("was_adjusted", False),
            "original_period": result.get("original_period", self.base_date_option),
            "actual_period": result.get("actual_period", self.base_date_option),
            "data_points": result.get("data_points", 0),
        }

        if buffet_data is None or buffet_data.empty:
            raise DataProcessingException(
                operation="fetch_buffet_indicator_data",
                message=f"No Buffet Indicator data returned for base_date: {base_date}",
                user_message=(
                    "Unable to fetch Buffet Indicator data. Please try a different "
                    "time period."
                ),
                context={
                    "base_date": str(base_date),
                    "base_date_option": self.base_date_option,
                },
            )

        return buffet_data, trend_data, adjustment_info

    async def get_vix_data(self, base_date: datetime) -> pd.DataFrame:
        """Get VIX data using workflow."""
        # Use the markets workflow to fetch VIX data
        result = await fetch_vix_data(base_date)

        # Extract the DataFrame
        vix_data = result.get("data")

        if vix_data is None or vix_data.empty:
            raise DataProcessingException(
                operation="fetch_vix_data",
                message=f"No VIX data returned for base_date: {base_date}",
                user_message=(
                    "Unable to fetch VIX data. Please try a different time period."
                ),
                context={
                    "base_date": str(base_date),
                    "base_date_option": self.base_date_option,
                },
            )

        return vix_data

    def _get_base_date(self) -> Optional[str]:
        """Convert base date option to actual date string."""
        base_date = calculate_base_date(self.base_date_option)
        if base_date is None:
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
                # For MAX option, use appropriate fallback date
                base_date = get_max_fallback_date("markets")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            async with self:
                message = format_date_range_message(
                    self.base_date_option,
                    base_date if self.base_date_option != "MAX" else None,
                )
                yield rx.toast.info(message)
            # Get Buffet Indicator data
            buffet_data, trend_data, adjustment_info = await self.get_buffet_data(
                base_date
            )

            if buffet_data is None or buffet_data.empty:
                async with self:
                    self.chart_figure_buffet = go.Figure()
                    yield rx.toast.warning(
                        "No Buffet Indicator data available for selected date range"
                    )
                return

            # Log successful data fetch and show adjustment message if needed
            async with self:
                if adjustment_info["was_adjusted"]:
                    from app.lib.periods import format_period_adjustment_message

                    adjustment_msg = format_period_adjustment_message(
                        adjustment_info["original_period"],
                        adjustment_info["actual_period"],
                        adjustment_info["data_points"],
                    )
                    yield rx.toast.info(adjustment_msg)
                else:
                    yield rx.toast.success(
                        f"Loaded Buffet Indicator data: {buffet_data.shape[0]} quarters"
                    )

            # Create plotly chart
            fig = go.Figure()

            # Use pre-calculated trend data (calculated on full dataset in workflow)
            # This ensures trend lines represent full historical context

            # Add trend lines first (so they appear behind the main line)
            if trend_data is not None:
                # Main trend line
                fig.add_trace(
                    go.Scatter(
                        x=trend_data["dates"],
                        y=trend_data["trend"],
                        mode="lines",
                        name="Exponential Trend",
                        line=dict(color="rgba(128,128,128,0.7)", width=2, dash="dash"),
                        hovertemplate="<b>Exponential Trend</b><br>"
                        # + "Date: %{x}<br>"
                        + "Value: %{y:.1f}<br>" + "<extra></extra>",
                    )
                )

                # +/- 2 std dev bands (outer)
                fig.add_trace(
                    go.Scatter(
                        x=trend_data["dates"],
                        y=trend_data["plus2_std"],
                        mode="lines",
                        name="+2 Std Dev",
                        line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dash"),
                        hoverinfo="skip",
                        # hovertemplate="<b>+2 Std Dev</b><br>"
                        # + "Date: %{x}<br>"
                        # "Value: %{y:.1f}<br>" + "<extra></extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=trend_data["dates"],
                        y=trend_data["minus2_std"],
                        mode="lines",
                        name="-2 Std Dev",
                        line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dash"),
                        showlegend=False,  # Don't show in legend to avoid duplication
                        hoverinfo="skip",
                        # hovertemplate="<b>-2 Std Dev</b><br>"
                        # + "Date: %{x}<br>"
                        # + "Value: %{y:.1f}<br>" + "<extra></extra>",
                    )
                )

                # +/- 1 std dev bands (inner)
                fig.add_trace(
                    go.Scatter(
                        x=trend_data["dates"],
                        y=trend_data["plus1_std"],
                        mode="lines",
                        name="+1 Std Dev",
                        line=dict(color="rgba(128,128,128,0.6)", width=1, dash="dash"),
                        hoverinfo="skip",
                        # hovertemplate="<b>+1 Std Dev</b><br>"
                        # + "Date: %{x}<br>"
                        # + "Value: %{y:.1f}<br>" + "<extra></extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=trend_data["dates"],
                        y=trend_data["minus1_std"],
                        mode="lines",
                        name="-1 Std Dev",
                        line=dict(color="rgba(128,128,128,0.6)", width=1, dash="dash"),
                        showlegend=False,  # Don't show in legend to avoid duplication
                        hoverinfo="skip",
                        # hovertemplate="<b>-1 Std Dev</b><br>"
                        # + "Date: %{x}<br>"
                        # + "Value: %{y:.1f}<br>" + "<extra></extra>",
                    )
                )

            # Plot Buffet Indicator (on top of trend lines)
            fig.add_trace(
                go.Scatter(
                    x=buffet_data.index,
                    y=buffet_data["Buffet_Indicator"],
                    mode="lines+markers",
                    name="Buffet Indicator",
                    line=dict(color="#2563eb", width=3),
                    marker=dict(size=4),
                    hovertemplate="<b>Buffet Indicator</b><br>"
                    # + "Date: %{x}<br>"
                    + "Value: %{y:.1f}<br>" + "<extra></extra>",
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

        except Exception as e:
            # Chart generation error - wrap in ChartException
            raise ChartException(
                chart_type="buffet_indicator",
                message=f"Failed to generate Buffet Indicator chart: {e}",
                user_message=(
                    "Failed to generate Buffet Indicator chart. Please try "
                    "refreshing the data."
                ),
                context={"base_date_option": self.base_date_option, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_buffet = False

    @rx.event(background=True)  # pylint: disable=not-callable
    async def update_vix_chart(self):
        """Update the VIX chart using background processing."""
        async with self:
            self.loading_vix = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                # For MAX option, use appropriate fallback date
                base_date = get_max_fallback_date("vix")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            async with self:
                message = format_date_range_message(
                    self.base_date_option,
                    base_date if self.base_date_option != "MAX" else None,
                )
                yield rx.toast.info(message)

            # Get VIX data
            vix_result = await fetch_vix_data(base_date)
            vix_data = vix_result.get("data")
            historical_mean = vix_result.get(
                "historical_mean", 20.0
            )  # Default fallback

            if vix_data is None or vix_data.empty:
                async with self:
                    self.chart_figure_vix = go.Figure()
                    yield rx.toast.warning(
                        "No VIX data available for selected date range"
                    )
                return

            # Log successful data fetch
            async with self:
                yield rx.toast.success(
                    f"Loaded VIX data: {vix_data.shape[0]} data points"
                )

            # Create plotly chart
            fig = go.Figure()

            # Plot VIX data
            fig.add_trace(
                go.Scatter(
                    x=vix_data.index,
                    y=vix_data["VIX"],
                    mode="lines",
                    name="VIX (Volatility Index)",
                    line=dict(color="#2563eb", width=2),  # Blue color for volatility
                    hovertemplate="<b>VIX</b><br>"
                    + "Value: %{y:.2f}<br>"
                    + "<extra></extra>",
                )
            )

            # Add 50-day moving average if available
            if "VIX_MA50" in vix_data.columns and not vix_data["VIX_MA50"].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=vix_data.index,
                        y=vix_data["VIX_MA50"],
                        mode="lines",
                        name="50-Day Moving Average",
                        line=dict(
                            color="#f59e0b", width=2, dash="dot"
                        ),  # Amber dotted line
                        hovertemplate="<b>50-Day MA</b><br>"
                        + "Value: %{y:.2f}<br>"
                        + "<extra></extra>",
                    )
                )

            # Add reference lines
            fig.add_hline(
                y=historical_mean,
                line_dash="dash",
                line_color="orange",
                opacity=0.7,
                annotation_text=f"Historical Mean (~{historical_mean:.1f})",
                annotation_position="bottom right",
            )
            fig.add_hline(
                y=10,
                line_dash="dash",
                line_color="green",
                opacity=0.7,
                annotation_text="Low Volatility (10)",
                annotation_position="top left",
            )
            fig.add_hline(
                y=30,
                line_dash="dash",
                line_color="red",
                opacity=0.7,
                annotation_text="High Volatility (30)",
                annotation_position="top right",
            )

            # Get theme-appropriate colors
            theme_colors = self.get_theme_colors()

            # Update layout
            title = "VIX (CBOE Volatility Index)"
            layout_props = {
                "title": title,
                "xaxis_title": "Date",
                "yaxis_title": "VIX Level",
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
                self.chart_figure_vix = fig
                yield rx.toast.success("VIX chart updated successfully")

        except Exception as e:
            # Chart generation error - wrap in ChartException
            raise ChartException(
                chart_type="vix",
                message=f"Failed to generate VIX chart: {e}",
                user_message=(
                    "Failed to generate VIX chart. Please try refreshing the data."
                ),
                context={"base_date_option": self.base_date_option, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_vix = False

    @rx.event(background=True)  # pylint: disable=not-callable
    async def update_yield_chart(self):
        """Update the yield curve chart using background processing."""
        async with self:
            self.loading_yield = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                # For MAX option, use appropriate fallback date
                base_date = get_max_fallback_date("yield_curve")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            async with self:
                message = format_date_range_message(
                    self.base_date_option,
                    base_date if self.base_date_option != "MAX" else None,
                )
                yield rx.toast.info(message)

            # Get yield curve data
            yield_result = await fetch_yield_curve_data(base_date)
            yield_data = yield_result.get("data")
            maturities = yield_result.get("maturities", [])
            latest_date = yield_result.get("latest_date")

            if yield_data is None or yield_data.empty:
                async with self:
                    self.chart_figure_yield = go.Figure()
                    yield rx.toast.warning(
                        "No yield curve data available for selected date range"
                    )
                return

            # Log successful data fetch
            async with self:
                yield rx.toast.success(
                    f"Loaded yield curve data: {yield_data.shape[0]} data points"
                )

            # Create plotly chart
            fig = go.Figure()

            # Get the most recent yield curve (latest available date)
            if latest_date:
                latest_data = yield_data.loc[yield_data.index == latest_date]
                if not latest_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=maturities,
                            y=latest_data.iloc[0].values,
                            mode="lines+markers",
                            name=f"Yield Curve ({latest_date.strftime('%Y-%m-%d')})",
                            line=dict(color="blue", width=3),
                            marker=dict(size=8, color="blue"),
                            hovertemplate="<b>%{fullData.name}</b><br>"
                            + "Maturity: %{x}<br>"
                            + "Yield: %{y:.2f}%<br>"
                            + "<extra></extra>",
                        )
                    )

            # Add historical perspective if we have enough data
            if len(yield_data) > 1 and latest_date is not None:
                # Add a few historical curves for context
                # (e.g., 1 year ago, 6 months ago)
                historical_dates = yield_data.index.sort_values(ascending=False)

                # Find dates approximately 1 year and 6 months ago
                for months_back, color, alpha in [
                    (12, "gray", 0.5),
                    (6, "orange", 0.7),
                ]:
                    target_date = latest_date - pd.DateOffset(months=months_back)
                    closest_date = historical_dates[historical_dates <= target_date]

                    if len(closest_date) > 0:
                        hist_date = closest_date[0]
                        hist_data = yield_data.loc[yield_data.index == hist_date]
                        if not hist_data.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=maturities,
                                    y=hist_data.iloc[0].values,
                                    mode="lines",
                                    name=(
                                        f"{months_back}M ago "
                                        f"({hist_date.strftime('%Y-%m-%d')})"
                                    ),
                                    line=dict(color=color, width=2, dash="dash"),
                                    opacity=alpha,
                                    hovertemplate="<b>%{fullData.name}</b><br>"
                                    + "Maturity: %{x}<br>"
                                    + "Yield: %{y:.2f}%<br>"
                                    + "<extra></extra>",
                                )
                            )

            # Get theme-appropriate colors
            theme_colors = self.get_theme_colors()

            # Update layout
            title = "US Treasury Yield Curve"
            layout_props = {
                "title": title,
                "xaxis_title": "Maturity",
                "yaxis_title": "Yield (%)",
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
                # Set explicit order for maturity labels
                categoryorder="array",
                categoryarray=maturities,
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
                self.chart_figure_yield = fig
                yield rx.toast.success("Yield curve chart updated successfully")

        except Exception as e:
            # Chart generation error - wrap in ChartException
            raise ChartException(
                chart_type="yield_curve",
                message=f"Failed to generate yield curve chart: {e}",
                user_message=(
                    "Failed to generate yield curve chart. "
                    "Please try refreshing the data."
                ),
                context={"base_date_option": self.base_date_option, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_yield = False

    @rx.event(background=True)  # pylint: disable=not-callable
    async def run_workflows(self):
        """Load initial chart data."""
        yield MarketState.update_buffet_chart
        yield MarketState.update_vix_chart
        yield MarketState.update_yield_chart


def plots_fear_and_greed_index() -> rx.Component:
    """Fear and Greed index plot."""
    return rx.box(rx.text("Fear and Greed index plot"))


def plots_buffet_indicator() -> rx.Component:
    """
    Buffet indicator plot.
    See https://www.currentmarketvaluation.com/models/buffett-indicator.php
    """
    return rx.cond(
        MarketState.loading_buffet,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=MarketState.chart_figure_buffet,
            width="100%",
            height="400px",
        ),
    )


def plots_yield_curve() -> rx.Component:
    """US Treasury yield curve plot."""
    return rx.cond(
        MarketState.loading_yield,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=MarketState.chart_figure_yield,
            width="100%",
            height="400px",
        ),
    )


def plots_vix_index() -> rx.Component:
    """VIX index plot."""
    return rx.cond(
        MarketState.loading_vix,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=MarketState.chart_figure_vix,
            width="100%",
            height="400px",
        ),
    )


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
        rx.grid(
            rx.card(plots_fear_and_greed_index()),
            rx.card(plots_buffet_indicator()),
            rx.card(plots_yield_curve()),
            rx.card(plots_vix_index()),
            columns="2",
            spacing="3",
            width="100%",
        ),
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
