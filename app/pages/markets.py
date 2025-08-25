"""
Markets page
"""

from typing import List, Optional
from datetime import datetime

import pandas as pd
import reflex as rx
import plotly.graph_objects as go

from app.lib.charts import (
    ChartConfig,
    TimeSeriesChartConfig,
    ThresholdLine,
    MARKET_COLORS,
    create_timeseries_chart,
    add_main_series,
    add_threshold_lines,
    add_trend_display,
    add_historical_curves,
    get_default_theme_colors,
)
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
        return get_default_theme_colors()

    def set_base_date(self, option: str):
        """Set base date option and update all market charts."""
        self.base_date_option = option
        yield rx.toast.info(f"Changed time period to {option}")
        yield from self.update_all_charts()

    def update_all_charts(self):
        """Update all market charts."""
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

            # Create time-series chart using utility functions
            config = TimeSeriesChartConfig(
                title="Buffet Indicator (Market Cap / GDP)",
                yaxis_title="Buffet Indicator (%)",
                hover_format="Value: %{y:.1f}<br>",
                height=400,
                primary_color=MARKET_COLORS["primary"],
            )

            # Get theme colors
            theme_colors = self.get_theme_colors()

            # Create empty chart with theme (no main series yet)
            fig = create_timeseries_chart(
                buffet_data,
                config,
                theme_colors,
                "Buffet_Indicator",
                include_main_series=False,
            )

            # Add background elements first (trend lines and thresholds)
            # Add pre-calculated trend data (preserves full dataset calculations)
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

            # Add main data series on top (drawn last, appears on top)
            add_main_series(fig, buffet_data, config, "Buffet_Indicator")

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

            # Create time-series chart using utility functions
            config = TimeSeriesChartConfig(
                title="VIX (CBOE Volatility Index)",
                yaxis_title="VIX Level",
                hover_format="Value: %{y:.2f}<br>",
                height=400,
                primary_color=MARKET_COLORS["primary"],
                primary_style="solid",  # Solid line without markers
            )

            # Get theme colors
            theme_colors = self.get_theme_colors()

            # Create empty chart with theme (no main series yet)
            fig = create_timeseries_chart(
                vix_data, config, theme_colors, "VIX", include_main_series=False
            )

            # Add background elements first (moving averages and thresholds)
            # Add 50-day moving average as historical curve if available
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

            # Add main VIX data series on top (drawn last, appears on top)
            add_main_series(fig, vix_data, config, "VIX")

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

            # Create yield curve chart using chart utilities
            fig = go.Figure()

            # Get theme colors
            theme_colors = self.get_theme_colors()

            # Store main series data for adding later
            main_series_data = None
            if latest_date:
                latest_data = yield_data.loc[yield_data.index == latest_date]
                if not latest_data.empty:
                    main_series_data = {
                        "x": maturities,
                        "y": latest_data.iloc[0].values,
                        "date": latest_date,
                    }

            # Add historical curves with semantic colors (background elements first)
            if len(yield_data) > 1 and latest_date is not None:
                historical_dates = yield_data.index.sort_values(ascending=False)

                curves = []
                # Find dates approximately 6 months and 12 months ago
                for months_back, color, name in [
                    (6, MARKET_COLORS["warning"], "6M ago"),
                    (12, MARKET_COLORS["trend"], "12M ago"),
                ]:
                    target_date = latest_date - pd.DateOffset(months=months_back)
                    closest_date = historical_dates[historical_dates <= target_date]

                    if len(closest_date) > 0:
                        hist_date = closest_date[0]
                        hist_data = yield_data.loc[yield_data.index == hist_date]
                        if not hist_data.empty:
                            curves.append(
                                {
                                    "x": maturities,
                                    "y": hist_data.iloc[0].values,
                                    "name": f"{name} "
                                    f"({hist_date.strftime('%Y-%m-%d')})",
                                    "color": color,
                                    "opacity": 0.7,
                                }
                            )

                add_historical_curves(fig, curves)

            # Add the current yield curve on top (drawn last, appears on top)
            if main_series_data:
                fig.add_trace(
                    go.Scatter(
                        x=main_series_data["x"],
                        y=main_series_data["y"],
                        mode="lines+markers",
                        name=f"Current "
                        f"({main_series_data['date'].strftime('%Y-%m-%d')})",
                        line={"color": MARKET_COLORS["primary"], "width": 3},
                        marker={"size": 6, "color": MARKET_COLORS["primary"]},
                        hovertemplate="<b>Current Yield Curve</b><br>"
                        + "Maturity: %{x}<br>"
                        + "Yield: %{y:.2f}%<br>"
                        + "<extra></extra>",
                    )
                )

            # Apply chart configuration and theme
            config = ChartConfig(
                title="US Treasury Yield Curve",
                yaxis_title="Yield (%)",
                hover_format="Yield: %{y:.2f}%<br>",
                height=400,
            )

            # Apply theme manually since this is a custom chart type
            fig.update_layout(
                title=config.title,
                xaxis_title="Maturity",
                yaxis_title=config.yaxis_title,
                hovermode="x unified",
                showlegend=True,
                height=config.height,
                margin={"l": 50, "r": 50, "t": 80, "b": 50},
                plot_bgcolor=theme_colors["plot_bgcolor"],
                paper_bgcolor=theme_colors["paper_bgcolor"],
                hoverlabel={
                    "bgcolor": theme_colors["hover_bgcolor"],
                    "bordercolor": theme_colors["hover_bordercolor"],
                    "font_size": 14,
                    "font_color": "white",
                },
            )

            # Only add font_color if it's not None
            if theme_colors["text_color"] is not None:
                fig.update_layout(font_color=theme_colors["text_color"])

            # Update axes with theme colors and maturity ordering
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=theme_colors["grid_color"],
                showline=True,
                linewidth=1,
                linecolor=theme_colors["line_color"],
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
