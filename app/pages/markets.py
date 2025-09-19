# pylint: disable=too-many-lines
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
    create_comparison_chart,
)
from app.flows.markets.buffet import fetch_buffet_indicator_data
from app.flows.markets.vix import fetch_vix_data
from app.flows.markets.yield_curve import fetch_yield_curve_data
from app.lib.logger import logger
from app.flows.markets import (
    fetch_currency_data,
    fetch_precious_metals_data,
    fetch_crypto_data,
    fetch_crude_oil_data,
    fetch_bloomberg_commodity_data,
    fetch_msci_world_data,
)
from app.lib.exceptions import PageOutputException
from app.lib.periods import (
    get_period_default,
    get_period_options,
    calculate_base_date,
    get_max_fallback_date,
    format_date_range_message,
)
from app.templates.template import template


# pylint: disable=inherit-non-class,too-many-instance-attributes
class MarketState(rx.State):
    """The app state."""

    active_tab: rx.Field[str] = rx.field("overview")

    # Chart settings
    period_option: rx.Field[str] = rx.field(default_factory=get_period_default)
    period_options: rx.Field[List[str]] = rx.field(default_factory=get_period_options)

    # Chart data
    chart_figure_buffet: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)
    chart_figure_vix: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)
    chart_figure_yield: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)
    chart_figure_currency: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)
    chart_figure_precious_metals: rx.Field[go.Figure] = rx.field(
        default_factory=go.Figure
    )
    chart_figure_crypto: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)
    chart_figure_crude_oil: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)
    chart_figure_bloomberg_commodity: rx.Field[go.Figure] = rx.field(
        default_factory=go.Figure
    )
    chart_figure_msci_world: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    # Loading state
    loading_buffet: rx.Field[bool] = rx.field(False)
    loading_vix: rx.Field[bool] = rx.field(False)
    loading_yield: rx.Field[bool] = rx.field(False)
    loading_currency: rx.Field[bool] = rx.field(False)
    loading_precious_metals: rx.Field[bool] = rx.field(False)
    loading_crypto: rx.Field[bool] = rx.field(False)
    loading_crude_oil: rx.Field[bool] = rx.field(False)
    loading_bloomberg_commodity: rx.Field[bool] = rx.field(False)
    loading_msci_world: rx.Field[bool] = rx.field(False)

    def set_active_tab(self, tab: str):
        """Switch between metrics and plot tabs."""
        self.active_tab = tab

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    def set_period_option(self, option: str):
        """Set base date option and update all market charts."""
        self.period_option = option
        yield rx.toast.info(f"Changed time period to {option}")
        yield from self.update_all_charts()

    def update_all_charts(self):
        """Update all market charts."""
        yield MarketState.update_buffet_chart
        yield MarketState.update_vix_chart
        yield MarketState.update_yield_chart
        yield MarketState.update_currency_chart
        yield MarketState.update_precious_metals_chart
        yield MarketState.update_crypto_chart
        yield MarketState.update_crude_oil_chart
        yield MarketState.update_bloomberg_commodity_chart
        yield MarketState.update_msci_world_chart

    async def get_buffet_data(
        self, base_date: datetime
    ) -> tuple[pd.DataFrame, dict | None, dict]:
        """Get Buffet Indicator data using workflow."""
        # Use the markets workflow to fetch and calculate data
        result = await fetch_buffet_indicator_data(base_date, self.period_option)

        # Extract the DataFrame and trend data
        buffet_data = result.get("data")
        trend_data = result.get("trend_data")
        data_points = result.get("data_points", 0)

        if buffet_data is None or buffet_data.empty:
            raise PageOutputException(
                output_type="Buffet Indicator data",
                message=f"No Buffet Indicator data returned for base_date: {base_date}",
                user_message=(
                    "Unable to fetch Buffet Indicator data. Please try a different "
                    "time period."
                ),
                context={
                    "base_date": str(base_date),
                    "period_option": self.period_option,
                },
            )

        return buffet_data, trend_data, data_points

    async def get_vix_data(self, base_date: datetime) -> pd.DataFrame:
        """Get VIX data using workflow."""
        # Use the markets workflow to fetch VIX data
        result = await fetch_vix_data(base_date)

        # Extract the DataFrame
        vix_data = result.get("data")

        if vix_data is None or vix_data.empty:
            raise PageOutputException(
                output_type="VIX data",
                message=f"No VIX data returned for base_date: {base_date}",
                user_message=(
                    "Unable to fetch VIX data. Please try a different time period."
                ),
                context={
                    "base_date": str(base_date),
                    "period_option": self.period_option,
                },
            )

        return vix_data

    async def get_currency_data(self, base_date: datetime) -> pd.DataFrame:
        """Get currency data using workflow."""
        # Use the markets workflow to fetch currency data
        result = await fetch_currency_data(base_date)

        # Extract the DataFrame
        currency_data = result.get("data")

        if currency_data is None or currency_data.empty:
            raise PageOutputException(
                output_type="currency data",
                message=f"No currency data returned for base_date: {base_date}",
                user_message=(
                    "Unable to fetch currency data. Please try a different time period."
                ),
                context={
                    "base_date": str(base_date),
                    "period_option": self.period_option,
                },
            )

        return currency_data

    def _get_base_date(self) -> Optional[str]:
        """Convert base date option to actual date string."""
        base_date = calculate_base_date(self.period_option)
        if base_date is None:
            return None
        return base_date.strftime("%Y-%m-%d")

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
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
                    self.period_option,
                    base_date if self.period_option != "MAX" else None,
                )
                yield rx.toast.info(message)
            # Get Buffet Indicator data
            buffet_data, trend_data, data_points = await self.get_buffet_data(base_date)

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
                    f"Loaded Buffet Indicator data: {data_points} data points"
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
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="Buffet Indicator chart",
                message=f"Failed to generate Buffet Indicator chart: {e}",
                user_message=(
                    "Failed to generate Buffet Indicator chart. Please try "
                    "refreshing the data."
                ),
                context={"period_option": self.period_option, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_buffet = False

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
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
                    self.period_option,
                    base_date if self.period_option != "MAX" else None,
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
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="VIX chart",
                message=f"Failed to generate VIX chart: {e}",
                user_message=(
                    "Failed to generate VIX chart. Please try refreshing the data."
                ),
                context={"period_option": self.period_option, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_vix = False

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
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
                    self.period_option,
                    base_date if self.period_option != "MAX" else None,
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

                    # Find closest date with more flexible approach
                    # First try exact target date or earlier
                    closest_dates = historical_dates[historical_dates <= target_date]

                    # If no dates found (edge case), try a bit more flexibility
                    if len(closest_dates) == 0:
                        # Look within ±1 week of target for edge cases
                        tolerance_start = target_date - pd.DateOffset(days=7)
                        tolerance_end = target_date + pd.DateOffset(days=7)
                        closest_dates = historical_dates[
                            (historical_dates >= tolerance_start)
                            & (historical_dates <= tolerance_end)
                        ]

                    # Debug logging to understand missing curves
                    logger.debug(
                        f"Yield curve {name}: target_date={target_date}, "
                        f"available_dates={len(closest_dates)}, "
                        f"latest_date={latest_date}, "
                        f"data_range=({yield_data.index.min()}, "
                        f"{yield_data.index.max()})"
                    )

                    if len(closest_dates) > 0:
                        hist_date = closest_dates[
                            0
                        ]  # Take the most recent within range
                        hist_data = yield_data.loc[yield_data.index == hist_date]
                        if not hist_data.empty:
                            logger.debug(f"Adding {name} curve from {hist_date}")
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
                        else:
                            logger.warning(
                                f"Empty data for {name} curve at {hist_date}"
                            )
                    else:
                        logger.warning(
                            f"No historical data available for {name} "
                            f"(target: {target_date}, data range: "
                            f"{yield_data.index.min()} to {yield_data.index.max()})"
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
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="yield curve chart",
                message=f"Failed to generate yield curve chart: {e}",
                user_message=(
                    "Failed to generate yield curve chart. "
                    "Please try refreshing the data."
                ),
                context={"period_option": self.period_option, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_yield = False

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
    async def update_currency_chart(self):
        """Update the currency exchange rate chart using background processing."""
        async with self:
            self.loading_currency = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                # For MAX option, use appropriate fallback date
                base_date = get_max_fallback_date("currency")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            async with self:
                message = format_date_range_message(
                    self.period_option,
                    base_date if self.period_option != "MAX" else None,
                )
                yield rx.toast.info(message)

            # Get currency data
            currency_result = await fetch_currency_data(base_date)
            currency_data = currency_result.get("data")

            if currency_data is None or currency_data.empty:
                async with self:
                    self.chart_figure_currency = go.Figure()
                    yield rx.toast.warning(
                        "No currency data available for selected date range"
                    )
                return

            # Log successful data fetch
            async with self:
                yield rx.toast.success(
                    f"Loaded currency data: {currency_data.shape[0]} data points"
                )

            # Create time-series chart using utility functions
            config = TimeSeriesChartConfig(
                title="Currency Exchange Rates",
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

            async with self:
                self.chart_figure_currency = fig
                yield rx.toast.success("Currency chart updated successfully")

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="currency chart",
                message=f"Failed to generate currency chart: {e}",
                user_message=(
                    "Failed to generate currency chart. Please try refreshing the data."
                ),
                context={"period_option": self.period_option, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_currency = False

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
    async def update_precious_metals_chart(self):
        """Update the precious metals (Gold Futures) chart in background."""
        async with self:
            self.loading_precious_metals = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                # For MAX option, use appropriate fallback date
                base_date = get_max_fallback_date("precious_metals")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            async with self:
                message = format_date_range_message(
                    self.period_option,
                    base_date if self.period_option != "MAX" else None,
                )
                yield rx.toast.info(message)

            # Get precious metals data
            precious_metals_result = await fetch_precious_metals_data(base_date)
            precious_metals_data = precious_metals_result.get("data")

            if precious_metals_data is None or precious_metals_data.empty:
                async with self:
                    self.chart_figure_precious_metals = go.Figure()
                    yield rx.toast.warning(
                        "No precious metals data available for selected date range"
                    )
                return

            # Log successful data fetch
            async with self:
                yield rx.toast.success(
                    f"Loaded gold data: {precious_metals_data.shape[0]} data points"
                )

            # Create time-series chart using utility functions
            config = TimeSeriesChartConfig(
                title="Gold Futures (COMEX)",
                yaxis_title="Price (USD/oz)",
                hover_format="Price: %{y:.2f}<br>",
                height=400,
                primary_color=MARKET_COLORS["primary"],
                primary_style="solid",  # Solid line without markers
            )

            # Get theme colors
            theme_colors = self.get_theme_colors()

            # Create empty chart with theme (no main series yet)
            fig = create_timeseries_chart(
                precious_metals_data,
                config,
                theme_colors,
                "Gold",
                include_main_series=False,
            )

            # Add background elements first (moving averages)
            curves = []

            # Add 50-day moving average if available
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

            # Add 200-day moving average if available
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

            # Add moving averages as historical curves
            if curves:
                add_historical_curves(fig, curves)

            # Add main gold data series on top (drawn last, appears on top)
            add_main_series(fig, precious_metals_data, config, "Gold")

            async with self:
                self.chart_figure_precious_metals = fig
                yield rx.toast.success("Gold chart updated successfully")

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="precious metals chart",
                message=f"Failed to generate precious metals chart: {e}",
                user_message=(
                    "Failed to generate precious metals chart. Please try "
                    "refreshing the data."
                ),
                context={"period_option": self.period_option, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_precious_metals = False

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
    async def update_bloomberg_commodity_chart(self):
        """Update the Bloomberg Commodity Index chart in background."""
        async with self:
            self.loading_bloomberg_commodity = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                # For MAX option, use appropriate fallback date
                base_date = get_max_fallback_date("bloomberg_commodity")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            async with self:
                message = format_date_range_message(
                    self.period_option,
                    base_date if self.period_option != "MAX" else None,
                )
                yield rx.toast.info(message)

            # Get Bloomberg Commodity Index data
            bloomberg_result = await fetch_bloomberg_commodity_data(base_date)
            bloomberg_data = bloomberg_result.get("data")

            if bloomberg_data is None or bloomberg_data.empty:
                async with self:
                    self.chart_figure_bloomberg_commodity = go.Figure()
                    yield rx.toast.warning(
                        "No Bloomberg Commodity Index data available for selected "
                        "date range"
                    )
                return

            # Log successful data fetch
            async with self:
                yield rx.toast.success(
                    f"Loaded Bloomberg Commodity Index data: "
                    f"{bloomberg_data.shape[0]} data points"
                )

            # Create time-series chart using utility functions
            config = TimeSeriesChartConfig(
                title="Bloomberg Commodity Index (^BCOM)",
                yaxis_title="Index Value",
                hover_format="Value: %{y:.2f}<br>",
                height=400,
                primary_color=MARKET_COLORS["primary"],
                primary_style="solid",  # Solid line without markers
            )

            # Get theme colors
            theme_colors = self.get_theme_colors()

            # Create empty chart with theme (no main series yet)
            fig = create_timeseries_chart(
                bloomberg_data,
                config,
                theme_colors,
                "BCOM",
                include_main_series=False,
            )

            # Add background elements first (moving averages)
            curves = []

            # Add 50-day moving average if available
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

            # Add 200-day moving average if available
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

            # Add moving averages as historical curves
            if curves:
                add_historical_curves(fig, curves)

            # Add main Bloomberg Commodity Index data series on top
            add_main_series(fig, bloomberg_data, config, "BCOM")

            async with self:
                self.chart_figure_bloomberg_commodity = fig
                yield rx.toast.success(
                    "Bloomberg Commodity Index chart updated successfully"
                )

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="Bloomberg Commodity Index chart",
                message=f"Failed to generate Bloomberg Commodity Index chart: {e}",
                user_message=(
                    "Failed to generate Bloomberg Commodity Index chart. Please try "
                    "refreshing the data."
                ),
                context={"period_option": self.period_option, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_bloomberg_commodity = False

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
    async def update_msci_world_chart(self):
        """Update the MSCI World Index chart using background processing."""
        async with self:
            self.loading_msci_world = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                # For MAX option, use appropriate fallback date
                base_date = get_max_fallback_date("msci_world")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            async with self:
                message = format_date_range_message(
                    self.period_option,
                    base_date if self.period_option != "MAX" else None,
                )
                yield rx.toast.info(message)

            # Get MSCI World data
            msci_result = await fetch_msci_world_data(base_date)
            msci_data = msci_result.get("data")

            if msci_data is None or msci_data.empty:
                async with self:
                    self.chart_figure_msci_world = go.Figure()
                    yield rx.toast.warning(
                        "No MSCI World data available for selected date range"
                    )
                return

            # Log successful data fetch
            async with self:
                yield rx.toast.success(
                    f"Loaded MSCI World data: {msci_data.shape[0]} data points"
                )

            # Create time-series chart using utility functions
            config = TimeSeriesChartConfig(
                title="MSCI World Index",
                yaxis_title="Index Value (USD)",
                hover_format="Value: %{y:.2f}<br>",
                height=400,
                primary_color=MARKET_COLORS["primary"],
                primary_style="solid",  # Solid line without markers
            )

            # Get theme colors
            theme_colors = self.get_theme_colors()

            # Create empty chart with theme (no main series yet)
            fig = create_timeseries_chart(
                msci_data,
                config,
                theme_colors,
                "MSCI_World",
                include_main_series=False,
            )

            # Add background elements first (moving averages and Bollinger bands)
            curves = []

            # Add 50-day moving average if available
            if (
                "MSCI_MA50" in msci_data.columns
                and not msci_data["MSCI_MA50"].isna().all()
            ):
                curves.append(
                    {
                        "x": msci_data.index,
                        "y": msci_data["MSCI_MA50"],
                        "name": "50-Day Moving Average",
                        "color": MARKET_COLORS["warning"],
                        "opacity": 0.8,
                    }
                )

            # Add 200-day moving average if available
            if (
                "MSCI_MA200" in msci_data.columns
                and not msci_data["MSCI_MA200"].isna().all()
            ):
                curves.append(
                    {
                        "x": msci_data.index,
                        "y": msci_data["MSCI_MA200"],
                        "name": "200-Day Moving Average",
                        "color": MARKET_COLORS["safe"],
                        "opacity": 0.8,
                    }
                )

            # Add Bollinger Bands as dashed light-grey lines
            if (
                "MSCI_BB_Upper" in msci_data.columns
                and not msci_data["MSCI_BB_Upper"].isna().all()
            ):
                curves.extend(
                    [
                        {
                            "x": msci_data.index,
                            "y": msci_data["MSCI_BB_Upper"],
                            "name": "Bollinger Upper",
                            "color": "rgba(128,128,128,0.5)",
                            "opacity": 0.6,
                        },
                        {
                            "x": msci_data.index,
                            "y": msci_data["MSCI_BB_Lower"],
                            "name": "Bollinger Lower",
                            "color": "rgba(128,128,128,0.5)",
                            "opacity": 0.6,
                        },
                    ]
                )

            # Add all curves as historical curves
            if curves:
                add_historical_curves(fig, curves)

            # Add main MSCI World data series on top (drawn last, appears on top)
            add_main_series(fig, msci_data, config, "MSCI_World")

            async with self:
                self.chart_figure_msci_world = fig
                yield rx.toast.success("MSCI World chart updated successfully")

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="MSCI World chart",
                message=f"Failed to generate MSCI World chart: {e}",
                user_message=(
                    "Failed to generate MSCI World chart. "
                    "Please try refreshing the data."
                ),
                context={"period_option": self.period_option, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_msci_world = False

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
    async def update_crypto_chart(self):
        """Update the cryptocurrency (Bitcoin and Ethereum) chart in background."""
        async with self:
            self.loading_crypto = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                # For MAX option, use appropriate fallback date
                base_date = get_max_fallback_date("crypto")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            async with self:
                message = format_date_range_message(
                    self.period_option,
                    base_date if self.period_option != "MAX" else None,
                )
                yield rx.toast.info(message)

            # Get cryptocurrency data
            crypto_result = await fetch_crypto_data(base_date)
            crypto_data = crypto_result.get("data")

            if crypto_data is None or crypto_data.empty:
                async with self:
                    self.chart_figure_crypto = go.Figure()
                    yield rx.toast.warning(
                        "No cryptocurrency data available for selected date range"
                    )
                return

            # Log successful data fetch
            async with self:
                yield rx.toast.success(
                    f"Loaded crypto data: {crypto_data.shape[0]} data points"
                )

            # Create comparison chart configuration
            config = ChartConfig(
                title="Cryptocurrency Prices",
                yaxis_title="Price (USD)",
                hover_format="Price: %{y:.2f}<br>",
                height=400,
            )

            # Get theme colors
            theme_colors = self.get_theme_colors()

            # Create comparison chart for Bitcoin vs Ethereum
            fig = create_comparison_chart(crypto_data, config, theme_colors)

            async with self:
                self.chart_figure_crypto = fig
                yield rx.toast.success("Cryptocurrency chart updated successfully")

        except Exception as e:
            async with self:
                self.chart_figure_crypto = go.Figure()
                yield rx.toast.error(f"Failed to update crypto chart: {str(e)}")
            # Re-raise as PageOutputException for better handling
            raise PageOutputException(
                output_type="crypto_chart",
                message=f"Failed to generate cryptocurrency chart: {e}",
                user_message="Failed to load cryptocurrency chart. Please try again.",
                context={"period_option": self.period_option},
            ) from e
        finally:
            async with self:
                self.loading_crypto = False

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
    async def update_crude_oil_chart(self):
        """Update the crude oil (WTI and Brent) chart in background."""
        async with self:
            self.loading_crude_oil = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                # For MAX option, use appropriate fallback date
                base_date = get_max_fallback_date("crude_oil")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            async with self:
                message = format_date_range_message(
                    self.period_option,
                    base_date if self.period_option != "MAX" else None,
                )
                yield rx.toast.info(message)

            # Get crude oil data
            crude_oil_result = await fetch_crude_oil_data(base_date)
            crude_oil_data = crude_oil_result.get("data")

            if crude_oil_data is None or crude_oil_data.empty:
                async with self:
                    self.chart_figure_crude_oil = go.Figure()
                    yield rx.toast.warning(
                        "No crude oil data available for selected date range"
                    )
                return

            # Log successful data fetch
            async with self:
                yield rx.toast.success(
                    f"Loaded crude oil data: {crude_oil_data.shape[0]} data points"
                )

            # Create comparison chart configuration
            config = ChartConfig(
                title="Crude Oil Prices",
                yaxis_title="Price (USD/bbl)",
                hover_format="Price: %{y:.2f}<br>",
                height=400,
            )

            # Get theme colors
            theme_colors = self.get_theme_colors()

            # Create the chart
            fig = create_comparison_chart(crude_oil_data, config, theme_colors)

            async with self:
                self.chart_figure_crude_oil = fig
                yield rx.toast.success("Crude oil chart updated successfully")

        except Exception as e:
            raise PageOutputException(
                output_type="crude_oil_chart",
                message=f"Failed to generate crude oil chart: {e}",
                user_message="Failed to load crude oil chart. Please try again.",
                context={"period_option": self.period_option},
            ) from e
        finally:
            async with self:
                self.loading_crude_oil = False

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
    async def run_workflows(self):
        """Load initial chart data."""
        yield MarketState.update_buffet_chart
        yield MarketState.update_vix_chart
        yield MarketState.update_yield_chart
        yield MarketState.update_currency_chart
        yield MarketState.update_precious_metals_chart
        yield MarketState.update_crypto_chart
        yield MarketState.update_crude_oil_chart
        yield MarketState.update_bloomberg_commodity_chart
        yield MarketState.update_msci_world_chart


def plots_currencies() -> rx.Component:
    """Currency exchange rates plot (USD/EUR and GBP/EUR)."""
    return rx.cond(
        MarketState.loading_currency,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=MarketState.chart_figure_currency,
            width="100%",
            height="400px",
        ),
    )


def plots_msci_world() -> rx.Component:
    """MSCI World Index plot with moving averages and Bollinger bands."""
    return rx.cond(
        MarketState.loading_msci_world,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=MarketState.chart_figure_msci_world,
            width="100%",
            height="400px",
        ),
    )


def tabs_overview() -> rx.Component:
    """Overview tab content."""
    return rx.vstack(
        rx.hstack(
            rx.text("Period:", font_weight="bold"),
            rx.select(
                MarketState.period_options,
                value=MarketState.period_option,
                on_change=MarketState.set_period_option,
            ),
            spacing="2",
            align="center",
            justify="start",
        ),
        rx.hstack(
            rx.grid(
                rx.card(plots_currencies()),
                rx.card(plots_msci_world()),
                rx.card("Card 3"),
                rx.card("Card 4"),
                columns="2",
                spacing="3",
                width="70%",
            ),
            rx.card(rx.text("News feed"), width="30%"),
            width="100%",
        ),
    )


def plots_fear_and_greed_index() -> rx.Component:
    """Fear and Greed index plot."""
    return rx.box(rx.text("Fear and Greed index plot"))


def plots_shiller_cape() -> rx.Component:
    """Shiller CAPE plot."""
    return rx.box(rx.text("Shiller CAPE plot"))


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


def tabs_us() -> rx.Component:
    """US tab content."""
    return rx.vstack(
        rx.hstack(
            rx.text("Period:", font_weight="bold"),
            rx.select(
                MarketState.period_options,
                value=MarketState.period_option,
                on_change=MarketState.set_period_option,
            ),
            spacing="2",
            align="center",
            justify="start",
        ),
        rx.grid(
            rx.card(plots_fear_and_greed_index()),
            rx.card(plots_buffet_indicator()),
            rx.card(plots_shiller_cape()),
            rx.card(plots_yield_curve()),
            rx.card(plots_vix_index()),
            columns="2",
            spacing="3",
            width="100%",
        ),
    )


def tabs_eu() -> rx.Component:
    """Europe tab content."""
    return rx.vstack(
        rx.hstack(
            rx.text("Period:", font_weight="bold"),
            rx.select(
                MarketState.period_options,
                value=MarketState.period_option,
                on_change=MarketState.set_period_option,
            ),
            spacing="2",
            align="center",
            justify="start",
        ),
        rx.grid(
            rx.card("Card 1"),
            rx.card("Card 2"),
            rx.card("Card 3"),
            rx.card("Card 4"),
            columns="2",
            spacing="3",
            width="100%",
        ),
    )


def plots_precious_metals() -> rx.Component:
    """Precious Metals plot (Gold Futures)."""
    return rx.cond(
        MarketState.loading_precious_metals,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=MarketState.chart_figure_precious_metals,
            width="100%",
            height="400px",
        ),
    )


def plots_crypto_currencies() -> rx.Component:
    """Cryptocurrency prices plot (Bitcoin and Ethereum)."""
    return rx.cond(
        MarketState.loading_crypto,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=MarketState.chart_figure_crypto,
            width="100%",
            height="400px",
        ),
    )


def plots_crude_oil() -> rx.Component:
    """Crude oil prices plot (WTI and Brent)."""
    return rx.cond(
        MarketState.loading_crude_oil,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=MarketState.chart_figure_crude_oil,
            width="100%",
            height="400px",
        ),
    )


def plots_commodity_index() -> rx.Component:
    """Bloomberg Commodity Index plot."""
    return rx.cond(
        MarketState.loading_bloomberg_commodity,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=MarketState.chart_figure_bloomberg_commodity,
            width="100%",
            height="400px",
        ),
    )


def tabs_commodities() -> rx.Component:
    """Commodities tab content."""
    return rx.vstack(
        rx.hstack(
            rx.text("Period:", font_weight="bold"),
            rx.select(
                MarketState.period_options,
                value=MarketState.period_option,
                on_change=MarketState.set_period_option,
            ),
            spacing="2",
            align="center",
            justify="start",
        ),
        rx.grid(
            rx.card(plots_precious_metals()),
            rx.card(plots_crypto_currencies()),
            rx.card(plots_crude_oil()),
            rx.card(plots_commodity_index()),
            columns="2",
            spacing="3",
            width="100%",
        ),
    )


# pylint: disable=not-callable
# pyright: ignore[reportArgumentType]
# pyrefly: ignore[not-callable,bad-argument-type]
@rx.page(route="/markets", on_load=MarketState.run_workflows)
@template
def page():
    """The markets page."""
    return rx.vstack(
        rx.heading("Markets", size="6", margin_bottom="1rem"),
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("Overview", value="overview"),
                rx.tabs.trigger("US", value="us"),
                rx.tabs.trigger("Europe", value="eu"),
                rx.tabs.trigger("Commodities", value="commodities"),
                # Emerging markets
            ),
            rx.tabs.content(tabs_overview(), value="overview", padding_top="1rem"),
            rx.tabs.content(tabs_us(), value="us", padding_top="1rem"),
            rx.tabs.content(tabs_eu(), value="eu", padding_top="1rem"),
            rx.tabs.content(
                tabs_commodities(), value="commodities", padding_top="1rem"
            ),
            value=MarketState.active_tab,
            on_change=MarketState.set_active_tab,
            width="100%",
        ),
        # height="100%",
        spacing="0",
    )
