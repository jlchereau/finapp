# pylint: disable=too-many-lines
"""
Compare page - Stock comparison and analysis
"""

from datetime import datetime
from random import uniform
from typing import List, Optional

import pandas as pd
import reflex as rx
import plotly.graph_objects as go

from app.components.combobox import combobox_wrapper as combobox
from app.flows.compare import (
    fetch_returns_data,
    fetch_volatility_data,
    fetch_volume_data,
    fetch_rsi_data,
)
from app.lib.exceptions import PageOutputException
from app.lib.charts import (
    ChartConfig,
    TimeSeriesChartConfig,
    ThresholdLine,
    MARKET_COLORS,
    create_comparison_chart,
    add_threshold_lines,
    get_default_theme_colors,
)
from app.lib.metrics import (
    show_metric_as_badge,
    integer_formatter,
    large_currency_formatter,
    # show_metric_as_gauge,  # not yet implemented
)
from app.lib.periods import (
    get_period_default,
    get_period_options,
    calculate_base_date,
    get_max_fallback_date,
    format_date_range_message,
)
from app.templates.template import template


# pylint: disable=inherit-non-class,too-many-instance-attributes
class CompareState(rx.State):
    """State management for the compare page."""

    # Currency selection
    currency: rx.Field[str] = rx.field("USD")
    currencies: rx.Field[List[str]] = rx.field(
        default_factory=lambda: ["USD", "EUR", "GBP"]
    )

    # Ticker input and selection
    ticker_input: rx.Field[str] = rx.field("")
    # ticker_input2: str | None = None
    selected_tickers: rx.Field[List[str]] = rx.field(default_factory=list)
    favorites: rx.Field[List[str]] = rx.field(
        default_factory=lambda: [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
        ]
    )

    # Chart settings
    active_tab: rx.Field[str] = rx.field("plots")
    period_option: rx.Field[str] = rx.field(default_factory=get_period_default)
    period_options: rx.Field[List[str]] = rx.field(default_factory=get_period_options)

    # Chart data
    chart_figure_returns: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)
    chart_figure_volatility: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)
    chart_figure_volume: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)
    chart_figure_rsi: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    # Loading states
    loading_returns: rx.Field[bool] = rx.field(False)
    loading_volatility: rx.Field[bool] = rx.field(False)
    loading_volume: rx.Field[bool] = rx.field(False)
    loading_rsi: rx.Field[bool] = rx.field(False)

    # get_theme_colors() method moved to app.lib.charts.get_default_theme_colors()

    async def get_returns_data(
        self, tickers: List[str], base_date: datetime
    ) -> pd.DataFrame:
        """Get normalized returns data for tickers from base_date using workflow."""
        if not tickers:
            return pd.DataFrame()

        # Use the wrapper function for graceful error handling
        result = await fetch_returns_data(tickers=tickers, base_date=base_date)

        # Extract the normalized DataFrame
        normalized_data = result.get("data")

        if normalized_data is None or normalized_data.empty:
            raise PageOutputException(
                output_type="returns data",
                message=f"No normalized returns data returned for tickers: {tickers}",
                user_message=(
                    "Unable to fetch returns data. Please check the selected tickers "
                    "and try again."
                ),
                context={"tickers": tickers, "base_date": str(base_date)},
            )

        return normalized_data

    async def get_volatility_data(
        self, tickers: List[str], base_date: datetime
    ) -> pd.DataFrame:
        """Get volatility data for tickers from base_date using workflow."""
        if not tickers:
            return pd.DataFrame()

        result = await fetch_volatility_data(tickers=tickers, base_date=base_date)

        volatility_data = result.get("data")

        if volatility_data is None or volatility_data.empty:
            raise PageOutputException(
                output_type="volatility data",
                message=f"No volatility data returned for tickers: {tickers}",
                user_message=(
                    "Unable to fetch volatility data. Please check the selected "
                    "tickers and try again."
                ),
                context={"tickers": tickers, "base_date": str(base_date)},
            )

        return volatility_data

    async def get_volume_data(
        self, tickers: List[str], base_date: datetime
    ) -> pd.DataFrame:
        """Get volume data for tickers from base_date using workflow."""
        if not tickers:
            return pd.DataFrame()

        result = await fetch_volume_data(tickers=tickers, base_date=base_date)

        volume_data = result.get("data")

        if volume_data is None or volume_data.empty:
            raise PageOutputException(
                output_type="volume data",
                message=f"No volume data returned for tickers: {tickers}",
                user_message=(
                    "Unable to fetch volume data. Please check the selected tickers "
                    "and try again."
                ),
                context={"tickers": tickers, "base_date": str(base_date)},
            )

        return volume_data

    async def get_rsi_data(
        self, tickers: List[str], base_date: datetime
    ) -> pd.DataFrame:
        """Get RSI data for tickers from base_date using workflow."""
        if not tickers:
            return pd.DataFrame()

        result = await fetch_rsi_data(tickers=tickers, base_date=base_date)

        rsi_data = result.get("data")

        if rsi_data is None or rsi_data.empty:
            raise PageOutputException(
                output_type="RSI data",
                message=f"No RSI data returned for tickers: {tickers}",
                user_message=(
                    "Unable to fetch RSI data. Please check the selected tickers "
                    "and try again."
                ),
                context={"tickers": tickers, "base_date": str(base_date)},
            )

        return rsi_data

    def set_currency(self, value: str):
        """Set the currency for the comparison."""
        self.currency = value

    def add_ticker(self):
        """Add ticker to selected list."""
        if not self.ticker_input:
            yield rx.toast.warning("Please enter a ticker symbol")
            return

        ticker = self.ticker_input.upper()
        if ticker in self.selected_tickers:
            yield rx.toast.warning(f"{ticker} is already in the comparison")
            return

        self.selected_tickers.append(ticker)
        self.ticker_input = ""
        yield rx.toast.info(f"Added {ticker} to comparison")
        yield CompareState.update_returns_chart
        yield CompareState.update_volatility_chart
        yield CompareState.update_volume_chart
        yield CompareState.update_rsi_chart

    def remove_ticker(self, ticker: str):
        """Remove ticker from selected list."""
        if ticker in self.selected_tickers:
            self.selected_tickers.remove(ticker)
            yield rx.toast.info(f"Removed {ticker} from comparison")
            yield CompareState.update_returns_chart
            yield CompareState.update_volatility_chart
            yield CompareState.update_volume_chart
            yield CompareState.update_rsi_chart
        else:
            yield rx.toast.warning(f"{ticker} not found in comparison list")

    def set_ticker_input(self, value: str | None):
        """Set ticker input value."""
        self.ticker_input = value or ""

    def toggle_favorite(self, ticker: str):
        """Toggle ticker in favorites list."""
        if ticker in self.favorites:
            self.favorites.remove(ticker)
        else:
            self.favorites.append(ticker)

    def set_active_tab(self, tab: str):
        """Switch between metrics and plot tabs."""
        self.active_tab = tab

    def set_period_option(self, option: str):
        """Set base date option and update all charts."""
        self.period_option = option
        yield rx.toast.info(f"Changed time period to {option}")
        yield CompareState.update_returns_chart
        yield CompareState.update_volatility_chart
        yield CompareState.update_volume_chart
        yield CompareState.update_rsi_chart

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
    async def update_returns_chart(self):
        """Update the returns chart using background processing."""
        if not self.selected_tickers:
            async with self:
                self.chart_figure_returns = go.Figure()
            return

        async with self:
            self.loading_returns = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                # For MAX option, use appropriate fallback date
                base_date = get_max_fallback_date("stocks")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            async with self:
                message = format_date_range_message(
                    self.period_option,
                    base_date if self.period_option != "MAX" else None,
                )
                yield rx.toast.info(message)

            # Get returns data
            returns_data = await self.get_returns_data(self.selected_tickers, base_date)

            if returns_data is None or returns_data.empty:
                async with self:
                    self.chart_figure_returns = go.Figure()
                    yield rx.toast.warning(
                        "No data available for selected tickers and date range"
                    )
                return

            # Log successful data fetch
            async with self:
                yield rx.toast.success(
                    f"Loaded data for {', '.join(returns_data.columns)}"
                )
                yield rx.toast.info(
                    f"Data: {returns_data.shape[0]} days, "
                    f"{returns_data.shape[1]} tickers"
                )

            # Create chart using reusable function
            config = ChartConfig(
                title="Returns",
                yaxis_title="Return (%)",
                hover_format="Return: %{y:.2f}%<br>",
            )
            theme_colors = get_default_theme_colors()
            fig = create_comparison_chart(returns_data, config, theme_colors)

            async with self:
                self.chart_figure_returns = fig
                yield rx.toast.success(
                    f"Returns chart updated with {len(returns_data.columns)} tickers"
                )

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="returns chart",
                message=f"Failed to generate returns chart: {e}",
                user_message=(
                    "Failed to generate returns chart. Please try refreshing the data."
                ),
                context={"tickers": self.selected_tickers, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_returns = False

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
    async def update_volatility_chart(self):
        """Update the volatility chart using background processing."""
        if not self.selected_tickers:
            async with self:
                self.chart_figure_volatility = go.Figure()
            return

        async with self:
            self.loading_volatility = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                base_date = get_max_fallback_date("stocks")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            # Get volatility data
            volatility_data = await self.get_volatility_data(
                self.selected_tickers, base_date
            )

            if volatility_data is None or volatility_data.empty:
                async with self:
                    self.chart_figure_volatility = go.Figure()
                return

            # Create chart using reusable function
            config = ChartConfig(
                title="Volatility",
                yaxis_title="Volatility (%)",
                hover_format="Volatility: %{y:.2f}%<br>",
            )
            theme_colors = get_default_theme_colors()
            fig = create_comparison_chart(volatility_data, config, theme_colors)

            async with self:
                self.chart_figure_volatility = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="volatility chart",
                message=f"Failed to generate volatility chart: {e}",
                user_message=(
                    "Failed to generate volatility chart. Please try refreshing "
                    "the data."
                ),
                context={"tickers": self.selected_tickers, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_volatility = False

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
    async def update_volume_chart(self):
        """Update the volume chart using background processing."""
        if not self.selected_tickers:
            async with self:
                self.chart_figure_volume = go.Figure()
            return

        async with self:
            self.loading_volume = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                base_date = get_max_fallback_date("stocks")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            # Get volume data
            volume_data = await self.get_volume_data(self.selected_tickers, base_date)

            if volume_data is None or volume_data.empty:
                async with self:
                    self.chart_figure_volume = go.Figure()
                return

            # Create chart using reusable function
            config = ChartConfig(
                title="Volume",
                yaxis_title="Volume",
                hover_format="Volume: %{y:,.0f}<br>",
            )
            theme_colors = get_default_theme_colors()
            fig = create_comparison_chart(volume_data, config, theme_colors)

            async with self:
                self.chart_figure_volume = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="volume chart",
                message=f"Failed to generate volume chart: {e}",
                user_message=(
                    "Failed to generate volume chart. Please try refreshing the data."
                ),
                context={"tickers": self.selected_tickers, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_volume = False

    # pylint: disable=not-callable
    # pyrefly: ignore[not-callable]
    @rx.event(background=True)
    async def update_rsi_chart(self):
        """Update the RSI chart using background processing."""
        if not self.selected_tickers:
            async with self:
                self.chart_figure_rsi = go.Figure()
            return

        async with self:
            self.loading_rsi = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                base_date = get_max_fallback_date("stocks")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            # Get RSI data
            rsi_data = await self.get_rsi_data(self.selected_tickers, base_date)

            if rsi_data is None or rsi_data.empty:
                async with self:
                    self.chart_figure_rsi = go.Figure()
                return

            # Create chart using reusable function with RSI-specific configuration
            config = TimeSeriesChartConfig(
                title="RSI",
                yaxis_title="RSI",
                hover_format="RSI: %{y:.1f}<br>",
                height=300,  # Comparison charts are shorter
                yaxis_range=[0, 100],  # RSI is bounded 0-100
            )
            theme_colors = get_default_theme_colors()
            fig = create_comparison_chart(rsi_data, config, theme_colors)

            # Add RSI threshold lines for overbought/oversold levels
            thresholds = [
                ThresholdLine(
                    value=70,
                    color=MARKET_COLORS["danger"],
                    label="Overbought (70)",
                    position="top right",
                ),
                ThresholdLine(
                    value=30,
                    color=MARKET_COLORS["safe"],
                    label="Oversold (30)",
                    position="bottom right",
                ),
            ]
            add_threshold_lines(fig, thresholds)

            async with self:
                self.chart_figure_rsi = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="RSI chart",
                message=f"Failed to generate RSI chart: {e}",
                user_message=(
                    "Failed to generate RSI chart. Please try refreshing the data."
                ),
                context={"tickers": self.selected_tickers, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_rsi = False

    def update_all_charts(self):
        """Update all charts."""
        yield CompareState.update_returns_chart
        yield CompareState.update_volatility_chart
        yield CompareState.update_volume_chart
        yield CompareState.update_rsi_chart

    def _get_base_date(self) -> Optional[str]:
        """Convert base date option to actual date string."""
        base_date = calculate_base_date(self.period_option)
        if base_date is None:
            return None
        return base_date.strftime("%Y-%m-%d")


def currency_input_section() -> rx.Component:
    """Currency input section with dropdown."""
    return rx.vstack(
        rx.text("Currency:", font_weight="bold"),
        rx.select(
            CompareState.currencies,
            value=CompareState.currency,
            on_change=CompareState.set_currency,
        ),
        width="100%",
        spacing="2",
    )


def ticker_input_section() -> rx.Component:
    """Ticker input section with favorites dropdown."""
    return rx.vstack(
        rx.text("Ticker:", font_weight="bold"),
        rx.hstack(
            combobox(
                options=CompareState.favorites,
                value=CompareState.ticker_input,
                on_change=CompareState.set_ticker_input,
            ),
            rx.button(
                "+",
                on_click=CompareState.add_ticker,
                size="2",
                variant="solid",
            ),
            spacing="2",
            align="center",
        ),
        width="100%",
        spacing="2",
    )


def ticker_list_section() -> rx.Component:
    """Selected tickers list with prices and remove buttons."""
    return rx.vstack(
        rx.foreach(
            CompareState.selected_tickers,
            ticker_item,
        ),
        width="100%",
        spacing="2",
    )


def ticker_item(ticker: rx.Var[str]) -> rx.Component:
    """Individual ticker item with actions."""
    return rx.hstack(
        rx.vstack(
            rx.text(ticker, font_weight="bold", font_size="sm"),
            rx.text(ticker.to_string() + " Inc", font_size="xs", color="gray"),
            align="start",
            spacing="0",
            flex="1",
        ),
        rx.text("$---.--", font_size="sm", color="gray"),
        rx.hstack(
            rx.button(
                rx.icon("heart", size=16),
                size="2",
                variant="solid",
                # Note using a partial resolves pylint and pyright linting issues
                # but then reflex compile fails
                # pylint: disable=no-value-for-parameter
                # pyrefly: ignore[missing-argument,bad-argument-type]
                on_click=lambda _: CompareState.toggle_favorite(ticker),
            ),
            rx.button(
                rx.icon("minus", size=16),
                size="2",
                variant="solid",
                color_scheme="red",
                # Note using a partial resolves pylint and pyright linting issues
                # but reflex compile then fails
                # pylint: disable=no-value-for-parameter
                # pyrefly: ignore[missing-argument,bad-argument-type]
                on_click=lambda _: CompareState.remove_ticker(ticker),
            ),
            spacing="1",
        ),
        justify="between",
        align="center",
        width="100%",
        padding="8px",
        border="1px solid",
        border_color=rx.color("gray", 6),
        border_radius="md",
    )


def left_sidebar() -> rx.Component:
    """Left sidebar with ticker selection."""
    return rx.vstack(
        currency_input_section(),
        ticker_input_section(),
        ticker_list_section(),
        width="400px",
        padding="1em",
        border_right="1px solid",
        border_color="var(--gray-6)",
        height="100%",
        # height="calc(100vh - 120px)",
        overflow_y="auto",
        spacing="4",
    )


def metrics_asset_valuation() -> rx.Component:
    """Asset valuation section."""
    return rx.table.root(
        rx.table.header(
            # Metrics column + one column per ticker
            rx.table.row(
                rx.table.column_header_cell("Metrics"),
                rx.table.column_header_cell("AMZN"),
                rx.table.column_header_cell("GOOG"),
            ),
        ),
        rx.table.body(
            # One row per metric
            rx.table.row(
                rx.table.row_header_cell("Quote"),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
            ),
            rx.table.row(
                rx.table.row_header_cell("DCF"),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
            ),
            # ---------------------------
            # TODO: Other valuations
            # ---------------------------
            # ---------------------------
            # TODO: Discount to fair value of at least 30%, ideally 50%
            # ---------------------------
        ),
        width="100%",
    )


def metrics_graham_indicators() -> rx.Component:
    """
    Graham indicators section.

    Metrics can be found in Chapter 14 pages 367 and after of
    The Intelligent Investor (4th revised edition) by Benjamin Graham.
    """
    return rx.table.root(
        # Metrics column + one column per ticker
        rx.table.header(
            rx.table.row(
                rx.table.column_header_cell("Metrics"),
                rx.table.column_header_cell("AMZN"),
                rx.table.column_header_cell("GOOG"),
            ),
        ),
        # One row per metric
        rx.table.body(
            rx.table.row(
                rx.table.row_header_cell("Market Capitalization"),
                # According to Graham, larger companies are generally more stable
                # Market cap should be at least $2 billion
                # and preferably over $5 billion
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(1000000000, 150000000000),
                        2000000000.0,
                        5000000000.0,
                        large_currency_formatter,
                    )
                ),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(1000000000, 150000000000),
                        2000000000.0,
                        5000000000.0,
                        large_currency_formatter,
                    )
                ),
            ),
            rx.table.row(
                rx.table.row_header_cell("Annual Revenue"),
                # According to Graham, larger companies are generally more stable
                # Annual revenue should be at least $1 billion
                # and preferably over $3 billion
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(1000000000, 150000000000),
                        1000000000.0,
                        3000000000.0,
                        large_currency_formatter,
                    )
                ),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(1000000000, 150000000000),
                        2000000000.0,
                        5000000000.0,
                        large_currency_formatter,
                    )
                ),
            ),
            rx.table.row(
                rx.table.row_header_cell("Current Ratio"),
                # According to Graham, a current ratio of 2 or higher is considered
                # healthy for a strong financial condition
                # (current assets >= 2 * current liabilities)
                rx.table.cell(show_metric_as_badge(uniform(0, 20), 1.5, 2)),
                rx.table.cell(show_metric_as_badge(uniform(0, 20), 1.5, 2)),
            ),
            # ---------------------------
            # TODO: Long term debt does not exceed working capital (p371)
            # (or net current assets but is this the same?)
            # ---------------------------
            rx.table.row(
                rx.table.row_header_cell("Years Positive Earnings"),
                # According to Graham, positive earnings for the past 10 years are
                # required to demonstrate earnings stability
                rx.table.cell(
                    show_metric_as_badge(uniform(0, 30), 10, 15, integer_formatter)
                ),
                rx.table.cell(
                    show_metric_as_badge(uniform(0, 30), 10, 15, integer_formatter)
                ),
            ),
            # ---------------------------
            # TODO: Years dividend paid (ideally 20 years)
            # Years dividend growth
            # ---------------------------
            # ---------------------------
            # TODO: EPS growth of 50% over 10 years (or 4% per year compounded)
            # ---------------------------
            # ---------------------------
            # TODO: Moderate PE ratio: use current price / average earnings
            # over the past 3 years <= 15
            # ---------------------------
            # ---------------------------
            # TODO: Moderate PB ratio (= price to assets?): less than 1.5 (best)
            # no more than 2.5 (worst)
            # ---------------------------
            # ---------------------------
            # TODO: Make sure PE * PB <= 22.5
            # ---------------------------
        ),
        width="100%",
    )


def metrics_analyst_ratings() -> rx.Component:
    """Analyst ratings section."""
    return rx.table.root(
        # Metrics column + one column per ticker
        rx.table.header(
            rx.table.row(
                rx.table.column_header_cell("Metrics"),
                rx.table.column_header_cell("AMZN"),
                rx.table.column_header_cell("GOOG"),
            ),
        ),
        # One row per metric
        rx.table.body(
            rx.table.row(
                rx.table.row_header_cell("Tipranks"),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
            ),
            rx.table.row(
                rx.table.row_header_cell("Zacks"),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
                rx.table.cell(
                    show_metric_as_badge(
                        uniform(100, 150),
                        uniform(80, 120),
                        uniform(130, 160),
                    )
                ),
            ),
        ),
        width="100%",
    )


def metrics_tab_content() -> rx.Component:
    """Metrics tab showing various financial metrics."""
    return rx.box(
        rx.section(
            rx.heading("Valuation", size="5", margin_bottom="1rem"),
            rx.divider(),
            metrics_asset_valuation(),
            padding_left="1rem",
            padding_right="1rem",
        ),
        rx.section(
            rx.heading("Graham Indicators", size="5", margin_bottom="1rem"),
            rx.divider(),
            metrics_graham_indicators(),
            padding_left="1rem",
            padding_right="1rem",
        ),
        rx.section(
            rx.heading("Analyst Ratings", size="5", margin_bottom="1rem"),
            rx.divider(),
            metrics_analyst_ratings(),
            padding_left="1rem",
            padding_right="1rem",
        ),
        width="100%",
    )


def plots_asset_returns() -> rx.Component:
    """Plotly chart for asset returns."""
    return rx.cond(
        CompareState.loading_returns,
        rx.center(rx.spinner(), height="300px"),
        rx.cond(
            CompareState.selected_tickers.length() > 0,
            rx.plotly(
                data=CompareState.chart_figure_returns,
                width="100%",
                height="300px",
            ),
            rx.center(
                rx.text("Select tickers...", color="gray"),
                height="300px",
            ),
        ),
    )


def plots_asset_volumes() -> rx.Component:
    """Plotly chart for asset volumes."""
    return rx.cond(
        CompareState.loading_volume,
        rx.center(rx.spinner(), height="300px"),
        rx.cond(
            CompareState.selected_tickers.length() > 0,
            rx.plotly(
                data=CompareState.chart_figure_volume,
                width="100%",
                height="300px",
            ),
            rx.center(
                rx.text("Select tickers...", color="gray"),
                height="300px",
            ),
        ),
    )


def plots_asset_relative_strength() -> rx.Component:
    """Plotly chart for asset relative strength (RSI)."""
    return rx.cond(
        CompareState.loading_rsi,
        rx.center(rx.spinner(), height="300px"),
        rx.cond(
            CompareState.selected_tickers.length() > 0,
            rx.plotly(
                data=CompareState.chart_figure_rsi,
                width="100%",
                height="300px",
            ),
            rx.center(
                rx.text("Select tickers...", color="gray"),
                height="300px",
            ),
        ),
    )


def plots_asset_volatility() -> rx.Component:
    """Plotly chart for asset volatility."""
    return rx.cond(
        CompareState.loading_volatility,
        rx.center(rx.spinner(), height="300px"),
        rx.cond(
            CompareState.selected_tickers.length() > 0,
            rx.plotly(
                data=CompareState.chart_figure_volatility,
                width="100%",
                height="300px",
            ),
            rx.center(
                rx.text("Select tickers...", color="gray"),
                height="300px",
            ),
        ),
    )


def plots_tab_content() -> rx.Component:
    """Plot tab showing several asset comparison charts."""
    return rx.vstack(
        rx.hstack(
            rx.text("Period:", font_weight="bold"),
            rx.select(
                CompareState.period_options,
                value=CompareState.period_option,
                on_change=CompareState.set_period_option,
            ),
            spacing="2",
            align="center",
            justify="start",
        ),
        rx.grid(
            rx.card(
                plots_asset_returns(),
            ),
            rx.card(plots_asset_relative_strength()),
            rx.card(
                plots_asset_volatility(),
            ),
            rx.card(
                plots_asset_volumes(),
            ),
            columns="2",
            spacing="4",
            width="100%",
        ),
        padding="1rem",
        spacing="4",
        width="100%",
    )


def main_content() -> rx.Component:
    """Main content area with tabs."""
    return rx.vstack(
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("Plots", value="plots"),
                rx.tabs.trigger("Metrics", value="metrics"),
            ),
            rx.tabs.content(
                plots_tab_content(),
                value="plots",
            ),
            rx.tabs.content(
                metrics_tab_content(),
                value="metrics",
            ),
            value=CompareState.active_tab,
            on_change=CompareState.set_active_tab,
            width="100%",
        ),
        flex="1",
        padding="1rem",
        width="100%",
    )


# pylint: disable=not-callable
# pyright: ignore[reportArgumentType]
# pyrefly: ignore[not-callable,bad-argument-type]
@rx.page(route="/compare", on_load=CompareState.update_all_charts)
@template
def page():
    """The compare page."""
    return rx.vstack(
        rx.heading("Compare", size="6", margin_bottom="1rem"),
        rx.hstack(
            left_sidebar(),
            main_content(),
            border="1px solid",
            border_color="var(--gray-6)",
            border_radius="10px",
            # height="100%",
            width="100%",
            spacing="0",
        ),
        # height="100%",
        spacing="0",
    )
