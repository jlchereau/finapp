"""
Compare page - Stock comparison and analysis
"""

from typing import List, Optional
from datetime import datetime

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
from app.lib.exceptions import DataProcessingException, ChartException
from app.lib.charts import (
    ChartConfig,
    RSIChartConfig,
    create_comparison_chart,
)
from app.lib.metrics import (
    show_metrics_as_badge,
)
from app.lib.periods import (
    get_period_options,
    calculate_base_date,
    get_max_fallback_date,
    format_date_range_message,
)
from app.templates.template import template


class CompareState(rx.State):  # pylint: disable=inherit-non-class
    """State management for the compare page."""

    # Ticker input and selection
    ticker_input: str = ""
    # ticker_input2: str | None = None
    selected_tickers: List[str] = []
    favorites: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]

    # Chart settings
    active_tab: str = "plots"
    base_date_option: str = "1Y"
    base_date_options: List[str] = get_period_options()

    # Chart data
    chart_figure_returns: go.Figure = go.Figure()
    chart_figure_volatility: go.Figure = go.Figure()
    chart_figure_volume: go.Figure = go.Figure()
    chart_figure_rsi: go.Figure = go.Figure()

    # Loading states
    loading_returns: bool = False
    loading_volatility: bool = False
    loading_volume: bool = False
    loading_rsi: bool = False

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

    async def get_returns_data(
        self, tickers: List[str], base_date: datetime
    ) -> pd.DataFrame:
        """Get normalized returns data for tickers from base_date using workflow."""
        if not tickers:
            return pd.DataFrame()

        # Use the compare workflow to fetch and normalize data
        result = await fetch_returns_data(tickers, base_date)

        # Extract the normalized DataFrame
        normalized_data = result.get("data")

        if normalized_data is None or normalized_data.empty:
            raise DataProcessingException(
                operation="normalize_returns_data",
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

        result = await fetch_volatility_data(tickers, base_date)
        volatility_data = result.get("data")

        if volatility_data is None or volatility_data.empty:
            raise DataProcessingException(
                operation="fetch_volatility_data",
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

        result = await fetch_volume_data(tickers, base_date)
        volume_data = result.get("data")

        if volume_data is None or volume_data.empty:
            raise DataProcessingException(
                operation="fetch_volume_data",
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

        result = await fetch_rsi_data(tickers, base_date)
        rsi_data = result.get("data")

        if rsi_data is None or rsi_data.empty:
            raise DataProcessingException(
                operation="fetch_rsi_data",
                message=f"No RSI data returned for tickers: {tickers}",
                user_message=(
                    "Unable to fetch RSI data. Please check the selected tickers "
                    "and try again."
                ),
                context={"tickers": tickers, "base_date": str(base_date)},
            )

        return rsi_data

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

    def set_base_date(self, option: str):
        """Set base date option and update all charts."""
        self.base_date_option = option
        yield rx.toast.info(f"Changed time period to {option}")
        yield CompareState.update_returns_chart
        yield CompareState.update_volatility_chart
        yield CompareState.update_volume_chart
        yield CompareState.update_rsi_chart

    @rx.event(background=True)  # pylint: disable=not-callable
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
                    self.base_date_option,
                    base_date if self.base_date_option != "MAX" else None,
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
            theme_colors = self.get_theme_colors()
            fig = create_comparison_chart(returns_data, config, theme_colors)

            async with self:
                self.chart_figure_returns = fig
                yield rx.toast.success(
                    f"Returns chart updated with {len(returns_data.columns)} tickers"
                )

        except Exception as e:
            # Chart generation error - wrap in ChartException
            raise ChartException(
                chart_type="returns",
                message=f"Failed to generate returns chart: {e}",
                user_message=(
                    "Failed to generate returns chart. Please try refreshing the data."
                ),
                context={"tickers": self.selected_tickers, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_returns = False

    @rx.event(background=True)  # pylint: disable=not-callable
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
            theme_colors = self.get_theme_colors()
            fig = create_comparison_chart(volatility_data, config, theme_colors)

            async with self:
                self.chart_figure_volatility = fig

        except Exception as e:
            # Chart generation error - wrap in ChartException
            raise ChartException(
                chart_type="volatility",
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

    @rx.event(background=True)  # pylint: disable=not-callable
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
            theme_colors = self.get_theme_colors()
            fig = create_comparison_chart(volume_data, config, theme_colors)

            async with self:
                self.chart_figure_volume = fig

        except Exception as e:
            # Chart generation error - wrap in ChartException
            raise ChartException(
                chart_type="volume",
                message=f"Failed to generate volume chart: {e}",
                user_message=(
                    "Failed to generate volume chart. Please try refreshing the data."
                ),
                context={"tickers": self.selected_tickers, "error": str(e)},
            ) from e
        finally:
            async with self:
                self.loading_volume = False

    @rx.event(background=True)  # pylint: disable=not-callable
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

            # Create chart using reusable function
            config = RSIChartConfig()
            theme_colors = self.get_theme_colors()
            fig = create_comparison_chart(rsi_data, config, theme_colors)

            async with self:
                self.chart_figure_rsi = fig

        except Exception as e:
            # Chart generation error - wrap in ChartException
            raise ChartException(
                chart_type="rsi",
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
        base_date = calculate_base_date(self.base_date_option)
        if base_date is None:
            return None
        return base_date.strftime("%Y-%m-%d")


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
            rx.text(ticker + " Inc", font_size="xs", color="gray"),
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
                # pylint: disable=no-value-for-parameter
                on_click=CompareState.toggle_favorite(ticker),
            ),
            rx.button(
                rx.icon("minus", size=16),
                size="2",
                variant="solid",
                color_scheme="red",
                # pylint: disable=no-value-for-parameter
                on_click=CompareState.remove_ticker(ticker),
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


def metrics_valuation() -> rx.Component:
    """Metrics valuation section."""
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.table.column_header_cell("Metrics"),
                # one column per ticker, on line per metric
                rx.table.column_header_cell("AMZN"),
                rx.table.column_header_cell("GOOG"),
            ),
        ),
        rx.table.body(
            rx.table.row(
                rx.table.row_header_cell("Quote"),
                rx.table.cell(show_metrics_as_badge(154.5, 125.2, 145.3)),
                rx.table.cell(show_metrics_as_badge(2729.3, 3500.1, 4600.5)),
            ),
            rx.table.row(
                rx.table.row_header_cell("DCF"),
                rx.table.cell(show_metrics_as_badge(154.5, 125.2, 145.3)),
                rx.table.cell(show_metrics_as_badge(2729.3, 3500.1, 4600.5)),
            ),
        ),
        width="100%",
    )


def metrics_graham() -> rx.Component:
    return rx.text("Graham metrics - coming soon...", color="gray")


def metrics_tab_content() -> rx.Component:
    """Metrics tab - cleared for simplification."""
    return rx.box(
        rx.section(
            rx.heading("Valuation", size="5"),
            rx.divider(),
            metrics_valuation(),
            padding_left="1rem",
            padding_right="1rem",
        ),
        rx.section(
            rx.heading("Graham", size="5"),
            rx.divider(),
            metrics_graham(),
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
            CompareState.selected_tickers.length() > 0,  # pylint: disable=no-member
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
            CompareState.selected_tickers.length() > 0,  # pylint: disable=no-member
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
            CompareState.selected_tickers.length() > 0,  # pylint: disable=no-member
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
            CompareState.selected_tickers.length() > 0,  # pylint: disable=no-member
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
                CompareState.base_date_options,
                value=CompareState.base_date_option,
                on_change=CompareState.set_base_date,
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
@rx.page(
    route="/compare",
    on_load=CompareState.update_all_charts,  # pyright: ignore[reportArgumentType]
)
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
