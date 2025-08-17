"""
Compare page - Stock comparison and analysis
"""

from typing import List, Optional
from datetime import datetime, timedelta

import pandas as pd
import reflex as rx
import plotly.graph_objects as go

from ..components.combobox import combobox_wrapper as combobox
from ..flows.compare import (
    fetch_returns_data,
    fetch_volatility_data,
    fetch_volume_data,
    fetch_rsi_data,
)
from ..templates.template import template


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
    base_date_options: List[str] = [
        "1W",
        "2W",
        "1M",
        "2M",
        "1Q",
        "2Q",
        "3Q",
        "1Y",
        "2Y",
        "3Y",
        "4Y",
        "5Y",
        "10Y",
        "YTD",
        "MAX",
    ]

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

        try:
            # Use the compare workflow to fetch and normalize data
            result = await fetch_returns_data(tickers, base_date)

            # Extract the normalized DataFrame
            normalized_data = result.get("data")

            if normalized_data is None or normalized_data.empty:
                return pd.DataFrame()

            return normalized_data

        except Exception:
            return pd.DataFrame()

    async def get_volatility_data(
        self, tickers: List[str], base_date: datetime
    ) -> pd.DataFrame:
        """Get volatility data for tickers from base_date using workflow."""
        if not tickers:
            return pd.DataFrame()

        try:
            result = await fetch_volatility_data(tickers, base_date)
            volatility_data = result.get("data")

            if volatility_data is None or volatility_data.empty:
                return pd.DataFrame()

            return volatility_data

        except Exception:
            return pd.DataFrame()

    async def get_volume_data(
        self, tickers: List[str], base_date: datetime
    ) -> pd.DataFrame:
        """Get volume data for tickers from base_date using workflow."""
        if not tickers:
            return pd.DataFrame()

        try:
            result = await fetch_volume_data(tickers, base_date)
            volume_data = result.get("data")

            if volume_data is None or volume_data.empty:
                return pd.DataFrame()

            return volume_data

        except Exception:
            return pd.DataFrame()

    async def get_rsi_data(
        self, tickers: List[str], base_date: datetime
    ) -> pd.DataFrame:
        """Get RSI data for tickers from base_date using workflow."""
        if not tickers:
            return pd.DataFrame()

        try:
            result = await fetch_rsi_data(tickers, base_date)
            rsi_data = result.get("data")

            if rsi_data is None or rsi_data.empty:
                return pd.DataFrame()

            return rsi_data

        except Exception:
            return pd.DataFrame()

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
                # For MAX option, use a very old date
                base_date = datetime(2000, 1, 1)
                async with self:
                    yield rx.toast.info("Loading maximum available data...")
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")
                async with self:
                    yield rx.toast.info(
                        f"Loading data from {self.base_date_option} ({base_date.strftime('%Y-%m-%d')})"
                    )

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
                    f"Data: {returns_data.shape[0]} days, {returns_data.shape[1]} tickers"
                )

            # Create plotly chart
            fig = go.Figure()

            # Plot each ticker
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
            ]

            for i, ticker in enumerate(returns_data.columns):
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=returns_data.index,
                        y=returns_data[ticker],
                        mode="lines",
                        name=ticker,
                        line=dict(color=color, width=2),
                        hovertemplate=f"<b>{ticker}</b><br>"
                        # + "Date: %{x}<br>"
                        + "Return: %{y:.2f}%<br>" + "<extra></extra>",
                    )
                )

            # Get theme-appropriate colors
            theme_colors = self.get_theme_colors()

            # Update layout
            title = f"Returns Comparison ({', '.join(self.selected_tickers)})"
            layout_props = {
                "title": title,
                "xaxis_title": "Date",
                "yaxis_title": "Return (%)",
                "hovermode": "x unified",
                "showlegend": True,
                "height": 300,
                "margin": dict(l=50, r=50, t=80, b=50),
                "plot_bgcolor": theme_colors["plot_bgcolor"],
                "paper_bgcolor": theme_colors["paper_bgcolor"],
                "hoverlabel": dict(
                    bgcolor=theme_colors["hover_bgcolor"],
                    bordercolor=theme_colors["hover_bordercolor"],
                    font_size=14,
                    font_color="white",  # White text on dark background
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
                self.chart_figure_returns = fig
                yield rx.toast.success(
                    f"Returns chart updated with {len(returns_data.columns)} tickers"
                )

        except Exception:
            async with self:
                self.chart_figure_returns = go.Figure()
                yield rx.toast.error("Returns chart update failed")
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
                base_date = datetime(2000, 1, 1)
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

            # Create plotly chart
            fig = go.Figure()

            # Plot each ticker
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
            ]

            for i, ticker in enumerate(volatility_data.columns):
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=volatility_data.index,
                        y=volatility_data[ticker],
                        mode="lines",
                        name=ticker,
                        line=dict(color=color, width=2),
                        hovertemplate=f"<b>{ticker}</b><br>"
                        # + "Date: %{x}<br>"
                        + "Volatility: %{y:.2f}%<br>" + "<extra></extra>",
                    )
                )

            # Get theme-appropriate colors
            theme_colors = self.get_theme_colors()

            # Update layout
            title = f"Volatility Comparison ({', '.join(self.selected_tickers)})"
            layout_props = {
                "title": title,
                "xaxis_title": "Date",
                "yaxis_title": "Volatility (%)",
                "hovermode": "x unified",
                "showlegend": True,
                "height": 300,
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
                self.chart_figure_volatility = fig

        except Exception:
            async with self:
                self.chart_figure_volatility = go.Figure()
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
                base_date = datetime(2000, 1, 1)
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            # Get volume data
            volume_data = await self.get_volume_data(self.selected_tickers, base_date)

            if volume_data is None or volume_data.empty:
                async with self:
                    self.chart_figure_volume = go.Figure()
                return

            # Create plotly chart
            fig = go.Figure()

            # Plot each ticker
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
            ]

            for i, ticker in enumerate(volume_data.columns):
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=volume_data.index,
                        y=volume_data[ticker],
                        mode="lines",
                        name=ticker,
                        line=dict(color=color, width=2),
                        hovertemplate=f"<b>{ticker}</b><br>"
                        # + "Date: %{x}<br>"
                        + "Volume: %{y:,.0f}<br>" + "<extra></extra>",
                    )
                )

            # Get theme-appropriate colors
            theme_colors = self.get_theme_colors()

            # Update layout
            title = f"Volume Comparison ({', '.join(self.selected_tickers)})"
            layout_props = {
                "title": title,
                "xaxis_title": "Date",
                "yaxis_title": "Volume",
                "hovermode": "x unified",
                "showlegend": True,
                "height": 300,
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
                self.chart_figure_volume = fig

        except Exception:
            async with self:
                self.chart_figure_volume = go.Figure()
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
                base_date = datetime(2000, 1, 1)
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            # Get RSI data
            rsi_data = await self.get_rsi_data(self.selected_tickers, base_date)

            if rsi_data is None or rsi_data.empty:
                async with self:
                    self.chart_figure_rsi = go.Figure()
                return

            # Create plotly chart
            fig = go.Figure()

            # Plot each ticker
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
            ]

            for i, ticker in enumerate(rsi_data.columns):
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=rsi_data.index,
                        y=rsi_data[ticker],
                        mode="lines",
                        name=ticker,
                        line=dict(color=color, width=2),
                        hovertemplate=f"<b>{ticker}</b><br>"
                        # + "Date: %{x}<br>"
                        + "RSI: %{y:.1f}<br>" + "<extra></extra>",
                    )
                )

            # Add reference lines for RSI
            fig.add_hline(
                y=70,
                line_dash="dash",
                line_color="red",
                opacity=0.7,
                annotation_text="Overbought (70)",
            )
            fig.add_hline(
                y=30,
                line_dash="dash",
                line_color="green",
                opacity=0.7,
                annotation_text="Oversold (30)",
            )

            # Get theme-appropriate colors
            theme_colors = self.get_theme_colors()

            # Update layout
            title = f"RSI Comparison ({', '.join(self.selected_tickers)})"
            layout_props = {
                "title": title,
                "xaxis_title": "Date",
                "yaxis_title": "RSI",
                "hovermode": "x unified",
                "showlegend": True,
                "height": 300,
                "margin": dict(l=50, r=50, t=80, b=50),
                "plot_bgcolor": theme_colors["plot_bgcolor"],
                "paper_bgcolor": theme_colors["paper_bgcolor"],
                "hoverlabel": dict(
                    bgcolor=theme_colors["hover_bgcolor"],
                    bordercolor=theme_colors["hover_bordercolor"],
                    font_size=14,
                    font_color="white",
                ),
                "yaxis": dict(range=[0, 100]),  # RSI is always 0-100
            }

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
                self.chart_figure_rsi = fig

        except Exception:
            async with self:
                self.chart_figure_rsi = go.Figure()
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
        today = datetime.now()

        if self.base_date_option == "1W":
            base_date = today - timedelta(weeks=1)
        elif self.base_date_option == "2W":
            base_date = today - timedelta(weeks=2)
        elif self.base_date_option == "1M":
            base_date = today - timedelta(days=30)
        elif self.base_date_option == "2M":
            base_date = today - timedelta(days=60)
        elif self.base_date_option == "1Q":
            base_date = today - timedelta(days=90)
        elif self.base_date_option == "2Q":
            base_date = today - timedelta(days=180)
        elif self.base_date_option == "3Q":
            base_date = today - timedelta(days=270)
        elif self.base_date_option == "1Y":
            base_date = today - timedelta(days=365)
        elif self.base_date_option == "2Y":
            base_date = today - timedelta(days=730)
        elif self.base_date_option == "3Y":
            base_date = today - timedelta(days=1095)
        elif self.base_date_option == "4Y":
            base_date = today - timedelta(days=1460)
        elif self.base_date_option == "5Y":
            base_date = today - timedelta(days=1825)
        elif self.base_date_option == "10Y":
            base_date = today - timedelta(days=3650)
        elif self.base_date_option == "YTD":
            base_date = datetime(today.year, 1, 1)
        else:  # MAX
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
                on_click=CompareState.toggle_favorite(ticker),
            ),
            rx.button(
                rx.icon("minus", size=16),
                size="2",
                variant="solid",
                color_scheme="red",
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


def metrics_tab_content() -> rx.Component:
    """Metrics tab - cleared for simplification."""
    return rx.center(
        rx.text("Metrics tab cleared - ready for new implementation", color="gray"),
        height="400px",
    )


def plots_asset_returns() -> rx.Component:
    """Plotly chart for asset returns."""
    return rx.cond(
        CompareState.loading_returns,
        rx.center(rx.spinner(), height="300px"),
        rx.cond(
            CompareState.selected_tickers.length(),
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
            CompareState.selected_tickers.length(),
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
            CompareState.selected_tickers.length(),
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
            CompareState.selected_tickers.length(),
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
            rx.text("Base:", font_weight="bold"),
            rx.select(
                CompareState.base_date_options,
                value=CompareState.base_date_option,
                on_change=CompareState.set_base_date,
            ),
            spacing="2",
            align="center",
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
