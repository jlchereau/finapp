"""
Compare page - Stock comparison and analysis
"""

from typing import List, Optional
from datetime import datetime, timedelta

import reflex as rx
import plotly.graph_objects as go

from ..components.combobox import combobox_wrapper as combobox
from ..models.yahoo import create_yahoo_history_provider
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
    base_date_options: List[str] = ["1W", "2W", "1M", "2M", "1Q", "2Q", "3Q", "1Y", "2Y", "3Y", "4Y", "5Y", "10Y", "YTD", "MAX"]

    # Chart data
    chart_figure: go.Figure = go.Figure()

    # Loading states
    loading_chart: bool = False

    @property
    def history_provider(self):
        """Get Yahoo history provider instance."""
        if not hasattr(self, "_history_provider"):
            self._history_provider = create_yahoo_history_provider()
        return self._history_provider

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

    async def get_price_data(self, tickers: List[str], base_date: datetime):
        """Get normalized price data for tickers from base_date."""
        if not tickers:
            return None

        import pandas as pd
        import yfinance as yf

        try:
            # Calculate period from base_date to now
            end_date = datetime.now()

            # Download data for all tickers
            data = yf.download(
                tickers,
                start=base_date,
                end=end_date,
                group_by="ticker" if len(tickers) > 1 else None,
                auto_adjust=True,
                prepost=True,
                threads=True,
            )

            if data.empty:
                return pd.DataFrame()

            # Normalize the data for comparison (set first value as 100)
            normalized_data = pd.DataFrame()

            if len(tickers) == 1:
                # Single ticker case - yfinance returns different structure for single ticker
                if "Close" in data.columns:
                    close_prices = data["Close"].dropna()
                    if not close_prices.empty:
                        normalized_data[tickers[0]] = (
                            (close_prices / close_prices.iloc[0]) - 1
                        ) * 100
                elif "Adj Close" in data.columns:
                    # Fallback to Adj Close if Close is not available
                    close_prices = data["Adj Close"].dropna()
                    if not close_prices.empty:
                        normalized_data[tickers[0]] = (
                            (close_prices / close_prices.iloc[0]) - 1
                        ) * 100
            else:
                # Multiple tickers case
                for ticker in tickers:
                    try:
                        close_prices = None
                        if hasattr(data, "columns"):
                            if (ticker, "Close") in data.columns:
                                close_prices = data[(ticker, "Close")].dropna()
                            elif (
                                ticker in data.columns
                                and hasattr(data[ticker], "columns")
                                and "Close" in data[ticker].columns
                            ):
                                close_prices = data[ticker]["Close"].dropna()

                        if close_prices is not None and not close_prices.empty:
                            normalized_data[ticker] = (
                                (close_prices / close_prices.iloc[0]) - 1
                            ) * 100
                    except (KeyError, IndexError, AttributeError):
                        continue

            return normalized_data

        except Exception as e:
            print(f"Error fetching price data: {e}")
            return pd.DataFrame()

    def add_ticker(self):
        """Add ticker to selected list."""
        if self.ticker_input and self.ticker_input.upper() not in self.selected_tickers:
            self.selected_tickers.append(self.ticker_input.upper())
            self.ticker_input = ""
            yield CompareState.update_chart

    def remove_ticker(self, ticker: str):
        """Remove ticker from selected list."""
        if ticker in self.selected_tickers:
            self.selected_tickers.remove(ticker)
            yield CompareState.update_chart

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
        """Set base date option and update chart."""
        self.base_date_option = option
        yield CompareState.update_chart

    @rx.event(background=True)
    async def update_chart(self):
        """Update the plotly chart using background processing."""
        if not self.selected_tickers:
            async with self:
                self.chart_figure = go.Figure()
            return

        async with self:
            self.loading_chart = True

        try:
            # Calculate base date
            base_date = self._get_base_date()
            if base_date is None:
                # For MAX option, use a very old date
                base_date = datetime(2000, 1, 1)
            else:
                base_date = datetime.strptime(base_date, "%Y-%m-%d")

            # Get price data
            price_data = await self.get_price_data(self.selected_tickers, base_date)

            if price_data is None or price_data.empty:
                async with self:
                    self.chart_figure = go.Figure()
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

            for i, ticker in enumerate(price_data.columns):
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data[ticker],
                        mode="lines",
                        name=ticker,
                        line=dict(color=color, width=2),
                        hovertemplate=f"<b>{ticker}</b><br>"
                        + "Date: %{x}<br>"
                        + "Return: %{y:.2f}%<br>"
                        + "<extra></extra>",
                    )
                )

            # Get theme-appropriate colors
            theme_colors = self.get_theme_colors()
            
            # Update layout
            title = f"Price Comparison ({', '.join(self.selected_tickers)})"
            layout_props = {
                "title": title,
                "xaxis_title": "Date",
                "yaxis_title": "Return (%)",
                "hovermode": "x unified",
                "showlegend": True,
                "height": 600,
                "margin": dict(l=50, r=50, t=80, b=50),
                "plot_bgcolor": theme_colors["plot_bgcolor"],
                "paper_bgcolor": theme_colors["paper_bgcolor"],
                "hoverlabel": dict(
                    bgcolor=theme_colors["hover_bgcolor"],
                    bordercolor=theme_colors["hover_bordercolor"],
                    font_size=14,
                    font_color="white",  # White text on dark semi-transparent background
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
                self.chart_figure = fig

        except Exception as e:
            print(f"Chart update error: {e}")
            async with self:
                self.chart_figure = go.Figure()
        finally:
            async with self:
                self.loading_chart = False

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


def plot_tab_content() -> rx.Component:
    """Plot tab showing price comparison chart."""
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
        rx.cond(
            CompareState.loading_chart,
            rx.center(rx.spinner(), height="400px"),
            rx.cond(
                CompareState.selected_tickers.length(),
                rx.plotly(
                    data=CompareState.chart_figure,
                    width="100%",
                    height="600px",
                ),
                rx.center(
                    rx.text("Select tickers to view comparison chart", color="gray"),
                    height="400px",
                ),
            ),
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
                plot_tab_content(),
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
    on_load=CompareState.update_chart,  # pyright: ignore[reportArgumentType]
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
