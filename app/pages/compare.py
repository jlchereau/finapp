"""
Compare page - Stock comparison and analysis
"""

from typing import List, Dict, Any, Optional
import io
import base64
from datetime import datetime, timedelta

import reflex as rx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ..templates.template import template
from ..models.ticker import TickerModel


class CompareState(rx.State):  # pylint: disable=inherit-non-class
    """State management for the compare page."""

    # Ticker input and selection
    ticker_input: str = ""
    selected_tickers: List[str] = []
    favorites: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]

    # Chart settings
    active_tab: str = "plot"
    base_date_option: str = "1Y"
    base_date_options: List[str] = ["1Y", "2Y", "5Y", "10Y", "YTD", "MAX"]

    # Chart data
    chart_image: str = ""

    # Metrics data - simplified for Reflex
    metrics_data: List[Dict[str, str]] = []

    # Loading states
    loading_chart: bool = False
    loading_metrics: bool = False

    @property
    def ticker_model(self) -> TickerModel:
        """Get ticker model instance."""
        if not hasattr(self, '_ticker_model'):
            self._ticker_model = TickerModel()
        return self._ticker_model

    def add_ticker(self):
        """Add ticker to selected list."""
        if self.ticker_input and self.ticker_input.upper() not in self.selected_tickers:
            self.selected_tickers.append(self.ticker_input.upper())
            self.ticker_input = ""
            self.update_chart()

    def remove_ticker(self, ticker: str):
        """Remove ticker from selected list."""
        if ticker in self.selected_tickers:
            self.selected_tickers.remove(ticker)
            self.update_chart()

    def select_from_favorites(self, ticker: str):
        """Select ticker from favorites dropdown."""
        self.ticker_input = ticker

    def toggle_favorite(self, ticker: str):
        """Toggle ticker in favorites list."""
        if ticker in self.favorites:
            self.favorites.remove(ticker)
        else:
            self.favorites.append(ticker)

    def set_active_tab(self, tab: str):
        """Switch between metrics and plot tabs."""
        self.active_tab = tab
        if tab == "metrics":
            self.update_metrics()

    def set_base_date(self, option: str):
        """Set base date option and update chart."""
        self.base_date_option = option
        self.update_chart()

    def update_chart(self):
        """Update the matplotlib chart."""
        if not self.selected_tickers:
            self.chart_image = ""
            return

        self.loading_chart = True
        yield

        try:
            # Calculate base date
            base_date = self._get_base_date()

            # Get price data
            price_data = self.ticker_model.get_price_data(
                self.selected_tickers, base_date
            )

            if price_data.empty:
                self.chart_image = ""
                return

            # Create matplotlib chart
            plt.style.use("default")
            _, ax = plt.subplots(figsize=(12, 8))

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
                ax.plot(
                    price_data.index,
                    price_data[ticker],
                    label=ticker,
                    linewidth=2,
                    color=color,
                )

            # Formatting
            ax.set_xlabel("Date")
            ax.set_ylabel("Return")
            title = f"Price Comparison ({', '.join(self.selected_tickers)})"
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.xticks(rotation=45)

            plt.tight_layout()

            # Convert to base64 image
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            self.chart_image = f"data:image/png;base64,{image_base64}"

        except Exception as e:
            print(f"Chart update error: {e}")
            self.chart_image = ""
        finally:
            self.loading_chart = False

    def update_metrics(self):
        """Update Graham metrics for selected tickers."""
        if not self.selected_tickers:
            self.metrics_data = []
            return

        self.loading_metrics = True
        yield

        try:
            flattened_metrics = []
            for ticker in self.selected_tickers:
                metrics = self.ticker_model.calculate_graham_metrics(ticker)
                if metrics:
                    for metric_name, metric_data in metrics.items():
                        flattened_metrics.append(
                            {
                                "ticker": ticker,
                                "metric": metric_name,
                                "value": f"{metric_data['value']:.2f}"
                                + ("%" if metric_name == "Dividend Yield" else ""),
                                "criterion": metric_data["criterion"],
                                "status": "✓" if metric_data["passes"] else "✗",
                                "status_color": (
                                    "green" if metric_data["passes"] else "red"
                                ),
                            }
                        )

            self.metrics_data = flattened_metrics
        except Exception as e:
            print(f"Metrics update error: {e}")
        finally:
            self.loading_metrics = False

    def _get_base_date(self) -> Optional[str]:
        """Convert base date option to actual date string."""
        today = datetime.now()

        if self.base_date_option == "1Y":
            base_date = today - timedelta(days=365)
        elif self.base_date_option == "2Y":
            base_date = today - timedelta(days=730)
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
            rx.input(
                placeholder="Enter ticker (e.g., MSFT)",
                value=CompareState.ticker_input,
                on_change=CompareState.set_ticker_input,
                width="150px",
            ),
            rx.select(
                CompareState.favorites,
                placeholder="Favorites",
                on_change=CompareState.select_from_favorites,
                width="120px",
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
                size="1",
                variant="ghost",
                color_scheme="red",
                on_click=lambda: CompareState.toggle_favorite(ticker),
            ),
            rx.button(
                "-",
                size="1",
                variant="ghost",
                color_scheme="red",
                on_click=lambda: CompareState.remove_ticker(ticker),
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
        border_color=rx.color("gray", 6),
        height="calc(100vh - 120px)",
        overflow_y="auto",
        spacing="4",
    )


def metrics_tab_content() -> rx.Component:
    """Metrics tab showing Graham's defensive investor criteria."""
    return rx.cond(
        CompareState.loading_metrics,
        rx.spinner(),
        rx.cond(
            CompareState.metrics_data,
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("Ticker"),
                        rx.table.column_header_cell("Metric"),
                        rx.table.column_header_cell("Value"),
                        rx.table.column_header_cell("Criterion"),
                        rx.table.column_header_cell("Status"),
                    ),
                ),
                rx.table.body(
                    rx.foreach(
                        CompareState.metrics_data,
                        lambda metric: rx.table.row(
                            rx.table.cell(metric["ticker"]),
                            rx.table.cell(metric["metric"]),
                            rx.table.cell(metric["value"]),
                            rx.table.cell(metric["criterion"]),
                            rx.table.cell(
                                rx.badge(
                                    metric["status"],
                                    color_scheme=metric["status_color"],
                                )
                            ),
                        ),
                    ),
                ),
                width="100%",
            ),
            rx.text("Select tickers to view metrics", color="gray"),
        ),
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
                CompareState.chart_image,
                rx.image(
                    src=CompareState.chart_image,
                    width="100%",
                    height="auto",
                    border_radius="md",
                ),
                rx.center(
                    rx.text("Select tickers to view comparison chart", color="gray"),
                    height="400px",
                ),
            ),
        ),
        spacing="4",
        width="100%",
    )


def main_content() -> rx.Component:
    """Main content area with tabs."""
    return rx.vstack(
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("Metrics", value="metrics"),
                rx.tabs.trigger("Plot", value="plot"),
            ),
            rx.tabs.content(
                metrics_tab_content(),
                value="metrics",
            ),
            rx.tabs.content(
                plot_tab_content(),
                value="plot",
            ),
            value=CompareState.active_tab,
            on_change=CompareState.set_active_tab,
            width="100%",
        ),
        flex="1",
        padding="1em",
        width="100%",
    )


@rx.page(
    route="/compare",
    on_load=CompareState.update_chart,  # pyright: ignore[reportArgumentType]
)
@template
def page():
    """The compare page."""
    return rx.hstack(
        left_sidebar(),
        main_content(),
        width="100%",
        height="calc(100vh - 80px)",
        spacing="0",
    )
