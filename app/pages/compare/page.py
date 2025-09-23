# pylint: disable=too-many-lines
"""
Compare page - Stock comparison and analysis
"""

from datetime import datetime
from typing import List

import reflex as rx

from app.components.combobox import combobox_wrapper as combobox
from app.lib.periods import (
    get_period_default,
    get_period_options,
    calculate_base_date,
    get_max_fallback_date,
)
from app.templates.template import template

# Import all components
from .returns_chart import returns_chart, update_returns_chart
from .volatility_chart import volatility_chart, update_volatility_chart
from .volume_chart import volume_chart, update_volume_chart
from .rsi_chart import rsi_chart, update_rsi_chart
from .metrics import metrics, update_metrics


# pylint: disable=inherit-non-class
class CompareState(rx.State):
    """State management for the compare page."""

    # Currency selection
    currency: rx.Field[str] = rx.field("USD")
    currencies: rx.Field[List[str]] = rx.field(
        default_factory=lambda: ["USD", "EUR", "GBP"]
    )

    # Ticker input and selection
    ticker_input: rx.Field[str] = rx.field("")
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

    @rx.var
    def base_date(self) -> datetime:
        """Convert period option to actual datetime."""
        base_date = calculate_base_date(self.period_option)
        if base_date is None:
            # For MAX option, use appropriate fallback date
            return get_max_fallback_date("stocks")
        return base_date

    @rx.event
    def set_currency(self, value: str):
        """Set the currency for the comparison."""
        self.currency = value

    @rx.event
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
        yield CompareState.run_workflows

    @rx.event
    def remove_ticker(self, ticker: str):
        """Remove ticker from selected list."""
        if ticker in self.selected_tickers:
            self.selected_tickers.remove(ticker)
            yield rx.toast.info(f"Removed {ticker} from comparison")
            yield CompareState.run_workflows
        else:
            yield rx.toast.warning(f"{ticker} not found in comparison list")

    @rx.event
    def set_ticker_input(self, value: str | None):
        """Set ticker input value."""
        self.ticker_input = value or ""

    def toggle_favorite(self, ticker: str):
        """Toggle ticker in favorites list."""
        if ticker in self.favorites:
            self.favorites.remove(ticker)
        else:
            self.favorites.append(ticker)

    @rx.event
    def set_active_tab(self, tab: str):
        """Switch between metrics and plot tabs."""
        self.active_tab = tab

    @rx.event
    def set_period_option(self, option: str):
        """Set base date option and update all charts."""
        self.period_option = option
        yield rx.toast.info(f"Changed time period to {option}")
        yield CompareState.run_workflows

    @rx.event
    def run_workflows(self):
        """Update data for all charts and metrics."""
        tickers = self.selected_tickers
        base_date = self.base_date
        return [
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_returns_chart(tickers=tickers, base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_volatility_chart(tickers=tickers, base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_volume_chart(tickers=tickers, base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_rsi_chart(tickers=tickers, base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_metrics(tickers=tickers, base_date=base_date),
        ]


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
        overflow_y="auto",
        spacing="4",
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
            rx.card(returns_chart()),
            rx.card(rsi_chart()),
            rx.card(volatility_chart()),
            rx.card(volume_chart()),
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
                metrics(),
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
# pyrefly: ignore[bad-argument-type]
@rx.page(route="/compare", on_load=CompareState.run_workflows)
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
            width="100%",
            spacing="0",
        ),
        spacing="0",
    )
