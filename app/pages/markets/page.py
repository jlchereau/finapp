"""
Markets page
"""

from typing import List
from datetime import datetime

import reflex as rx

from app.lib.periods import (
    get_period_default,
    get_period_options,
    calculate_base_date,
    get_max_fallback_date,
)
from app.templates.template import template

from .msci_world import msci_world, update_msci_world
from .buffet_indicator import buffet_indicator, update_buffet_indicator
from .vix_chart import vix_chart, update_vix_chart
from .yield_curve import yield_curve, update_yield_curve
from .currency_chart import currency_chart, update_currency_chart
from .precious_metals import precious_metals, update_precious_metals
from .crypto_chart import crypto_chart, update_crypto_chart
from .crude_oil import crude_oil, update_crude_oil
from .bloomberg_commodity import bloomberg_commodity, update_bloomberg_commodity

# from .placeholder_charts import (
#     fear_and_greed_chart,
#     shiller_cape_chart,
#     update_fear_and_greed,
#     update_shiller_cape,
# )


# pylint: disable=inherit-non-class
class PageState(rx.State):
    """The markets page state."""

    active_tab: rx.Field[str] = rx.field("overview")

    # Chart settings
    period_option: rx.Field[str] = rx.field(default_factory=get_period_default)
    period_options: rx.Field[List[str]] = rx.field(default_factory=get_period_options)

    @rx.event
    def set_active_tab(self, tab: str):
        """Switch between metrics and plot tabs."""
        self.active_tab = tab

    @rx.var
    def base_date(self) -> datetime:
        """
        Convert periodoption to actual datetime.
        See https://reflex.dev/docs/vars/computed-vars/
        """
        base_date = calculate_base_date(self.period_option)
        if base_date is None:
            # For MAX option, use appropriate fallback date
            return get_max_fallback_date("markets")
        return base_date

    @rx.event
    def set_period_option(self, option: str):
        """Set base date option and update all market charts."""
        self.period_option = option
        for w in self.run_workflows():
            yield w

    @rx.event
    def run_workflows(self):
        """Update data for all charts and metrics."""
        base_date = self.base_date
        return [
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_msci_world(base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_buffet_indicator(base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_vix_chart(base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_yield_curve(base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_currency_chart(base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_precious_metals(base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_crypto_chart(base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_crude_oil(base_date=base_date),
            # pylint: disable=no-value-for-parameter
            # pyrefly: ignore[no-matching-overload]
            update_bloomberg_commodity(base_date=base_date),
        ]


def tabs_overview() -> rx.Component:
    """Overview tab content."""
    return rx.vstack(
        rx.hstack(
            rx.text("Period:", font_weight="bold"),
            rx.select(
                PageState.period_options,
                value=PageState.period_option,
                on_change=PageState.set_period_option,
            ),
            spacing="2",
            align="center",
            justify="start",
        ),
        rx.hstack(
            rx.grid(
                rx.card(msci_world()),
                rx.card(precious_metals()),
                rx.card(bloomberg_commodity()),
                rx.card(crypto_chart()),
                columns="2",
                spacing="3",
                width="70%",
            ),
            rx.card(rx.text("News feed"), width="30%"),
            width="100%",
        ),
    )


def tabs_us() -> rx.Component:
    """US tab content."""
    return rx.vstack(
        rx.hstack(
            rx.text("Period:", font_weight="bold"),
            rx.select(
                PageState.period_options,
                value=PageState.period_option,
                on_change=PageState.set_period_option,
            ),
            spacing="2",
            align="center",
            justify="start",
        ),
        rx.grid(
            rx.card(buffet_indicator()),
            rx.card(vix_chart()),
            rx.card(yield_curve()),
            rx.card(currency_chart()),
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
                PageState.period_options,
                value=PageState.period_option,
                on_change=PageState.set_period_option,
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


def tabs_commodities() -> rx.Component:
    """Commodities tab content."""
    return rx.vstack(
        rx.hstack(
            rx.text("Period:", font_weight="bold"),
            rx.select(
                PageState.period_options,
                value=PageState.period_option,
                on_change=PageState.set_period_option,
            ),
            spacing="2",
            align="center",
            justify="start",
        ),
        rx.grid(
            rx.card(crude_oil()),
            rx.card(precious_metals()),
            rx.card(crypto_chart()),
            rx.card(bloomberg_commodity()),
            columns="2",
            spacing="3",
            width="100%",
        ),
    )


# pylint: disable=not-callable
# pyright: ignore[reportArgumentType]
# pyrefly: ignore[bad-argument-type]
@rx.page(route="/markets", on_load=PageState.run_workflows)
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
            value=PageState.active_tab,
            on_change=PageState.set_active_tab,
            width="100%",
        ),
        # height="100%",
        spacing="0",
    )
