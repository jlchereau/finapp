"""
Markets page
"""

import reflex as rx
from ..templates.template import template
from ..components.combobox import combobox


class State(rx.State):
    """The app state."""

    # Sample market data for combobox testing
    markets: list[str] = [
        "SPY - SPDR S&P 500 ETF",
        "QQQ - Invesco QQQ Trust", 
        "VTI - Vanguard Total Stock Market",
        "AAPL - Apple Inc.",
        "MSFT - Microsoft Corporation",
        "GOOGL - Alphabet Inc.",
        "AMZN - Amazon.com Inc.",
        "TSLA - Tesla Inc.",
    ]

    selected_market: str = ""


@rx.page(route="/markets")
@template
def portfolio():
    """The markets page."""
    return rx.vstack(
        rx.heading("Markets", size="9"),
        rx.text("Test Combobox Component:", size="4", margin_top="6"),
        combobox(
            options=State.markets,
            placeholder="Search markets...",
            aria_label="Select market",
            class_name="w-96",
        ),
        rx.cond(
            State.selected_market,
            rx.box(
                rx.text("Selected Market:", size="3", weight="medium"),
                rx.text(State.selected_market, size="2"),
                margin_top="4",
                padding="4",
                border="1px solid",
                border_color="zinc.200",
                border_radius="md",
            )
        ),
        spacing="5",
        justify="center",
        min_height="85vh",
        align="center",
    )
