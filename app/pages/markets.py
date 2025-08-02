"""
Markets page
"""

import reflex as rx
from ..templates.template import template


class State(rx.State):  # pylint: disable=inherit-non-class
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


@rx.page(route="/markets")  # pyright: ignore[reportArgumentType]
@template
def portfolio():
    """The markets page."""
    return rx.vstack(
        rx.heading("Markets", size="9"),
        rx.text("Test Combobox Component:", size="4", margin_top="6"),
        spacing="5",
        justify="center",
        min_height="85vh",
    )
