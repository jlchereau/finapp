"""
Backtest page
"""

import reflex as rx
from ..templates.template import template


class State(rx.State):
    """The app state."""
    pass


@rx.page(route="/backtest")
@template
def portfolio():
    """The backtest page."""
    return rx.vstack(
        rx.heading("Backtest", size="9"),
        spacing="5",
        justify="center",
        min_height="85vh",
    )
