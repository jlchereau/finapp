"""
Backtest page
"""

import reflex as rx
from app.templates.template import template


# pylint: disable=inherit-non-class
class State(rx.State):
    """The app state."""


# pylint: disable=not-callable
# pyright: ignore[reportArgumentType]
# pyrefly: ignore[bad-argument-type]
@rx.page(route="/backtest")
@template
def page():
    """The backtest page."""
    return rx.vstack(
        rx.heading("Backtest", size="6", margin_bottom="1rem"),
        # rx.hstack(),
        # height="100%",
        spacing="0",
    )
