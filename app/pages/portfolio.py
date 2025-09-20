"""Portfolio page"""

import reflex as rx
from app.templates.template import template


# pylint: disable=inherit-non-class
class PortfolioState(rx.State):
    """The portfolio page state."""


# pylint: disable=not-callable
# pyright: ignore[reportArgumentType]
# pyrefly: ignore[bad-argument-type]
@rx.page(route="/portfolio")
@template
def page():
    """The portfolio page."""
    return rx.vstack(
        rx.heading("Portfolio", size="6", margin_bottom="1rem"),
        # rx.hstack(),
        # height="100%",
        spacing="0",
    )
