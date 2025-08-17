"""Portfolio page"""

import reflex as rx
from ..templates.template import template


class PortfolioState(rx.State):  # pylint: disable=inherit-non-class
    """The portfolio page state."""


# pylint: disable=not-callable
@rx.page(route="/portfolio")  # pyright: ignore[reportArgumentType]
@template
def page():
    """The portfolio page."""
    return rx.vstack(
        rx.heading("Portfolio", size="6", margin_bottom="1rem"),
        # rx.hstack(),
        # height="100%",
        spacing="0",
    )
