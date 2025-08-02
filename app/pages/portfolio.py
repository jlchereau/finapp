""" Portfolio page """

import reflex as rx
from ..templates.template import template


class PortfolioState(rx.State):  # pylint: disable=inherit-non-class
    """The portfolio page state."""
    pass


@rx.page(route="/portfolio")  # pyright: ignore[reportArgumentType]
@template
def page():
    """The portfolio page."""
    return rx.vstack(
        rx.heading("Portfolio", size="9"),
        spacing="5",
        justify="center",
        min_height="85vh"
    )
